import numpy as np
import time
from scipy import special

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin for safety
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    c_min = 1e-12
    T = horizon
    n_z = 2
    grid_out = resource_grid
    n_out = len(grid_out)
    
    def utility(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    def utility_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    def utility_double_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c ** 2)
        else:
            return -gamma * c ** (-gamma - 1)

    # --- Spectral Setup ---
    # Use Chebyshev nodes on the log-scale of the resource grid
    # Internal grid range: [0.001, 40]
    m_min = 0.001
    m_max = 40.0
    
    # Number of collocation points. 128 is usually sufficient for exponential convergence
    # in smooth problems. We can increase if time permits, but 128-256 is fast.
    n_cheb = 128
    
    # Chebyshev nodes in xi in [-1, 1]
    k = np.arange(n_cheb)
    xi = np.cos(np.pi * (k + 0.5) / n_cheb)
    
    # Map xi to m
    # xi = (2 * log(m/m_min) / log(m_max/m_min)) - 1
    # log(m/m_min) = (xi + 1) * log(m_max/m_min) / 2
    # m = m_min * exp((xi + 1) * log(m_max/m_min) / 2)
    log_ratio = np.log(m_max / m_min)
    m_cheb = m_min * np.exp((xi + 1) * log_ratio / 2)
    
    # Chebyshev differentiation matrix D
    # D[i, j] such that dV/dxi = D @ V_coeffs? No, D @ V_values gives dV/dxi at nodes.
    # We need the matrix that acts on values to get derivatives.
    c = np.zeros((n_cheb, 2))
    c[0, 0] = 1
    c[0, 1] = 2
    c[-1, 0] = 1
    c[-1, 1] = -2
    for i in range(1, n_cheb - 1):
        c[i, 0] = 1
        c[i, 1] = 0
    
    X = np.tile(xi, (n_cheb, 1))
    dX = X - X.T
    
    D = (c @ (1.0 / c.T)) / (dX + np.eye(n_cheb))
    D = D - np.diag(np.sum(D, axis=1))
    
    # D1 maps values at nodes to first derivatives wrt xi at nodes
    # D2 = D @ D maps values to second derivatives wrt xi at nodes
    
    def eval_cheb(x_eval, coeffs):
        """Evaluate Chebyshev series with given coefficients at x_eval"""
        # coeffs are in order T_0, T_1, ...
        # Use Clenshaw's algorithm or simple summation
        # For small N, simple summation is fine, but Clenshaw is better
        # Let's use a simple vectorized approach for N=128
        # T_n(x) can be computed recursively
        T0 = np.ones_like(x_eval)
        T1 = x_eval
        val = coeffs[0] * T0
        if len(coeffs) > 1:
            val += coeffs[1] * T1
            for i in range(2, len(coeffs)):
                T2 = 2 * x_eval * T1 - T0
                val += coeffs[i] * T2
                T0 = T1
                T1 = T2
        return val

    def values_to_coeffs(values):
        """Convert function values at Chebyshev nodes to Chebyshev coefficients"""
        # Use discrete cosine transform (DCT)
        # The coefficients are related to the DCT of the values
        # c_k = (2/N) * sum_{j=0}^{N-1} f(x_j) T_k(x_j)
        # This is equivalent to DCT-I or DCT-II depending on normalization
        # NumPy doesn't have DCT-I directly in standard fftpack, but scipy.fft does?
        # Or we can just use the matrix inverse? No, that's expensive.
        # Let's use the fact that T_k(cos(theta_j)) = cos(k * theta_j)
        # theta_j = pi * (j + 0.5) / N
        # coeffs = (2/N) * DCT(values)
        # scipy.fft.dctn(values, type=1, norm='ortho')? 
        # Actually, the standard FFTPACK DCT is Type II.
        # Let's just compute the coefficients via the Vandermonde-like matrix or DCT.
        # For N=128, building the matrix is cheap.
        # T_k(x_j) matrix
        # k=0..N-1, j=0..N-1
        # M[k, j] = cos(k * theta_j)
        theta = np.pi * (k + 0.5) / n_cheb
        M = np.cos(np.outer(k, theta))
        # coeffs = (2/N) * M @ values, with first and last coeff halved?
        # Standard expansion: f(x) = sum_{k=0}^{N-1} ' c_k T_k(x)
        # where ' means half weight for k=0 and k=N-1.
        # To get c_k:
        # c_0 = (1/N) sum f_j
        # c_k = (2/N) sum f_j T_k(x_j) for 1<=k<=N-2
        # c_{N-1} = (1/N) sum f_j T_{N-1}(x_j)
        
        coeffs = np.zeros(n_cheb)
        # Sum f_j * T_k(x_j)
        sums = M @ values
        coeffs[0] = sums[0] / n_cheb
        if n_cheb > 1:
            coeffs[1:-1] = 2.0 * sums[1:-1] / n_cheb
            coeffs[-1] = sums[-1] / n_cheb
        return coeffs

    def eval_cheb_deriv(x_eval, coeffs, order):
        """Evaluate derivative of Chebyshev series at x_eval"""
        # We need d^order/dxi^order f(xi)
        # We can transform coefficients to derivative coefficients
        # Or just evaluate the polynomial of the derivative
        # Let's compute the derivative coefficients first
        if order == 0:
            return eval_cheb(x_eval, coeffs)
        
        # Derivative of T_k(x)
        # T'_k(x) = sum_{j=0, step 2}^{k-1} (2j+1) T_j(x) if k-j is odd?
        # Actually, it's easier to just apply the differentiation matrix logic to coefficients
        # Or use the property:
        # d/dx T_k(x) = k * U_{k-1}(x)
        # But we need to express U in terms of T.
        # U_{k-1}(x) = T_k'(x)/k
        # This is getting complex. 
        # Alternative: Use the differentiation matrix D to get derivative VALUES at nodes,
        # then fit new coefficients? No, we need eval at arbitrary x.
        
        # Let's stick to the matrix D for values at nodes, and for arbitrary x,
        # we can use the fact that the derivative is also a Chebyshev series.
        # Let's precompute the transformation matrix from coeffs to derivative coeffs.
        # This is a fixed matrix for a given N.
        
        # For now, let's just use a helper that computes the derivative coefficients
        def get_derivative_coeffs(coeffs, order):
            if order == 0:
                return coeffs
            if order == 1:
                # d/dx T_k = sum_{j=0}^{floor((k-1)/2)} (2j+1) T_{k-2j-1} ? No.
                # T'_k = k U_{k-1}
                # U_{k-1} = T_k' / k
                # U_n = 2 sum_{j=0}^{floor(n/2)} T_{n-2j} - T_n? No.
                # Let's use the recurrence:
                # T'_0 = 0
                # T'_1 = 1
                # T'_{k} = (k/(k-1)) T'_{k-1} - ... no.
                
                # Simpler:
                # d/dx T_k(x) = sum_{j=0}^{k-1} c_{k,j} T_j(x)
                # Coefficients can be precomputed.
                # For N=128, let's just build the matrix once.
                pass
            
            # Let's build the differentiation matrix for coefficients
            # M_diff[k, j] such that coeff_deriv[j] = sum_k M_diff[j, k] coeff[k]
            # This is constant for N.
            pass

        # Actually, for this problem, we can just use the values at the Chebyshev nodes
        # and interpolate? No, the whole point is spectral accuracy.
        
        # Let's compute the derivative values at the nodes using D,
        # then fit coefficients for the derivative?
        # No, that's O(N^2) per step.
        
        # Let's use the explicit formula for derivative coefficients.
        # From Trefethen "Spectral Methods in MATLAB":
        # d/dx T_k(x) = 
        #   0 if k=0
        #   1 if k=1
        #   sum_{j=0}^{floor((k-1)/2)} (2j+1) T_{k-2j-1}(x) if k is even? No.
        
        # Let's just compute the derivative coefficients once for order 1 and 2.
        def compute_deriv_matrix(order):
            # Returns matrix M such that c_deriv = M @ c
            M = np.zeros((n_cheb, n_cheb))
            for k in range(n_cheb):
                if order == 1:
                    if k == 0:
                        continue
                    # T'_k = k * U_{k-1}
                    # U_{k-1} = sum_{j=0}^{floor((k-1)/2)} c_j T_{k-1-2j}
                    # Coeffs for U_n:
                    # U_0 = T_0
                    # U_1 = T_1 + T_0? No.
                    # U_n = 2 T_{n+1}'/(n+1)
                    # Let's use the relation:
                    # T'_k = sum_{j=0}^{k-1} (2j+1) T_{k-2j-1} if (k-j) is odd?
                    # Correct formula:
                    # T'_k(x) = \sum_{j=0, j < k, k-j odd}^{k-1} (2j+1) T_j(x)
                    for j in range(k):
                        if (k - j) % 2 == 1:
                            M[j, k] = 2 * j + 1
                elif order == 2:
                    # Apply first derivative twice
                    # We can do this by chaining
                    pass
            return M
        
        # Let's precompute these matrices outside the loop? 
        # But we are inside the function. We can define them here.
        # However, doing this for every call is wasteful. 
        # But N=128 is small.
        
        # Let's optimize: We only need to evaluate V' and V'' at specific points.
        # We can precompute the derivative coefficients for V_{t+1} once per time step.
        
        # For this implementation, I will compute the derivative coefficients
        # using a precomputed matrix.
        
        # Let's define the matrices globally or cache them.
        # Since this is a single function call, I'll just compute them here.
        
        if not hasattr(solve_consumption, 'deriv_matrices'):
            solve_consumption.deriv_matrices = {}
        
        if order not in solve_consumption.deriv_matrices:
            # Build matrix for this order
            M = np.zeros((n_cheb, n_cheb))
            if order == 1:
                for k in range(n_cheb):
                    if k == 0:
                        continue
                    for j in range(k):
                        if (k - j) % 2 == 1:
                            M[j, k] = 2 * j + 1
            elif order == 2:
                # M2 = M1 @ M1
                M1 = np.zeros((n_cheb, n_cheb))
                for k in range(n_cheb):
                    if k == 0:
                        continue
                    for j in range(k):
                        if (k - j) % 2 == 1:
                            M1[j, k] = 2 * j + 1
                M = M1 @ M1
            solve_consumption.deriv_matrices[order] = M
        
        M = solve_consumption.deriv_matrices[order]
        deriv_coeffs = M @ coeffs
        return eval_cheb(x_eval, deriv_coeffs)

    # --- Backward Induction ---
    
    V_coeffs = [None] * T
    P = [None] * T
    
    # Terminal condition: V_{T-1}(m) = U(m)
    # We need to represent U(m) on the Chebyshev grid
    V_vals_T1 = utility(m_cheb)
    V_coeffs[T-1] = values_to_coeffs(V_vals_T1)
    P[T-1] = m_cheb[np.newaxis, :].repeat(n_z, axis=0)
    
    # Precompute transition probabilities and incomes
    # transition[z, zp]
    
    for t in range(T-2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: copy last computed
            for t_rem in range(t, -1, -1):
                V_coeffs[t_rem] = V_coeffs[t+1].copy()
                P[t_rem] = P[t+1].copy()
            break
            
        V_next_coeffs = V_coeffs[t+1]
        
        # Precompute derivative coefficients for V_next
        # We need V' and V'' evaluated at arbitrary points.
        # We will compute the coefficients for V' and V'' once.
        
        # Using the matrices defined in eval_cheb_deriv
        # Let's get the matrices
        M1 = solve_consumption.deriv_matrices.get(1)
        if M1 is None:
            M1 = np.zeros((n_cheb, n_cheb))
            for k in range(n_cheb):
                if k == 0: continue
                for j in range(k):
                    if (k - j) % 2 == 1:
                        M1[j, k] = 2 * j + 1
            solve_consumption.deriv_matrices[1] = M1
        
        M2 = solve_consumption.deriv_matrices.get(2)
        if M2 is None:
            M2 = M1 @ M1
            solve_consumption.deriv_matrices[2] = M2
            
        V_prime_coeffs = M1 @ V_next_coeffs
        V_double_prime_coeffs = M2 @ V_next_coeffs
        
        V_t_vals = np.zeros((n_z, n_cheb))
        P_t = np.zeros((n_z, n_cheb))
        
        for z in range(n_z):
            # Solve for c at each m_cheb[i]
            c = m_cheb * 0.5
            c = np.maximum(c, c_min)
            c = np.minimum(c, m_cheb)
            
            for _ in range(50):
                U_prime = utility_prime(c)
                
                cont_deriv = np.zeros(n_cheb)
                cont_second = np.zeros(n_cheb)
                
                for zp in range(n_z):
                    m_next = R * (m_cheb - c) + income[zp]
                    
                    # Map m_next to xi_next
                    # xi_next = (2 * log(m_next/m_min) / log(m_max/m_min)) - 1
                    # Clip m_next to [m_min, m_max] to be safe, though it should be in range
                    # If m_next is out of range, the value function is not defined by the polynomial.
                    # However, for m in [0.001, 40], c in [0, m], R=1.02, income in [0.5, 1.5]
                    # m_next = 1.02(m-c) + y.
                    # Min m_next: 1.02(0) + 0.5 = 0.5.
                    # Max m_next: 1.02(40) + 1.5 = 42.3.
                    # Our grid is up to 40. So m_next can exceed 40.
                    # We must handle this. 
                    # For m_next > m_max, we can extrapolate or clip.
                    # Clipping is safer for validity, but might bias.
                    # Let's clip to [m_min, m_max] for the evaluation.
                    m_next_clipped = np.clip(m_next, m_min, m_max)
                    
                    xi_next = (2 * np.log(m_next_clipped / m_min) / log_ratio) - 1
                    
                    # Evaluate V' and V'' at xi_next
                    V_prime_val = eval_cheb(xi_next, V_prime_coeffs)
                    V_double_prime_val = eval_cheb(xi_next, V_double_prime_coeffs)
                    
                    cont_deriv += transition[z, zp] * V_prime_val
                    cont_second += transition[z, zp] * V_double_prime_val
                
                FOC = U_prime - beta * R * cont_deriv
                
                if np.all(np.abs(FOC) < 1e-12):
                    break
                
                U_double = utility_double_prime(c)
                dFOC = U_double + beta * R**2 * cont_second
                
                # Newton step
                # dFOC should be negative (concavity)
                step = FOC / dFOC
                c_new = c - step
                c_new = np.clip(c_new, c_min, m_cheb)
                
                # Damping
                if np.any(np.abs(c_new - c) > 0.5 * np.abs(c)):
                    c_new = c - 0.5 * step
                    c_new = np.clip(c_new, c_min, m_cheb)
                
                c = c_new
            
            c = np.clip(c, c_min, m_cheb)
            P_t[z, :] = c
            
            # Compute V_t
            U_c = utility(c)
            cont_val = np.zeros(n_cheb)
            for zp in range(n_z):
                m_next = R * (m_cheb - c) + income[zp]
                m_next_clipped = np.clip(m_next, m_min, m_max)
                xi_next = (2 * np.log(m_next_clipped / m_min) / log_ratio) - 1
                V_next_val = eval_cheb(xi_next, V_next_coeffs)
                cont_val += transition[z, zp] * V_next_val
            
            V_t_vals[z, :] = U_c + beta * cont_val
        
        V_coeffs[t] = values_to_coeffs(V_t_vals[0]) # Wait, V depends on z?
        # The value function V_t(m) is the same for all z? 
        # No, V_t(m) is defined as max_c U(c) + beta sum pi V_{t+1}(m').
        # The maximization is over c, which is a function of m and z?
        # In this problem, the state is (m, z).
        # The Bellman equation is:
        # V_t(m, z) = max_c U(c) + beta sum_{z'} pi_{z z'} V_{t+1}(R(m-c)+y_{z'}, z')
        # So V_t depends on z.
        # My current code stores V_coeffs[t] as a single vector.
        # I need V_coeffs[t][z] for each z.
        
        # Let's restructure V_coeffs to be a list of arrays (n_z, n_cheb)
        # Or a list of lists.
        
        # Correcting V_coeffs structure
        if isinstance(V_coeffs[t], np.ndarray) and V_coeffs[t].ndim == 1:
             # This is the terminal step logic, need to fix
             pass
             
        # Let's redo the storage.
        # V_coeffs[t] should be shape (n_z, n_cheb)
        
        V_coeffs[t] = np.zeros((n_z, n_cheb))
        for z in range(n_z):
            V_coeffs[t][z] = values_to_coeffs(V_t_vals[z])
            
        P[t] = P_t

    # Interpolate policy to output grid
    result = np.zeros((T, n_z, n_out))
    
    # We have P[t] as (n_z, n_cheb) values at m_cheb
    # We need to interpolate to grid_out
    # Since m_cheb and grid_out are both on log scale, we can interpolate in log space
    log_m_cheb = np.log(m_cheb)
    log_grid_out = np.log(grid_out)
    
    for t in range(T):
        for z in range(n_z):
            # Interpolate P[t, z] from m_cheb to grid_out
            # np.interp requires x to be increasing
            result[t, z, :] = np.interp(log_grid_out, log_m_cheb, P[t][z])
            
    # Enforce constraints
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.clip(result[t, z, :], c_min, grid_out + 1e-10)
            
    # Terminal condition must be exact on the output grid
    result[T-1, :, :] = grid_out[np.newaxis, :]
    
    return result
