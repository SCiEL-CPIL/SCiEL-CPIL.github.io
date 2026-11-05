import numpy as np
import time
import sys

def _build_hamiltonians():
    """
    Builds A and B for the 6-site XXZ Heisenberg chain.
    Sites 0..5.
    Bonds: (0,1), (1,2), (2,3), (3,4), (4,5).
    Odd bonds (in 0-indexed pair counting 1-2, 3-4, 5-6? No, standard is 1-2, 2-3 etc).
    Task says: "odd- and even-bond halves".
    Usually bond 1 is between site 1 and 2.
    Let's assume sites 0,1,2,3,4,5.
    Bonds:
    0: (0,1) -> Odd
    1: (1,2) -> Even
    2: (2,3) -> Odd
    3: (3,4) -> Even
    4: (4,5) -> Odd
    
    So A contains bonds 0, 2, 4.
    B contains bonds 1, 3.
    
    Field: Uniform field 0.3.
    "each including half of the uniform field".
    So A has field term 0.15 * sum S^z, B has 0.15 * sum S^z.
    Wait, "half of the uniform field" usually means the field term is split.
    H_field = h * sum S^z_j.
    H_A_field = 0.5 * H_field.
    H_B_field = 0.5 * H_field.
    
    Anisotropy lambda = 0.8.
    """
    N = 6
    dim = 2**N
    lambda_xz = 0.8
    h_field = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    # Spin operators are 0.5 * Pauli
    Sx = 0.5 * sx
    Sy = 0.5 * sy
    Sz = 0.5 * sz
    
    def op_on_site(op, site):
        # Apply op on 'site' (0-indexed), identity on others
        # Tensor product order: site 0 is leftmost or rightmost?
        # Standard quantum info: |q0 q1 ... qn-1>. Operator on site k is I ⊗ ... ⊗ O ⊗ ... ⊗ I.
        # Let's use site 0 as the most significant bit (leftmost in Kronecker product).
        ops = []
        for i in range(N):
            if i == site:
                ops.append(op)
            else:
                ops.append(np.eye(2, dtype=complex))
        # np.kron is associative, fold
        res = ops[0]
        for i in range(1, N):
            res = np.kron(res, ops[i])
        return res
    
    def bond_term(site1, site2):
        # Sx1 Sx2 + Sy1 Sy2 + lambda Sz1 Sz2
        # Note: Sx1 Sx2 = Sx(1) ⊗ Sx(2) ⊗ I...
        # We can construct this by applying ops on both sites.
        # Easier: construct full operator by placing Sx/Sy/Sz on the two sites.
        # Term 1: Sx on s1, Sx on s2
        # Term 2: Sy on s1, Sy on s2
        # Term 3: lambda Sz on s1, Sz on s2
        
        # Helper to place two ops
        def two_site_op(op1, site1, op2, site2):
            ops = []
            for i in range(N):
                if i == site1:
                    ops.append(op1)
                elif i == site2:
                    ops.append(op2)
                else:
                    ops.append(np.eye(2, dtype=complex))
            res = ops[0]
            for i in range(1, N):
                res = np.kron(res, ops[i])
            return res

        return (two_site_op(Sx, site1, Sx, site2) + 
                two_site_op(Sy, site1, Sy, site2) + 
                lambda_xz * two_site_op(Sz, site1, Sz, site2))

    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Bonds
    # 0-1 (Odd) -> A
    # 1-2 (Even) -> B
    # 2-3 (Odd) -> A
    # 3-4 (Even) -> B
    # 4-5 (Odd) -> A
    
    A += bond_term(0, 1)
    B += bond_term(1, 2)
    A += bond_term(2, 3)
    B += bond_term(3, 4)
    A += bond_term(4, 5)
    
    # Field
    # H_field = h * sum Sz_j
    H_field = np.zeros((dim, dim), dtype=complex)
    for j in range(N):
        H_field += op_on_site(Sz, j)
        
    A += 0.5 * h_field * H_field
    B += 0.5 * h_field * H_field
    
    return A, B

def _nelder_mead(f, x0, max_iter=1000, alpha=1.0, gamma=2.0, rho=0.5, sigma=2.0, tol=1e-8, time_limit=None, start_time=None):
    """
    Pure NumPy Nelder-Mead implementation.
    f: objective function
    x0: initial point (numpy array)
    """
    if start_time is None:
        start_time = time.time()
    n = len(x0)
    
    # Initialize simplex
    simplex = [x0.copy()]
    for i in range(n):
        x_new = x0.copy()
        x_new[i] += 0.05 if x_new[i] == 0 else 0.05 * x_new[i]
        simplex.append(x_new)
        
    values = [f(x) for x in simplex]
    
    for it in range(max_iter):
        if time_limit is not None and time.time() - start_time > time_limit:
            break
            
        # Sort simplex by function value
        order = np.argsort(values)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[0] - values[-1]) < tol:
            break
            
        x_best = simplex[0]
        x_worst = simplex[-1]
        x_second_worst = simplex[-2]
        
        # Centroid of all but worst
        x_centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = x_centroid + alpha * (x_centroid - x_worst)
        f_r = f(x_r)
        
        if f_r < values[0]:
            # Expansion
            x_e = x_centroid + gamma * (x_r - x_centroid)
            f_e = f(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                values[-1] = f_e
            else:
                simplex[-1] = x_r
                values[-1] = f_r
        elif f_r < values[-2]:
            # Accept reflection
            simplex[-1] = x_r
            values[-1] = f_r
        else:
            # Contraction
            x_c = x_centroid + rho * (x_worst - x_centroid)
            f_c = f(x_c)
            if f_c < values[-1]:
                simplex[-1] = x_c
                values[-1] = f_c
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex[i] = x_best + sigma * (simplex[i] - x_best)
                    values[i] = f(simplex[i])
                    
    # Return best point
    order = np.argsort(values)
    return simplex[order[0]], values[order[0]]

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    A, B = _build_hamiltonians()
    H = A + B
    
    # Precompute exact propagators
    # Use eigen-decomposition for stability and speed
    eigvals, eigvecs = np.linalg.eigh(H)
    
    times = [0.25, 0.5, 1.0]
    U_exact = []
    for t in times:
        # U = V exp(-i t E) V^H
        phase = np.exp(-1j * t * eigvals)
        U_t = eigvecs @ np.diag(phase) @ eigvecs.conj().T
        U_exact.append(U_t)
        
    # Precompute A and B eigendecomps
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    eigvals_B, eigvecs_B = np.linalg.eigh(B)
    
    def get_exp_A(t_coeff):
        # exp(-i t_coeff A)
        if t_coeff == 0:
            return np.eye(64, dtype=complex)
        phase = np.exp(-1j * t_coeff * eigvals_A)
        return eigvecs_A @ np.diag(phase) @ eigvecs_A.conj().T
        
    def get_exp_B(t_coeff):
        # exp(-i t_coeff B)
        if t_coeff == 0:
            return np.eye(64, dtype=complex)
        phase = np.exp(-1j * t_coeff * eigvals_B)
        return eigvecs_B @ np.diag(phase) @ eigvecs_B.conj().T

    # Fourier-series parameterization for coefficients
    # a_k = 1/12 + sum_{m=1}^{3} [A_m * cos(2*pi*m*k/12) + B_m * sin(2*pi*m*k/12)]
    # b_k = 1/12 + sum_{m=1}^{3} [C_m * cos(2*pi*m*k/12) + D_m * sin(2*pi*m*k/12)]
    # This gives 12 free parameters (4 for a, 4 for b) instead of 22
    # The sum constraint is automatically satisfied since sum of cos/sin over full period is 0
    
    k_vals = np.arange(12)
    # Precompute Fourier basis
    fourier_cos = np.zeros((12, 3))
    fourier_sin = np.zeros((12, 3))
    for m in range(1, 4):
        fourier_cos[:, m-1] = np.cos(2 * np.pi * m * k_vals / 12)
        fourier_sin[:, m-1] = np.sin(2 * np.pi * m * k_vals / 12)
    
    def fourier_to_coeffs(params):
        # params: 12 values [A1, A2, A3, B1, B2, B3, C1, C2, C3, D1, D2, D3]
        a_params = params[:6]
        b_params = params[6:12]
        
        # a_k = 1/12 + A1*cos1 + A2*cos2 + A3*cos3 + B1*sin1 + B2*sin2 + B3*sin3
        a = np.ones(12) / 12.0
        a += a_params[0] * fourier_cos[:, 0] + a_params[1] * fourier_cos[:, 1] + a_params[2] * fourier_cos[:, 2]
        a += a_params[3] * fourier_sin[:, 0] + a_params[4] * fourier_sin[:, 1] + a_params[5] * fourier_sin[:, 2]
        
        b = np.ones(12) / 12.0
        b += b_params[0] * fourier_cos[:, 0] + b_params[1] * fourier_cos[:, 1] + b_params[2] * fourier_cos[:, 2]
        b += b_params[3] * fourier_sin[:, 0] + b_params[4] * fourier_sin[:, 1] + b_params[5] * fourier_sin[:, 2]
        
        return a, b
    
    def objective(params):
        a, b = fourier_to_coeffs(params)
        
        # Check bounds [-50, 50]
        if np.any(np.abs(a) > 50) or np.any(np.abs(b) > 50):
            return 1e10
            
        err_sum = 0.0
        for i, t in enumerate(times):
            U_approx = np.eye(64, dtype=complex)
            for k in range(12):
                E_A_k = get_exp_A(a[k] * t)
                E_B_k = get_exp_B(b[k] * t)
                U_approx = E_B_k @ E_A_k @ U_approx
                
            U_target = U_exact[i]
            diff = U_approx - U_target
            err = np.linalg.norm(diff, 2)
            err_sum += err
            
        return err_sum / len(times)
        
    # Initial guess: all Fourier coefficients zero (uniform distribution)
    x0 = np.zeros(12)
    
    # Run Nelder-Mead with reduced parameter space
    x_best, val_best = _nelder_mead(objective, x0, max_iter=5000, time_limit=130.0, start_time=start_time)
    
    # Extract coefficients using Fourier parameterization
    a, b = fourier_to_coeffs(x_best)
    
    # Ensure float types
    a = [float(x) for x in a]
    b = [float(x) for x in b]
    
    return a, b

if __name__ == "__main__":
    # Self test
    a, b = coefficients(12)
    print(f"Sum a: {sum(a)}, Sum b: {sum(b)}")
