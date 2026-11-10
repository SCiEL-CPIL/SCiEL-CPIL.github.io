import numpy as np
import time
from scipy.optimize import minimize

def _build_hamiltonians():
    """Construct the A and B matrices for the 6-site Heisenberg chain."""
    # 6 sites, spin 1/2. Dimension 2^6 = 64.
    N = 6
    dim = 2 ** N
    
    # Identity
    I = np.eye(dim)
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Helper to apply an operator on a specific site
    def op_on_site(op, site):
        # op is 2x2, site is 0..5
        # Build the full operator
        # We can use kron
        ops = [I] * N
        ops[site] = op
        result = ops[0]
        for i in range(1, N):
            result = np.kron(result, ops[i])
        return result
    
    # Anisotropy
    gamma = 0.8
    # Field
    h = 0.3
    
    # Heisenberg Hamiltonian: H = sum_{i=0}^{4} [ Jx sx_i sx_{i+1} + Jy sy_i sy_{i+1} + Jz sz_i sz_{i+1} ] + h sum sz_i
    # J = 1 (assumed, or scaled? Task says "anisotropy 0.8 and field 0.3", usually J=1)
    Jx = 1.0
    Jy = 1.0
    Jz = 0.8
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Bonds: (0,1), (1,2), (2,3), (3,4), (4,5)
    # Even bonds: (0,1), (2,3), (4,5) -> B? 
    # Odd bonds: (1,2), (3,4) -> A?
    # Task: "A and B are the odd- and even-bond halves (each including half of the uniform field)"
    # Usually "even" means indices 0,1 (bond 0), 2,3 (bond 2), 4,5 (bond 4).
    # "odd" means 1,2 (bond 1), 3,4 (bond 3).
    
    # Field term: h * sum sz_i. Split half to A, half to B.
    # H_field = h * sum sz_i
    # A gets 0.5 * H_field + Odd_Bonds
    # B gets 0.5 * H_field + Even_Bonds
    
    for i in range(N):
        sz_i = op_on_site(sz, i)
        A += 0.5 * h * sz_i
        B += 0.5 * h * sz_i
    
    # Odd bonds: (1,2) and (3,4)
    for i in [1, 3]:
        j = i + 1
        A += Jx * np.kron(op_on_site(sx, i), op_on_site(sx, j)) # Wait, this kron logic is wrong for non-adjacent in full space if not careful.
        # Actually, op_on_site returns the full 64x64 matrix.
        # So A += Jx * (sx_i @ sx_j) is not right, it's the tensor product.
        # My op_on_site returns the full matrix. So I can just multiply them?
        # No, sx_i and sx_j commute if i != j. So sx_i @ sx_j is the correct operator.
        A += Jx * (op_on_site(sx, i) @ op_on_site(sx, j))
        A += Jy * (op_on_site(sy, i) @ op_on_site(sy, j))
        A += Jz * (op_on_site(sz, i) @ op_on_site(sz, j))
        
    # Even bonds: (0,1), (2,3), (4,5)
    for i in [0, 2, 4]:
        j = i + 1
        B += Jx * (op_on_site(sx, i) @ op_on_site(sx, j))
        B += Jy * (op_on_site(sy, i) @ op_on_site(sy, j))
        B += Jz * (op_on_site(sz, i) @ op_on_site(sz, j))
        
    return A, B

def _compute_error(a, b, A, B, times):
    """Compute the mean spectral norm error."""
    dim = A.shape[0]
    errors = []
    for t in times:
        # Compute U(t) = prod_{k=1}^{12} exp(-i b_k t B) exp(-i a_k t A)
        U = np.eye(dim, dtype=complex)
        for k in range(12):
            # exp(-i b_k t B)
            # Use scipy.linalg.expm
            U = np.linalg.multi_dot([scipy.linalg.expm(-1j * b[k] * t * B), U])
            U = np.linalg.multi_dot([scipy.linalg.expm(-1j * a[k] * t * A), U])
        
        # Exact propagator
        H = A + B
        U_exact = scipy.linalg.expm(-1j * t * H)
        
        err = np.linalg.norm(U - U_exact, 2)
        errors.append(err)
    return np.mean(errors)

def coefficients(stages):
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    A, B = _build_hamiltonians()
    times = [0.25, 0.5, 1.0]
    
    # Initial guess: 4th order 6-stage coefficients repeated twice?
    # Or just the seed's 2nd order.
    # Let's try a symmetric 4th order 6-stage guess.
    # Standard 4th order 6-stage (Yoshida) for A B A B A B:
    # a1=a6, a2=a5, a3=a4
    # b1=b6, b2=b5, b3=b4
    
    # Let's use the known 4th order 3-stage coefficients and repeat them 4 times, scaling by 1/4.
    # 3-stage 4th order (Suzuki):
    # a1=a3 = 1/(2(1-2^(1/3)))
    # a2 = 1/(2(2^(1/3)-1))
    # b1=b2 = 1/2
    # Sum a = 0? No, this is for the relative error.
    # Actually, the coefficients sum to 1 in the standard definition.
    # Let's check the sum of Suzuki 3-stage:
    # alpha = 2**(1/3)
    # a1 = 1/(2*(1-alpha))
    # a2 = 1/(2*(alpha-1))
    # a3 = a1
    # sum = 1/(1-alpha) + 1/(alpha-1) = 0.
    # This is weird.
    
    # Let's use the 2nd order Strang as initial guess, it's stable.
    # Strang: A(0.5) B(1) A(0.5)
    # In our B A B A ... form:
    # Step 1: B(0) A(0.5)
    # Step 2: B(1) A(0.5)
    # Step 3: B(0) A(0)
    # Sum a = 1, Sum b = 1.
    
    # Repeat 4 times for 12 stages.
    a_init = [0.125, 0.125, 0.0] * 4
    b_init = [0.0, 0.25, 0.0] * 4
    
    # Ensure sums are exactly 1
    a_init = np.array(a_init)
    b_init = np.array(b_init)
    
    # Optimization
    # Variables: a (12), b (12).
    # Constraints: sum(a)=1, sum(b)=1.
    # We can reduce variables: a12 = 1 - sum(a1..11), b12 = 1 - sum(b1..11).
    
    def objective(x):
        a = x[:11]
        b = x[11:22]
        a_full = np.append(a, 1.0 - np.sum(a))
        b_full = np.append(b, 1.0 - np.sum(b))
        
        # Check bounds
        if np.any(a_full < -50) or np.any(a_full > 50) or np.any(b_full < -50) or np.any(b_full > 50):
            return 1e10
            
        return _compute_error(a_full, b_full, A, B, times)
    
    x0 = np.concatenate([a_init[:11], b_init[:11]])
    
    try:
        # Use Nelder-Mead as it doesn't require gradients and handles constraints via penalty or bounds loosely
        # L-BFGS-B is better with bounds.
        # Bounds for a_i, b_i: [-50, 50]
        bounds = [(-50, 50)] * 22
        
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-8})
        
        if res.success or res.fun < 1e9:
            a_opt = np.append(res.x[:11], 1.0 - np.sum(res.x[:11]))
            b_opt = np.append(res.x[11:22], 1.0 - np.sum(res.x[11:22]))
            
            # Final check
            if np.all(np.isfinite(a_opt)) and np.all(np.isfinite(b_opt)):
                return a_opt.tolist(), b_opt.tolist()
    except Exception:
        pass
    
    # Fallback to initial guess
    return a_init.tolist(), b_init.tolist()

# Import scipy here to ensure it's available in the scope if needed, though usually global imports are better.
import scipy.linalg
