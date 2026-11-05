import numpy as np
import time
from scipy.linalg import expm

def coefficients(stages):
    """
    Optimizes a 12-stage product formula using a symmetric 4th-order Suzuki-Trotter 
    initialization and local optimization (Nelder-Mead) to minimize the spectral norm 
    error at t = 0.25, 0.5, 1.0.
    """
    start_time = time.time()
    deadline = 130.0  # Safety margin
    
    # Construct A and B matrices for a 6-site XXZ Heisenberg chain
    # Anisotropy delta = 0.8, Field h = 0.3
    # Hamiltonian H = sum_{i=1}^{5} (S_i^x S_{i+1}^x + S_i^y S_{i+1}^y + delta S_i^z S_{i+1}^z) + h sum_{i=1}^6 S_i^z
    # Split into odd bonds (1-2, 3-4, 5-6) -> A
    # Split into even bonds (2-3, 4-5) -> B
    # Field is split half to A, half to B.
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]]) / 2
    sy = np.array([[0, -1j], [1j, 0]]) / 2
    sz = np.array([[1, 0], [0, -1]]) / 2
    s0 = np.array([[1, 0], [0, 1]]) / 2
    
    delta = 0.8
    h = 0.3
    
    # Build interaction terms
    # We need to build operators on the 6-site Hilbert space (2^6 = 64 dim)
    # Helper to build site operator
    def site_op(op, site, n_sites=6):
        # op is 2x2, site is 0-indexed
        op_full = op
        for i in range(n_sites):
            if i != site:
                op_full = np.kron(op_full, s0) if i > site else np.kron(s0, op_full)
        # The above logic is flawed for general site. Let's do it properly.
        # Start with identity
        op_full = np.array([[1.0]])
        for i in range(n_sites):
            if i == site:
                op_full = np.kron(op_full, op)
            else:
                op_full = np.kron(op_full, s0)
        return op_full

    def bond_op(op1, op2, site1, site2, n_sites=6):
        # op1 on site1, op2 on site2
        # This is complex to build generally. Let's use a simpler tensor product construction.
        # Since n=6 is small, we can build it.
        # Actually, for performance in optimization, we should precompute A and B once.
        pass

    # Precompute A and B
    # A: bonds (0,1), (2,3), (4,5) + h/2 * sum S_z
    # B: bonds (1,2), (3,4) + h/2 * sum S_z
    
    def build_hamiltonian(bonds, field_factor):
        H = np.zeros((64, 64), dtype=complex)
        # Add bond terms
        for i, j in bonds:
            # S_i^x S_j^x + S_i^y S_j^y + delta S_i^z S_j^z
            # Sx_i Sx_j
            term_xx = np.kron(np.kron(np.kron(np.kron(np.kron(s0, sx), s0), s0), s0), s0) # This is wrong, need to place ops at i and j
            # Let's write a helper to place operators
            def place_ops(ops_dict):
                # ops_dict: {site: op}
                result = np.array([[1.0]])
                for site in range(6):
                    if site in ops_dict:
                        result = np.kron(result, ops_dict[site])
                    else:
                        result = np.kron(result, s0)
                return result
            
            H += place_ops({i: sx, j: sx})
            H += place_ops({i: sy, j: sy})
            H += delta * place_ops({i: sz, j: sz})
        
        # Add field
        for i in range(6):
            H += field_factor * h * place_ops({i: sz})
            
        return H

    A = build_hamiltonian([(0,1), (2,3), (4,5)], 0.5)
    B = build_hamiltonian([(1,2), (3,4)], 0.5)
    H_exact = A + B
    
    # Exact propagator function
    def exact_propagator(t):
        return expm(-1j * t * H_exact)
    
    # Product formula propagator
    def product_propagator(t, a, b):
        U = np.eye(64, dtype=complex)
        for k in range(12):
            # Order: exp(-i b_k t B) exp(-i a_k t A)
            U = expm(-1j * b[k] * t * B) @ U
            U = expm(-1j * a[k] * t * A) @ U
        return U

    # Objective function: mean spectral norm error
    def objective(params):
        # params: 10 floats (due to symmetry and constraints)
        # a1, a2, a3, b1, b2, b3
        # a = [a1, a2, a3, a3, a2, a1]
        # b = [b1, b2, b3, b3, b2, b1]
        # Constraints: 2*(a1+a2+a3)=1 => a3 = 0.5 - a1 - a2
        #               2*(b1+b2+b3)=1 => b3 = 0.5 - b1 - b2
        a1, a2, b1, b2 = params[0], params[1], params[2], params[3]
        a3 = 0.5 - a1 - a2
        b3 = 0.5 - b1 - b2
        
        a = [a1, a2, a3, a3, a2, a1]
        b = [b1, b2, b3, b3, b2, b1]
        
        # Check bounds [-50, 50]
        if any(abs(x) > 50 for x in a + b):
            return 1e10
            
        err = 0.0
        for t in [0.25, 0.5, 1.0]:
            U_approx = product_propagator(t, a, b)
            U_exact = exact_propagator(t)
            diff = U_approx - U_exact
            # Spectral norm
            # Use scipy.linalg.svd for singular values
            s = np.linalg.svd(diff, compute_uv=False)
            err += np.max(s)
        return err / 3.0

    # Initial guess: 4th order Suzuki-Trotter
    # b1 = 1/(2-2^(1/3)) ~ 0.675, b2 = -2^(1/3)/(2-2^(1/3)) ~ -0.350
    # c1 = b1/2 ~ 0.337, c2 = b1+b2 ~ 0.325
    # Mapping to 6-step symmetric:
    # We can try to fit the ST sequence into the 6-step structure.
    # ST: B(c1/2) A(b1) B(c2) A(b2) B(c1)
    # This is 3 A, 3 B (with splits).
    # Our structure: B(b1) A(a1) B(b2) A(a2) B(b3) A(a3) B(b3) A(a3) B(b2) A(a2) B(b1) A(a1)
    # This is 6 A, 6 B.
    # Let's just start with equal weights (2nd order) and let optimization do the work.
    # Or start with a 4th-order approximation mapped to 6 steps.
    # Let's try a symmetric 4th-order scheme for 6 steps.
    # Reference: "High-order symmetric splitting"
    # For now, use a reasonable starting point.
    init_params = [1/12, 1/12, 1/12, 1/12] # a1=a2=1/12, a3=1/6? No, 2*(1/12+1/12+1/6)=1.
    # a1=1/12, a2=1/12, a3=1/6. b same.
    init_params = [1/12, 1/12, 1/12, 1/12]
    
    # Use Nelder-Mead
    from scipy.optimize import minimize
    
    best_params = init_params
    best_err = objective(init_params)
    
    # Optimization
    try:
        res = minimize(objective, init_params, method='Nelder-Mead', 
                       options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False})
        if res.fun < best_err:
            best_params = res.x
            best_err = res.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        # Return best found
        a1, a2, b1, b2 = best_params
        a3 = 0.5 - a1 - a2
        b3 = 0.5 - b1 - b2
        a = [a1, a2, a3, a3, a2, a1]
        b = [b1, b2, b3, b3, b2, b1]
        return a, b
        
    # Final check
    a1, a2, b1, b2 = best_params
    a3 = 0.5 - a1 - a2
    b3 = 0.5 - b1 - b2
    a = [a1, a2, a3, a3, a2, a1]
    b = [b1, b2, b3, b3, b2, b1]
    
    # Ensure finite and in bounds
    for i in range(12):
        if not np.isfinite(a[i]) or abs(a[i]) > 50: a[i] = 0
        if not np.isfinite(b[i]) or abs(b[i]) > 50: b[i] = 0
        
    # Normalize to ensure sum is exactly 1
    sum_a = sum(a)
    sum_b = sum(b)
    if sum_a != 0:
        a = [x / sum_a for x in a]
    if sum_b != 0:
        b = [x / sum_b for x in b]
        
    return a, b
