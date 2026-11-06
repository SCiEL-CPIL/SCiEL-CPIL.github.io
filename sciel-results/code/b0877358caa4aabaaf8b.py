import numpy as np
import time
import math

def coefficients(stages):
    """
    Optimizes the coefficients of a 12-stage product formula to minimize
    the mean spectral norm error for a 6-site XXZ Heisenberg chain.
    """
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin for 60s budget (or 150s if that's the hard limit, but 50s is safe for 60s sub-process)
    
    # 1. Construct the Hamiltonians A and B
    # 6-site chain, spin-1/2. Dimension 2^6 = 64.
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def kron_all(*mats):
        res = mats[0]
        for m in mats[1:]:
            res = np.kron(res, m)
        return res

    # Identity for single site
    I2 = np.eye(2, dtype=complex)
    
    # Build operators for each site
    # We need Sx_j, Sy_j, Sz_j for j=0..5
    # To save memory/time, we can build the terms directly.
    # H = sum_{j=1}^{5} (Sx_j Sx_{j+1} + Sy_j Sy_{j+1} + Delta Sz_j Sz_{j+1}) + h sum Sz_j
    # A: Odd bonds (1-2, 3-4, 5-6) -> indices (0,1), (2,3), (4,5)
    #    Field: half of h * sum Sz_j. Let's say A takes Sz on sites 0, 2, 4? 
    #    Prompt: "each including half of the uniform field".
    #    This usually means H_A = H_odd + 0.5 * H_field and H_B = H_even + 0.5 * H_field.
    #    Since H_field commutes with everything, the split doesn't affect the product 
    #    structure's commutativity with the field part, but it does affect the 
    #    specific matrices A and B. However, since exp(-i t (A+B)) = exp(-i t H_total),
    #    and the field part is in both, it will factor out or just add up.
    #    Let's define A and B explicitly.
    
    Delta = 0.8
    h = 0.3
    
    A = np.zeros((64, 64), dtype=complex)
    B = np.zeros((64, 64), dtype=complex)
    
    # Helper to get operator on site j
    def op_on_site(op, j):
        ops = [I2] * 6
        ops[j] = op
        return kron_all(*ops)
        
    # Bonds for A (odd): (0,1), (2,3), (4,5)
    odd_bonds = [(0, 1), (2, 3), (4, 5)]
    # Bonds for B (even): (1, 2), (3, 4)
    even_bonds = [(1, 2), (3, 4)]
    
    def bond_term(j, k):
        # Sx_j Sx_k + Sy_j Sy_k + Delta Sz_j Sz_k
        # Sx_j Sx_k
        term = np.zeros((64, 64), dtype=complex)
        for op1 in [sx, sy, sz]:
            coeff = 1.0 if op1 in [sx, sy] else Delta
            ops = [I2] * 6
            ops[j] = op1
            ops[k] = op1
            term += coeff * kron_all(*ops)
        return term

    for j, k in odd_bonds:
        A += bond_term(j, k)
        
    for j, k in even_bonds:
        B += bond_term(j, k)
        
    # Field: 0.5 * h * sum Sz_j in both A and B
    field_half = 0.0
    for j in range(6):
        field_half += 0.5 * h * op_on_site(sz, j)
        
    A += field_half
    B += field_half

    # Eigen-decomposition of A and B
    wA, vA = np.linalg.eigh(A) # A is Hermitian
    wB, vB = np.linalg.eigh(B) # B is Hermitian
    
    def get_expA(a_k, t):
        return (vA * np.exp(-1j * a_k * t * wA)) @ vA.conj().T

    def get_expB(b_k, t):
        return (vB * np.exp(-1j * b_k * t * wB)) @ vB.conj().T

    ts = [0.25, 0.5, 1.0]
    
    # Precompute H_total eig
    H_total = A + B
    wT, vT = np.linalg.eigh(H_total)
    
    def get_expT(t):
        return (vT * np.exp(-1j * t * wT)) @ vT.conj().T

    def calculate_error_full(params):
        a_free = params[:11]
        b_free = params[11:22]
        
        a12 = 1.0 - np.sum(a_free)
        b12 = 1.0 - np.sum(b_free)
        
        a = np.append(a_free, a12)
        b = np.append(b_free, b12)
        
        # Check bounds
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e6
            
        error_sum = 0.0
        for t in ts:
            U_approx = np.eye(64, dtype=complex)
            for k in range(12):
                U_approx = get_expB(b[k], t) @ U_approx
                U_approx = get_expA(a[k], t) @ U_approx
            
            U_exact = get_expT(t)
            
            # Spectral norm
            # Use svd for 64x64
            s = np.linalg.svd(U_approx - U_exact, compute_uv=False)
            error_sum += s[0]
            
        return error_sum / 3.0

    # Initial guess: Strang splitting (uniform)
    x0 = np.array([1.0/12.0] * 22)
    
    best_params = x0.copy()
    best_error = calculate_error_full(x0)
    
    # Custom Nelder-Mead implementation to avoid scipy dependency
    def nelder_mead(func, x0, max_iter=200, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
        n = len(x0)
        # Initialize simplex
        simplex = [x0.copy()]
        for i in range(n):
            x_new = x0.copy()
            if x_new[i] == 0:
                x_new[i] = 0.00025
            else:
                x_new[i] = 1.0 + x_new[i]
            simplex.append(x_new)
        
        # Evaluate initial simplex
        fvals = [func(v) for v in simplex]
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            # Check convergence
            if abs(fvals[-1] - fvals[0]) < 1e-8:
                break
            
            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = func(x_r)
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                f_e = func(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                x_c = centroid + sigma * (simplex[-1] - centroid)
                f_c = func(x_c)
                if f_c < fvals[-1]:
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
        
        # Find best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]

    try:
        best_params, best_error = nelder_mead(calculate_error_full, x0, max_iter=100)
    except Exception:
        pass

    # Final result construction
    a = np.append(best_params[:11], 1.0 - np.sum(best_params[:11]))
    b = np.append(best_params[11:22], 1.0 - np.sum(best_params[11:22]))
    
    # Final sanity check
    if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
        return [1.0/12.0]*12, [1.0/12.0]*12
        
    if abs(np.sum(a) - 1.0) > 1e-5 or abs(np.sum(b) - 1.0) > 1e-5:
        return [1.0/12.0]*12, [1.0/12.0]*12

    return list(a), list(b)
