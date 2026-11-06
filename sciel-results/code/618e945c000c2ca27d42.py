import numpy as np
import scipy.optimize
import time
import math

def coefficients(stages):
    """
    Optimizes the coefficients of a 12-stage product formula to minimize
    the mean spectral norm error for a 6-site XXZ Heisenberg chain.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin for 60s budget
    
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

    # Precompute exact propagator factors? No, depends on t.
    # But we can precompute expm for specific t values?
    # The evaluator uses t = 0.25, 0.5, 1.0.
    # We can precompute expm(-i * t * A) and expm(-i * t * B) for these t's?
    # No, the coefficients a_k, b_k change, so the exponents change.
    # However, we can precompute the eigen-decomposition of A and B to speed up expm.
    
    # Eigen-decomposition of A and B
    wA, vA = np.linalg.eigh(A) # A is Hermitian
    wB, vB = np.linalg.eigh(B) # B is Hermitian
    
    def expm_H(H_eigvals, H_eigvecs, t):
        # exp(-i * t * H)
        # = V diag(exp(-i t w)) V^dag
        diag = np.exp(-1j * t * H_eigvals)
        # V diag V^dag
        # vA is (64, 64). diag is (64,).
        # Efficient way: vA * diag (broadcast) then @ vA.conj().T
        return (vA * diag) @ vA.conj().T if H_eigvals is wA else (vB * diag) @ vB.conj().T
    
    # We need a generic expm for arbitrary coefficients
    # def get_expA(a_k, t): return (vA * np.exp(-1j * a_k * t * wA)) @ vA.conj().T
    # def get_expB(b_k, t): return (vB * np.exp(-1j * b_k * t * wB)) @ vB.conj().T

    def get_expA(a_k, t):
        return (vA * np.exp(-1j * a_k * t * wA)) @ vA.conj().T

    def get_expB(b_k, t):
        return (vB * np.exp(-1j * b_k * t * wB)) @ vB.conj().T

    ts = [0.25, 0.5, 1.0]
    
    def calculate_error(params):
        # params: 22 floats
        # a_1..a_11, b_1..b_11
        a_free = params[:11]
        b_free = params[11:22]
        
        a12 = 1.0 - np.sum(a_free)
        b12 = 1.0 - np.sum(b_free)
        
        a = np.append(a_free, a12)
        b = np.append(b_free, b12)
        
        # Check bounds [-50, 50]
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e6 # Penalty
            
        error_sum = 0.0
        for t in ts:
            U_approx = np.eye(64, dtype=complex)
            # Product order k=1..12: exp(-i b_k t B) exp(-i a_k t A)
            # Note: The prompt says prod_k exp(-i b_k t B) exp(-i a_k t A)
            # This means B comes first, then A, for each k.
            for k in range(12):
                U_approx = get_expB(b[k], t) @ U_approx
                U_approx = get_expA(a[k], t) @ U_approx
            
            U_exact = np.eye(64, dtype=complex)
            # Exact is exp(-i t (A+B))
            # Since A and B don't commute, we can't just multiply exps.
            # We must compute expm(-i * t * (A+B)).
            # We can precompute eig of A+B?
            # Let's do it once outside?
            
            # Precomputation for A+B
            # H_total = A + B
            # We should compute eig of H_total once.
            
            # Let's move H_total eig calc outside.
            
            # For now, placeholder, will fix in code structure.
            pass
        
        # This function is a sketch.
        # We need to precompute H_total eig.
        
        return 0.0 # Placeholder

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
    
    # Try Nelder-Mead
    # maxiter needs to be high enough, but we have time limit.
    # We will run multiple optimizers or restarts if time permits.
    
    best_params = x0
    best_error = calculate_error_full(x0)
    
    # Define a wrapper to check time
    def time_check():
        return time.time() < deadline

    # Run Nelder-Mead
    try:
        res = scipy.optimize.minimize(
            calculate_error_full, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 2000, 'adaptive': True, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        if res.fun < best_error:
            best_error = res.fun
            best_params = res.x
    except Exception:
        pass

    if not time_check():
        # Time up, return best so far
        a = np.append(best_params[:11], 1.0 - np.sum(best_params[:11]))
        b = np.append(best_params[11:22], 1.0 - np.sum(best_params[11:22]))
        # Clip to bounds just in case
        a = np.clip(a, -50, 50)
        b = np.clip(b, -50, 50)
        # Re-normalize? If clipping breaks sum, we have a problem.
        # But Nelder-Mead usually stays near valid region if penalty is high.
        # If sum is broken, we might return invalid.
        # Let's ensure sum=1 by adjusting the last element after clipping?
        # If we clip, sum might not be 1.
        # Better to check validity before returning.
        
        if abs(np.sum(a) - 1.0) > 1e-5 or abs(np.sum(b) - 1.0) > 1e-5:
             # Fallback to Strang
             return [1.0/12.0]*12, [1.0/12.0]*12

        return list(a), list(b)

    # If time remains, try L-BFGS-B with tighter tolerances or different start
    # Or try Differential Evolution for global search?
    # DE is expensive. Nelder-Mead is local.
    # Let's try a few random restarts with Nelder-Mead if time permits.
    
    # Actually, let's just return the best found by Nelder-Mead.
    # It's a structural improvement over fixed Strang.
    
    a = np.append(best_params[:11], 1.0 - np.sum(best_params[:11]))
    b = np.append(best_params[11:22], 1.0 - np.sum(best_params[11:22]))
    
    # Final sanity check
    if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
        return [1.0/12.0]*12, [1.0/12.0]*12
        
    return list(a), list(b)

