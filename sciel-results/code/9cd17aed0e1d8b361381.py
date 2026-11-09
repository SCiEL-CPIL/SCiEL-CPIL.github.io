import numpy as np
import time
import sys
from scipy.optimize import minimize

def _build_hamiltonians():
    """
    Build A (odd bonds + half field) and B (even bonds + half field) for a 6-site
    open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
    """
    N = 6
    lam = 0.8
    h = 0.3
    
    sx = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
    sy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
    
    I2 = np.eye(2, dtype=complex)
    
    def op_on_site(op, site):
        """Apply operator op on given site (0-indexed) to full 2^N space."""
        ops = [I2] * N
        ops[site] = op
        result = ops[0]
        for i in range(1, N):
            result = np.kron(result, ops[i])
        return result
    
    # Build A: odd bonds (0-1, 2-3, 4-5) + half field
    A = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(0, N, 2):
        j = i + 1
        A += op_on_site(sx, i) @ op_on_site(sx, j)
        A += op_on_site(sy, i) @ op_on_site(sy, j)
        A += lam * op_on_site(sz, i) @ op_on_site(sz, j)
    for i in range(N):
        A += (h/2) * op_on_site(sz, i)
    
    # Build B: even bonds (1-2, 3-4) + half field
    B = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(1, N-1, 2):
        j = i + 1
        B += op_on_site(sx, i) @ op_on_site(sx, j)
        B += op_on_site(sy, i) @ op_on_site(sy, j)
        B += lam * op_on_site(sz, i) @ op_on_site(sz, j)
    for i in range(N):
        B += (h/2) * op_on_site(sz, i)
    
    return A, B

# Cache for eigen-decompositions
_eig_cache = {}

def _get_eig(A, B):
    """Get eigen-decompositions of A, B, and A+B, cached."""
    key = (id(A), id(B))
    if key not in _eig_cache:
        wA, vA = np.linalg.eigh(A)
        wB, vB = np.linalg.eigh(B)
        H = A + B
        wH, vH = np.linalg.eigh(H)
        _eig_cache[key] = (wA, vA, wB, vB, wH, vH)
    return _eig_cache[key]

def _compute_error_full(a, b, A, B):
    """Compute mean spectral-norm error at t = 0.25, 0.5, 1.0."""
    wA, vA, wB, vB, wH, vH = _get_eig(A, B)
    dim = A.shape[0]
    
    errors = []
    for t in [0.25, 0.5, 1.0]:
        U = np.eye(dim, dtype=complex)
        
        # Pre-compute exponents for A and B to save some computation?
        # expA_k = vA @ diag(exp(-i a_k t wA)) @ vA.conj().T
        # expB_k = vB @ diag(exp(-i b_k t wB)) @ vB.conj().T
        
        for k in range(len(a)):
            # exp(-i a_k t A)
            exp_vals_A = np.exp(-1j * a[k] * t * wA)
            expA = vA @ np.diag(exp_vals_A) @ vA.conj().T
            U = U @ expA
            
            # exp(-i b_k t B)
            exp_vals_B = np.exp(-1j * b[k] * t * wB)
            expB = vB @ np.diag(exp_vals_B) @ vB.conj().T
            U = U @ expB
        
        # Exact: exp(-i t (A+B))
        exp_vals_H = np.exp(-1j * t * wH)
        U_exact = vH @ np.diag(exp_vals_H) @ vH.conj().T
        
        diff = U - U_exact
        err = np.linalg.norm(diff, 2)
        errors.append(err)
    
    return np.mean(errors)

def coefficients(stages):
    """Optimize 12-stage product formula coefficients using symmetric local search."""
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    A, B = _build_hamiltonians()
    
    # Symmetric parameterization:
    # a = [x0, x1, x2, x3, x4, x5, x5, x4, x3, x2, x1, x0]
    # sum(a) = 2*(x0+x1+x2+x3+x4+x5) = 1 => x5 = 0.5 - (x0+x1+x2+x3+x4)
    # So 5 free parameters for a: x0..x4
    # Same for b.
    
    def _unpack(params):
        # params = [a0..a4, b0..b4]
        pa = params[:5]
        pb = params[5:]
        
        a = np.zeros(12)
        b = np.zeros(12)
        
        a[0] = a[11] = pa[0]
        a[1] = a[10] = pa[1]
        a[2] = a[9] = pa[2]
        a[3] = a[8] = pa[3]
        a[4] = a[7] = pa[4]
        a5 = 0.5 - sum(pa)
        a[5] = a[6] = a5
        
        b[0] = b[11] = pb[0]
        b[1] = b[10] = pb[1]
        b[2] = b[9] = pb[2]
        b[3] = b[8] = pb[3]
        b[4] = b[7] = pb[4]
        b5 = 0.5 - sum(pb)
        b[5] = b[6] = b5
        
        return a, b

    def _objective(params):
        a, b = _unpack(params)
        # Clamp to [-50, 50] is usually not needed if we start near 1/12, 
        # but let's be safe. If clamped, sum constraint might break slightly, 
        # but _unpack enforces sum=1 structurally. 
        # The clamp in the final output is what matters for validity.
        return _compute_error_full(a, b, A, B)
    
    # Initial guess: Uniform 1/12
    # a_i = 1/12. 
    # In symmetric param: 2*(x0+...+x4+x5)=1.
    # If uniform, x0=...=x5=1/12.
    # Then x5 = 0.5 - (5/12) = (6-5)/12 = 1/12. Consistent.
    x0 = 1.0/12
    init_params = np.array([x0]*5 + [x0]*5)
    
    # Run Nelder-Mead
    # We have 10 parameters.
    # Set maxiter high to ensure convergence.
    # Time limit check is handled by the outer loop or we trust scipy's speed.
    # 10 params, ~2000 iterations usually enough.
    # Each eval takes ~10-20ms? 12 steps * 3 times * matrix ops.
    # Let's estimate: 12*3 = 36 matrix exps. 64x64.
    # 36 * (64^3 * 10 flops) ~ 36 * 260k * 10 ~ 90M flops.
    # NumPy does 100M flops in ~1s? Maybe faster.
    # If 1 eval is 50ms, 1000 evals is 50s.
    # So we have budget for ~2000-3000 iterations.
    
    result = None
    try:
        result = minimize(_objective, init_params, method='Nelder-Mead', 
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
    except Exception as e:
        # If scipy fails, fallback to a simpler approach
        pass
    
    if result and result.success and result.fun < _objective(init_params):
        best_params = result.x
    else:
        # Fallback or try Powell if NM failed
        try:
            result2 = minimize(_objective, init_params, method='Powell', 
                               options={'maxiter': 1000, 'xtol': 1e-8, 'ftol': 1e-10})
            if result2 and result2.fun < _objective(init_params):
                best_params = result2.x
            else:
                best_params = init_params
        except:
            best_params = init_params
            
    # If time remains, do a local refinement with smaller steps?
    # Nelder-Mead adaptive should handle this.
    
    a, b = _unpack(best_params)
    
    # Final clamp
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    
    # Ensure sum is exactly 1 (fix any floating point issues from clamping)
    a_sum = sum(a)
    if abs(a_sum - 1.0) > 1e-10:
        corr = (1.0 - a_sum) / 12.0
        a = [x + corr for x in a]
        
    b_sum = sum(b)
    if abs(b_sum - 1.0) > 1e-10:
        corr = (1.0 - b_sum) / 12.0
        b = [x + corr for x in b]
    
    return a, b
