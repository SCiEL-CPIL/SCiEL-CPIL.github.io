"""Optimized product formula design for Heisenberg-chain time evolution."""
import numpy as np
import math
import time

DEADLINE = 145.0
EVAL_TIMES = (0.25, 0.5, 1.0)
N_SITES = 6
ANISOTROPY = 0.8
FIELD = 0.3
STAGES = 12

def build_hamiltonians():
    n = N_SITES
    sx = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
    sy = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
    sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
    zero = np.array([[0, 0], [0, 0]], dtype=complex)

    def site_op(op, j):
        ops = [zero] * n
        ops[j] = op
        result = np.array([[1.0+0j]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result

    A = np.zeros((2**n, 2**n), dtype=complex)
    B = np.zeros((2**n, 2**n), dtype=complex)

    for j in range(n - 1):
        term = (site_op(sx, j) @ site_op(sx, j+1) +
                site_op(sy, j) @ site_op(sy, j+1) +
                ANISOTROPY * site_op(sz, j) @ site_op(sz, j+1))
        if j % 2 == 0:
            B += term
        else:
            A += term

    for j in range(n):
        half_field = FIELD / 2.0
        term = half_field * site_op(sz, j)
        A += term
        B += term

    return A, B

# Precompute eigen-decompositions for fast expm
_HAM_CACHE = None
def get_hamiltonians():
    global _HAM_CACHE
    if _HAM_CACHE is None:
        A, B = build_hamiltonians()
        # Eigen-decompose A and B for fast expm via V @ diag(exp) @ V^-1
        wA, VA = np.linalg.eig(A)
        wB, VB = np.linalg.eig(B)
        # Precompute V^{-1}
        VAI = np.linalg.inv(VA)
        VBI = np.linalg.inv(VB)
        # Also precompute exact propagator eigen-decomposition
        AB = A + B
        wAB, VAB = np.linalg.eig(AB)
        VABI = np.linalg.inv(VAB)
        _HAM_CACHE = (A, B, wA, VA, VAI, wB, VB, VBI, wAB, VAB, VABI)
    return _HAM_CACHE

def fast_expm_eig(w, V, Vi, c):
    """Compute exp(-i*c*H) where H has eigenvalues w, eigenvectors V, Vi=V^{-1}."""
    return Vi @ (V * np.exp(-1j * c * w[:, None]))

def eval_coefficients_fast(a, b):
    """Compute mean spectral-norm error using precomputed eigen-decompositions."""
    _, _, wA, VA, VAI, wB, VB, VBI, wAB, VAB, VABI = get_hamiltonians()
    total_error = 0.0
    n_times = len(EVAL_TIMES)
    dim = 2 ** N_SITES
    for t in EVAL_TIMES:
        U = np.eye(dim, dtype=complex)
        for k in range(STAGES):
            U = U @ fast_expm_eig(wB, VB, VBI, b[k] * t)
            U = U @ fast_expm_eig(wA, VA, VAI, a[k] * t)
        U_exact = fast_expm_eig(wAB, VAB, VABI, t)
        diff = U - U_exact
        s = np.linalg.svd(diff, compute_uv=False)
        total_error += s[0]
    return total_error / n_times

def normalize_coefficients(a, b):
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-15:
        a = [x / sa for x in a]
    else:
        a = [1.0/STAGES] * STAGES
    if abs(sb) > 1e-15:
        b = [x / sb for x in b]
    else:
        b = [1.0/STAGES] * STAGES
    a = [max(-50.0, min(50.0, x)) for x in a]
    b = [max(-50.0, min(50.0, x)) for x in b]
    sa = sum(a)
    sb = sum(b)
    if abs(sa) > 1e-15:
        a = [x / sa for x in a]
    else:
        a = [1.0/STAGES] * STAGES
    if abs(sb) > 1e-15:
        b = [x / sb for x in b]
    else:
        b = [1.0/STAGES] * STAGES
    return a, b

def generate_candidates():
    """Generate diverse candidate splitting schemes."""
    candidates = []
    
    # 1. Uniform (2nd order)
    a0 = [1.0/STAGES] * STAGES
    b0 = [1.0/STAGES] * STAGES
    candidates.append((a0[:], b0[:]))
    
    # 2. Strang-like: alternate emphasis
    a_strang = [1.0/STAGES] * STAGES
    b_strang = [1.0/STAGES] * STAGES
    # Make first and last b larger (Strang symmetry)
    b_strang[0] = 0.5/STAGES + 0.05
    b_strang[-1] = 0.5/STAGES + 0.05
    b_strang[1] = 0.5/STAGES - 0.025
    b_strang[-2] = 0.5/STAGES - 0.025
    candidates.append((a_strang[:], b_strang[:]))
    
    # 3. 4th-order Suzuki-Trotter coefficients replicated
    # Standard 4th order for 2-part: 
    # a = [c, 0.5-c, 0.5-c, c], b = [2c-0.5, 0.5-2c, 0.5-2c, 2c-0.5]
    # where c = 1/(4 - 4^(1/4))
    c4 = 1.0 / (4.0 - 4.0**(0.25))
    a4_block = [c4, 0.5 - c4, 0.5 - c4, c4]
    b4_block = [2.0*c4 - 0.5, 0.5 - 2.0*c4, 0.5 - 2.0*c4, 2.0*c4 - 0.5]
    
    # Replicate 3 times to get 12 stages
    a_4th = []
    b_4th = []
    for _ in range(3):
        a_4th.extend([x/3.0 for x in a4_block])
        b_4th.extend([x/3.0 for x in b4_block])
    candidates.append((a_4th[:], b_4th[:]))
    
    # 4. Symmetric 6th-order-like pattern
    # Use a pattern that emphasizes middle stages less
    a_sym = [1.0/STAGES] * STAGES
    b_sym = [1.0/STAGES] * STAGES
    # Create a bell-shaped distribution for better error cancellation
    for i in range(STAGES):
        pos = i / (STAGES - 1)
        weight = 1.0 + 0.3 * math.cos(2 * math.pi * pos)
        a_sym[i] = weight / sum(1.0 + 0.3 * math.cos(2 * math.pi * j / (STAGES - 1)) for j in range(STAGES))
        b_sym[i] = weight / sum(1.0 + 0.3 * math.cos(2 * math.pi * j / (STAGES - 1)) for j in range(STAGES))
    candidates.append((a_sym[:], b_sym[:]))
    
    # 5. Random symmetric candidates
    np.random.seed(42)
    for i in range(30):
        a_half = np.random.uniform(-0.2, 0.2, 6)
        b_half = np.random.uniform(-0.2, 0.2, 6)
        a_full = np.concatenate([a_half, a_half[::-1]])
        b_full = np.concatenate([b_half, b_half[::-1]])
        sa = np.sum(a_full)
        sb = np.sum(b_full)
        if abs(sa) > 1e-10:
            a_full = a_full / sa
        else:
            a_full = np.ones(STAGES)/STAGES
        if abs(sb) > 1e-10:
            b_full = b_full / sb
        else:
            b_full = np.ones(STAGES)/STAGES
        a_full = np.clip(a_full, -50.0, 50.0)
        b_full = np.clip(b_full, -50.0, 50.0)
        sa = np.sum(a_full)
        sb = np.sum(b_full)
        if abs(sa) > 1e-10:
            a_full = a_full / sa
        if abs(sb) > 1e-10:
            b_full = b_full / sb
        candidates.append((a_full.tolist(), b_full.tolist()))
    
    return candidates

def optimize_coefficients():
    """Main optimization routine with deadline guard."""
    start_time = time.time()
    
    # Generate candidates
    candidates = generate_candidates()
    
    best_a = None
    best_b = None
    best_err = float('inf')
    
    # Evaluate all candidates
    for i, (a_cand, b_cand) in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 60:
            break
        
        a_cand, b_cand = normalize_coefficients(a_cand[:], b_cand[:])
        try:
            err = eval_coefficients_fast(a_cand, b_cand)
        except Exception:
            continue
        
        if err < best_err:
            best_err = err
            best_a = a_cand[:]
            best_b = b_cand[:]
    
    if best_a is None:
        best_a = [1.0/STAGES] * STAGES
        best_b = [1.0/STAGES] * STAGES
        best_err = eval_coefficients_fast(best_a, best_b)
    
    # Local optimization with Nelder-Mead
    if time.time() - start_time < DEADLINE - 30:
        from scipy.optimize import minimize
        
        a0 = np.array(best_a)
        b0 = np.array(best_b)
        
        def objective(x):
            a_opt = x[:STAGES].tolist()
            b_opt = x[STAGES:].tolist()
            a_opt, b_opt = normalize_coefficients(a_opt, b_opt)
            try:
                return eval_coefficients_fast(a_opt, b_opt)
            except Exception:
                return 1e10
        
        x0 = np.concatenate([a0, b0])
        
        # Try multiple restarts if time permits
        n_restarts = 3
        for r in range(n_restarts):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            if r == 0:
                x_start = x0
            else:
                # Perturb from best
                perturbation = np.random.normal(0, 0.05, len(x0))
                x_start = x0 + perturbation
            
            options = {'maxiter': 150, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            try:
                result = minimize(objective, x_start, method='Nelder-Mead', options=options)
                if result.fun < best_err:
                    a_opt, b_opt = normalize_coefficients(result.x[:STAGES].tolist(), result.x[STAGES:].tolist())
                    best_err = result.fun
                    best_a = a_opt
                    best_b = b_opt
                    x0 = result.x.copy()
            except Exception:
                pass
    
    return best_a, best_b, best_err

def coefficients(stages):
    """Return optimized coefficients for the 12-stage product formula."""
    start_time = time.time()
    
    try:
        a_best, b_best, err = optimize_coefficients()
    except Exception:
        # Fallback to Strang splitting
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    # Deadline guard
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        a_best, b_best = normalize_coefficients(a_best, b_best)
        return a_best, b_best
    
    a_best, b_best = normalize_coefficients(a_best, b_best)
    return a_best, b_best
