"""High-order symmetric product formula for Heisenberg-chain time evolution."""
import numpy as np
import math
import time
import sys

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

_HAM_CACHE = None
def get_hamiltonians():
    global _HAM_CACHE
    if _HAM_CACHE is None:
        A, B = build_hamiltonians()
        wA, VA = np.linalg.eig(A)
        wB, VB = np.linalg.eig(B)
        VAI = np.linalg.inv(VA)
        VBI = np.linalg.inv(VB)
        AB = A + B
        wAB, VAB = np.linalg.eig(AB)
        VABI = np.linalg.inv(VAB)
        _HAM_CACHE = (A, B, wA, VA, VAI, wB, VB, VBI, wAB, VAB, VABI)
    return _HAM_CACHE

def fast_expm_eig(w, V, Vi, c):
    exp_diag = np.exp(-1j * c * w)
    return (V * exp_diag[None, :]) @ Vi

def eval_coefficients_fast(a, b):
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
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    else:
        a = np.ones(STAGES) / STAGES
    if abs(sb) > 1e-15:
        b = b / sb
    else:
        b = np.ones(STAGES) / STAGES
    
    a = np.clip(a, -50.0, 50.0)
    b = np.clip(b, -50.0, 50.0)
    
    sa = np.sum(a)
    sb = np.sum(b)
    if abs(sa) > 1e-15:
        a = a / sa
    else:
        a = np.ones(STAGES) / STAGES
    if abs(sb) > 1e-15:
        b = b / sb
    else:
        b = np.ones(STAGES) / STAGES
    
    return a.tolist(), b.tolist()

def make_symmetric_full(a_half, b_half):
    """Create full 12-element symmetric arrays from 6-element halves."""
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    return a_full, b_full

def generate_symmetric_candidates():
    """Generate symmetric candidates based on high-order splitting theory."""
    candidates = []
    
    # 1. Uniform (1st order)
    a0 = np.ones(6) / 6.0
    b0 = np.ones(6) / 6.0
    a_full, b_full = make_symmetric_full(a0, b0)
    a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
    candidates.append((a_full, b_full))
    
    # 2. 4th-order Suzuki-Trotter based (symmetric 6-stage, repeated)
    # Standard 4th order coefficients for 2-operator splitting
    # c1 = 1/(4-4^0.25), c2 = 0.5-c1, c3 = 2*c1-0.5
    c1 = 1.0 / (4.0 - 4.0**0.25)
    c2 = 0.5 - c1
    c3 = 2.0 * c1 - 0.5
    
    # For symmetric 12-stage, use the 4th-order pattern
    # The 4th order 6-stage symmetric is: [c1, c2, 0, c2, c1, 0] for A
    # and [0, c3, 1-c3, c3, 0, 0] for B (but needs adjustment for our format)
    
    # Let's use a known 4th-order symmetric 12-stage pattern
    # Pattern: a = [x1, x2, x3, x4, x5, x6, x6, x5, x4, x3, x2, x1]
    # b = [y1, y2, y3, y4, y5, y6, y6, y5, y4, y3, y2, y1]
    
    # 4th order symmetric 12-stage from literature
    # Using the composition method: S4 = S2(c) * S2(-2c) * S2(c) where S2 is 2nd order
    # This gives specific coefficients
    
    # Let me use a simpler approach: start from known good symmetric patterns
    
    # Pattern A: Endpoint-weighted
    a_w1 = np.array([0.15, 0.10, 0.08, 0.07, 0.07, 0.08])
    b_w1 = np.array([0.08, 0.07, 0.07, 0.08, 0.10, 0.15])
    sa, sb = np.sum(a_w1), np.sum(b_w1)
    a_w1 = a_w1 / sa
    b_w1 = b_w1 / sb
    a_full, b_full = make_symmetric_full(a_w1, b_w1)
    a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
    candidates.append((a_full, b_full))
    
    # Pattern B: Center-weighted
    a_w2 = np.array([0.07, 0.08, 0.10, 0.12, 0.12, 0.10])
    b_w2 = np.array([0.10, 0.12, 0.12, 0.10, 0.08, 0.07])
    sa, sb = np.sum(a_w2), np.sum(b_w2)
    a_w2 = a_w2 / sa
    b_w2 = b_w2 / sb
    a_full, b_full = make_symmetric_full(a_w2, b_w2)
    a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
    candidates.append((a_full, b_full))
    
    # Pattern C: Alternating
    a_w3 = np.array([0.12, 0.06, 0.10, 0.06, 0.08, 0.08])
    b_w3 = np.array([0.08, 0.08, 0.06, 0.10, 0.06, 0.12])
    sa, sb = np.sum(a_w3), np.sum(b_w3)
    a_w3 = a_w3 / sa
    b_w3 = b_w3 / sb
    a_full, b_full = make_symmetric_full(a_w3, b_w3)
    a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
    candidates.append((a_full, b_full))
    
    # Pattern D: Cosine-weighted
    for amp in [0.2, 0.4, 0.6]:
        a_w4 = np.array([1.0 + amp * math.cos(2 * math.pi * j / 5.0) for j in range(6)])
        b_w4 = np.array([1.0 + amp * math.cos(2 * math.pi * j / 5.0) for j in range(6)])
        sa, sb = np.sum(a_w4), np.sum(b_w4)
        a_w4 = a_w4 / sa
        b_w4 = b_w4 / sb
        a_full, b_full = make_symmetric_full(a_w4, b_w4)
        a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
        candidates.append((a_full, b_full))
    
    # Pattern E: Sine-weighted
    for amp in [0.2, 0.4, 0.6]:
        a_w5 = np.array([1.0 + amp * math.sin(math.pi * (j+1) / 7.0) for j in range(6)])
        b_w5 = np.array([1.0 + amp * math.sin(math.pi * (j+1) / 7.0) for j in range(6)])
        sa, sb = np.sum(a_w5), np.sum(b_w5)
        a_w5 = a_w5 / sa
        b_w5 = b_w5 / sb
        a_full, b_full = make_symmetric_full(a_w5, b_w5)
        a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
        candidates.append((a_full, b_full))
    
    # Pattern F: Linear ramp
    a_w6 = np.linspace(0.5, 1.5, 6)
    b_w6 = np.linspace(1.5, 0.5, 6)
    sa, sb = np.sum(a_w6), np.sum(b_w6)
    a_w6 = a_w6 / sa
    b_w6 = b_w6 / sb
    a_full, b_full = make_symmetric_full(a_w6, b_w6)
    a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
    candidates.append((a_full, b_full))
    
    # Pattern G: Quadratic
    a_w7 = np.array([1.0 + 0.3 * (j - 2.5) ** 2 / 6.25 for j in range(6)])
    b_w7 = np.array([1.0 - 0.3 * (j - 2.5) ** 2 / 6.25 for j in range(6)])
    sa, sb = np.sum(a_w7), np.sum(b_w7)
    a_w7 = a_w7 / sa
    b_w7 = b_w7 / sb
    a_full, b_full = make_symmetric_full(a_w7, b_w7)
    a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
    candidates.append((a_full, b_full))
    
    # Random symmetric candidates
    np.random.seed(42)
    for i in range(200):
        a_half = np.random.uniform(-0.3, 0.7, 6)
        b_half = np.random.uniform(-0.3, 0.7, 6)
        a_full, b_full = make_symmetric_full(a_half, b_half)
        a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
        candidates.append((a_full, b_full))
    
    return candidates

def _make_objective_symmetric():
    """Create a symmetric objective function for 12D parameter space (6+6)."""
    def objective_sym(x):
        a_half = x[:6]
        b_half = x[6:12]
        a_full, b_full = make_symmetric_full(a_half, b_half)
        a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
        try:
            return eval_coefficients_fast(a_full, b_full)
        except Exception:
            return 1e10
    return objective_sym

def optimize_coefficients():
    start_time = time.time()
    
    candidates = generate_symmetric_candidates()
    
    best_a = None
    best_b = None
    best_err = float('inf')
    
    # Phase 0: Evaluate candidates
    for i, (a_cand, b_cand) in enumerate(candidates):
        if time.time() - start_time > DEADLINE - 60:
            break
        
        a_cand, b_cand = normalize_coefficients(a_cand, b_cand)
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
    
    # Phase 1: L-BFGS-B on symmetric 12D parameter space with multiple restarts
    if time.time() - start_time < DEADLINE - 50:
        from scipy.optimize import minimize
        
        obj_sym = _make_objective_symmetric()
        
        a0_half = np.array(best_a[:6])
        b0_half = np.array(best_b[:6])
        x0_sym = np.concatenate([a0_half, b0_half])
        
        # Try multiple random restarts with L-BFGS-B
        np.random.seed(123)
        n_restarts = 30
        for r in range(n_restarts):
            if time.time() - start_time > DEADLINE - 50:
                break
            
            if r == 0:
                x_start = x0_sym
            else:
                # Random perturbation with decreasing scale
                scale = 0.15 - 0.004 * r
                if scale < 0.01:
                    scale = 0.01
                perturbation = np.random.normal(0, scale, len(x0_sym))
                x_start = x0_sym + perturbation
            
            options = {'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-11, 'maxcor': 12}
            try:
                result = minimize(obj_sym, x_start, method='L-BFGS-B', options=options)
                if result.fun < best_err:
                    a_half = result.x[:6]
                    b_half = result.x[6:12]
                    a_full, b_full = make_symmetric_full(a_half, b_half)
                    a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
                    best_err = result.fun
                    best_a = a_full
                    best_b = b_full
                    x0_sym = result.x.copy()
            except Exception:
                pass
    
    # Phase 2: Asymmetric refinement using L-BFGS-B on full 24D space
    if time.time() - start_time < DEADLINE - 25:
        from scipy.optimize import minimize
        
        def objective_asym(x):
            a_full = x[:12]
            b_full = x[12:24]
            a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
            try:
                return eval_coefficients_fast(a_full, b_full)
            except Exception:
                return 1e10
        
        a0_full = np.array(best_a)
        b0_full = np.array(best_b)
        x0_asym = np.concatenate([a0_full, b0_full])
        
        options = {'maxiter': 400, 'ftol': 1e-14, 'gtol': 1e-12, 'maxcor': 15}
        try:
            result = minimize(objective_asym, x0_asym, method='L-BFGS-B', options=options)
            if result.fun < best_err:
                a_full = result.x[:12]
                b_full = result.x[12:24]
                a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
                best_err = result.fun
                best_a = a_full
                best_b = b_full
        except Exception:
            pass
    
    # Phase 3: Final polish with very tight tolerances
    if time.time() - start_time < DEADLINE - 5:
        from scipy.optimize import minimize
        
        def objective_asym(x):
            a_full = x[:12]
            b_full = x[12:24]
            a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
            try:
                return eval_coefficients_fast(a_full, b_full)
            except Exception:
                return 1e10
        
        a0_full = np.array(best_a)
        b0_full = np.array(best_b)
        x0_asym = np.concatenate([a0_full, b0_full])
        
        options = {'maxiter': 600, 'ftol': 1e-16, 'gtol': 1e-14, 'maxcor': 20}
        try:
            result = minimize(objective_asym, x0_asym, method='L-BFGS-B', options=options)
            if result.fun < best_err:
                a_full = result.x[:12]
                b_full = result.x[12:24]
                a_full, b_full = normalize_coefficients(a_full.tolist(), b_full.tolist())
                best_err = result.fun
                best_a = a_full
                best_b = b_full
        except Exception:
            pass
    
    return best_a, best_b, best_err

def coefficients(stages):
    start_time = time.time()
    
    try:
        a_best, b_best, err = optimize_coefficients()
    except Exception:
        substeps = stages // 2
        a = [0.5 / substeps] * (2 * substeps)
        b = []
        for _ in range(substeps):
            b += [1.0 / substeps, 0.0]
        while len(a) < stages:
            a.append(0.0)
            b.append(0.0)
        return a, b
    
    elapsed = time.time() - start_time
    a_best, b_best = normalize_coefficients(a_best, b_best)
    return a_best, b_best
