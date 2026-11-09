import numpy as np
import time
from scipy.linalg import expm
from scipy.optimize import minimize

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    # 6-site open chain, sites 0..5
    # XXZ: Jz * Sz_i Sz_j + Jxy/2 * (Sx_i Sx_j + Sy_i Sy_j)
    # anisotropy delta = 0.8 means Jxy/Jz = 0.8
    # field h = 0.3
    n = 6
    dim = 2**n
    delta = 0.8
    h = 0.3
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    # Build terms for each bond
    # Bond i connects site i and i+1
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    # Even bonds: (1,2), (3,4) -> B
    # Field: h * sum Sz_i, split half to A and half to B
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # XXZ interaction: Jz * Sz_i Sz_j + (Jz*delta)/2 * (Sx_i Sx_j + Sy_i Sy_j)
    # Let Jz = 1, so Jxy = delta = 0.8
    Jz = 1.0
    Jxy = delta * Jz
    
    def add_bond(H, i, j):
        """Add XXZ bond between sites i and j to H."""
        # Sz_i Sz_j
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        ops[j] = sz
        H += Jz * kron_all(ops)
        
        # Sx_i Sx_j
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sx
        ops[j] = sx
        H += (Jxy / 2) * kron_all(ops)
        
        # Sy_i Sy_j
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sy
        ops[j] = sy
        H += (Jxy / 2) * kron_all(ops)
    
    # Odd bonds: (0,1), (2,3), (4,5)
    for (i, j) in [(0, 1), (2, 3), (4, 5)]:
        add_bond(A, i, j)
    
    # Even bonds: (1,2), (3,4)
    for (i, j) in [(1, 2), (3, 4)]:
        add_bond(B, i, j)
    
    # Field: h * sum_i Sz_i, split half to each
    for i in range(n):
        ops = [np.eye(2) for _ in range(n)]
        ops[i] = sz
        field_term = h * 0.5 * kron_all(ops)
        A += field_term
        B += field_term
    
    return A, B

def _spectral_norm_error(a, b, A, B, t):
    """Compute spectral norm of U(t) - exp(-i t (A+B))."""
    # Build product formula
    U = np.eye(A.shape[0], dtype=complex)
    for k in range(len(a)):
        U = expm(-1j * b[k] * t * B) @ expm(-1j * a[k] * t * A) @ U
    # Exact propagator
    U_exact = expm(-1j * t * (A + B))
    # Spectral norm of difference
    diff = U - U_exact
    # Use SVD for spectral norm
    s = np.linalg.svd(diff, compute_uv=False)
    return s[0]

def coefficients(stages):
    """Optimize 12-stage product formula coefficients for Heisenberg chain."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    A, B = _build_hamiltonians()
    dim = A.shape[0]
    times = [0.25, 0.5, 1.0]
    
    # Initial guess: symmetric Strang-like splitting
    # For 12 stages, use a symmetric pattern
    # Pattern: a = [c1, c2, c3, c4, c5, c6, c6, c5, c4, c3, c2, c1]
    #          b = [d1, d2, d3, d4, d5, d6, d6, d5, d4, d3, d2, d1]
    # But we need sum(a)=sum(b)=1
    
    # Start with uniform-ish symmetric guess
    # 12 stages, symmetric means first 6 determine last 6
    n_half = stages // 2  # 6
    
    # Initial: simple symmetric with equal weights
    a_init = np.ones(n_half) / stages  # each = 1/12, sum of first half = 6/12 = 0.5
    # Full a: [a_init, a_init[::-1]] -> sum = 1
    a_full_init = np.concatenate([a_init, a_init[::-1]])
    
    b_init = np.ones(n_half) / stages
    b_full_init = np.concatenate([b_init, b_init[::-1]])
    
    # Actually, let's use a better initial guess based on known high-order splitting
    # For 2nd order Strang: a = [1/2, 1/2], b = [1, 0] per substep
    # For higher order, use Yoshida or similar coefficients
    
    # Let's try a 4th-order-like symmetric pattern
    # For a 12-stage symmetric formula, we can use nested Strang or explicit coefficients
    # Known 4th order coefficients for 2-term splitting (from Suzuki):
    # x1 = 1/(2-2^(1/3)), x0 = 1 - x1
    # For 12 stages, we can do 2 substeps of 6-stage, or use a direct 12-stage formula
    
    # Let's just optimize numerically. Use the symmetric parameterization.
    
    def objective(params):
        """Objective: mean spectral norm error over 3 times."""
        # params: 12 a's + 12 b's, but enforce symmetry and sum constraints
        # Use 6 a_half + 6 b_half, construct full with symmetry
        a_half = params[:6]
        b_half = params[6:12]
        
        a_full = np.concatenate([a_half, a_half[::-1]])
        b_full = np.concatenate([b_half, b_half[::-1]])
        
        # Enforce sum constraints by scaling
        a_sum = np.sum(a_full)
        b_sum = np.sum(b_full)
        
        if abs(a_sum) < 1e-12:
            a_full = np.ones(12) / 12
        else:
            a_full = a_full / a_sum
        
        if abs(b_sum) < 1e-12:
            b_full = np.ones(12) / 12
        else:
            b_full = b_full / b_sum
        
        # Clip to [-50, 50]
        a_full = np.clip(a_full, -50, 50)
        b_full = np.clip(b_full, -50, 50)
        
        # Re-normalize after clipping
        a_sum = np.sum(a_full)
        b_sum = np.sum(b_full)
        if abs(a_sum) < 1e-12:
            a_full = np.ones(12) / 12
        else:
            a_full = a_full / a_sum
        if abs(b_sum) < 1e-12:
            b_full = np.ones(12) / 12
        else:
            b_full = b_full / b_sum
        
        # Compute mean spectral norm error
        total_err = 0.0
        for t in times:
            U = np.eye(dim, dtype=complex)
            for k in range(12):
                U = expm(-1j * b_full[k] * t * B) @ expm(-1j * a_full[k] * t * A) @ U
            U_exact = expm(-1j * t * (A + B))
            diff = U - U_exact
            s = np.linalg.svd(diff, compute_uv=False)
            total_err += s[0]
        
        return total_err / len(times)
    
    # Initial parameters (symmetric half)
    x0 = np.concatenate([a_full_init[:6], b_full_init[:6]])
    
    # Try a few different initializations
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Try some structured initializations
    # 1. Strang-like: a_half = [1/12]*6, b_half = [1/6, 0, 1/6, 0, 1/6, 0] scaled
    # Actually for Strang with 2 substeps: each substep has a=[0.5,0.5], b=[1,0]
    # For 12 stages = 2 substeps * 6 stages each? No, 12 stages total.
    
    # Let me try: treat it as 6 substeps of 2 stages each (Strang)
    # Each Strang: exp(-i b t B) exp(-i a t A) exp(-i b t B) but we only have one B and one A per stage
    # So our formula is strictly alternating B, A, B, A, ...
    
    # For 12 stages: B A B A B A B A B A B A
    # Symmetric: b = [b1,b2,b3,b4,b5,b6,b6,b5,b4,b3,b2,b1]
    #            a = [a1,a2,a3,a4,a5,a6,a6,a5,a4,a3,a2,a1]
    
    # Let's try optimization with L-BFGS-B
    results = []
    
    for init_idx, x_init in enumerate([x0]):
        if time.time() > deadline - 60:
            break
        
        try:
            result = minimize(
                objective,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x.copy()
            results.append((init_idx, result.fun, result.success))
        except Exception as e:
            results.append((init_idx, float('inf'), str(e)))
        
        if time.time() > deadline - 30:
            break
    
    # Try a second round with perturbed initial conditions if time allows
    if time.time() < deadline - 40:
        for trial in range(3):
            if time.time() > deadline - 20:
                break
            x_perturbed = best_x + 0.01 * np.random.randn(12)
            try:
                result = minimize(
                    objective,
                    x_perturbed,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'maxfun': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_x = result.x.copy()
            except:
                pass
    
    # Extract final coefficients
    a_half = best_x[:6]
    b_half = best_x[6:12]
    
    a_full = np.concatenate([a_half, a_half[::-1]])
    b_full = np.concatenate([b_half, b_half[::-1]])
    
    # Enforce sum constraints
    a_sum = np.sum(a_full)
    b_sum = np.sum(b_full)
    if abs(a_sum) > 1e-12:
        a_full = a_full / a_sum
    else:
        a_full = np.ones(12) / 12
    if abs(b_sum) > 1e-12:
        b_full = b_full / b_sum
    else:
        b_full = np.ones(12) / 12
    
    # Clip to [-50, 50] and re-normalize
    a_full = np.clip(a_full, -50, 50)
    b_full = np.clip(b_full, -50, 50)
    a_sum = np.sum(a_full)
    b_sum = np.sum(b_full)
    if abs(a_sum) > 1e-12:
        a_full = a_full / a_sum
    if abs(b_sum) > 1e-12:
        b_full = b_full / b_sum
    
    # Final check
    if time.time() < deadline - 5:
        final_obj = objective(best_x)
        if final_obj < best_obj:
            # Re-extract
            a_half = best_x[:6]
            b_half = best_x[6:12]
            a_full = np.concatenate([a_half, a_half[::-1]])
            b_full = np.concatenate([b_half, b_half[::-1]])
            a_sum = np.sum(a_full)
            b_sum = np.sum(b_full)
            if abs(a_sum) > 1e-12:
                a_full = a_full / a_sum
            if abs(b_sum) > 1e-12:
                b_full = b_full / b_sum
    
    # Convert to lists
    a_list = [float(x) for x in a_full]
    b_list = [float(x) for x in b_full]
    
    # Final safety: ensure sums are exactly 1 (within floating point)
    a_sum = sum(a_list)
    b_sum = sum(b_list)
    if abs(a_sum - 1.0) > 1e-10:
        a_list = [x / a_sum for x in a_list]
    if abs(b_sum - 1.0) > 1e-10:
        b_list = [x / b_sum for x in b_list]
    
    return a_list, b_list
