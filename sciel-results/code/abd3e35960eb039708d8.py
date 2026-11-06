"""Optimized product formula for Heisenberg chain time evolution."""
import time
import numpy as np

def _build_hamiltonians():
    """Build A (odd bonds + half field) and B (even bonds + half field) for 6-site XXZ chain."""
    N = 6
    anis = 0.8
    field = 0.3
    dim = 2 ** N
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1.0]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def op_on_site(site, op):
        ops = [I2] * N
        ops[site] = op
        return kron_all(ops)
    
    def bond_op(j, op1, op2):
        ops = [I2] * N
        ops[j] = op1
        ops[j+1] = op2
        return kron_all(ops)
    
    A = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    
    # Odd bonds: (0,1), (2,3), (4,5) -> A
    for j in [0, 2, 4]:
        A += bond_op(j, sx, sx) + bond_op(j, sy, sy) + anis * bond_op(j, sz, sz)
    
    # Even bonds: (1,2), (3,4) -> B
    for j in [1, 3]:
        B += bond_op(j, sx, sx) + bond_op(j, sy, sy) + anis * bond_op(j, sz, sz)
    
    # Uniform field: half in A, half in B
    for j in range(N):
        A += 0.5 * field * op_on_site(j, sz)
        B += 0.5 * field * op_on_site(j, sz)
    
    return A, B


def _precompute_eigs(A, B, H):
    """Precompute eigendecompositions of A, B, and H using eigh for Hermitian matrices."""
    A_evals, A_evecs = np.linalg.eigh(A)
    B_evals, B_evecs = np.linalg.eigh(B)
    H_evals, H_evecs = np.linalg.eigh(H)
    
    return (A_evals, A_evecs), (B_evals, B_evecs), (H_evals, H_evecs)


def _expm_from_eigh(evals, evecs, scale):
    """Compute exp(scale * M) from precomputed eigh decomposition."""
    exp_vals = np.exp(scale * evals)
    return (evecs * exp_vals) @ evecs.conj().T


def _objective_fast(a, b, A_eig, B_eig, H_eig, dim):
    """Fast objective using precomputed eigendecompositions."""
    t_vals = [0.25, 0.5, 1.0]
    errors = []
    
    for t in t_vals:
        # Exact propagator
        exact = _expm_from_eigh(H_eig[0], H_eig[1], -1j * t)
        
        # Product formula: U = prod_k exp(-i b_k t B) exp(-i a_k t A)
        # Applied in order k = 0, 1, ..., 11
        U = np.eye(dim, dtype=complex)
        
        for k in range(len(a)):
            exp_a = _expm_from_eigh(A_eig[0], A_eig[1], -1j * a[k] * t)
            exp_b = _expm_from_eigh(B_eig[0], B_eig[1], -1j * b[k] * t)
            U = exp_b @ exp_a @ U
        
        diff = U - exact
        s = np.linalg.svd(diff, compute_uv=False)
        errors.append(s[0] if len(s) > 0 else 0.0)
    
    return np.mean(errors)


def coefficients(stages):
    """
    12-stage product formula with precomputed eigendecompositions and
    multi-start gradient-free optimization.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build Hamiltonians once
    A, B = _build_hamiltonians()
    H = A + B
    dim = A.shape[0]
    
    # Precompute eigendecompositions
    A_eig, B_eig, H_eig = _precompute_eigs(A, B, H)
    
    def eval_score(a, b):
        """Evaluate with bounds and sum checks."""
        if abs(np.sum(a) - 1.0) > 1e-6 or abs(np.sum(b) - 1.0) > 1e-6:
            return 1e10
        if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
            return 1e10
        return _objective_fast(a, b, A_eig, B_eig, H_eig, dim)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Equal weights baseline
    eq = np.ones(stages) / stages
    candidates.append((eq.copy(), eq.copy()))
    
    # 2. Symmetric 4th-order Suzuki-Trotter (Yoshida) coefficients
    c = 2.0 ** (1.0 / 3.0)
    x1 = 1.0 / (2.0 - c)
    x2 = -c / (2.0 - c)
    x3 = 1.0 - 2.0 * x1
    
    # Symmetric 6-step pattern, duplicated for 12 steps
    a_sym = np.array([x3/2, x1/2, x1/2, x3/2, x1/2, x1/2,
                       x3/2, x1/2, x1/2, x3/2, x1/2, x1/2])
    b_sym = np.array([x2/2, x2/2, 0, 0, x2/2, x2/2,
                       x2/2, x2/2, 0, 0, x2/2, x2/2])
    a_sym = a_sym / np.sum(a_sym)
    b_sym = b_sym / np.sum(b_sym)
    candidates.append((a_sym.copy(), b_sym.copy()))
    
    # 3. Alternating pattern (Strang-like)
    a_alt = np.zeros(stages)
    b_alt = np.zeros(stages)
    for k in range(stages):
        if k % 2 == 0:
            a_alt[k] = 2.0 / stages
            b_alt[k] = 0.0
        else:
            a_alt[k] = 0.0
            b_alt[k] = 2.0 / stages
    a_alt = a_alt / np.sum(a_alt)
    b_alt = b_alt / np.sum(b_alt)
    candidates.append((a_alt.copy(), b_alt.copy()))
    
    # 4. Linear ramp
    a_ramp = np.linspace(0.5, 1.5, stages)
    b_ramp = np.linspace(1.5, 0.5, stages)
    a_ramp = a_ramp / np.sum(a_ramp)
    b_ramp = b_ramp / np.sum(b_ramp)
    candidates.append((a_ramp.copy(), b_ramp.copy()))
    
    # 5. Center-weighted
    center = np.arange(stages) - stages / 2.0
    a_cw = np.exp(-0.5 * (center / (stages/2.0))**2)
    b_cw = np.exp(-0.5 * (center / (stages/2.0))**2)
    a_cw = a_cw / np.sum(a_cw)
    b_cw = b_cw / np.sum(b_cw)
    candidates.append((a_cw.copy(), b_cw.copy()))
    
    # Evaluate all candidates
    best_a = candidates[0][0].copy()
    best_b = candidates[0][1].copy()
    best_score = eval_score(best_a, best_b)
    
    for a_c, b_c in candidates[1:]:
        s = eval_score(a_c, b_c)
        if s < best_score:
            best_a = a_c.copy()
            best_b = b_c.copy()
            best_score = s
    
    # Greedy coordinate descent with line search
    # Optimize each (a_k, b_k) pair sequentially, then repeat passes
    def line_search_1d(f, x0, dx, lo=-50.0, hi=50.0, max_evals=20):
        """Golden-section line search along direction dx from x0."""
        # Find bracket
        step = 0.05
        best_f = f(x0)
        best_x = x0.copy()
        for _ in range(max_evals):
            if time.time() > deadline - 10:
                break
            x_trial = x0 + step * dx
            if np.any(x_trial < lo) or np.any(x_trial > hi):
                step *= 0.5
                continue
            f_trial = f(x_trial)
            if f_trial < best_f:
                best_f = f_trial
                best_x = x_trial.copy()
            else:
                step *= 0.5
        return best_x, best_f
    
    def line_search_2d(f, a0, b0, da, db, lo=-50.0, hi=50.0, max_evals=30):
        """2D line search: try perturbing a and b along given directions."""
        best_f = f(a0, b0)
        best_a = a0.copy()
        best_b = b0.copy()
        
        # Try pure a direction
        for step in [0.05, 0.1, 0.2, 0.02, 0.5]:
            if time.time() > deadline - 10:
                break
            a_trial = a0 + step * da
            b_trial = b0.copy()
            if np.any(a_trial < lo) or np.any(a_trial > hi):
                continue
            f_trial = f(a_trial, b_trial)
            if f_trial < best_f:
                best_f = f_trial
                best_a = a_trial.copy()
                best_b = b_trial.copy()
        
        # Try pure b direction
        for step in [0.05, 0.1, 0.2, 0.02, 0.5]:
            if time.time() > deadline - 10:
                break
            a_trial = a0.copy()
            b_trial = b0 + step * db
            if np.any(b_trial < lo) or np.any(b_trial > hi):
                continue
            f_trial = f(a_trial, b_trial)
            if f_trial < best_f:
                best_f = f_trial
                best_a = a_trial.copy()
                best_b = b_trial.copy()
        
        # Try combined direction
        for step in [0.03, 0.07, 0.15, 0.01, 0.3]:
            if time.time() > deadline - 10:
                break
            a_trial = a0 + step * da
            b_trial = b0 + step * db
            if np.any(a_trial < lo) or np.any(a_trial > hi) or np.any(b_trial < lo) or np.any(b_trial > hi):
                continue
            f_trial = f(a_trial, b_trial)
            if f_trial < best_f:
                best_f = f_trial
                best_a = a_trial.copy()
                best_b = b_trial.copy()
        
        return best_a, best_b, best_f
    
    # Coordinate descent: optimize each stage k's (a_k, b_k)
    # We keep the sum constraint by adjusting the last element
    a_cd = best_a.copy()
    b_cd = best_b.copy()
    
    max_passes = 20
    for p in range(max_passes):
        if time.time() > deadline - 15:
            break
        
        improved = False
        for k in range(stages - 1):  # Leave last element as the slack
            if time.time() > deadline - 10:
                break
            
            # Current values
            ak = a_cd[k]
            bk = b_cd[k]
            
            # Try perturbing a_k and b_k independently
            # Direction: try to reduce error by adjusting this stage
            # Use small random perturbations as search directions
            np.random.seed(k + p * 100 + 42)
            da = np.zeros(stages)
            db = np.zeros(stages)
            
            # Try several random directions
            best_local_f = eval_score(a_cd, b_cd)
            best_local_a = a_cd.copy()
            best_local_b = b_cd.copy()
            
            for trial in range(8):
                if time.time() > deadline - 10:
                    break
                
                # Random direction in the (a_k, b_k) plane, keeping sum=1
                da_k = np.random.uniform(-0.3, 0.3)
                db_k = np.random.uniform(-0.3, 0.3)
                
                # Apply: increase a_k by da_k, decrease a_{last} by da_k to keep sum=1
                a_trial = a_cd.copy()
                b_trial = b_cd.copy()
                a_trial[k] += da_k
                a_trial[-1] -= da_k
                b_trial[k] += db_k
                b_trial[-1] -= db_k
                
                if np.any(a_trial < -50) or np.any(a_trial > 50) or np.any(b_trial < -50) or np.any(b_trial > 50):
                    continue
                
                f_trial = eval_score(a_trial, b_trial)
                if f_trial < best_local_f:
                    best_local_f = f_trial
                    best_local_a = a_trial.copy()
                    best_local_b = b_trial.copy()
                    improved = True
            
            if f_trial < best_score:
                best_a = best_local_a.copy()
                best_b = best_local_b.copy()
                best_score = best_local_f
            
            a_cd = best_local_a
            b_cd = best_local_b
        
        if not improved:
            # No improvement in this pass; try with larger steps
            np.random.seed(p * 200 + 7)
            for k in range(stages - 1):
                if time.time() > deadline - 10:
                    break
                
                best_local_f = eval_score(a_cd, b_cd)
                best_local_a = a_cd.copy()
                best_local_b = b_cd.copy()
                
                for trial in range(6):
                    if time.time() > deadline - 10:
                        break
                    
                    da_k = np.random.uniform(-1.0, 1.0)
                    db_k = np.random.uniform(-1.0, 1.0)
                    
                    a_trial = a_cd.copy()
                    b_trial = b_cd.copy()
                    a_trial[k] += da_k
                    a_trial[-1] -= da_k
                    b_trial[k] += db_k
                    b_trial[-1] -= db_k
                    
                    if np.any(a_trial < -50) or np.any(a_trial > 50) or np.any(b_trial < -50) or np.any(b_trial > 50):
                        continue
                    
                    f_trial = eval_score(a_trial, b_trial)
                    if f_trial < best_local_f:
                        best_local_f = f_trial
                        best_local_a = a_trial.copy()
                        best_local_b = b_trial.copy
                
                if best_local_f < best_score:
                    best_a = best_local_a.copy()
                    best_b = best_local_b.copy()
                    best_score = best_local_f
                
                a_cd = best_local_a
                b_cd = best_local_b
        
        # If no improvement across entire pass, break early
        if time.time() > deadline - 20:
            break
    
    # Final cleanup
    best_a = np.clip(best_a, -50, 50)
    best_b = np.clip(best_b, -50, 50)
    best_a = best_a + (1.0 - np.sum(best_a)) / stages
    best_b = best_b + (1.0 - np.sum(best_b)) / stages
    
    if np.any(best_a < -50) or np.any(best_a > 50) or np.any(best_b < -50) or np.any(best_b > 50):
        best_a = np.ones(stages) / stages
        best_b = np.ones(stages) / stages
    
    # Final verification
    final_score = eval_score(best_a, best_b)
    if final_score < best_score:
        best_score = final_score
    
    return list(best_a), list(best_b)
