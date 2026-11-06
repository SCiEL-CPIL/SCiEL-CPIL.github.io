"""
Optimized product formula for Heisenberg-chain time evolution.
Uses a fast local search with precomputed exponentials and symmetry
constraints to minimize mean spectral-norm error over three evolution times.
"""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _build_hamiltonians():
    """
    Build A (odd bonds + half uniform field) and B (even bonds + half uniform field)
    for a 6-site open XXZ Heisenberg chain with anisotropy 0.8 and field 0.3.
    """
    N = 6
    anis = 0.8
    field = 0.3
    dim = 2 ** N
    
    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    def kron_all(ops):
        result = np.array([[1.0]], dtype=complex)
        for op in ops:
            result = np.kron(result, op)
        return result
    
    def op_on_site(site, op, N):
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
        A += 0.5 * field * op_on_site(j, sz, N)
        B += 0.5 * field * op_on_site(j, sz, N)
    
    return A, B


def _precompute_eig(H):
    """Precompute eigen-decomposition for Hermitian H."""
    evals, evecs = np.linalg.eigh(H)
    return evals, evecs


def _expm_fast(evals, evecs, t):
    """Compute exp(-i*t*H) using precomputed eigen-decomposition."""
    phase = np.exp(-1j * t * evals)
    return evecs @ (phase[:, None] * evecs.conj().T)


def _objective(coeffs_vec, A_eig, B_eig, H_eig, dim):
    """
    Objective function: mean spectral norm error over t = 0.25, 0.5, 1.0.
    coeffs_vec: 24-element vector = [a_1..a_12, b_1..b_12]
    """
    stages = 12
    a = coeffs_vec[:stages]
    b = coeffs_vec[stages:]
    
    # Check consistency
    if abs(np.sum(a) - 1.0) > 1e-6 or abs(np.sum(b) - 1.0) > 1e-6:
        return 1e10
    
    # Check bounds
    if np.any(a < -50) or np.any(a > 50) or np.any(b < -50) or np.any(b > 50):
        return 1e10
    
    t_vals = [0.25, 0.5, 1.0]
    errors = []
    
    for t in t_vals:
        exact = _expm_fast(H_eig[0], H_eig[1], t)
        U = np.eye(dim, dtype=complex)
        
        for k in range(stages):
            exp_a = _expm_fast(A_eig[0], A_eig[1], a[k] * t)
            exp_b = _expm_fast(B_eig[0], B_eig[1], b[k] * t)
            U = exp_b @ exp_a @ U
        
        err = np.linalg.svd(U - exact, compute_uv=False)[0]
        errors.append(err)
    
    return np.mean(errors)


def _fast_local_search(objective, x0, max_iter=200, timeout=None, start_time=None):
    """
    Fast coordinate-descent style local search.
    Much faster than Nelder-Mead in high dimensions.
    """
    if start_time is None:
        start_time = time.time()
    
    n = len(x0)
    x_best = np.array(x0, dtype=float)
    f_best = objective(x_best)
    
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    current_step = 0
    
    for iteration in range(max_iter):
        if timeout is not None and (time.time() - start_time) > timeout:
            break
        
        improved = False
        
        for i in range(n):
            if timeout is not None and (time.time() - start_time) > timeout:
                break
            
            step = step_sizes[current_step]
            
            # Try positive and negative steps
            for sign in [1.0, -1.0]:
                x_trial = x_best.copy()
                x_trial[i] += sign * step
                
                # Check bounds
                if x_trial[i] < -50 or x_trial[i] > 50:
                    continue
                
                f_trial = objective(x_trial)
                
                if f_trial < f_best:
                    x_best = x_trial
                    f_best = f_trial
                    improved = True
        
        if not improved:
            current_step = min(current_step + 1, len(step_sizes) - 1)
        
        if current_step >= len(step_sizes) - 1 and not improved:
            break
    
    return x_best, f_best


def coefficients(stages):
    """
    Return 12-stage product formula coefficients (a, b) that minimize
    the mean spectral-norm error of the time evolution approximation.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build Hamiltonians once
    A, B = _build_hamiltonians()
    H = A + B
    dim = 2 ** 6
    
    # Precompute eigen-decompositions
    A_eig = _precompute_eig(A)
    B_eig = _precompute_eig(B)
    H_eig = _precompute_eig(H)
    
    # Generate multiple initial candidates using different splitting patterns
    candidates = []
    
    # Candidate 1: Symmetric 4th-order Yoshida-based
    c = 2.0 ** (1.0 / 3.0)
    x1 = 1.0 / (2.0 - c)
    x2 = -c / (2.0 - c)
    x3 = 1.0 - x1
    
    # 12-stage symmetric pattern: [w1,w2,w3,w4,w5,w6,w6,w5,w4,w3,w2,w1]
    # For A: distribute x3 at ends, x1 in middle
    a1 = np.array([x3, x1, x1, x1, x1, x3, x3, x1, x1, x1, x1, x3])
    a1 = a1 / np.sum(a1)
    
    # For B: distribute x2 evenly (negative values allowed)
    b1 = np.array([x2] * 12)
    b1 = b1 / np.sum(b1)
    
    candidates.append(np.concatenate([a1, b1]))
    
    # Candidate 2: Uniform distribution
    a2 = np.ones(12) / 12.0
    b2 = np.ones(12) / 12.0
    candidates.append(np.concatenate([a2, b2]))
    
    # Candidate 3: Strang-inspired with negative middle
    a3 = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15])
    b3 = np.array([0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    candidates.append(np.concatenate([a3, b3]))
    
    # Candidate 4: Concentrated at ends
    a4 = np.array([0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2])
    a4 = a4 / np.sum(a4)
    b4 = np.array([0.08, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.08])
    b4 = b4 / np.sum(b4)
    candidates.append(np.concatenate([a4, b4]))
    
    # Define objective
    def obj(vec):
        return _objective(vec, A_eig, B_eig, H_eig, dim)
    
    # Evaluate all candidates and pick the best as starting point
    best_candidate = None
    best_f = float('inf')
    
    for cand in candidates:
        f = obj(cand)
        if f < best_f:
            best_f = f
            best_candidate = cand
    
    if best_candidate is None:
        return (np.ones(12) / 12.0).tolist(), (np.ones(12) / 12.0).tolist()
    
    # Run fast local search from the best candidate
    remaining = deadline - time.time() - 5.0
    if remaining < 3.0:
        a_opt = best_candidate[:stages].tolist()
        b_opt = best_candidate[stages:].tolist()
    else:
        x_opt, f_opt = _fast_local_search(
            obj, best_candidate, 
            max_iter=100,
            timeout=remaining, 
            start_time=start_time
        )
        a_opt = x_opt[:stages].tolist()
        b_opt = x_opt[stages:].tolist()
    
    # Clamp to [-50, 50]
    a_opt = [max(-50.0, min(50.0, v)) for v in a_opt]
    b_opt = [max(-50.0, min(50.0, v)) for v in b_opt]
    
    # Normalize to ensure sum = 1
    sum_a = sum(a_opt)
    sum_b = sum(b_opt)
    if abs(sum_a) > 1e-10:
        a_opt = [v / sum_a for v in a_opt]
    if abs(sum_b) > 1e-10:
        b_opt = [v / sum_b for v in b_opt]
    
    # Final validation
    if len(a_opt) != stages or len(b_opt) != stages:
        return (np.ones(12) / 12.0).tolist(), (np.ones(12) / 12.0).tolist()
    
    return a_opt, b_opt

# EVOLVE-BLOCK-END
