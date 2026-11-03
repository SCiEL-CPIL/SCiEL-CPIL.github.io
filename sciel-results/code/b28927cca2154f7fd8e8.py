import time
import numpy as np
from scipy.optimize import minimize

N = 100
SCALE = 2.0 / N

def _compute_O(h):
    """
    Compute the overlap vector O for all shifts k=0..N-1.
    O_pos[k] = SCALE * sum_{j=k}^{N-1} h[j]*(1-h[j-k])
    O_neg[k] = SCALE * sum_{j=0}^{N-1-k} h[j]*(1-h[j+k])
    """
    S1 = np.cumsum(h[::-1])[::-1]
    S2 = np.array([np.dot(h[k:], h[:N - k]) for k in range(N)])
    O_pos = SCALE * (S1 - S2)
    
    csum = np.cumsum(h)
    S1n = csum[N - 1 - np.arange(N)]
    S2n = np.array([np.dot(h[:N - k], h[k:]) for k in range(N)])
    O_neg = SCALE * (S1n - S2n)
    
    return np.concatenate([O_pos, O_neg])

def _compute_grad_O(h):
    """
    Compute the gradient of each O[k] with respect to h.
    Returns a (2N, N) matrix where row k is dO[k]/dh.
    """
    grad = np.zeros((2 * N, N))
    
    for k in range(N):
        # O_pos[k] = SCALE * sum_{j=k}^{N-1} h[j]*(1-h[j-k])
        # dO_pos[k]/dh[j] for j in [k, N-1]:
        #   term h[j]*(1-h[j-k]): dh[j] = (1-h[j-k]), dh[j-k] = -h[j]
        # dO_pos[k]/dh[j] for j in [0, k-1]:
        #   term h[j+k]*(1-h[j]): dh[j] = -h[j+k]
        
        for j in range(k, N):
            grad[k, j] += SCALE * (1.0 - h[j - k])
            if j - k >= 0:
                grad[k, j - k] += SCALE * (-h[j])
        
        for j in range(0, k):
            grad[k, j] += SCALE * (-h[j + k])
        
        # O_neg[k] = SCALE * sum_{j=0}^{N-1-k} h[j]*(1-h[j+k])
        # dO_neg[k]/dh[j] for j in [0, N-1-k]:
        #   term h[j]*(1-h[j+k]): dh[j] = (1-h[j+k]), dh[j+k] = -h[j]
        
        for j in range(0, N - k):
            grad[k + N, j] += SCALE * (1.0 - h[j + k])
            if j + k < N:
                grad[k + N, j + k] += SCALE * (-h[j])
    
    return grad

def _soft_max(O, beta=50.0):
    """
    Smooth approximation of max(O) using log-sum-exp.
    """
    m = O.max()
    return m + np.log(np.sum(np.exp(beta * (O - m)))) / beta

def _soft_max_grad(O, beta=50.0):
    """
    Gradient of soft-max with respect to O.
    """
    m = O.max()
    exp_terms = np.exp(beta * (O - m))
    weights = exp_terms / np.sum(exp_terms)
    return weights

def _project(z):
    """
    Project z onto {h in [0,1], sum h = N/2}
    """
    lo = float(z.min()) - 1.0
    hi = float(z.max()) + 1.0
    target = N / 2.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        s = np.clip(z - mid, 0.0, 1.0).sum()
        if s > target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return np.clip(z - t, 0.0, 1.0)

def _soft_max_objective(h, beta=50.0):
    """
    Objective: soft-max of overlap values.
    """
    O = _compute_O(h)
    return _soft_max(O, beta)

def _soft_max_gradient(h, beta=50.0):
    """
    Gradient of soft-max objective with respect to h.
    """
    O = _compute_O(h)
    grad_O = _compute_grad_O(h)
    w = _soft_max_grad(O, beta)
    # Gradient = grad_O.T @ w
    return grad_O.T @ w

def _fista(h_init, deadline, max_iter=2000, beta=50.0):
    """
    FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for minimizing
    soft-max of overlaps subject to h in [0,1] and sum(h) = N/2.
    """
    h = _project(np.asarray(h_init, dtype=float).copy())
    best_h = h.copy()
    best_val = float(_compute_O(best_h).max())
    
    # FISTA parameters
    L = 10.0  # Lipschitz constant estimate
    t_k = 1.0
    z = h.copy()
    
    start_time = time.time()
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        # Compute gradient at z
        grad_z = _soft_max_gradient(z, beta)
        
        # Gradient step
        y = z - grad_z / L
        
        # Project onto constraint set
        x_new = _project(y)
        
        # Update FISTA momentum
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k * t_k)) / 2.0
        z = x_new + ((t_k - 1.0) / t_next) * (x_new - h)
        t_k = t_next
        h = x_new
        
        # Track best
        val = float(_compute_O(h).max())
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h, best_val

def _slsqp_optimize(h_init, deadline, beta=50.0):
    """
    Use SLSQP to minimize soft-max objective.
    """
    h0 = _project(np.asarray(h_init, dtype=float).copy())
    
    def objective(x):
        return _soft_max_objective(x, beta)
    
    def gradient(x):
        return _soft_max_gradient(x, beta)
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0,
                    'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N
    
    start_time = time.time()
    try:
        result = minimize(objective, h0, jac=gradient, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 100, 'ftol': 1e-10})
        
        if result.success and time.time() < deadline - 5.0:
            h_opt = _project(result.x)
            val = float(_compute_O(h_opt).max())
            return h_opt, val
    except Exception:
        pass
    
    return h_init, float(_compute_O(h_init).max())

def _simulated_annealing(h_init, deadline, max_iter=50000):
    """
    Simulated annealing on the continuous space [0,1]^N with sum constraint.
    Uses a cooling schedule and random perturbations.
    """
    h = _project(np.asarray(h_init, dtype=float).copy())
    best_h = h.copy()
    best_val = float(_compute_O(h).max())
    
    T_start = 0.05
    T_end = 0.001
    rng = np.random.default_rng(42)
    
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        # Cooling schedule
        progress = it / max_iter
        T = T_start * (T_end / T_start) ** progress
        
        # Generate a random perturbation
        n_perturb = rng.integers(1, max(2, N // 10) + 1)
        indices = rng.choice(N, n_perturb, replace=False)
        deltas = rng.normal(0, 0.1, size=n_perturb)
        
        h_new = h.copy()
        h_new[indices] += deltas
        h_new = _project(h_new)
        
        val_new = float(_compute_O(h_new).max())
        delta_val = val_new - best_val
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta_val < 0 or rng.random() < np.exp(-delta_val / max(T, 1e-10)):
            h = h_new
            if val_new < best_val:
                best_val = val_new
                best_h = h_new.copy()
    
    return best_h, best_val

def _local_refine(h, deadline, max_iter=10000):
    """
    Local refinement using small random steps with acceptance criterion.
    """
    h = h.copy()
    best_h = h.copy()
    best_val = float(_compute_O(h).max())
    
    rng = np.random.default_rng(123)
    
    for it in range(max_iter):
        if time.time() > deadline - 1.0:
            break
        
        # Small random perturbation
        idx = rng.integers(0, N)
        delta = rng.normal(0, 0.01)
        
        h_new = h.copy()
        h_new[idx] += delta
        h_new = _project(h_new)
        
        val_new = float(_compute_O(h_new).max())
        
        if val_new < best_val - 1e-12:
            h = h_new
            best_val = val_new
            best_h = h_new.copy()
        elif val_new < best_val:
            h = h_new
    
    return best_h, best_val

def _bang_bang_local(h_init, deadline, max_iter=20000):
    """
    Local search on bang-bang (0/1) solutions with pair swaps.
    """
    h = _project(np.asarray(h_init, dtype=float).copy())
    h_bb = (h > 0.5).astype(float)
    
    num_ones = int(h_bb.sum())
    rng = np.random.default_rng(42)
    if num_ones > N // 2:
        idxs = np.where(h_bb == 1.0)[0]
        to_remove = rng.choice(idxs, num_ones - N // 2, replace=False)
        h_bb[to_remove] = 0.0
    elif num_ones < N // 2:
        idxs = np.where(h_bb == 0.0)[0]
        to_add = rng.choice(idxs, N // 2 - num_ones, replace=False)
        h_bb[to_add] = 1.0
    
    best_h = h_bb.copy()
    best_val = float(_compute_O(h_bb).max())
    
    rng2 = np.random.default_rng(123)
    
    for it in range(max_iter):
        if time.time() > deadline - 1.0:
            break
        
        i, j = rng2.choice(N, 2, replace=False)
        if h_bb[i] == h_bb[j]:
            continue
        
        h_bb[i], h_bb[j] = h_bb[j], h_bb[i]
        val = float(_compute_O(h_bb).max())
        
        if val < best_val - 1e-12:
            best_val = val
            best_h = h_bb.copy()
        else:
            h_bb[i], h_bb[j] = h_bb[j], h_bb[i]
    
    return best_h, best_val

def construct():
    start = time.time()
    deadline = start + 145.0
    
    idx = np.arange(N)
    rng = np.random.default_rng(42)
    
    # Generate diverse initial solutions
    inits = []
    
    inits.append(np.full(N, 0.5))
    
    for f in (1, 2, 3, 5, 7, 11, 13, 17, 23, 31):
        inits.append(0.5 + 0.4 * np.cos(2.0 * np.pi * f * idx / N))
    
    for _ in range(20):
        inits.append(rng.random(N))
    
    for _ in range(20):
        bb = np.zeros(N)
        bb[rng.choice(N, N // 2, replace=False)] = 1.0
        inits.append(bb)
    
    for period in [2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 17, 19, 23, 29]:
        bb = np.zeros(N)
        ones = period // 2
        pattern = np.zeros(period)
        pattern[:ones] = 1.0
        bb = np.tile(pattern, N // period + 1)[:N]
        if bb.sum() > N/2:
            idxs = np.where(bb==1)[0]
            to_remove = rng.choice(idxs, int(bb.sum() - N/2), replace=False)
            bb[to_remove] = 0
        elif bb.sum() < N/2:
            idxs = np.where(bb==0)[0]
            to_add = rng.choice(idxs, int(N/2 - bb.sum()), replace=False)
            bb[to_add] = 1
        inits.append(bb)
    
    # Phase 1: Simulated annealing on each init
    best_h = inits[0].copy()
    best_val = float(_compute_O(best_h).max())
    
    per_init_time = 5.0
    for init in inits:
        if time.time() > deadline - 40.0:
            break
        try:
            h_sa, val_sa = _simulated_annealing(init, start + (time.time() - start) + per_init_time)
            if val_sa < best_val:
                best_val = val_sa
                best_h = h_sa
        except Exception:
            continue
    
    # Phase 2: FISTA with soft-max surrogate from best SA result
    if time.time() < deadline - 30.0:
        try:
            h_fista, val_fista = _fista(best_h, deadline - 25.0, beta=50.0)
            if val_fista < best_val:
                best_val = val_fista
                best_h = h_fista
        except Exception:
            pass
    
    # Phase 3: SLSQP from FISTA result
    if time.time() < deadline - 20.0:
        try:
            h_slsqp, val_slsqp = _slsqp_optimize(best_h, deadline - 15.0, beta=50.0)
            if val_slsqp < best_val:
                best_val = val_slsqp
                best_h = h_slsqp
        except Exception:
            pass
    
    # Phase 4: Local refinement
    if time.time() < deadline - 15.0:
        h_ref, val_ref = _local_refine(best_h, deadline - 10.0)
        if val_ref < best_val:
            best_val = val_ref
            best_h = h_ref
    
    # Phase 5: Bang-bang local search
    if time.time() < deadline - 10.0:
        h_bb, val_bb = _bang_bang_local(best_h, deadline - 5.0)
        if val_bb < best_val:
            best_val = val_bb
            best_h = h_bb
    
    # Phase 6: Final FISTA polish with higher beta
    if time.time() < deadline - 8.0:
        try:
            h_fista2, val_fista2 = _fista(best_h, deadline - 3.0, beta=100.0)
            if val_fista2 < best_val:
                best_val = val_fista2
                best_h = h_fista2
        except Exception:
            pass
    
    # Phase 7: Final local refinement
    if time.time() < deadline - 3.0:
        h_final, val_final = _local_refine(best_h, deadline - 1.0)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    # Ensure exact sum and bounds
    best_h = _project(best_h)
    for _ in range(10):
        d = N / 2.0 - best_h.sum()
        if abs(d) < 1e-12:
            break
        best_h = np.clip(best_h + d / N, 0.0, 1.0)
        best_h = _project(best_h)
        
    return best_h
