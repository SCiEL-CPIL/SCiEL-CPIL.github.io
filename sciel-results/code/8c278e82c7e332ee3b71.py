"""Combined approach: Bang-bang local search + continuous subgradient polish."""
import time
import numpy as np

N = 100
W = 2.0 / N
TARGET_SUM = N / 2.0

# EVOLVE-BLOCK-START
def _compute_overlaps(h):
    """
    Compute the maximum overlap of h with shifted 1-h for all shifts.
    """
    S1 = np.cumsum(h[::-1])[::-1] # S1[k] = sum h[k:]
    S2 = np.array([np.dot(h[k:], h[:N - k]) for k in range(N)])
    O_pos = W * (S1 - S2)
    
    csum = np.cumsum(h)
    # S1n[k] = sum h[:N-k]
    S1n = csum[N - 1 - np.arange(N)]
    S2n = np.array([np.dot(h[:N - k], h[k:]) for k in range(N)])
    O_neg = W * (S1n - S2n)
    
    return np.concatenate([O_pos, O_neg])

def _max_overlap(h):
    return _compute_overlaps(h).max()

def _project(h):
    """Project h onto [0,1]^N with sum N/2."""
    lo = float(h.min()) - 1.0
    hi = float(h.max()) + 1.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        s = np.clip(h - mid, 0.0, 1.0).sum()
        if s > TARGET_SUM:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return np.clip(h - t, 0.0, 1.0)

def _grad_max_overlap(h, O=None):
    """
    Compute the gradient of the maximum overlap.
    Uses the active set of shifts.
    """
    if O is None:
        O = _compute_overlaps(h)
    Omax = O.max()
    grad = np.zeros(N)
    active = np.where(O > Omax - 1e-8)[0]
    
    # Precompute masks and indices for efficiency? 
    # N is small, so looping is fine.
    
    for k_idx in active:
        if k_idx < N:
            k = k_idx
            # O_pos[k]
            # grad[j] += W * (1 - h[j-k]) for j>=k
            # grad[j] -= W * h[j+k] for j<N-k
            j_range = np.arange(N)
            mask1 = j_range >= k
            mask2 = j_range < N - k
            grad[mask1] += W * (1 - h[j_range[mask1] - k])
            grad[mask2] -= W * h[j_range[mask2] + k]
        else:
            k = k_idx - N
            # O_neg[k]
            # grad[j] += W * (1 - h[j+k]) for j<N-k
            # grad[j] -= W * h[j-k] for j>=k
            j_range = np.arange(N)
            mask1 = j_range < N - k
            mask2 = j_range >= k
            grad[mask1] += W * (1 - h[j_range[mask1] + k])
            grad[mask2] -= W * h[j_range[mask2] - k]
            
    return grad

def _subgrad_polish(h, deadline):
    """
    Subgradient descent on the max overlap function.
    """
    h = _project(h.copy())
    best_h = h.copy()
    best_val = _max_overlap(h)
    
    step = 0.05
    t_fista = 1.0
    h_prev = h.copy()
    patience = 0
    
    for i in range(5000):
        if time.time() > deadline - 5.0:
            break
            
        O = _compute_overlaps(h)
        Omax = O.max()
        grad = _grad_max_overlap(h, O)
        
        # FISTA step
        y = h - step * grad
        h_new = _project(y)
        
        # Check improvement
        val_new = _max_overlap(h_new)
        if val_new < best_val:
            best_h = h_new.copy()
            best_val = val_new
            patience = 0
        else:
            patience += 1
            if patience > 100:
                step *= 0.9
                patience = 0
        
        # Update FISTA variables
        t_new = (1 + np.sqrt(1 + 4 * t_fista * t_fista)) / 2
        h_prev = h.copy()
        h = _project(h + ((t_fista - 1) / t_new) * (h - h_prev))
        t_fista = t_new
        
        # Simple step size adjustment
        if val_new < Omax:
            step *= 1.05
            if step > 0.5: step = 0.5
        else:
            step *= 0.95
            if step < 1e-5: step = 1e-5

    return best_h, best_val

def _local_search_bb(h, deadline):
    """
    Convert to bang-bang and perform local search.
    """
    h_bb = (h >= 0.5).astype(float)
    # Ensure sum is N/2
    diff = TARGET_SUM - h_bb.sum()
    if diff > 0:
        zeros = np.where(h_bb == 0)[0]
        if len(zeros) >= int(diff):
            rng = np.random.default_rng(42)
            idx = rng.choice(zeros, size=int(diff), replace=False)
            h_bb[idx] = 1.0
    elif diff < 0:
        ones = np.where(h_bb == 1)[0]
        if len(ones) >= int(-diff):
            rng = np.random.default_rng(42)
            idx = rng.choice(ones, size=int(-diff), replace=False)
            h_bb[idx] = 0.0
            
    best_val = _max_overlap(h_bb)
    best_h = h_bb.copy()
    
    rng = np.random.default_rng(0)
    for _ in range(5000):
        if time.time() > deadline - 5.0:
            break
        ones = np.where(h_bb == 1)[0]
        zeros = np.where(h_bb == 0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        i = rng.choice(ones)
        j = rng.choice(zeros)
        
        h_test = h_bb.copy()
        h_test[i] = 0.0
        h_test[j] = 1.0
        
        val = _max_overlap(h_test)
        if val < best_val - 1e-9:
            h_bb = h_test
            best_val = val
            best_h = h_bb.copy()
            
    return best_h, best_val

def construct():
    start = time.time()
    deadline = start + 140.0
    
    # Start with bang-bang patterns
    # Try a few different initial bang-bang configurations
    inits = []
    # 1. Alternating
    h1 = np.tile([1, 0], N // 2)
    inits.append(h1)
    # 2. Blocks
    h2 = np.zeros(N)
    h2[:N//2] = 1.0
    inits.append(h2)
    # 3. Random
    rng = np.random.default_rng(1)
    for _ in range(5):
        h_rand = rng.random(N)
        h_bb = (h_rand >= 0.5).astype(float)
        # Adjust sum
        diff = TARGET_SUM - h_bb.sum()
        if diff > 0:
            zeros = np.where(h_bb == 0)[0]
            h_bb[rng.choice(zeros, size=int(diff), replace=False)] = 1.0
        elif diff < 0:
            ones = np.where(h_bb == 1)[0]
            h_bb[rng.choice(ones, size=int(-diff), replace=False)] = 0.0
        inits.append(h_bb)
        
    best_h = None
    best_val = float('inf')
    
    for init in inits:
        if time.time() > deadline - 20.0:
            break
            
        # 1. Local search on bang-bang
        h_bb, val_bb = _local_search_bb(init, deadline)
        if val_bb < best_val:
            best_val = val_bb
            best_h = h_bb.copy()
            
        # 2. Subgradient polish from bang-bang
        h_pol, val_pol = _subgrad_polish(h_bb, deadline)
        if val_pol < best_val:
            best_val = val_pol
            best_h = h_pol.copy()
            
        # 3. Subgradient polish from random continuous
        if time.time() > deadline - 10.0:
            continue
        h_cont_init = rng.random(N)
        h_pol2, val_pol2 = _subgrad_polish(h_cont_init, deadline)
        if val_pol2 < best_val:
            best_val = val_pol2
            best_h = h_pol2.copy()
            
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    return best_h
# EVOLVE-BLOCK-END
