import time
import numpy as np
from scipy.optimize import linprog

N = 100
SCALE = 2.0 / N

def _compute_O(h):
    """Compute overlaps for all shifts."""
    S1 = np.cumsum(h[::-1])[::-1]
    S2 = np.array([np.dot(h[k:], h[:N - k]) for k in range(N)])
    O_pos = SCALE * (S1 - S2)
    csum = np.cumsum(h)
    S1n = csum[N - 1 - np.arange(N)]
    S2n = np.array([np.dot(h[:N - k], h[k:]) for k in range(N)])
    O_neg = SCALE * (S1n - S2n)
    return np.concatenate([O_pos, O_neg])

def _project(z):
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

def _solve_qp(h_init, deadline):
    # Formulate as: minimize t
    # subject to:
    #   sum(h) = N/2
    #   0 <= h_i <= 1
    #   O_k(h) <= t for all k
    
    # O_k(h) = SCALE * (S1_k - sum_{j} h[j] h[j-k])
    # S1_k is linear in h.
    # sum h[j]h[j-k] is quadratic.
    # This is a QCQP. We can relax it or use a different approach.
    # Since N=100, we can try to solve the QCQP directly if we had a solver, but we don't.
    # Instead, we can use a heuristic based on the spectral properties or just improve the local search.
    
    # Let's try a simple subgradient method with a better step size rule.
    h = _project(h_init.copy())
    best_h = h.copy()
    best_val = float(_compute_O(h).max())
    
    # Use a fixed step size that decreases over time
    step_size = 0.1
    decay = 0.99
    
    for i in range(5000):
        if time.time() > deadline - 5.0:
            break
            
        O = _compute_O(h)
        Omax = float(O.max())
        
        # Subgradient of max(O_k)
        # grad = SCALE * (grad_S1 - grad_corr)
        # This is complex to compute exactly for the max.
        # Approximate by picking the active constraint (max O_k)
        idx = np.argmax(O)
        if idx < N:
            k = idx
            # O_pos[k] = SCALE * (sum_{j=k}^{N-1} h[j] - sum_{j=k}^{N-1} h[j]h[j-k])
            # grad w.r.t h[j] for j >= k:
            #   d/dh[j] O = SCALE * (1 - h[j-k])
            #   d/dh[j-k] O = -SCALE * h[j]
            # For j < k: 0
            grad = np.zeros(N)
            for j in range(k, N):
                grad[j] += SCALE * (1 - h[j-k])
                if j-k < N:
                    grad[j-k] -= SCALE * h[j]
        else:
            k = idx - N
            # O_neg[k] = SCALE * (sum_{j=0}^{N-1-k} h[j] - sum_{j=0}^{N-1-k} h[j]h[j+k])
            # grad w.r.t h[j] for j < N-k:
            #   d/dh[j] O = SCALE * (1 - h[j+k])
            #   d/dh[j+k] O = -SCALE * h[j]
            grad = np.zeros(N)
            for j in range(N-k):
                grad[j] += SCALE * (1 - h[j+k])
                if j+k < N:
                    grad[j+k] -= SCALE * h[j]
        
        # Projected subgradient step
        h_new = _project(h - step_size * grad)
        val_new = float(_compute_O(h_new).max())
        
        if val_new < best_val:
            best_h = h_new.copy()
            best_val = val_new
        
        h = h_new
        step_size *= decay
    
    return best_h

def _local_search_bb(h):
    # Convert to bang-bang
    h_bb = (h >= 0.5).astype(float)
    # Ensure sum is N/2
    diff = N / 2.0 - h_bb.sum()
    if diff > 0:
        # Need more 1s
        zeros = np.where(h_bb == 0)[0]
        if len(zeros) >= int(diff):
            # Pick the ones that reduce overlap most? Just pick random for now
            h_bb[zeros[:int(diff)]] = 1.0
    elif diff < 0:
        ones = np.where(h_bb == 1)[0]
        if len(ones) >= int(-diff):
            h_bb[ones[:int(-diff)]] = 0.0
            
    # Local search
    best_val = _compute_O(h_bb).max()
    for _ in range(5000):
        # Pick a random 1 and a random 0
        ones = np.where(h_bb == 1)[0]
        zeros = np.where(h_bb == 0)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        i = np.random.choice(ones)
        j = np.random.choice(zeros)
        
        h_test = h_bb.copy()
        h_test[i] = 0.0
        h_test[j] = 1.0
        
        val = _compute_O(h_test).max()
        if val < best_val - 1e-9:
            h_bb = h_test
            best_val = val
    return h_bb

def construct():
    start = time.time()
    deadline = start + 140.0
    
    seed = np.full(N, 0.5)
    best_h = seed.copy()
    best_val = float(_compute_O(seed).max())
    
    idx = np.arange(N)
    inits = [seed.copy()]
    for f in (1, 2, 3, 5, 8):
        inits.append(0.5 + 0.30 * np.cos(2.0 * np.pi * f * idx / N))
        inits.append(0.5 + 0.40 * np.sign(np.cos(2.0 * np.pi * f * idx / N)))
    
    rng = np.random.default_rng(0)
    for _ in range(5):
        inits.append(rng.random(N))
    
    for init in inits:
        if time.time() > deadline - 10.0:
            break
        try:
            h_cont = _solve_qp(init, deadline)
            val_cont = float(_compute_O(h_cont).max())
            if val_cont < best_val:
                best_h, best_val = h_cont, val_cont
            
            # Try bang-bang local search
            h_bb = _local_search_bb(h_cont)
            val_bb = float(_compute_O(h_bb).max())
            if val_bb < best_val:
                best_h, best_val = h_bb, val_bb
                
        except Exception:
            continue
            
    return best_h
