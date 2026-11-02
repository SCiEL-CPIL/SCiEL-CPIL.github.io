import time
import numpy as np

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
    """Project onto {h: 0<=h<=1, sum(h)=N/2} via bisection on threshold."""
    lo = float(z.min()) - 1.0
    hi = float(z.max()) + 1.0
    target = N / 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        s = np.clip(z - mid, 0.0, 1.0).sum()
        if s > target:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return np.clip(z - t, 0.0, 1.0)

def _soft_max_grad(O, beta):
    """
    Compute gradient of the soft-max approximation:
    f(O) = (1/beta) * log(sum(exp(beta * O_k)))
    grad = softmax(beta * O) * (1/beta) * beta = softmax(beta * O)
    Wait, let's derive properly.
    f(O) = (1/beta) * log(sum_k exp(beta * O_k))
    d f / d O_j = (1/beta) * (beta * exp(beta * O_j) / sum_k exp(beta * O_k))
                = exp(beta * O_j) / sum_k exp(beta * O_k)
                = softmax(beta * O)_j
    
    So the gradient of the soft-max objective w.r.t O is the softmax vector.
    We need grad w.r.t h.
    O_k(h) is the overlap.
    We can compute the gradient of each O_k w.r.t h, then take a weighted sum with softmax weights.
    """
    beta = 10.0  # Temperature
    # Clip O to avoid overflow in exp
    O_shifted = O - np.max(O)
    exp_O = np.exp(beta * O_shifted)
    weights = exp_O / np.sum(exp_O)
    
    # Compute gradient of each O_k w.r.t h
    # O has length 2N. First N are positive shifts, next N are negative.
    grad = np.zeros(N)
    
    # We need to be efficient.
    # O_pos[k] = SCALE * (sum_{j=k}^{N-1} h[j] - sum_{j=k}^{N-1} h[j]h[j-k])
    # O_neg[k] = SCALE * (sum_{j=0}^{N-1-k} h[j] - sum_{j=0}^{N-1-k} h[j]h[j+k])
    
    # Instead of looping over all k and computing full grad, we can accumulate.
    # But N=100, so 200 gradients of length 100 is 20,000 ops. Very fast.
    
    for k in range(N):
        w = weights[k]
        if w > 1e-10:
            # Positive shift k
            # d O_pos[k] / d h[j]
            # Term 1: sum_{j=k}^{N-1} h[j] -> grad is 1 for j>=k
            # Term 2: - sum_{j=k}^{N-1} h[j]h[j-k]
            #   d/dh[j] (h[j]h[j-k]) = h[j-k] for j>=k
            #   d/dh[j-k] (h[j]h[j-k]) = h[j] for j>=k -> index j-k
            # So for j >= k:
            #   grad[j] += 1 - h[j-k]
            #   grad[j-k] += -h[j]
            
            # Vectorized:
            # j from k to N-1
            # grad[k:] += 1 - h[:N-k]
            # grad[:N-k] += -h[k:]
            
            grad[k:] += (1.0 - h[:N-k])
            grad[:N-k] += -h[k:]
            
            grad *= SCALE  # Wait, this scales the whole thing. Better to scale at the end or inside.
            # Actually, let's just accumulate the raw grad and multiply by SCALE and w at the end?
            # No, w is specific to k.
            # Let's just do it directly.
            
            # Reset and redo properly
            # This is getting messy. Let's just loop j.
            # N=100, so 100*100 = 10,000 ops per k? No, 100 ops per k. 200*100 = 20,000 total.
            # It's fine to be simple.
            
    # Let's restart the gradient calculation cleanly.
    grad = np.zeros(N)
    
    for k in range(N):
        w_pos = weights[k]
        if w_pos > 1e-10:
            for j in range(k, N):
                grad[j] += w_pos * SCALE * (1.0 - h[j-k])
                grad[j-k] += -w_pos * SCALE * h[j]
                
        w_neg = weights[N+k]
        if w_neg > 1e-10:
            for j in range(N-k):
                grad[j] += w_neg * SCALE * (1.0 - h[j+k])
                grad[j+k] += -w_neg * SCALE * h[j]
                
    return grad

def _soft_max_value(O, beta):
    O_shifted = O - np.max(O)
    return (1.0/beta) * np.log(np.sum(np.exp(beta * O_shifted)))

def _solve_smooth(h_init, deadline):
    """
    Minimize soft-max of overlaps using projected gradient descent.
    """
    h = _project(h_init.copy())
    best_h = h.copy()
    best_val = float(_compute_O(h).max())
    best_soft_val = _soft_max_value(_compute_O(h), 10.0)
    
    # Adaptive step size
    step_size = 0.05
    patience = 0
    
    for i in range(2000):
        if time.time() > deadline - 5.0:
            break
            
        O = _compute_O(h)
        current_max = float(O.max())
        
        # Update best
        if current_max < best_val:
            best_val = current_max
            best_h = h.copy()
            patience = 0
        else:
            patience += 1
            
        if patience > 100:
            # Reduce step size
            step_size *= 0.9
            patience = 0
            
        # Compute gradient of soft-max
        grad = _soft_max_grad(O, 10.0)
        
        # Projected gradient step
        h_new = _project(h - step_size * grad)
        
        # Check if step improved soft-max value (for step size adjustment)
        O_new = _compute_O(h_new)
        soft_new = _soft_max_value(O_new, 10.0)
        
        if soft_new > best_soft_val:
            # Bad step, reduce step size
            step_size *= 0.8
            # Don't update h? Or update h to h_new but keep best_h?
            # Usually in projected gradient, we take the step.
            # But if it's bad, maybe we should skip or backtrack.
            # Let's just skip the update if it's significantly worse.
            if soft_new > best_soft_val + 0.01:
                continue
        else:
            # Good step, maybe increase step size slightly
            if i > 100:
                step_size *= 1.01
            best_soft_val = soft_new
            
        h = h_new
        
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
            # Use smooth optimization
            h_cont = _solve_smooth(init, deadline)
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
