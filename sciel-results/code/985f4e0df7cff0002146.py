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

def _objective(h):
    """
    Objective: maximize t such that O_i <= t for all i.
    We minimize -t with constraints O_i - t <= 0.
    """
    h = np.clip(h, 0, 1)
    O = _compute_O(h)
    return -O.max()

def _objective_grad(h):
    """
    Gradient of the max overlap (we want to minimize this).
    """
    O = _compute_O(h)
    Omax = float(O.max())
    # Softmax gradient
    tau = 50.0
    z = tau * (O - Omax)
    ez = np.exp(z)
    w = ez / ez.sum()
    w_pos, w_neg = w[:N], w[N:]
    
    K = np.arange(N)[:, None]
    I = np.arange(N)[None, :]
    
    ge = (I >= K)
    lt = (I < (N - K))
    
    idx_p = np.clip(I - K, 0, N - 1)
    idx_m = np.clip(I + K, 0, N - 1)
    
    Gpos = SCALE * (ge.astype(float) - ge * h[idx_p] - lt * h[idx_m])
    Gneg = SCALE * (lt.astype(float) - lt * h[idx_m] - ge * h[idx_p])
    
    grad = w_pos @ Gpos + w_neg @ Gneg
    return grad

def _project(z):
    """
    Project z onto {h in [0,1], sum h = N/2}
    """
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

def _optimize(init, deadline):
    h = _project(np.asarray(init, dtype=float).copy())
    
    tau_schedule = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    best_h_local = h.copy()
    best_val_local = float(_compute_O(h).max())
    
    giter = 0
    for tau in tau_schedule:
        base_eta = 0.5 / (1.0 + 0.1 * tau)
        
        for _ in range(200):
            if time.time() > deadline - 2.0:
                return best_h_local, best_val_local
                
            Omax, grad = _step(h, tau)
            if not np.all(np.isfinite(grad)):
                break
            
            eta = base_eta
            h_new = _project(h - eta * grad)
            O_new = _compute_O(h_new)
            val_new = float(O_new.max())
            
            if val_new < best_val_local:
                best_h_local = h_new.copy()
                best_val_local = val_new
                
            h = h_new
            giter += 1

    return best_h_local, best_val_local

def _step(h, tau):
    """
    Compute the gradient of the softmax-smoothed max overlap.
    """
    O = _compute_O(h)
    Omax = float(O.max())
    z = tau * (O - Omax)
    ez = np.exp(z)
    w = ez / ez.sum()
    w_pos, w_neg = w[:N], w[N:]
    
    K = np.arange(N)[:, None]
    I = np.arange(N)[None, :]
    
    ge = (I >= K)
    lt = (I < (N - K))
    
    idx_p = np.clip(I - K, 0, N - 1)
    idx_m = np.clip(I + K, 0, N - 1)
    
    Gpos = SCALE * (ge.astype(float) - ge * h[idx_p] - lt * h[idx_m])
    Gneg = SCALE * (lt.astype(float) - lt * h[idx_m] - ge * h[idx_p])
    
    grad = w_pos @ Gpos + w_neg @ Gneg
    return Omax, grad

def construct():
    start = time.time()
    deadline = start + 145.0  # Safety margin

    seed = np.full(N, 0.5)
    best_h = seed.copy()
    best_val = float(_compute_O(seed).max())  # 0.5

    idx = np.arange(N)
    inits = []
    
    # 1. Bang-bang random inits
    rng = np.random.default_rng(42)
    for _ in range(5):
        bb = np.zeros(N)
        bb[rng.choice(N, N // 2, replace=False)] = 1.0
        inits.append(bb)
        
    # 2. Structured continuous inits
    for f in (1, 2, 3, 5):
        inits.append(0.5 + 0.4 * np.cos(2.0 * np.pi * f * idx / N))
        
    # 3. Random continuous
    for _ in range(5):
        inits.append(rng.random(N))
        
    # 4. "Sawtooth" or periodic bang-bang
    for period in [3, 4, 5, 7, 11]:
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

    # 5. Low-rank tensor product inits
    for f1 in [1, 2, 3]:
        for f2 in [1, 2, 3, 4]:
            a = 0.5 + 0.3 * np.cos(2.0 * np.pi * f1 * idx / N)
            b = 0.5 + 0.3 * np.cos(2.0 * np.pi * f2 * idx / N)
            prod = a * b
            prod = prod - prod.mean() + 0.5
            inits.append(prod)

    # 6. Markov chain inits with different numbers of states
    for n_states in [2, 3, 4, 5]:
        try:
            h_markov = _markov_init(n_states, rng, n_iter=150, deadline=deadline)
            inits.append(h_markov)
        except Exception:
            continue

    for init in inits:
        if time.time() > deadline - 10.0:
            break
        try:
            h, val = _optimize(init, deadline)
            if np.all(np.isfinite(h)) and val < best_val:
                best_h = h
                best_val = val
        except Exception:
            continue

    # Final polish with adaptive tau and eta
    h = best_h.copy()
    tau = 500.0
    eta = 0.01
    for _ in range(1000):
        if time.time() > deadline - 1.0:
            break
        Omax, grad = _step(h, tau)
        if not np.all(np.isfinite(grad)):
            break
        h_new = _project(h - eta * grad)
        val_new = float(_compute_O(h_new).max())
        if val_new < best_val:
            best_h = h_new
            best_val = val_new
            h = h_new
        else:
            eta *= 0.95

    # Ensure exact sum and bounds
    best_h = _project(best_h)
    for _ in range(10):
        d = N / 2.0 - best_h.sum()
        if abs(d) < 1e-12:
            break
        best_h = np.clip(best_h + d / N, 0.0, 1.0)
        best_h = _project(best_h)
        
    return best_h

def _markov_init(n_states, rng, n_iter=150, deadline=None):
    """
    Generate a sequence using a Markov chain with n_states states.
    Each state has an emission probability of 1.
    """
    # Initialize transition matrix
    P = np.ones((n_states, n_states)) / n_states
    # Emission probabilities (probability of emitting 1)
    emissions = np.full(n_states, 0.5)
    
    obj = float('inf')
    h = np.full(N, 0.5)
    
    for iter in range(n_iter):
        if deadline is not None and time.time() > deadline - 2.0:
            break
            
        # Generate sequence
        state = rng.integers(n_states)
        h = np.zeros(N)
        for i in range(N):
            h[i] = 1.0 if rng.random() < emissions[state] else 0.0
            state = rng.choice(n_states, p=P[state])
        
        # Adjust to have exactly N/2 ones
        if h.sum() > N/2:
            idxs = np.where(h==1)[0]
            to_remove = rng.choice(idxs, int(h.sum() - N/2), replace=False)
            h[to_remove] = 0
        elif h.sum() < N/2:
            idxs = np.where(h==0)[0]
            to_add = rng.choice(idxs, int(N/2 - h.sum()), replace=False)
            h[to_add] = 1
        
        O = _compute_O(h)
        obj = float(O.max())
        
        # Try perturbing emission probabilities
        improved = False
        for s in range(n_states):
            for delta in [-0.05, 0.05]:
                old_em = emissions[s]
                emissions[s] = np.clip(old_em + delta, 0.0, 1.0)
                
                # Regenerate
                state = rng.integers(n_states)
                h_test = np.zeros(N)
                for i in range(N):
                    h_test[i] = 1.0 if rng.random() < emissions[s] else 0.0
                    state = rng.choice(n_states, p=P[state])
                
                if h_test.sum() > N/2:
                    idxs = np.where(h_test==1)[0]
                    to_remove = rng.choice(idxs, int(h_test.sum() - N/2), replace=False)
                    h_test[to_remove] = 0
                elif h_test.sum() < N/2:
                    idxs = np.where(h_test==0)[0]
                    to_add = rng.choice(idxs, int(N/2 - h_test.sum()), replace=False)
                    h_test[to_add] = 1
                
                O_test = _compute_O(h_test)
                obj_test = float(O_test.max())
                
                if obj_test < obj - 1e-10:
                    obj = obj_test
                    h = h_test
                    improved = True
                else:
                    emissions[s] = old_em
                    break
        
        # Try perturbing transition matrix
        for s in range(n_states):
            for t in range(n_states):
                if s == t:
                    continue
                delta = 0.05
                if P[s, t] + delta <= 1.0:
                    P_old = P[s].copy()
                    P[s, t] += delta
                    P[s] /= P[s].sum()
                    
                    # Regenerate
                    state = rng.integers(n_states)
                    h_test = np.zeros(N)
                    for i in range(N):
                        h_test[i] = 1.0 if rng.random() < emissions[state] else 0.0
                        state = rng.choice(n_states, p=P[state])
                    
                    if h_test.sum() > N/2:
                        idxs = np.where(h_test==1)[0]
                        to_remove = rng.choice(idxs, int(h_test.sum() - N/2), replace=False)
                        h_test[to_remove] = 0
                    elif h_test.sum() < N/2:
                        idxs = np.where(h_test==0)[0]
                        to_add = rng.choice(idxs, int(N/2 - h_test.sum()), replace=False)
                        h_test[to_add] = 1
                    
                    O_test = _compute_O(h_test)
                    obj_test = float(O_test.max())
                    
                    if obj_test < obj - 1e-10:
                        obj = obj_test
                        h = h_test
                        improved = True
                    else:
                        P[s] = P_old
        
        if not improved:
            break
    
    return h
