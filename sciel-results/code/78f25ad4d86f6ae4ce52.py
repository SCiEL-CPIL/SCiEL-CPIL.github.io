import numpy as np
import time

N = 100
HALF = N // 2

# Precompute index matrix for circular shifts: IDX[k, i] = (i - k) mod N
_IDX = (np.arange(N)[None, :] - np.arange(N)[:, None]) % N

def _F(h):
    # max over k>=1 of sum_i min(h[i], 1 - h[(i-k)%N])
    M = h[_IDX]
    ov = np.minimum(h[None, :], 1.0 - M).sum(axis=1)
    return float(ov[1:].max())

def _rand_balanced(rng):
    x = np.zeros(N, dtype=np.float64)
    x[rng.choice(N, HALF, replace=False)] = 1.0
    return x

def _local_search(h, deadline, rng, patience=1):
    h = h.copy()
    best = _F(h)
    improved = True
    no_improve = 0
    while improved and time.time() < deadline - 8:
        improved = False
        ones = np.nonzero(h > 0.5)[0]
        zeros = np.nonzero(h < 0.5)[0]
        best_move = None
        best_val = best
        # evaluate all single moves
        for a in ones:
            for b in zeros:
                if a == b:
                    continue
                h[a] = 0.0; h[b] = 1.0
                v = _F(h)
                h[a] = 1.0; h[b] = 0.0
                if v < best_val - 1e-12:
                    best_val = v
                    best_move = (a, b)
        if best_move is not None:
            a, b = best_move
            h[a] = 0.0; h[b] = 1.0
            best = best_val
            improved = True
        else:
            break
    return h, best

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    rng = np.random.default_rng(42)
    
    # Generate initial candidate
    h = _rand_balanced(rng)
    
    # Run local search
    h, _ = _local_search(h, deadline, rng)
    
    return h
