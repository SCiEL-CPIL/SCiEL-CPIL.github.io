import numpy as np
import time

N = 100

def overlap_max(h):
    # O(s) = sum_{i=0}^{N-1-s} h[i]*(1-h[i+s]) for s=0..N-1
    # return max over s
    n = len(h)
    ones = 1.0 - h
    best = 0.0
    for s in range(n):
        # i from 0 to n-1-s
        ov = np.dot(h[:n-s], ones[s:])
        if ov > best:
            best = ov
    return best

def construct():
    t0 = time.time()
    deadline = t0 + 120.0  # safety margin before 150s kill (evaluator says 120s in contract, 150s kill)
    
    # initial guesses
    def ramp():
        return (np.arange(N) + 0.5) / N
    def constant():
        return np.full(N, 0.5)
    # 0/1 alternating-ish: put ones at even positions
    def ones_even():
        h = np.zeros(N)
        h[0::2] = 1.0
        # need exactly N/2 ones; N=100 even, 0::2 gives 50 ones
        return h
    # 0/1 spread: ones at positions floor(k*N/N2)
    def ones_spread():
        h = np.zeros(N)
        k = N//2
        positions = (np.arange(k) * N + k//2) // k
        h[positions] = 1.0
        return h
    
    candidates = [ramp(), constant(), ones_even(), ones_spread()]
    best_h = min(candidates, key=overlap_max)
    best_val = overlap_max(best_h)
    
    # local search
    h = best_h.copy()
    rng = np.random.default_rng(0)
    n = N
    iters = 0
    no_improve = 0
    while time.time() < deadline:
        # pick a move: transfer delta from i to j
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        # choose delta magnitude
        # try to push toward 0/1 (extreme) to reduce sum h^2
        delta = rng.uniform(0.01, 0.5)
        # direction: reduce h[i], increase h[j] or vice versa
        for sign in (+1, -1):
            hi = h[i] + sign*delta
            hj = h[j] - sign*delta
            if hi < 0 or hi > 1 or hj < 0 or hj > 1:
                continue
            h[i] = hi
            h[j] = hj
            val = overlap_max(h)
            if val < best_val - 1e-12:
                best_val = val
                best_h = h.copy()
                no_improve = 0
                iters += 1
                break
            else:
                h[i] = h[i] - sign*delta
                h[j] = h[j] + sign*delta
                no_improve += 1
        if no_improve > 2000:
            # restart from best with random perturbation
            h = best_h.copy()
            # perturb
            idx = rng.integers(0, n, 5)
            for k in idx:
                h[k] = np.clip(h[k] + rng.uniform(-0.3, 0.3), 0, 1)
            # fix sum
            diff = N/2 - h.sum()
            h += diff/n
            h = np.clip(h, 0, 1)
            no_improve = 0
    
    # ensure exact sum
    diff = N/2 - best_h.sum()
    best_h += diff / N
    best_h = np.clip(best_h, 0, 1)
    # re-fix sum after clip (clip may change sum)
    # do a few iterations
    for _ in range(10):
        diff = N/2 - best_h.sum()
        if abs(diff) < 1e-12:
            break
        # add diff to the element with most room
        if diff > 0:
            room = 1.0 - best_h
        else:
            room = best_h
        # distribute to elements with most room
        order = np.argsort(-room)
        # just add uniformly then clip, iterate
        best_h += diff / N
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
