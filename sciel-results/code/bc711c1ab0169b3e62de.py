import numpy as np
import time

N = 100

def _overlaps(h):
    # h: array of length N in [0,1]
    # overlap(k) = sum_{i=0}^{N-k-1} h[i]*(1-h[i+k]) for k=0..N-1
    # return array of length N
    n = len(h)
    ov = np.empty(n)
    for k in range(n):
        m = n - k
        ov[k] = np.sum(h[:m] * (1.0 - h[k:k+m]))
    return ov

def _grad_overlaps(h):
    # gradient of each overlap(k) w.r.t. h[j]
    # d overlap(k)/d h[j] = [j < n-k] - [h[j+k] if j < n-k] - [h[j-k] if j >= k]
    n = len(h)
    # We'll return a function or compute soft-max gradient directly
    pass

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s (evaluator kills at 150s)
    
    # initial guess: ramp
    h = np.linspace(0.0, 1.0, N)
    # adjust sum to N/2: ramp sum = (N-1)/2 ... linspace(0,1,N) sum = (N-1)/2
    # we need sum = N/2, difference = N/2 - (N-1)/2 = 0.5
    # add 0.5/N to each? that shifts. Better: recompute.
    # Actually linspace(0,1,N) has sum (N-1)/2. We need N/2. Add 0.5/N to each element: new sum = (N-1)/2 + 0.5 = N/2. But then max becomes 1+0.5/N > 1. Clip.
    # Instead, use a ramp in [0, 1] with exact sum: h[i] = a*i + b, sum = a*N(N-1)/2 + b*N = N/2.
    # Let's just do h = np.linspace(0,1,N) and then project to sum N/2.
    h = _project(h, N/2.0)
    
    best_h = h.copy()
    best_val = _max_overlap(h)
    
    # optimize with L-BFGS-B on smooth objective
    ...
    
    return best_h
