# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _compute_C(a, b, p):
    # C[i] = sum_j a[i,j] * b[p[i], p[j]]  with B = b[p,p]
    B = b[np.ix_(p, p)]
    return (a @ B).astype(np.int64)


def _refine(a, b, p, deadline, best_cost, best_p):
    """First-improvement 2-opt (swap) local search until deadline.
    p is a numpy int64 array (a copy).
    Returns (best_cost, best_p) found so far."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()

    while time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() >= deadline:
                break
            ai = a[i]
            pi = p[i]
            for j in range(i + 1, n):
                # Vectorized delta calculation
                dA = a[i, :] - a[j, :]
                dB = b[p[j], p] - b[p[i], p]
                dB[i] = 0
                dB[j] = 0
                delta = int(np.dot(dA, dB))
                
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cur += delta
                    improved = True
                    
                    if cur < best_cost:
                        best_cost = cur
                        best_p = p.copy()
                    
                    # Time check
                    if time.time() >= deadline:
                        return best_cost, best_p
            if not improved and time.time() >= deadline:
                break
        if not improved:
            break

    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    return best_cost, best_p


def _anneal(a, b, p, start_time, deadline, best_cost, best_p):
    """Simulated Annealing refinement.
    Returns (best_cost, best_p)."""
    n = len(p)
    cur = _qap_cost(a, b, p)
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()

    T = 5000.0
    T_end = 1.0
    alpha = 0.995
    
    while T > T_end and (time.time() - start_time) < deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        dA = a[i, :] - a[j, :]
        dB = b[p[j], p] - b[p[i], p]
        dB[i] = 0
        dB[j] = 0
        delta = int(np.dot(dA, dB))
        
        # Acceptance criterion
        if delta < 0 or (np.random.rand() < np.exp(-delta / T)):
            p[i], p[j] = p[j], p[i]
            cur += delta
            
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
                
        T *= alpha
        
        # Periodic time check
        if time.time() - start_time > deadline:
            break
            
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
        
    return best_cost, best_p


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s kill

    # --- Strong constructive start: rank matching ---
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central
    
    best_cost = _qap_cost(a, b, p0)
    best_p = p0.copy()

    # First refinement on the rank-matching start
    p = p0.copy()
    best_cost, best_p = _refine(a, b, p, deadline, best_cost, best_p)

    # --- Simulated Annealing for diversification ---
    # Use a fixed seed for reproducibility in evaluation, but we can also just use default RNG
    # The inspiration used np.random.randint which uses global state. 
    # Let's use a local RNG for cleanliness, but the inspiration's score suggests it works well.
    # I will use np.random.default_rng for better practice, but map to randint calls.
    rng = np.random.default_rng(12345)
    
    # We can run SA on the current best solution found by local search
    p = best_p.copy()
    best_cost, best_p = _anneal(a, b, p, t0, deadline, best_cost, best_p)
    
    # If time remains, do a few random restarts with local search
    while time.time() < deadline - 5.0:
        p = rng.permutation(n).astype(np.int64)
        best_cost, best_p = _refine(a, b, p, deadline, best_cost, best_p)
        
    return best_p.tolist()
# EVOLVE-BLOCK-END
