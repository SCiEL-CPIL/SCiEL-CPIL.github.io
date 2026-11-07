"""Erdos minimum overlap: minimize max autocorrelation of h vs 1-h."""
import time
import random
import numpy as np

N = 100
DEADLINE = 150.0
START = time.time()
TIME_MARGIN = 10.0  # safety margin before the 150s wall-clock kill


def remaining_time():
    return DEADLINE - TIME_MARGIN - (time.time() - START)


def _value(h):
    """The evaluator's raw objective (lower is better)."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr)) / N * 2.0


def _valid(h):
    h = np.asarray(h, dtype=float)
    if h.shape != (N,) or not np.all(np.isfinite(h)):
        return False
    if np.any(h < -1e-8) or np.any(h > 1.0 + 1e-8):
        return False
    if abs(float(np.sum(h)) - N / 2.0) > 1e-9 * N:
        return False
    return True


def _project(h):
    """Project onto {0<=h<=1, sum(h)=N/2} via bisection on the box threshold."""
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    target = N / 2.0
    s = float(np.sum(h))
    if s <= target:
        return h
    lo, hi = 0.0, 1.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if float(np.sum(np.clip(h - mid, 0.0, 1.0))) >= target:
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return np.clip(h - mid, 0.0, 1.0)


def _make_anti_periodic(first_half):
    """Create full h from first 50 values using anti-periodic structure."""
    first_half = np.clip(first_half, 0.0, 1.0)
    h = np.empty(N)
    h[:50] = first_half
    h[50:] = 1.0 - first_half
    return h


def _local_search(h, max_iters=5000, temp=0.05):
    """Simulated annealing with swap/flip moves on anti-periodic representation."""
    h = _project(h.copy())
    best_h = h.copy()
    best_v = _value(best_h)
    current_v = best_v
    
    for it in range(max_iters):
        if remaining_time() < 1.0:
            break
        
        # Choose move type: flip one element or swap two
        if random.random() < 0.5:
            # Flip: pick a random index in first half, perturb it
            i = random.randint(0, 49)
            delta = random.uniform(-0.3, 0.3)
            h_new = h.copy()
            h_new[i] = np.clip(h[i] + delta, 0.0, 1.0)
            h_new[50 + i] = 1.0 - h_new[i]  # maintain anti-periodic
        else:
            # Swap: pick two indices in first half, swap their values
            i, j = random.sample(range(50), 2)
            h_new = h.copy()
            h_new[i], h_new[j] = h[j], h[i]
            h_new[50 + i], h_new[50 + j] = h[50 + j], h[50 + i]
        
        h_new = _project(h_new)
        v_new = _value(h_new)
        
        # Accept if better or with probability exp(-delta/temp)
        if v_new < current_v or random.random() < np.exp(-(v_new - current_v) / temp):
            h = h_new
            current_v = v_new
            if v_new < best_v:
                best_v = v_new
                best_h = h.copy()
        
        # Cool down
        temp *= 0.9995
        temp = max(temp, 0.001)
    
    return best_h, best_v


def _multi_start_local_search(n_starts=30, max_iters_per_start=2000):
    """Multi-start local search with random anti-periodic seeds."""
    best_h = np.full(N, 0.5)
    best_v = _value(best_h)
    
    for start in range(n_starts):
        if remaining_time() < 3.0:
            break
        
        # Generate random anti-periodic seed
        if start == 0:
            # Start from all 0.5
            first_half = np.full(50, 0.5)
        elif start == 1:
            # Binary seed
            first_half = np.random.choice([0.0, 1.0], size=50)
        elif start == 2:
            # Ternary seed
            first_half = np.random.choice([0.0, 0.5, 1.0], size=50)
        else:
            # Random uniform seed with some structure
            first_half = np.random.uniform(0.0, 1.0, size=50)
            # Add some periodic structure
            if start % 3 == 0:
                for i in range(0, 50, 10):
                    first_half[i:i+5] = first_half[i] + np.random.uniform(-0.1, 0.1)
        
        h_seed = _make_anti_periodic(first_half)
        h_seed = _project(h_seed)
        
        # Local search from this seed
        h_result, v_result = _local_search(h_seed, max_iters=max_iters_per_start)
        
        if v_result < best_v:
            best_v = v_result
            best_h = h_result
    
    return best_h, best_v


def construct():
    # Phase 1: Multi-start local search with anti-periodic structure
    best_h, best_v = _multi_start_local_search(n_starts=40, max_iters_per_start=1500)
    
    # Phase 2: If time remains, do more starts with longer searches
    if remaining_time() > 15.0:
        h2, v2 = _multi_start_local_search(n_starts=20, max_iters_per_start=3000)
        if v2 < best_v:
            best_h, best_v = h2, v2
    
    # Phase 3: Final refinement on best solution
    if remaining_time() > 5.0:
        h3, v3 = _local_search(best_h, max_iters=3000, temp=0.01)
        if v3 < best_v:
            best_h, best_v = h3, v3
    
    # Final safety: guarantee a valid, sum-corrected output
    best_h = _project(best_h)
    return best_h
