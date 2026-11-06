"""Optimized additive basis: local search to minimize peak representation multiplicity."""
import time
import random
import numpy as np

_N = 128
_LO = 128
_HI = 256

def _compute_counts(A_set):
    """Compute representation counts for each target in [128,256].
    Uses convolution via numpy for speed.
    """
    arr = np.zeros(_N + 1, dtype=np.int64)
    for a in A_set:
        arr[a] = 1
    # Convolution: count of ordered pairs (a,b) with a+b = s
    conv = np.convolve(arr, arr, mode='full')  # index s gives ordered pair count
    # For unordered pairs with equal summands permitted:
    # unordered_count(s) = (ordered_count(s) + diagonal(s)) / 2
    # where diagonal(s) = 1 if s is even and s/2 in A, else 0
    diag = np.zeros(_HI + 1, dtype=np.int64)
    for a in A_set:
        s = 2 * a
        if s <= _HI:
            diag[s] = 1
    # conv has length 2*_N+1, index s corresponds to sum s (0 to 2*_N=256)
    ordered = conv[:_HI + 1]  # sums 0..256
    unordered = (ordered + diag) // 2
    return unordered[_LO:_HI + 1]  # sums 128..256

def _peak(counts):
    return int(np.max(counts))

def _coverage_ok(counts):
    return np.all(counts > 0)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s safety margin
    
    # Initial seed: interval [64, 128]
    A = set(range(64, _N + 1))
    counts = _compute_counts(A)
    best_peak = _peak(counts)
    best_A = set(A)
    
    rng = random.Random(42)
    
    # Local search: add/remove/swap moves
    # Strategy: prefer removing elements that are in the "middle" (high contribution)
    # and adding low elements to spread out representations
    
    # Precompute: for each element x, how many targets does it contribute to
    # and with what multiplicity impact
    while time.time() < deadline:
        # Try a batch of random moves
        for _ in range(200):
            if time.time() >= deadline:
                break
            
            move_type = rng.choice(['remove', 'add', 'swap'])
            
            if move_type == 'remove':
                # Try removing a random element
                candidates = list(A)
                if len(candidates) <= 40:
                    continue
                x = rng.choice(candidates)
                A.remove(x)
                new_counts = _compute_counts(A)
                new_peak = _peak(new_counts)
                if _coverage_ok(new_counts) and new_peak < best_peak:
                    best_peak = new_peak
                    best_A = set(A)
                else:
                    A.add(x)
                    
            elif move_type == 'add':
                # Try adding a random element not in A
                if len(A) >= _N:
                    continue
                x = rng.randint(0, _N)
                if x in A:
                    continue
                A.add(x)
                new_counts = _compute_counts(A)
                new_peak = _peak(new_counts)
                if new_peak <= best_peak:
                    # Accept if peak doesn't increase (keep for further exploration)
                    if new_peak < best_peak:
                        best_peak = new_peak
                        best_A = set(A)
                else:
                    A.discard(x)
                    
            elif move_type == 'swap':
                # Remove one, add another
                if len(A) < 40:
                    continue
                x = rng.choice(list(A))
                y = rng.randint(0, _N)
                if y in A:
                    continue
                A.remove(x)
                A.add(y)
                new_counts = _compute_counts(A)
                new_peak = _peak(new_counts)
                if new_peak <= best_peak:
                    if new_peak < best_peak:
                        best_peak = new_peak
                        best_A = set(A)
                else:
                    A.discard(y)
                    A.add(x)
        
        # Periodically reset to best found to avoid getting stuck
        if time.time() > start_time + 5:
            A = set(best_A)
            counts = _compute_counts(A)
    
    return sorted(best_A)
