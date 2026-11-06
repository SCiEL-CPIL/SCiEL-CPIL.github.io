import time
import random
import numpy as np

N = 17
DEADLINE = time.time() + 80.0   # safety margin under 90s

def _time_left():
    return DEADLINE - time.time()

def _apply_cmp(arr, i, j):
    if arr[i] > arr[j]:
        arr[i], arr[j] = arr[j], arr[i]
    return arr

def _validity(comps):
    """Check all 2**17 Boolean inputs. Returns True if sorted on all inputs."""
    n = len(comps)
    for mask in range(1 << N):
        arr = np.fromiter(((mask >> k) & 1 for k in range(N)), dtype=np.int8, count=N)
        for i, j in comps:
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
        if np.any(np.diff(arr) < 0):
            return False
    return True

def _is_sorted(comps, mask):
    arr = np.fromiter(((mask >> k) & 1 for k in range(N)), dtype=np.int8, count=N)
    for i, j in comps:
        if arr[i] > arr[j]:
            arr[i], arr[j] = arr[j], arr[i]
    return bool(np.all(np.diff(arr) >= 0))

def _greedy_remove(comps):
    """Reorder-aware greedy removal: try removing each comparator; if invalid,
    also try swapping it with neighbors. Return the shortest valid network."""
    comps = list(comps)
    cur = list(comps)
    # Precompute sortedness for all masks once for current network
    while True:
        improved = False
        for idx in range(len(cur)):
            if _time_left() < 2.0:
                return cur
            # Try removing idx
            trial = cur[:idx] + cur[idx+1:]
            if _validity(trial):
                cur = trial
                improved = True
                break
            # Try swapping with next
            if idx + 1 < len(cur):
                trial2 = cur[:idx] + [cur[idx+1]] + cur[idx:idx+1] + cur[idx+2:]
                if _validity(trial2):
                    cur = trial2
                    improved = True
                    break
        if not improved:
            break
    return cur

def _seed_network_a():
    """
    Seed from a 16-wire network + merge wire 16.
    Build a known 16-wire network (from known patterns) and extend.
    We use a simple but compact 16-wire construction.
    """
    comps = []
    # Use a known 16-wire network (depth ~12, ~70 comparators)
    # We'll build it layer by layer based on known optimal-ish patterns.
    # For 16 wires, a known network has 70 comparators.
    # Here we construct one:
    # Layer 0: (0,8),(1,9),(2,10),(3,11),(4,12),(5,13),(6,14),(7,15)
    # Layer 1: (0,4),(1,5),(2,6),(3,7),(8,12),(9,13),(10,14),(11,15)
    # Layer 2: (0,2),(1,3),(4,6),(5,7),(8,10),(9,11),(12,14),(13,15)
    # Layer 3: (0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)
    # Layer 4: (0,4),(2,6),(1,5),(3,7),(8,12),(10,14),(9,13),(11,15)
    # Layer 5: (0,8),(1,9),(2,10),(3,11),(4,12),(5,13),(6,14),(7,15)
    # Layer 6: (0,2),(1,3),(4,6),(5,7),(8,10),(9,11),(12,14),(13,15)
    # Layer 7: (0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)
    # Layer 8: (0,4),(1,5),(2,6),(3,7),(8,12),(9,13),(10,14),(11,15)
    # Layer 9: (0,8),(1,9),(2,10),(3,11),(4,12),(5,13),(6,14),(7,15)
    # Layer 10: (0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15)
    # Layer 11: (0,8),(1,9),(2,10),(3,11),(4,12),(5,13),(6,14),(7,15)
    # This is 12 layers x 8 = 96. Too many. Let me use a known smaller one.
    # Actually, let me just use the odd-even transposition for 17 wires as base
    # and rely on greedy removal to reduce it.
    pass
    # Use odd-even transposition for 17 wires (136 comparators) as the base seed.
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _seed_network_b():
    """
    A different seed: use a known 17-wire construction.
    We'll use a slightly different pattern.
    """
    comps = []
    # Batch 0: compare (0,1),(2,3),...,(14,15)
    for i in range(0, N - 1, 2):
        comps.append((i, i + 1))
    # Batch 1: compare (1,2),(3,4),...,(13,14)
    for i in range(1, N - 1, 2):
        comps.append((i, i + 1))
    # Repeat for N phases
    for phase in range(1, N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def construct():
    """
    Return an ordered list of (i, j) comparators that sorts all 17-input sequences.
    Uses seeded networks + greedy removal with deadline guard.
    """
    random.seed(42)
    seeds = [
        _seed_network_a(),
        _seed_network_b(),
    ]
    
    best = None
    best_len = float('inf')
    
    for seed in seeds:
        if _time_left() < 10:
            break
        # Verify seed is valid
        if not _validity(seed):
            continue
        # Apply greedy removal
        result = _greedy_remove(seed)
        if len(result) < best_len:
            best = result
            best_len = len(result)
        if _time_left() < 5:
            break
    
    if best is None:
        # Fallback: return the odd-even transposition network
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    return best
