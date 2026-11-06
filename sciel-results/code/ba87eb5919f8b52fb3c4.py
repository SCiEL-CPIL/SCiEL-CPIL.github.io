"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import random

def _is_sidon_with_diagonals(subset):
    """Check if subset has all unordered pair sums unique including diagonals (i.e. i<=j)."""
    n = len(subset)
    s = set()
    for i in range(n):
        for j in range(i, n):
            v = subset[i] + subset[j]
            if v in s:
                return False
            s.add(v)
    return True

def _max_sidon_size_ambient(ambient):
    """Find maximum k such that some k-subset of ambient is Sidon (with diagonals).
    ambient has 12 elements, so k is in [1..12]."""
    n = len(ambient)
    # Try k from n down to 1; return first k with any k-subset being Sidon
    for k in range(n, 0, -1):
        for combo in itertools.combinations(ambient, k):
            if _is_sidon_with_diagonals(list(combo)):
                return k
    return 0

def _count_5_sidons(ambient):
    """Count how many 5-element subsets of ambient are Sidon (with diagonals)."""
    count = 0
    for combo in itertools.combinations(ambient, 5):
        if _is_sidon_with_diagonals(list(combo)):
            count += 1
    return count

def _local_search_ambient(ambient, deadline, rng):
    """Local search to minimize the number of 5-Sidon subsets."""
    ambient = list(ambient)
    best_ambient = list(ambient)
    best_count = _count_5_sidons(ambient)
    
    while time.time() < deadline:
        # Pick a random element to perturb
        idx = rng.randint(0, 11)
        old_val = ambient[idx]
        
        # Try perturbing to a nearby value or a random value in [0,128]
        if rng.random() < 0.5:
            # Small perturbation
            delta = rng.randint(-5, 5)
            new_val = old_val + delta
        else:
            # Larger perturbation
            new_val = rng.randint(0, 128)
        
        if new_val < 0 or new_val > 128:
            continue
        
        # Check if new_val is already in ambient (excluding idx)
        if any(ambient[j] == new_val for j in range(12) if j != idx):
            continue
        
        ambient[idx] = new_val
        new_count = _count_5_sidons(ambient)
        
        if new_count < best_count:
            best_count = new_count
            best_ambient = list(ambient)
            if best_count == 0:
                break
        else:
            # Revert with some probability (simulated annealing-like)
            if rng.random() < 0.3:
                ambient[idx] = old_val
            # Otherwise keep the change to explore
    
    return best_ambient, best_count

def _random_ambient(rng):
    """Generate a random ambient set of 12 distinct integers in [0,128]."""
    return sorted(rng.sample(range(0, 129), 12))

def _dense_ambient(rng):
    """Generate a dense ambient set by starting with consecutive and adding noise."""
    base = list(range(12))
    # Add random offsets
    for i in range(12):
        if rng.random() < 0.3:
            base[i] += rng.randint(-3, 3)
    # Ensure distinct and in range
    seen = set()
    result = []
    for v in base:
        v = max(0, min(128, v))
        while v in seen:
            v = (v + 1) % 129
        seen.add(v)
        result.append(v)
    return sorted(result)

def _select_best_ambient():
    deadline = time.time() + 80
    best = None
    best_k = None
    best_count = None
    
    rng = random.Random(42)
    
    # Precomputed initial candidates
    cands = [
        list(range(12)),
        list(range(0, 24, 2)),
        list(range(0, 36, 3)),
        list(range(0, 48, 4)),
        list(range(0, 60, 5)),
        list(range(0, 72, 6)),
        list(range(0, 96, 8)),
        list(range(0, 120, 10)),
        list(range(0, 132, 11)),
        [0,1,2,3,4,5,6,7,8,9,10,12],
        list(range(1, 13)),
        [0,1,2,3,4,5,6,7,8,9,11,13],
    ]
    
    # Evaluate initial candidates
    for cand in cands:
        if len(cand) != 12 or len(set(cand)) != 12 or any(x < 0 or x > 128 for x in cand):
            continue
        k = _max_sidon_size_ambient(cand)
        if best_k is None or k < best_k:
            best = cand
            best_k = k
            best_count = _count_5_sidons(cand)
            if best_k <= 3:
                return best
    
    # If we haven't found k<=3, do local search from various starting points
    if best_k is not None and best_k > 3:
        # Try local search from the best candidate
        search_deadline = deadline - 5  # Leave 5s margin
        for start in [best] + [_dense_ambient(rng) for _ in range(3)] + [_random_ambient(rng) for _ in range(3)]:
            if time.time() > search_deadline:
                break
            result, count = _local_search_ambient(start, search_deadline, rng)
            k = _max_sidon_size_ambient(result)
            if best_k is None or k < best_k:
                best = result
                best_k = k
                best_count = count
                if best_k <= 3:
                    return best
            elif k == best_k and count < best_count:
                best = result
                best_count = count
    
    # Fallback
    if best is None:
        return list(range(12))
    return best

def construct():
    # Deterministically return the best ambient set found from the small candidate list
    return _select_best_ambient()
# EVOLVE-BLOCK-END
