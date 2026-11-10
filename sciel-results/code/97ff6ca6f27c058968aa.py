import time
import itertools
import random

def is_sidon(subset):
    """Check if subset is Sidon (all unordered pair sums including diagonals are unique)."""
    s = sorted(subset)
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            val = s[i] + s[j]
            if val in sums:
                return False
            sums.add(val)
    return True

def largest_sidon_size(ambient):
    """Find the largest Sidon subset size of ambient set."""
    n = len(ambient)
    # Try from largest possible down
    for size in range(n, 0, -1):
        for subset in itertools.combinations(ambient, size):
            if is_sidon(list(subset)):
                return size
    return 0

def count_collisions(ambient):
    """Count the number of pair-sum collisions in the ambient set."""
    s = sorted(ambient)
    sums = {}
    for i in range(len(s)):
        for j in range(i, len(s)):
            val = s[i] + s[j]
            sums[val] = sums.get(val, 0) + 1
    # Total pairs
    total = len(s) * (len(s) + 1) // 2
    unique = len(sums)
    return total - unique  # number of "extra" pairs beyond unique sums

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    best_ambient = None
    best_size = 12  # worst case: all 12 form a Sidon set
    
    # Strategy 1: Greedy collision-maximizing construction
    def greedy_construct():
        ambient = [0]  # start with 0
        used = {0}
        for _ in range(11):
            best_candidate = None
            best_score = -1
            for cand in range(1, 129):
                if cand in used:
                    continue
                # Count collisions this candidate creates with existing elements
                temp = ambient + [cand]
                s = sorted(temp)
                sums = {}
                for i in range(len(s)):
                    for j in range(i, len(s)):
                        val = s[i] + s[j]
                        sums[val] = sums.get(val, 0) + 1
                collisions = sum(v - 1 for v in sums.values() if v > 1)
                if collisions > best_score:
                    best_score = collisions
                    best_candidate = cand
            if best_candidate is not None:
                ambient.append(best_candidate)
                used.add(best_candidate)
        return ambient
    
    # Strategy 2: Arithmetic progressions
    def ap_construct():
        results = []
        for start in range(0, 129):
            for d in range(1, 12):
                end = start + 11 * d
                if end <= 128:
                    results.append([start + i * d for i in range(12)])
        return results
    
    # Strategy 3: Near-arithmetic progressions (perturb AP)
    def near_ap_construct():
        results = []
        for start in range(0, 120):
            for d in range(1, 12):
                base = [start + i * d for i in range(12)]
                if max(base) > 128:
                    continue
                # Try small perturbations
                for delta in range(-2, 3):
                    perturbed = [x + delta * (i % 2) for i, x in enumerate(base)]
                    if all(0 <= x <= 128 for x in perturbed) and len(set(perturbed)) == 12:
                        results.append(perturbed)
        return results
    
    # Evaluate candidates
    candidates = []
    
    # Add greedy
    candidates.append(greedy_construct())
    
    # Add APs
    candidates.extend(ap_construct())
    
    # Add some near-APs (limit to keep it fast)
    near_aps = near_ap_construct()
    candidates.extend(near_aps[:500])
    
    # Evaluate all candidates
    for amb in candidates:
        if time.time() > deadline:
            break
        if len(set(amb)) != 12 or not all(0 <= x <= 128 for x in amb):
            continue
        size = largest_sidon_size(amb)
        if size < best_size:
            best_size = size
            best_ambient = list(amb)
    
    # Local search / hill climbing on best found
    if best_ambient is None:
        best_ambient = greedy_construct()
    
    current = list(best_ambient)
    current_size = best_size
    
    # Hill climbing: try swapping elements
    for iteration in range(5000):
        if time.time() > deadline:
            break
        improved = False
        for i in range(12):
            if time.time() > deadline:
                break
            for new_val in range(0, 129):
                if new_val in current and new_val != current[i]:
                    continue
                if current[i] == new_val:
                    continue
                trial = list(current)
                trial[i] = new_val
                if len(set(trial)) != 12:
                    continue
                size = largest_sidon_size(trial)
                if size < current_size:
                    current = trial
                    current_size = size
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    if current_size < best_size:
        best_ambient = current
        best_size = current_size
    
    return list(best_ambient)
