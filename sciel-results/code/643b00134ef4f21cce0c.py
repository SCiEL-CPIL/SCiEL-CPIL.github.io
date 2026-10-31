"""Improved program: deterministic spectral/structured construction + local search with deadline guard."""
import math
import time
import random
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 150.0  # wall-clock limit
_SAFETY_MARGIN = 5.0  # seconds to stay under limit

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return _TIME_LIMIT - _SAFETY_MARGIN - _elapsed()

def _score_of(U):
    """Compute the score: 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)."""
    U = sorted(set(U))
    if len(U) < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute U+U (sumset) and U-U (difference set)
    U_set = set(U)
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    # |U-U| counts distinct differences; |U+U| counts distinct sums
    num_diff = len(diffset)
    num_sum = len(sumset)
    if num_sum == 0:
        return 0.0
    denom = math.log(2 * max_u + 1, 2)
    if denom == 0:
        return 0.0
    score = 1.0 + math.log(num_diff / num_sum, 2) / denom
    return score

def _score_of_fast(U):
    """Faster score using numpy for larger sets."""
    U = sorted(set(U))
    if len(U) < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    U_arr = np.array(U, dtype=np.int64)
    # Sumset: all pairwise sums
    sums = U_arr[:, None] + U_arr[None, :]
    sumset = set(sums.flatten().tolist())
    # Diffset: all pairwise differences
    diffs = U_arr[:, None] - U_arr[None, :]
    diffset = set(diffs.flatten().tolist())
    num_diff = len(diffset)
    num_sum = len(sumset)
    if num_sum == 0:
        return 0.0
    denom = math.log(2 * max_u + 1, 2)
    if denom == 0:
        return 0.0
    score = 1.0 + math.log(num_diff / num_sum, 2) / denom
    return score

def _generate_golomb_like(n, scale=1):
    """Generate a Golomb-ruler-like set: positions that maximize distinct differences."""
    # Use a simple greedy Golomb ruler construction
    U = [0]
    for i in range(1, n):
        # Try to place next element as far as possible while keeping all differences distinct
        best_pos = None
        best_diff_count = -1
        # Search in a reasonable range
        max_search = i * i * 2 + 10
        for pos in range(max_search, i * i, -1):
            candidate = U + [pos]
            diffs = set()
            valid = True
            for a_idx in range(len(candidate)):
                for b_idx in range(a_idx + 1, len(candidate)):
                    d = abs(candidate[a_idx] - candidate[b_idx])
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                best_pos = pos
                break
        if best_pos is None:
            # Fallback: just use a large gap
            best_pos = U[-1] + i * i + 1
        U.append(best_pos)
    # Scale
    U = [x * scale for x in U]
    return U

def _generate_fibonacci(n):
    """Generate set based on Fibonacci-like growth."""
    U = [0]
    if n > 1:
        U.append(1)
    for i in range(2, n):
        U.append(U[-1] + U[-2])
    return U

def _generate_powers(n, base=2):
    """Generate set based on powers."""
    return [base ** i for i in range(n)]

def _generate_structured(n, pattern="mixed"):
    """Generate various structured sets."""
    if pattern == "golomb":
        return _generate_golomb_like(n)
    elif pattern == "fib":
        return _generate_fibonacci(n)
    elif pattern == "powers":
        return _generate_powers(n)
    elif pattern == "mixed":
        # Mix: start with small Golomb, extend with larger gaps
        U = _generate_golomb_like(min(n, 10))
        while len(U) < n:
            U.append(U[-1] + len(U) ** 2 + len(U))
        return U
    return U

def _local_search(U, max_iters=200, time_budget=30.0):
    """Local search: try adding, removing, or moving elements."""
    U = sorted(set(U))
    if not U:
        return U
    best_score = _score_of(U)
    best_U = list(U)
    start = time.time()
    
    for iteration in range(max_iters):
        if time.time() - start > time_budget:
            break
        if _remaining() < 10.0:
            break
        
        improved = False
        n = len(best_U)
        max_u = max(best_U)
        
        # Try adding a new element
        candidates_to_add = []
        # Near existing elements
        for x in best_U:
            for delta in range(1, 20):
                candidates_to_add.append(x + delta)
                candidates_to_add.append(max(0, x - delta))
        # Far elements
        candidates_to_add.extend(range(max_u + 1, max_u + 100))
        # Random
        for _ in range(50):
            candidates_to_add.append(random.randint(0, max_u * 2))
        
        for cand in candidates_to_add:
            if cand in best_U:
                continue
            new_U = sorted(best_U + [cand])
            s = _score_of(new_U)
            if s > best_score + 1e-12:
                best_score = s
                best_U = new_U
                improved = True
                break
        
        if improved:
            continue
        
        # Try removing an element (if size > 2)
        if len(best_U) > 2:
            for i in range(len(best_U)):
                if best_U[i] == 0:
                    continue  # Must keep 0
                new_U = best_U[:i] + best_U[i+1:]
                s = _score_of(new_U)
                if s > best_score + 1e-12:
                    best_score = s
                    best_U = new_U
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try moving a non-zero element
        for i in range(len(best_U)):
            if best_U[i] == 0:
                continue
            orig = best_U[i]
            for delta in range(1, 15):
                for new_val in [orig + delta, max(0, orig - delta)]:
                    if new_val in best_U:
                        continue
                    new_U = list(best_U)
                    new_U[i] = new_val
                    new_U = sorted(new_U)
                    s = _score_of(new_U)
                    if s > best_score + 1e-12:
                        best_score = s
                        best_U = new_U
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try a random restart-like perturbation
            if random.random() < 0.3:
                new_U = list(best_U)
                # Perturb 1-2 elements
                for _ in range(random.randint(1, 2)):
                    idx = random.choice([i for i in range(len(new_U)) if new_U[i] != 0])
                    new_U[idx] = max(0, new_U[idx] + random.randint(-20, 20))
                    new_U = sorted(set(new_U))
                s = _score_of(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U
    
    return best_U

def _multi_start_local_search(initial_sets, total_time_budget=100.0):
    """Run local search from multiple starting points."""
    best_U = None
    best_score = -float('inf')
    start = time.time()
    
    for init_set in initial_sets:
        if time.time() - start > total_time_budget:
            break
        if _remaining() < 15.0:
            break
        
        time_per_set = max(5.0, (total_time_budget - (time.time() - start)) / max(1, len(initial_sets)))
        result = _local_search(init_set, max_iters=300, time_budget=time_per_set)
        s = _score_of(result)
        if s > best_score:
            best_score = s
            best_U = result
    
    return best_U, best_score

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    random.seed(42)
    
    # Generate diverse initial candidates
    initial_sets = []
    
    # Small Golomb rulers
    for n in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]:
        U = _generate_golomb_like(n)
        initial_sets.append(U)
    
    # Fibonacci-based
    for n in [5, 7, 9, 11, 13]:
        U = _generate_fibonacci(n)
        initial_sets.append(U)
    
    # Powers
    for base in [2, 3]:
        for n in [5, 7, 9]:
            U = _generate_powers(n, base)
            initial_sets.append(U)
    
    # Mixed structured
    for n in [6, 8, 10, 12, 15]:
        U = _generate_structured(n, "mixed")
        initial_sets.append(U)
    
    # Known good: [0, 1, 3]
    initial_sets.append([0, 1, 3])
    
    # Some random structured sets
    for trial in range(10):
        n = random.randint(4, 12)
        U = [0]
        current = 0
        for i in range(1, n):
            current += random.randint(1, i * 3 + 2)
            U.append(current)
        initial_sets.append(U)
    
    # Deduplicate
    unique_sets = []
    seen = set()
    for s in initial_sets:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_sets.append(list(s))
    
    # Evaluate all initial sets
    best_init = None
    best_init_score = -float('inf')
    for s in unique_sets:
        sc = _score_of(s)
        if sc > best_init_score:
            best_init_score = sc
            best_init = s
    
    # Run multi-start local search
    # Prioritize better initial sets
    scored_sets = [( _score_of(s), s) for s in unique_sets]
    scored_sets.sort(key=lambda x: -x[0])
    top_sets = [s for _, s in scored_sets[:15]]
    
    best_U, best_score = _multi_start_local_search(top_sets, total_time_budget=90.0)
    
    if best_U is None:
        best_U = best_init if best_init else [0, 1, 3]
    
    # Final polish: more aggressive local search on the best result
    if _remaining() > 20.0:
        polished = _local_search(best_U, max_iters=500, time_budget=min(30.0, _remaining() - 10.0))
        s = _score_of(polished)
        if s > _score_of(best_U):
            best_U = polished
    
    # Ensure 0 is in the set and all are nonnegative integers
    best_U = sorted(set(int(x) for x in best_U if x >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
    
    return best_U
