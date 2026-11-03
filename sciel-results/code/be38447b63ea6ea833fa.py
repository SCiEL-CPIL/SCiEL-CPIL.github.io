import time
import math
import random
from itertools import combinations
import numpy as np

def _score_fast(U):
    """Compute the score for a given set U using NumPy for speed."""
    U_arr = np.array(sorted(set(U)), dtype=np.int64)
    n = len(U_arr)
    if n == 0:
        return 0.0
    max_u = int(U_arr[-1])
    if max_u == 0:
        return 1.0
    
    # Compute U+U and U-U using broadcasting
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    
    sum_set_size = len(np.unique(sums))
    diff_set_size = len(np.unique(diffs))
    
    ratio = diff_set_size / sum_set_size
    if ratio <= 1:
        return 1.0
    
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
    
    return 1.0 + log_ratio / log_denom

def _score(U):
    """Compute the score for a given set U (as a sorted list)."""
    return _score_fast(U)

def _local_search(U, deadline, max_steps=2000):
    """Local search: try adding, removing, or swapping elements."""
    current_U = sorted(set(U))
    current_score = _score(current_U)
    best_U = current_U[:]
    best_score = current_score
    
    max_val = 40
    step = 0
    while step < max_steps and time.time() < deadline:
        step += 1
        improved = False
        
        # Try removing an element (except 0)
        for i in range(1, len(current_U)):
            new_U = current_U[:i] + current_U[i+1:]
            s = _score(new_U)
            if s > best_score:
                best_score = s
                best_U = new_U[:]
                current_U = new_U
                improved = True
                break
        
        if improved:
            continue
        
        # Try adding an element
        current_set = set(current_U)
        for v in range(1, max_val + 1):
            if v not in current_set:
                new_U = sorted(current_U + [v])
                s = _score(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U[:]
                    current_U = new_U
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swapping: remove one, add another
        for i in range(1, len(current_U)):
            removed = current_U[i]
            base = current_U[:i] + current_U[i+1:]
            base_set = set(base)
            for v in range(1, max_val + 1):
                if v not in base_set and v != removed:
                    new_U = sorted(base + [v])
                    s = _score(new_U)
                    if s > best_score:
                        best_score = s
                        best_U = new_U[:]
                        current_U = new_U
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Stuck, try random perturbation
            if random.random() < 0.4:
                # Random restart with small modification
                if len(current_U) > 2:
                    idx = random.randint(1, len(current_U) - 1)
                    new_val = random.randint(1, max_val)
                    if new_val not in set(current_U):
                        current_U = current_U[:idx] + [new_val] + current_U[idx+1:]
                        current_U = sorted(set(current_U))
                        current_score = _score(current_U)
                        if current_score > best_score:
                            best_score = current_score
                            best_U = current_U[:]
                else:
                    # Add a random element
                    new_val = random.randint(1, max_val)
                    if new_val not in set(current_U):
                        current_U = sorted(current_U + [new_val])
                        current_score = _score(current_U)
                        if current_score > best_score:
                            best_score = current_score
                            best_U = current_U[:]
            else:
                break
    
    return best_U, best_score

def _simulated_annealing(U, deadline, max_iters=5000, T_start=2.0, T_end=0.01):
    """Simulated annealing to escape local optima."""
    current_U = sorted(set(U))
    current_score = _score(current_U)
    best_U = current_U[:]
    best_score = current_score
    
    max_val = 50
    iters = 0
    T = T_start
    
    while iters < max_iters and time.time() < deadline:
        iters += 1
        # Annealing schedule
        T = T_start * (T_end / T_start) ** (iters / max_iters)
        
        # Generate neighbor
        current_set = set(current_U)
        move_type = random.random()
        
        if move_type < 0.3 and len(current_U) > 1:
            # Remove element
            i = random.randint(1, len(current_U) - 1)
            new_U = current_U[:i] + current_U[i+1:]
        elif move_type < 0.6:
            # Add element
            candidates = [v for v in range(1, max_val + 1) if v not in current_set]
            if candidates:
                v = random.choice(candidates)
                new_U = sorted(current_U + [v])
            else:
                continue
        elif move_type < 0.85:
            # Swap
            if len(current_U) < 2:
                continue
            i = random.randint(1, len(current_U) - 1)
            removed = current_U[i]
            base = current_U[:i] + current_U[i+1:]
            base_set = set(base)
            candidates = [v for v in range(1, max_val + 1) if v not in base_set and v != removed]
            if candidates:
                v = random.choice(candidates)
                new_U = sorted(base + [v])
            else:
                continue
        else:
            # Random jump: replace 2-3 elements
            if len(current_U) < 2:
                continue
            num_replace = random.randint(1, min(3, len(current_U) - 1))
            indices = random.sample(range(1, len(current_U)), num_replace)
            base = [x for i, x in enumerate(current_U) if i not in indices]
            base_set = set(base)
            new_elements = []
            for _ in range(num_replace):
                candidates = [v for v in range(1, max_val + 1) if v not in base_set and v not in new_elements]
                if candidates:
                    v = random.choice(candidates)
                    new_elements.append(v)
                    base_set.add(v)
            if len(new_elements) == num_replace:
                new_U = sorted(base + new_elements)
            else:
                continue
        
        new_score = _score(new_U)
        
        # Accept or reject
        if new_score > current_score or random.random() < math.exp((new_score - current_score) / T):
            current_U = new_U
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_U = new_U[:]
    
    return best_U, best_score

def _search_small_sets(deadline):
    """Search over small sets of nonnegative integers containing 0."""
    best_score = 0.0
    best_set = [0]
    
    # Try all subsets of {0, 1, 2, ..., max_val} of size up to max_size
    max_val = 18
    max_size = 5
    
    # We must include 0
    candidates = list(range(1, max_val + 1))
    
    for size in range(1, max_size + 1):
        if time.time() > deadline:
            break
        for combo in combinations(candidates, size):
            if time.time() > deadline:
                break
            U = [0] + list(combo)
            U.sort()
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # First, get a good baseline from brute force
    best_set, best_score = _search_small_sets(deadline)
    
    # Try some known good structures and local search from them
    initial_candidates = []
    
    # Powers of 2
    for n in range(2, 10):
        initial_candidates.append([2**i for i in range(n)])
    
    # Powers of 3
    for n in range(2, 7):
        initial_candidates.append([3**i for i in range(n)])
    
    # Mixed structures
    initial_candidates.append([0, 1, 2, 4, 8])
    initial_candidates.append([0, 1, 2, 4, 8, 16])
    initial_candidates.append([0, 1, 2, 4, 7])
    initial_candidates.append([0, 1, 2, 4, 8, 15])
    initial_candidates.append([0, 1, 2, 3, 6])
    initial_candidates.append([0, 1, 2, 4, 5])
    initial_candidates.append([0, 1, 3, 4, 7])
    initial_candidates.append([0, 1, 2, 5, 8])
    initial_candidates.append([0, 1, 2, 4, 9])
    initial_candidates.append([0, 1, 2, 5, 10])
    initial_candidates.append([0, 1, 3, 6, 12])
    initial_candidates.append([0, 1, 2, 4, 6, 12])
    initial_candidates.append([0, 1, 2, 3, 4, 8])
    initial_candidates.append([0, 1, 2, 4, 8, 14])
    initial_candidates.append([0, 1, 2, 5, 10, 20])
    initial_candidates.append([0, 1, 3, 4, 7, 14])
    
    # Add best_set as a candidate
    initial_candidates.append(best_set)
    
    # Local search from each candidate
    for candidate in initial_candidates:
        if time.time() > deadline:
            break
        candidate = sorted(set(candidate))
        if 0 not in candidate:
            candidate = [0] + candidate
        # Ensure no duplicates and sorted
        candidate = sorted(set(candidate))
        
        s = _score(candidate)
        if s > best_score:
            best_score = s
            best_set = candidate[:]
        
        # Do local search
        searched_set, searched_score = _local_search(candidate, deadline, max_steps=500)
        if searched_score > best_score:
            best_score = searched_score
            best_set = searched_set[:]
    
    # Try simulated annealing from best found so far
    if time.time() < deadline:
        sa_set, sa_score = _simulated_annealing(best_set, deadline, max_iters=3000)
        if sa_score > best_score:
            best_score = sa_score
            best_set = sa_set[:]
    
    # Try a few random restarts with local search
    for _ in range(15):
        if time.time() > deadline:
            break
        # Random initial set
        size = random.randint(3, 8)
        vals = random.sample(range(1, 35), size)
        U = [0] + sorted(vals)
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
        
        searched_set, searched_score = _local_search(U, deadline, max_steps=400)
        if searched_score > best_score:
            best_score = searched_score
            best_set = searched_set[:]
    
    # Final simulated annealing polish on best result
    if time.time() < deadline - 5:
        sa_set, sa_score = _simulated_annealing(best_set, deadline, max_iters=2000)
        if sa_score > best_score:
            best_score = sa_score
            best_set = sa_set[:]
    
    return best_set
