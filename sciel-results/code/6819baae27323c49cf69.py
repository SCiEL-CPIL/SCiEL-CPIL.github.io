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

def _modular_greedy_search(p, max_k, deadline):
    """
    Greedy search restricted to U = {k * p | k in V} where V contains 0.
    This structurally eliminates low-order sum collisions.
    """
    # V contains 0, we select other elements from 1..max_k
    V = [0]
    current_score = _score([0])
    best_V = V[:]
    best_score = current_score
    
    candidates = list(range(1, max_k + 1))
    
    # Greedy: repeatedly add the element that maximizes score increase
    while time.time() < deadline:
        best_gain = 0
        best_candidate = None
        
        for c in candidates:
            if c in V:
                continue
            new_V = V + [c]
            new_U = [k * p for k in new_V]
            s = _score(new_U)
            gain = s - current_score
            if gain > best_gain:
                best_gain = gain
                best_candidate = c
        
        if best_candidate is None or best_gain <= 0:
            break
        
        V.append(best_candidate)
        V.sort()
        current_score = _score([k * p for k in V])
        
        if current_score > best_score:
            best_score = current_score
            best_V = V[:]
    
    # Try removing elements to see if score improves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(1, len(best_V)):
            new_V = best_V[:i] + best_V[i+1:]
            new_U = [k * p for k in new_V]
            s = _score(new_U)
            if s > best_score:
                best_score = s
                best_V = new_V[:]
                improved = True
    
    return [k * p for k in best_V], best_score

def _modular_local_search(U, p, deadline, max_steps=1000):
    """
    Local search restricted to modular structure U = {k*p | k in V}.
    """
    # Extract V from U
    V = [u // p for u in U if u % p == 0]
    V = sorted(set(V))
    if 0 not in V:
        V = [0] + V
    
    current_V = V[:]
    current_score = _score([k * p for k in current_V])
    best_V = current_V[:]
    best_score = current_score
    
    max_k = max(current_V) + 20 if current_V else 30
    step = 0
    
    while step < max_steps and time.time() < deadline:
        step += 1
        improved = False
        
        # Try removing an element from V (except 0)
        for i in range(1, len(current_V)):
            new_V = current_V[:i] + current_V[i+1:]
            new_U = [k * p for k in new_V]
            s = _score(new_U)
            if s > best_score:
                best_score = s
                best_V = new_V[:]
                current_V = new_V
                improved = True
                break
        
        if improved:
            continue
        
        # Try adding an element to V
        current_set = set(current_V)
        for v in range(1, max_k + 1):
            if v not in current_set:
                new_V = sorted(current_V + [v])
                new_U = [k * p for k in new_V]
                s = _score(new_U)
                if s > best_score:
                    best_score = s
                    best_V = new_V[:]
                    current_V = new_V
                    improved = True
                    break
        
        if improved:
            continue
        
        # Try swapping in V
        for i in range(1, len(current_V)):
            removed = current_V[i]
            base = current_V[:i] + current_V[i+1:]
            base_set = set(base)
            for v in range(1, max_k + 1):
                if v not in base_set and v != removed:
                    new_V = sorted(base + [v])
                    new_U = [k * p for k in new_V]
                    s = _score(new_U)
                    if s > best_score:
                        best_score = s
                        best_V = new_V[:]
                        current_V = new_V
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Random perturbation
            if random.random() < 0.3:
                if len(current_V) > 2:
                    idx = random.randint(1, len(current_V) - 1)
                    new_val = random.randint(1, max_k)
                    if new_val not in set(current_V):
                        current_V = current_V[:idx] + [new_val] + current_V[idx+1:]
                        current_V = sorted(set(current_V))
                        current_score = _score([k * p for k in current_V])
                        if current_score > best_score:
                            best_score = current_score
                            best_V = current_V[:]
                else:
                    new_val = random.randint(1, max_k)
                    if new_val not in set(current_V):
                        current_V = sorted(current_V + [new_val])
                        current_score = _score([k * p for k in current_V])
                        if current_score > best_score:
                            best_score = current_score
                            best_V = current_V[:]
            else:
                break
    
    return [k * p for k in best_V], best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_score = 0.0
    best_set = [0]
    
    # Phase 1: Modular arithmetic search with various primes
    primes_to_try = [2, 3, 5, 7, 11, 13, 17, 19]
    for p in primes_to_try:
        if time.time() > deadline - 20:
            break
        
        # Determine max_k based on time budget
        time_left = deadline - time.time()
        max_k = min(30, max(10, int(time_left / 10)))
        
        # Greedy search
        mod_set, mod_score = _modular_greedy_search(p, max_k, deadline)
        if mod_score > best_score:
            best_score = mod_score
            best_set = mod_set[:]
        
        # Local search on the result
        if time.time() < deadline - 10:
            ls_set, ls_score = _modular_local_search(mod_set, p, deadline, max_steps=500)
            if ls_score > best_score:
                best_score = ls_score
                best_set = ls_set[:]
    
    # Phase 2: Try some known good structures and local search from them
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
        if time.time() > deadline - 15:
            break
        candidate = sorted(set(candidate))
        if 0 not in candidate:
            candidate = [0] + candidate
        candidate = sorted(set(candidate))
        
        s = _score(candidate)
        if s > best_score:
            best_score = s
            best_set = candidate[:]
        
        # Do local search
        searched_set, searched_score = _local_search(candidate, deadline, max_steps=300)
        if searched_score > best_score:
            best_score = searched_score
            best_set = searched_set[:]
    
    # Phase 3: Simulated annealing from best found so far
    if time.time() < deadline - 10:
        sa_set, sa_score = _simulated_annealing(best_set, deadline, max_iters=2000)
        if sa_score > best_score:
            best_score = sa_score
            best_set = sa_set[:]
    
    # Phase 4: Try a few random restarts with local search
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        # Random initial set
        size = random.randint(3, 8)
        vals = random.sample(range(1, 35), size)
        U = [0] + sorted(vals)
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
        
        searched_set, searched_score = _local_search(U, deadline, max_steps=300)
        if searched_score > best_score:
            best_score = searched_score
            best_set = searched_set[:]
    
    # Final simulated annealing polish on best result
    if time.time() < deadline - 5:
        sa_set, sa_score = _simulated_annealing(best_set, deadline, max_iters=1500)
        if sa_score > best_score:
            best_score = sa_score
            best_set = sa_set[:]
    
    return best_set
