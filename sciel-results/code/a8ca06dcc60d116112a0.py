import time
import math
import random
from itertools import combinations
import numpy as np

def _score(U):
    """Compute the score for a given set U (as a sorted list)."""
    U_set = set(U)
    n = len(U_set)
    if n == 0:
        return 0.0
    max_u = max(U_set)
    if max_u == 0:
        return 1.0  # degenerate case
    
    # Compute U+U
    sum_set = set()
    for a in U_set:
        for b in U_set:
            sum_set.add(a + b)
    
    # Compute U-U
    diff_set = set()
    for a in U_set:
        for b in U_set:
            diff_set.add(a - b)
    
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 1:
        return 1.0
    
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    
    if log_denom == 0:
        return 1.0
    
    return 1.0 + log_ratio / log_denom

def _local_search(U, deadline, max_steps=2000):
    """Local search: try adding, removing, or swapping elements."""
    current_U = sorted(set(U))
    current_score = _score(current_U)
    best_U = current_U[:]
    best_score = current_score
    
    max_val = 30
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
            if random.random() < 0.3:
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

def _search_small_sets(deadline):
    """Search over small sets of nonnegative integers containing 0."""
    best_score = 0.0
    best_set = [0]
    
    # Try all subsets of {0, 1, 2, ..., max_val} of size up to max_size
    max_val = 15
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
    
    # Try a few random restarts
    for _ in range(20):
        if time.time() > deadline:
            break
        # Random initial set
        size = random.randint(3, 7)
        vals = random.sample(range(1, 25), size)
        U = [0] + sorted(vals)
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
        
        searched_set, searched_score = _local_search(U, deadline, max_steps=300)
        if searched_score > best_score:
            best_score = searched_score
            best_set = searched_set[:]
    
    return best_set
