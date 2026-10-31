"""Program for sumset/difference-set exponent lower bound maximization."""
import time
import math
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _score(U):
    """Compute the score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    if not U:
        return 0.0
    U = sorted(set(U))
    if not U or U[0] != 0:
        return 0.0
    n = len(U)
    max_u = U[-1]
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for i in range(n):
        for j in range(i, n):  # Since U+U is symmetric, we can use i<=j
            sumset.add(U[i] + U[j])
    
    # Compute difference set U-U
    diffset = set()
    for i in range(n):
        for j in range(n):
            diffset.add(U[i] - U[j])
    
    sum_count = len(sumset)
    diff_count = len(diffset)
    
    if sum_count == 0:
        return 0.0
    
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 0.0
    
    return 1.0 + math.log(diff_count / sum_count) / log_denom


def _score_fast(U_set, max_u):
    """Faster score computation using the set directly."""
    if not U_set:
        return 0.0
    n = len(U_set)
    
    # Compute sumset
    sumset = set()
    U_list = list(U_set)
    for i in range(n):
        ui = U_list[i]
        for j in range(i, n):
            sumset.add(ui + U_list[j])
    
    # Compute difference set
    diffset = set()
    for i in range(n):
        ui = U_list[i]
        for j in range(n):
            diffset.add(ui - U_list[j])
    
    sum_count = len(sumset)
    diff_count = len(diffset)
    
    if sum_count == 0:
        return 0.0
    
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 0.0
    
    return 1.0 + math.log(diff_count / sum_count) / log_denom


def _initial_set(size, max_val):
    """Generate an initial set containing 0."""
    if size <= 1:
        return [0]
    # Start with 0 and random other elements
    candidates = list(range(1, max_val + 1))
    random.shuffle(candidates)
    return [0] + candidates[:size - 1]


def _construct_greedy(max_val, target_size, deadline):
    """Greedy construction: add elements that improve score most."""
    U = [0]
    best_score = _score(U)
    
    # Try adding elements greedily
    for _ in range(target_size - 1):
        if time.time() > deadline:
            break
        
        best_elem = None
        best_new_score = best_score - 1e-18
        
        # Sample candidate elements to try
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        # Limit candidates for speed
        candidates = candidates[:min(500, max_val)]
        
        for elem in candidates:
            if elem in U:
                continue
            new_U = U + [elem]
            new_max = max(new_U)
            s = _score_fast(set(new_U), new_max)
            if s > best_new_score:
                best_new_score = s
                best_elem = elem
        
        if best_elem is not None:
            U.append(best_elem)
            best_score = best_new_score
        else:
            break
    
    return sorted(U), best_score


def _local_search(U, max_val, deadline, temperature=1000):
    """Local search with insertions, deletions, and swaps."""
    U = sorted(set(U))
    best_score = _score(U)
    current_score = best_score
    U_set = set(U)
    
    iteration = 0
    no_improve = 0
    
    while time.time() < deadline and no_improve < 50:
        iteration += 1
        
        # Try random insertions
        if random.random() < 0.4:
            elem = random.randint(1, max_val)
            if elem not in U_set:
                new_U = U_set | {elem}
                new_max = max(new_U)
                s = _score_fast(new_U, new_max)
                if s > current_score:
                    U_set = new_U
                    current_score = s
                    if s > best_score:
                        best_score = s
                    no_improve = 0
                else:
                    no_improve += 1
        
        # Try random deletions (keep 0)
        elif random.random() < 0.3:
            candidates = [x for x in U_set if x != 0]
            if candidates:
                elem = random.choice(candidates)
                new_U = U_set - {elem}
                if new_U:
                    new_max = max(new_U)
                    s = _score_fast(new_U, new_max)
                    if s > current_score:
                        U_set = new_U
                        current_score = s
                        if s > best_score:
                            best_score = s
                        no_improve = 0
                    else:
                        no_improve += 1
        
        # Try swaps
        else:
            candidates = [x for x in U_set if x != 0]
            if candidates:
                elem = random.choice(candidates)
                new_elem = random.randint(1, max_val)
                if new_elem not in U_set:
                    new_U = U_set - {elem} | {new_elem}
                    new_max = max(new_U)
                    s = _score_fast(new_U, new_max)
                    if s > current_score:
                        U_set = new_U
                        current_score = s
                        if s > best_score:
                            best_score = s
                        no_improve = 0
                    else:
                        no_improve += 1
    
    return sorted(U_set), best_score


def _simulated_annealing(U, max_val, deadline, initial_temp=1000, cooling=0.995):
    """Simulated annealing for set optimization."""
    U_set = set(U)
    current_score = _score_fast(U_set, max(U_set))
    best_score = current_score
    best_U = set(U_set)
    temp = initial_temp
    
    while time.time() < deadline and temp > 1e-6:
        temp *= cooling
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.4:  # Insert
            elem = random.randint(1, max_val)
            if elem not in U_set:
                new_U = U_set | {elem}
                new_max = max(new_U)
                s = _score_fast(new_U, new_max)
                delta = s - current_score
                if delta > 0 or random.random() < math.exp(delta / max(temp, 1e-9)):
                    U_set = new_U
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_U = set(U_set)
        
        elif move_type < 0.7:  # Delete
            candidates = [x for x in U_set if x != 0]
            if candidates:
                elem = random.choice(candidates)
                new_U = U_set - {elem}
                if new_U:
                    new_max = max(new_U)
                    s = _score_fast(new_U, new_max)
                    delta = s - current_score
                    if delta > 0 or random.random() < math.exp(delta / max(temp, 1e-9)):
                        U_set = new_U
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_U = set(U_set)
        
        else:  # Swap
            candidates = [x for x in U_set if x != 0]
            if candidates:
                elem = random.choice(candidates)
                new_elem = random.randint(1, max_val)
                if new_elem not in U_set:
                    new_U = U_set - {elem} | {new_elem}
                    new_max = max(new_U)
                    s = _score_fast(new_U, new_max)
                    delta = s - current_score
                    if delta > 0 or random.random() < math.exp(delta / max(temp, 1e-9)):
                        U_set = new_U
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_U = set(U_set)
    
    return sorted(best_U), best_score


def construct():
    """Main entry point: construct a set U to maximize the score."""
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    random.seed(42)
    
    # Try different strategies and keep the best
    best_U = [0]
    best_score = _score(best_U)
    
    # Strategy 1: Greedy construction with various sizes
    for target_size in [10, 30, 50, 80, 120, 200]:
        if time.time() > deadline - 30:
            break
        max_val = target_size * 20
        U, score = _construct_greedy(max_val, target_size, deadline - 10)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Strategy 2: Simulated annealing from greedy results
    if time.time() < deadline - 20:
        max_val = max(best_U) * 2 + 100 if best_U else 200
        U, score = _simulated_annealing(best_U, max_val, deadline - 5, initial_temp=500, cooling=0.99)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Strategy 3: Local search refinement
    if time.time() < deadline - 5:
        max_val = max(best_U) + 50 if best_U else 200
        U, score = _local_search(best_U, max_val, deadline - 2, temperature=500)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U = [0] + best_U
    
    return sorted(set(best_U))

# EVOLVE-BLOCK-END
