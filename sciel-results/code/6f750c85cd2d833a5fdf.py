import time
import random
from math import log
import itertools
import numpy as np

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    ratio = len(diffs) / len(sums)
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def fast_score(U):
    """
    Faster scoring using numpy for larger sets.
    """
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return 0.0
    
    U_arr = np.array(U)
    
    # Compute all pairwise sums (i <= j)
    i_idx, j_idx = np.triu_indices(n)
    sums = U_arr[i_idx] + U_arr[j_idx]
    num_sums = len(np.unique(sums))
    
    # Compute all pairwise differences (i < j)
    i_idx_d, j_idx_d = np.triu_indices(n, k=1)
    diffs = U_arr[j_idx_d] - U_arr[i_idx_d]
    num_pos_diffs = len(np.unique(diffs))
    # Total diffs = 2 * num_pos_diffs + 1 (for 0)
    num_diffs = 2 * num_pos_diffs + 1
    
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
    return 1 + log(ratio) / log(D)

def greedy_add(U, deadline, max_steps=500):
    """
    Structural Change: Greedy Maximal Independent Set Heuristic.
    Vertices are candidate integers.
    We iteratively add the vertex that maximizes the score improvement (net gain).
    This is a direct, greedy approach to the 'graph coloring' analogy where we pick
    the best 'independent' addition at each step.
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current
    best_score = current_score
    
    # Candidate pool: 
    # 1. Gaps within current range
    # 2. Extensions beyond max
    max_val = max(current) * 2 + 100
    if max_val < 100:
        max_val = 100
    
    # Pre-compute a list of potential candidates to avoid generating them repeatedly
    # We focus on integers that are "far" from existing ones to encourage dispersion
    # but also check nearby to fill gaps if beneficial.
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
            
        n = len(current)
        
        # Generate a diverse set of candidates to evaluate
        # 1. Random samples from the full range
        # 2. Specific strategic points (gaps, extensions)
        candidates = set()
        
        # Random samples
        for _ in range(50):
            candidates.add(random.randint(1, max_val))
            
        # Strategic: Just above max
        for d in range(1, 20):
            candidates.add(max(current) + d)
            
        # Strategic: Fill large gaps
        for i in range(1, n):
            gap = current[i] - current[i-1]
            if gap > 5:
                mid = (current[i] + current[i-1]) // 2
                candidates.add(mid)
                candidates.add(mid - 1)
                candidates.add(mid + 1)
        
        # Remove existing elements
        current_set = set(current)
        candidates = [c for c in candidates if c not in current_set and c >= 1]
        
        if not candidates:
            break
            
        # Evaluate candidates and pick the best one
        best_candidate = None
        best_candidate_score = current_score
        
        # To save time, we can sample if there are too many, but 100 candidates * O(N^2) is usually fine for N<50
        # Let's limit to top 50 random candidates if the list is huge, though it shouldn't be.
        if len(candidates) > 100:
            candidates = random.sample(candidates, 100)
        
        for c in candidates:
            new_U = sorted(current_set | {c})
            s = fast_score(new_U)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = c
        
        if best_candidate is not None:
            current = sorted(current_set | {best_candidate})
            current_score = best_candidate_score
            if current_score > best_score:
                best_score = current_score
                best = current
            # Expand search range if we added a large number
            if best_candidate > max_val / 2:
                max_val = int(max_val * 1.5)
        else:
            # No improvement found in this batch, try a slightly different range or break?
            # If we can't improve, maybe we should try removing or swapping? 
            # But the strategy is "Greedy Add". If stuck, we might break or try a different random seed for candidates.
            # Let's try a few more rounds with fresh randoms.
            if step > 50:
                break
    
    return best, best_score

def local_search(U, deadline, max_iter=200):
    """
    Local search to fix local optima found by greedy or to polish.
    Uses perturbations (move, swap, add, remove).
    """
    current = sorted(set(U))
    current_score = fast_score(current)
    if current_score < 0:
        return current, current_score
    
    best = current
    best_score = current_score
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        n = len(current)
        move_type = random.choice(['move', 'swap', 'add', 'remove'])
        
        new_set = current[:]
        
        if move_type == 'move' and n > 1:
            i = random.randint(1, n-1)
            old_val = current[i]
            # Move within a window
            window = max(5, int(max(current) * 0.1))
            new_val = old_val + random.randint(-window, window)
            if new_val <= 0:
                new_val = old_val + random.randint(1, window)
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            if len(new_set) != n:
                continue # Skip if collision
            
        elif move_type == 'swap' and n > 1:
            i = random.randint(1, n-1)
            old_val = current[i]
            # Swap with a new value
            max_val = max(current) + 20
            new_val = random.randint(1, max_val)
            if new_val in current:
                continue
            new_set[i] = new_val
            new_set = sorted(set(new_set))
            
        elif move_type == 'add' and n < 30:
            max_val = max(current) + 50
            new_val = random.randint(1, max_val)
            if new_val in current:
                continue
            new_set.append(new_val)
            new_set = sorted(new_set)
            
        elif move_type == 'remove' and n > 3:
            i = random.randint(1, n-1)
            new_set.pop(i)
            
        else:
            continue
            
        new_score = fast_score(new_set)
        
        if new_score > current_score:
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best = current
        else:
            # Accept worse with low probability to escape local minima
            if random.random() < 0.1:
                current = new_set
                current_score = new_score
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_set = [0, 1, 4, 9, 11]
    best_score = fast_score(best_set)
    
    # Initial candidates: Golomb Rulers and Geometric Progressions
    golomb_rulers = [
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 41],
        [0, 1, 4, 10, 18, 23, 29, 37, 44, 52],
    ]
    
    geometric = []
    for ratio in [1.5, 2.0, 2.5, 3.0]:
        U = [0]
        val = 1
        for _ in range(12):
            U.append(val)
            val = int(val * ratio)
            if val > 2000:
                break
        geometric.append(U)
        
    candidates = golomb_rulers + geometric
    
    # Add some random sparse sets
    random.seed(42)
    for _ in range(10):
        n_elem = random.randint(4, 10)
        max_val = random.randint(50, 300)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        candidates.append(U)
        
    # Evaluate all candidates
    scored_candidates = []
    for U in candidates:
        s = fast_score(U)
        if s > 0:
            scored_candidates.append((s, U))
            
    scored_candidates.sort(reverse=True, key=lambda x: x[0])
    
    # Apply Greedy Add to the top candidates
    # This is the structural change: instead of just local search, we start with a good seed
    # and greedily build up the set by picking the best single addition at each step.
    
    for score, U in scored_candidates[:5]:
        if time.time() > deadline - 20:
            break
        
        refined, refined_score = greedy_add(U, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_set = refined
            
    # Apply Local Search to the best set found so far to polish
    if time.time() < deadline - 10:
        polished, polished_score = local_search(best_set, deadline, max_iter=300)
        if polished_score > best_score:
            best_score = polished_score
            best_set = polished
            
    # Try a few more random restarts with Greedy + Local Search if time permits
    for _ in range(3):
        if time.time() > deadline - 10:
            break
        n_elem = random.randint(4, 8)
        max_val = random.randint(30, 150)
        U = [0] + sorted(random.sample(range(1, max_val), n_elem - 1))
        
        # Greedy phase
        g_refined, g_score = greedy_add(U, deadline)
        # Local phase
        l_refined, l_score = local_search(g_refined, deadline, max_iter=100)
        
        if l_score > best_score:
            best_score = l_score
            best_set = l_refined
            
    return best_set
