"""Superpermutation search: Greedy Hamiltonian Path + fast local search."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import permutations

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def _classic_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!."""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            best_overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, 0, -1):
                if new[-o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new
    return word


def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def _precompute_costs(n):
    """
    Precompute transition costs between all permutations.
    Returns (all_perms, costs_matrix).
    """
    all_perms = list(permutations(range(1, n + 1)))
    N = len(all_perms)
    
    if HAS_NUMPY:
        costs = np.zeros((N, N), dtype=np.uint8)
        for i in range(N):
            p_i = all_perms[i]
            row = costs[i]
            for j in range(N):
                if i == j:
                    continue
                p_j = all_perms[j]
                max_k = 0
                # Check overlaps from largest to smallest
                # k is the length of the matching suffix of p_i and prefix of p_j
                for k in range(n - 1, 0, -1):
                    if p_i[n-k:] == p_j[:k]:
                        max_k = k
                        break
                row[j] = n - max_k
        return all_perms, costs
    else:
        costs = [[0] * N for _ in range(N)]
        for i in range(N):
            p_i = all_perms[i]
            for j in range(N):
                if i == j:
                    continue
                p_j = all_perms[j]
                max_k = 0
                for k in range(n - 1, 0, -1):
                    if p_i[n-k:] == p_j[:k]:
                        max_k = k
                        break
                costs[i][j] = n - max_k
        return all_perms, costs


def _greedy_construct(n, all_perms, costs, seed):
    """
    Construct a superpermutation using a greedy heuristic with random tie-breaking.
    """
    random.seed(seed)
    N = len(all_perms)
    visited = [False] * N
    current = 0
    visited[0] = True
    path = [0]
    total_len = n
    
    # Get the cost row for the current node
    if HAS_NUMPY:
        # costs is a numpy array
        pass
    else:
        # costs is a list of lists
        pass

    for _ in range(N - 1):
        min_cost = 255 # Max possible cost is n (which is <= 7)
        best_candidates = []
        
        if HAS_NUMPY:
            row = costs[current]
            # Iterate through unvisited
            for j in range(N):
                if not visited[j]:
                    c = row[j]
                    if c < min_cost:
                        min_cost = c
                        best_candidates = [j]
                    elif c == min_cost:
                        best_candidates.append(j)
        else:
            row = costs[current]
            for j in range(N):
                if not visited[j]:
                    c = row[j]
                    if c < min_cost:
                        min_cost = c
                        best_candidates = [j]
                    elif c == min_cost:
                        best_candidates.append(j)
        
        if not best_candidates:
            break # Should not happen
            
        next_node = random.choice(best_candidates)
        visited[next_node] = True
        path.append(next_node)
        total_len += min_cost
        current = next_node
        
    # Reconstruct word
    word = list(all_perms[path[0]])
    for idx in range(1, len(path)):
        prev_id = path[idx-1]
        curr_id = path[idx]
        p_prev = all_perms[prev_id]
        p_curr = all_perms[curr_id]
        
        # Find overlap
        max_k = 0
        for k in range(n - 1, 0, -1):
            if p_prev[n-k:] == p_curr[:k]:
                max_k = k
                break
        word.extend(p_curr[max_k:])
        
    return word


def _build_perm_index(n):
    """Build a dict mapping permutation tuples to unique IDs, and return list of all perms."""
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    return all_perms, perm_to_id


def _init_counts(word, n, perm_to_id):
    """Initialize count array: for each permutation, how many windows cover it."""
    num_perms = _factorial(n)
    counts = [0] * num_perms
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        pid = perm_to_id.get(p)
        if pid is not None:
            counts[pid] += 1
    return counts


def _is_all_covered(counts):
    """Check if all permutations have count >= 1."""
    for c in counts:
        if c == 0:
            return False
    return True


def _try_remove_chars_fast(word, n, time_limit):
    """
    Shorten superpermutation using fast count-tracking deletion.
    """
    word = list(word)
    target_perms = _factorial(n)
    deadline = time.time() + time_limit
    
    # Build permutation index
    all_perms, perm_to_id = _build_perm_index(n)
    num_perms = target_perms
    
    # Initialize counts
    counts = _init_counts(word, n, perm_to_id)
    
    if not _is_all_covered(counts):
        return word
    
    best_len = len(word)
    
    # Fast deletion: try removing each position, use incremental count updates
    improved = True
    while improved:
        if time.time() > deadline - 2:
            break
        improved = False
        
        for i in range(len(word) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            if len(word) - 1 < n:
                break
            
            L = len(word)
            
            # Save state for rollback
            old_counts = []
            
            # Step 1: Decrement counts for old windows that include position i
            k_start = max(0, i - n + 1)
            k_end = min(i, L - n)
            
            for k in range(k_start, k_end + 1):
                p = tuple(word[k:k + n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    old_counts.append((k, pid))
                    counts[pid] -= 1
            
            # Step 2: Compute new windows formed after deletion
            new_k_start = max(0, i - n + 1)
            new_k_end = min(i - 1, L - 1 - n)
            
            new_pids = []
            for k in range(new_k_start, new_k_end + 1):
                if k < i:
                    w1 = word[k:i]
                    w2 = word[i + 1:k + n + 1]
                    p = tuple(w1 + w2)
                else:
                    p = tuple(word[k + 1:k + 1 + n])
                
                pid = perm_to_id.get(p)
                if pid is not None:
                    counts[pid] += 1
                    new_pids.append(pid)
            
            # Step 3: Check if all permutations still covered
            valid = True
            for pid in range(num_perms):
                if counts[pid] == 0:
                    valid = False
                    break
            
            if valid:
                # Accept deletion
                del word[i]
                improved = True
                best_len = len(word)
            else:
                # Rollback
                for k, pid in old_counts:
                    counts[pid] += 1
                for pid in new_pids:
                    counts[pid] -= 1
    
    return word


def construct(n):
    """
    Construct a superpermutation on n symbols.
    Returns a string of digits 1..n.
    """
    start_time = time.time()
    time_limit = 120
    
    if n <= 1:
        return "1" if n == 1 else ""
    
    # Step 1: Precompute costs
    all_perms, costs = _precompute_costs(n)
    
    # Step 2: Greedy Construction
    best_word = None
    best_len = float('inf')
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 30 # Reserve 30s for local search
    
    # Estimate number of runs
    # Each run is O(N^2). N=5040 -> 25M ops.
    # In Python, 25M simple ops might take ~1-2s.
    # Let's try 5 runs first.
    max_runs = 5
    if remaining > 60:
        max_runs = 10
    elif remaining > 30:
        max_runs = 5
    else:
        max_runs = 1
        
    for run in range(max_runs):
        if time.time() > start_time + time_limit - 40:
            break
        word = _greedy_construct(n, all_perms, costs, seed=run)
        if len(word) < best_len:
            best_len = len(word)
            best_word = word
            
    if best_word is None:
        # Fallback
        best_word = _classic_superperm(n)
        best_len = len(best_word)
        
    # Step 3: Local Search
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 5
    if remaining > 5:
        best_word = _try_remove_chars_fast(best_word, n, remaining)
        
    return "".join(map(str, best_word))
# EVOLVE-BLOCK-END
