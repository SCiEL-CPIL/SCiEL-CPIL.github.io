"""
Strategy: Constraint Satisfaction Problem (CSP) with Backtracking Search.
Instead of stochastic local search (Simulated Annealing) which can get stuck
in local minima and requires careful temperature tuning, we model the problem
as a CSP.

Variables: Each row is a variable.
Domains: Sets of valid rows (tuples of increasing integers starting at 0) 
         with max element <= current_scope.
Constraints: All differences within all rows must be unique.

Heuristics:
1. MRV (Minimum Remaining Values): Order rows by the number of available
   candidates (rows that don't conflict with already placed rows).
   This is approximated by picking the "hardest" row to place next, 
   or simply iterating and checking conflicts.
   Actually, for a static CSP, we can precompute domains. But domains depend on scope.
   
   Better approach: Dynamic Backtracking.
   1. Start with a small scope M (close to counting bound).
   2. Generate all valid rows with max <= M.
   3. Use Backtracking to select n_rows rows such that their difference sets are disjoint.
   
   Optimization:
   - Precompute difference masks for all candidate rows.
   - Use bitmasks for fast conflict checking.
   - Order rows dynamically: at each step, pick the row index that has the 
     fewest compatible candidates given the current partial assignment? 
     No, rows are indistinguishable in the output, but we need n_rows of them.
     We can just pick any available row. To guide search, we can prefer rows 
     that "leave more room" for others, but MRV is usually best.
     Since rows are symmetric, we can fix the order of rows (e.g., sort by last element 
     or just pick the first available). However, to break symmetry and reduce search space,
     we can enforce an ordering on the rows, e.g., row_i[-1] <= row_{i+1}[-1].
     This reduces the permutation factor.

   Algorithm:
   1. Estimate initial M from counting bound.
   2. Loop M from M_start to M_max:
      a. Generate all valid rows with max <= M.
      b. For each row, compute its difference mask.
      c. Sort rows by some heuristic (e.g., sparsity, or just random).
      d. Backtracking search:
         - State: used_mask, list of chosen rows.
         - If len(chosen) == n_rows: Success.
         - Else: Find a row from candidates that is disjoint from used_mask.
           To reduce branching, we can filter candidates first.
           Since rows are symmetric, we can just pick the first valid one? 
           No, we need to explore.
           However, if we sort rows and enforce that we only pick rows 
           "later" in the sorted list than the last picked one, we avoid permutations.
           
   3. If found, return. Else increase M.
"""

import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START

def _row_diffs(row):
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs

def _row_mask(row):
    m = 0
    for d in _row_diffs(row):
        m |= (1 << d)
    return m

def _scope(rows):
    return max(r[-1] for r in rows)

def _is_valid(rows):
    all_diffs = []
    for row in rows:
        all_diffs.extend(_row_diffs(row))
    return len(all_diffs) == len(set(all_diffs))

def _generate_valid_rows(max_scope, num_elements):
    """
    Generate all valid rows of length num_elements starting with 0, 
    with max element <= max_scope.
    Returns list of (row_tuple, mask).
    """
    valid_rows = []
    # We need to choose num_elements - 1 distinct values from 1..max_scope
    # and sort them.
    # This is equivalent to choosing a subset of size num_elements-1 from {1, ..., max_scope}.
    # The row is (0, v1, v2, ..., v_{k}) where 0 < v1 < ... < vk <= max_scope.
    
    # To avoid duplicates and ensure we cover the space, we iterate through combinations.
    # However, the number of combinations C(max_scope, num_elements-1) can be large.
    # For (7,5), num_elements=6, so we choose 5 from 1..M.
    # If M=110, C(110, 5) is ~ 41 million. Too many to generate all at once in Python.
    
    # Alternative: Generate rows on the fly during backtracking?
    # Or, use a smarter generation.
    # Since we need disjoint difference sets, and difference sets are unique per row,
    # we can't just pick any row.
    
    # Let's use a different approach for generation:
    # We don't need ALL rows. We need a SET of n_rows rows.
    # We can use a randomized backtracking or a heuristic search.
    
    # Given the time limit, generating all combinations is risky if M is large.
    # Let's stick to a hybrid: 
    # 1. Try to find a solution with M close to bound using a greedy + local repair? 
    #    The current program does SA. 
    #    The prompt asks for a DIFFERENT strategy.
    
    # Let's try a "Conflict-Driven" Backtracking with limited candidate generation.
    # At each step, generate a batch of random valid rows, filter those compatible with 
    # current used_mask, and pick one. If stuck, backtrack.
    
    # But "Backtracking" implies a systematic search.
    # Let's try a systematic search for small M, and switch to SA for larger M?
    # No, let's stick to one strategy: CSP with Heuristics.
    
    # Refined Strategy:
    # 1. Start M at counting_bound.
    # 2. Use a recursive backtracking function.
    #    - Parameters: current_depth, used_mask, list_of_chosen_rows.
    #    - If depth == n_rows: return success.
    #    - Generate a list of candidate rows (randomly sampled from the space of valid rows 
    #      with max <= M). 
    #    - Filter candidates: must be disjoint from used_mask.
    #    - If no candidates: return failure.
    #    - Pick a candidate (e.g., one with "smallest" max element to leave room? 
    #      or random).
    #    - Recurse.
    #    - If recursion fails, try next candidate.
    
    # To make this efficient, we need good candidate generation.
    # Random sampling might miss the solution if the space is sparse.
    # But for (7,5), the space is not that sparse at M=112.
    
    # Let's combine: 
    # 1. Precompute a set of "good" rows? No.
    # 2. Use a "WalkSAT" style local search on the CSP? That's basically SA.
    
    # Let's try a **Greedy with Backtracking** approach.
    # 1. Sort all potential "structures" by some metric.
    # 2. Pick the best one.
    # 3. If stuck, backtrack.
    
    # Actually, a very effective CSP heuristic is **Forward Checking**.
    # When we pick a row, we remove all conflicting rows from the domains of subsequent rows.
    # If any domain becomes empty, fail.
    
    # Implementation Plan:
    # 1. Determine M.
    # 2. Generate a large pool of candidate rows (e.g., 10,000 random valid rows).
    # 3. Compute masks.
    # 4. Backtracking:
    #    - Maintain a list of available candidates.
    #    - At each step, pick a candidate that is compatible with used_mask.
    #    - To ensure we don't pick the same row twice, we track indices.
    #    - Heuristic: Pick the candidate that minimizes the "conflict potential" 
    #      with the remaining pool? (Hard to compute).
    #    - Simpler: Pick the candidate with the smallest last element (greedy packing).
    #    - If stuck, backtrack and try the next best candidate.
    
    # This is essentially a Depth-First Search with a specific branching order.
    
    pass

def _construct_csp(n_rows, k, time_limit):
    """
    Attempt to construct the set using CSP Backtracking.
    """
    start_time = time.time()
    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    counting_bound = n_rows * num_pairs
    
    # Start from counting bound + small margin
    M_start = counting_bound
    M_max = counting_bound + 50
    
    rng = random.Random(123)
    
    for M in range(M_start, M_max + 1):
        if time.time() - start_time > time_limit - 5:
            break
            
        # Generate a pool of candidate rows
        # We want diverse candidates.
        # Random sampling from 1..M.
        pool = []
        pool_masks = []
        seen_rows = set()
        
        # Generate enough candidates
        # C(M, k) is the total space. We sample.
        # For M=105, k=5, C(105,5) ~ 96M. We can't generate all.
        # We generate 5000 random candidates.
        
        candidates_found = 0
        attempts = 0
        max_attempts = 50000
        
        while candidates_found < 5000 and attempts < max_attempts:
            attempts += 1
            if M < num_elements - 1:
                break
            vals = rng.sample(range(1, M + 1), num_elements - 1)
            vals.sort()
            row = (0,) + tuple(vals)
            
            if row in seen_rows:
                continue
            seen_rows.add(row)
            
            diffs = _row_diffs(row)
            if len(diffs) != len(set(diffs)):
                continue
                
            mask = _row_mask(row)
            pool.append(row)
            pool_masks.append(mask)
            candidates_found += 1
            
        if candidates_found < n_rows:
            continue
            
        # Backtracking Search
        # We want to select n_rows rows from pool such that masks are disjoint.
        # Order: Sort pool by last element (ascending) to pack tightly?
        # Or sort by "popularity" (how many other rows it conflicts with)?
        # Let's sort by last element to encourage small scope usage.
        
        # To avoid symmetry, we can enforce that we pick rows in an order 
        # such that row_i <= row_{i+1} in some ordering.
        # Since rows are tuples, we can sort the pool lexicographically.
        # And in backtracking, we only pick rows with index > last_picked_index.
        
        sorted_indices = sorted(range(len(pool)), key=lambda i: pool[i])
        sorted_pool = [pool[i] for i in sorted_indices]
        sorted_masks = [pool_masks[i] for i in sorted_indices]
        
        # Backtracking function
        def backtrack(depth, used_mask, chosen_indices, start_idx):
            if time.time() - start_time > time_limit - 2:
                return None
                
            if depth == n_rows:
                return [sorted_pool[i] for i in chosen_indices]
                
            # Try candidates from start_idx to end
            # Heuristic: Try candidates that are "least conflicting" with the rest?
            # For now, just try in order.
            
            for i in range(start_idx, len(sorted_pool)):
                m = sorted_masks[i]
                if m & used_mask:
                    continue
                    
                # Forward Checking: 
                # If we pick this, can we still find n_rows - depth - 1 rows 
                # from the remaining indices?
                # This is expensive to check exactly. 
                # Approximation: Check if there are at least n_rows - depth - 1 
                # rows remaining that are disjoint from (used_mask | m).
                # We can do a quick count or just rely on backtracking.
                
                # Pick this row
                chosen_indices.append(i)
                result = backtrack(depth + 1, used_mask | m, chosen_indices, i + 1)
                if result is not None:
                    return result
                chosen_indices.pop()
                
            return None
            
        # Run backtracking
        # To improve success rate, we might want to shuffle the pool slightly 
        # or use a different heuristic.
        # Let's try a few random shuffles of the sorted pool to escape local traps?
        # No, the "index > last_picked" constraint breaks shuffling.
        
        # Alternative: Don't enforce index ordering. Just ensure we don't pick the same row.
        # This increases search space by n_rows! but might find solutions faster 
        # if the "sorted" order is bad.
        # Given n_rows=7, 7! = 5040. Manageable.
        
        # Let's try without index ordering first, but with a limit on branching.
        
        def backtrack_no_order(depth, used_mask, chosen_rows, available_indices):
            if time.time() - start_time > time_limit - 2:
                return None
                
            if depth == n_rows:
                return chosen_rows
                
            # Heuristic: Pick the row with the smallest last element 
            # among available candidates to keep scope low?
            # Or pick the one that conflicts with the fewest other available rows?
            
            # Let's try: Pick the first available row that is compatible.
            # To diversify, we can rotate the start point or shuffle.
            
            # For now, simple DFS:
            # Find all compatible rows
            compatible = []
            for idx in available_indices:
                if not (sorted_masks[idx] & used_mask):
                    compatible.append(idx)
                    
            if not compatible:
                return None
                
            # Sort compatible by last element (smallest first)
            compatible.sort(key=lambda idx: sorted_pool[idx][-1])
            
            for idx in compatible:
                new_available = [i for i in available_indices if i != idx]
                chosen_rows.append(sorted_pool[idx])
                result = backtrack_no_order(depth + 1, used_mask | sorted_masks[idx], chosen_rows, new_available)
                if result is not None:
                    return result
                chosen_rows.pop()
                
            return None
            
        result = backtrack_no_order(0, 0, [], list(range(len(sorted_pool))))
        
        if result is not None:
            # Convert to lists
            return [list(r) for r in result]
            
    return None

def construct(n_rows, k):
    start_time = time.time()
    DEADLINE = start_time + 140.0  # Safety margin

    num_elements = k + 1
    num_pairs = num_elements * (num_elements - 1) // 2
    total_diffs_needed = n_rows * num_pairs
    counting_bound = n_rows * num_pairs
    
    # Try CSP approach first
    # Give it 60 seconds
    csp_limit = 60
    if time.time() < DEADLINE - csp_limit:
        result = _construct_csp(n_rows, k, csp_limit)
        if result is not None:
            if _is_valid(result):
                return result
            else:
                print("CSP result invalid, falling back")
                
    # Fallback to a simple greedy/SA if CSP fails
    # Re-use a simplified SA from the previous logic or just greedy
    
    rng = random.Random(42)
    
    # Greedy construction
    def _generate_random_row(max_val, num_elements, rng):
        if num_elements == 1:
            return [0]
        if num_elements == 2:
            return [0, rng.randint(1, max_val)]
        if max_val < num_elements - 1:
            return None
        vals = rng.sample(range(1, max_val + 1), num_elements - 1)
        vals.sort()
        return [0] + vals

    def _greedy_construct(n_rows, num_elements, max_scope, rng, max_restarts=20):
        for restart in range(max_restarts):
            used_mask = 0
            rows = []
            success = True
            for _ in range(n_rows):
                found = False
                for _ in range(100):
                    row = _generate_random_row(max_scope, num_elements, rng)
                    if row is None:
                        continue
                    diffs = _row_diffs(row)
                    if len(diffs) != len(set(diffs)):
                        continue
                    m = _row_mask(row)
                    if m & used_mask:
                        continue
                    rows.append(row)
                    used_mask |= m
                    found = True
                    break
                if not found:
                    success = False
                    break
            if success:
                return rows
        return None

    # Try greedy with expanding scope
    scope_try = counting_bound + 10
    while scope_try < 300 and time.time() < DEADLINE - 10:
        rows = _greedy_construct(n_rows, num_elements, scope_try, rng, max_restarts=10)
        if rows is not None:
            if _is_valid(rows):
                return rows
        scope_try += 5
        
    # Final fallback: Sequential construction (always valid, but large scope)
    rows = []
    used = set()
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < num_elements:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
        
    return rows

# EVOLVE-BLOCK-END
