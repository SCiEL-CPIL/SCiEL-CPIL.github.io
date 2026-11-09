"""Rewritten program: CSP-based backtracking to maximize cardinality-separated subset sums."""
import sys
import random
import time

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a set of distinct integers in 1..256 maximizing size such that
    no two subsets of different cardinalities have the same sum.
    
    Uses a backtracking search with forward checking.
    """
    start_time = time.time()
    time_limit = 55.0  # Leave 5s buffer for overhead/evaluation
    
    best_set = []
    
    def backtrack_idx(current_set, sums_by_card, candidates, start_idx):
        nonlocal best_set, start_time
        
        if time.time() - start_time > time_limit:
            return
        
        # Update best if current is better
        if len(current_set) > len(best_set):
            best_set = current_set[:]
        
        # If we run out of candidates, stop
        if start_idx >= len(candidates):
            return
            
        # Pruning: Even if we pick all remaining, can we beat best?
        if len(current_set) + (len(candidates) - start_idx) <= len(best_set):
            return
            
        for i in range(start_idx, len(candidates)):
            x = candidates[i]
            
            # Generate new sums
            max_k = len(current_set) + 1
            new_sums = [set() for _ in range(max_k + 1)]
            
            for k in range(1, max_k + 1):
                if k - 1 < len(sums_by_card) and sums_by_card[k-1]:
                    new_sums[k] = {s + x for s in sums_by_card[k-1]}
            
            # Check collision
            valid = True
            for k in range(1, max_k + 1):
                if not new_sums[k]:
                    continue
                for s in new_sums[k]:
                    for j in range(len(sums_by_card)):
                        if j == k:
                            continue
                        if s in sums_by_card[j]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Prepare new state
            new_sums_by_card = [set() for _ in range(max_k + 1)]
            for k in range(max_k + 1):
                if k < len(sums_by_card):
                    new_sums_by_card[k] = sums_by_card[k].copy()
                if k < len(new_sums):
                    new_sums_by_card[k] = new_sums_by_card[k].union(new_sums[k])
            
            # Recurse
            backtrack_idx(current_set + [x], new_sums_by_card, candidates, i + 1)
            
            if time.time() - start_time > time_limit:
                return

    # Initial state
    init_sums = [set() for _ in range(1)]
    init_sums[0] = {0}
    candidates = list(range(1, 257))
    
    # Run deterministic search first (ascending order)
    backtrack_idx([], init_sums, candidates, 0)
    
    # If we have time left, try some randomized restarts to escape local optima
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        for restart in range(20):
            if time.time() - start_time > time_limit:
                break
            
            # Shuffle candidates
            rand_candidates = list(range(1, 257))
            random.shuffle(rand_candidates)
            
            # Run search
            backtrack_idx([], init_sums, rand_candidates, 0)
            
            if time.time() - start_time > time_limit:
                break

    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
