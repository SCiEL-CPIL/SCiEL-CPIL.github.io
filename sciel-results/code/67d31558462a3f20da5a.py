# EVOLVE-BLOCK-START
import math
import time
import random
from fractions import Fraction

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    N = 255  # numbers 2..256
    nums = list(range(2, 257))
    
    # Precompute conflict bitmasks
    # conflict_mask[i] is a bitset of vertices that conflict with i
    conflict_mask = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if math.lcm(nums[i], nums[j]) <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Precompute reciprocal values
    recip = [1.0 / nums[i] for i in range(N)]
    
    # Greedy initialization
    def greedy(order):
        selected_mask = 0
        for v in order:
            # Check if v conflicts with any selected
            # selected_mask has bits set for selected vertices
            # conflict_mask[v] has bits set for vertices conflicting with v
            # If (selected_mask & conflict_mask[v]) != 0, there is a conflict
            if (selected_mask & conflict_mask[v]) == 0:
                selected_mask |= (1 << v)
        return selected_mask

    # Local Search with 1-for-1 and 2-for-1 exchanges
    def local_search(selected_mask, time_limit=5.0):
        sel_mask = selected_mask
        # Calculate initial value
        best_val = 0.0
        for i in range(N):
            if (sel_mask >> i) & 1:
                best_val += recip[i]
        
        t_start = time.time()
        improved = True
        while improved and time.time() - t_start < time_limit:
            improved = False
            
            # List of selected vertices
            sel_list = [i for i in range(N) if (sel_mask >> i) & 1]
            
            # 1-for-1 Exchange: Remove v, add u
            for v in sel_list:
                # Candidates to add: not in sel_mask, not conflicting with (sel_mask - v)
                # Let's just iterate all u not in sel_mask
                # To speed up, we can precompute candidates, but N is small.
                # We need u such that u is not in sel_mask and u does not conflict with any w in sel_mask \ {v}
                # This is equivalent to: u not in sel_mask, and (conflict_mask[u] & (sel_mask ^ (1 << v))) == 0
                potential_u_mask = 0
                # We can iterate u from 0 to N-1
                for u in range(N):
                    if (sel_mask >> u) & 1:
                        continue
                    # Check conflict with rest of selection
                    if (conflict_mask[u] & (sel_mask ^ (1 << v))) == 0:
                        # Check if adding u and removing v improves value
                        delta = recip[u] - recip[v]
                        if delta > 1e-12:
                            # Apply move
                            sel_mask = (sel_mask ^ (1 << v)) | (1 << u)
                            best_val += delta
                            improved = True
                            break
                if improved:
                    break
            
            # 2-for-1 Exchange: Remove v1, v2, add u
            # This allows removing two low-value blockers to add one high-value vertex
            # Iterate over pairs of selected vertices
            # Optimization: only consider pairs where sum of reciprocals is small
            # And u where reciprocal is large
            # For now, simple loop. N is ~20-30 in typical solutions, so pairs are ~500-1000
            # If no improvement in 1-for-1, try 2-for-1
            if not improved:
                # Get list of selected again
                sel_list = [i for i in range(N) if (sel_mask >> i) & 1]
                # Sort by value ascending to try removing low-value ones first?
                # Or just iterate all pairs
                for i_idx in range(len(sel_list)):
                    if time.time() - t_start > time_limit:
                        break
                    v1 = sel_list[i_idx]
                    for j_idx in range(i_idx + 1, len(sel_list)):
                        v2 = sel_list[j_idx]
                        # Remove v1, v2
                        # Find u not in sel_mask, not conflicting with sel_mask - {v1, v2}
                        # And recip[u] > recip[v1] + recip[v2]
                        # To speed up, we can precompute max recip of valid u
                        # But let's just check candidates
                        # Candidates: u not in sel_mask
                        # Constraint: (conflict_mask[u] & (sel_mask ^ (1 << v1) ^ (1 << v2))) == 0
                        
                        # Optimization: Only check u if recip[u] is potentially better
                        # We can find the max recip[u] satisfying the constraint
                        # But for simplicity and given small N, we iterate.
                        # To make it faster, we can iterate u in descending order of recip
                        # and break on first valid one that improves.
                        
                        # Let's pre-sort u's by recip descending once outside?
                        # No, the constraint depends on v1, v2.
                        
                        # Iterate u in descending order of value
                        # To avoid sorting every time, we have a global sorted list
                        pass 

                # Better 2-for-1 implementation:
                # Pre-sort vertices by reciprocal descending
                sorted_by_recip = sorted(range(N), key=lambda x: -recip[x])
                
                for i_idx in range(len(sel_list)):
                    if time.time() - t_start > time_limit:
                        break
                    v1 = sel_list[i_idx]
                    for j_idx in range(i_idx + 1, len(sel_list)):
                        v2 = sel_list[j_idx]
                        base_mask = sel_mask ^ (1 << v1) ^ (1 << v2)
                        # Find best u not in sel_mask, not conflicting with base_mask
                        # such that recip[u] > recip[v1] + recip[v2]
                        
                        # We can iterate sorted_by_recip and pick the first valid u
                        found = False
                        for u in sorted_by_recip:
                            if (sel_mask >> u) & 1:
                                continue
                            if (conflict_mask[u] & base_mask) != 0:
                                continue
                            # Check if it improves
                            if recip[u] > recip[v1] + recip[v2] + 1e-12:
                                # Apply move
                                sel_mask = base_mask | (1 << u)
                                best_val += recip[u] - recip[v1] - recip[v2]
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if improved:
                        break
        
        return sel_mask, best_val

    # Exhaustive Polish: Ensure no single add/swap is missed
    def exhaustive_polish(selected_mask):
        sel_mask = selected_mask
        best_val = 0.0
        for i in range(N):
            if (sel_mask >> i) & 1:
                best_val += recip[i]
        
        # Try adding any vertex that doesn't conflict
        # Try swapping any vertex with any other
        # This is essentially local_search with a tighter tolerance or just one pass
        
        # 1. Try adding any valid u
        for u in range(N):
            if (sel_mask >> u) & 1:
                continue
            if (conflict_mask[u] & sel_mask) == 0:
                sel_mask |= (1 << u)
                best_val += recip[u]
        
        # 2. Try swapping any v with any u
        # This is 1-for-1
        sel_list = [i for i in range(N) if (sel_mask >> i) & 1]
        for v in sel_list:
            for u in range(N):
                if (sel_mask >> u) & 1:
                    continue
                if (conflict_mask[u] & (sel_mask ^ (1 << v))) == 0:
                    if recip[u] > recip[v] + 1e-12:
                        sel_mask = (sel_mask ^ (1 << v)) | (1 << u)
                        best_val += recip[u] - recip[v]
                        # Restart swap check? Or just continue?
                        # Since we want to be sure, we can break and restart the loop
                        # But for "polish", one pass is often enough if we iterate multiple times
                        break
        
        return sel_mask, best_val

    # Multi-start
    best_result_mask = 0
    best_score = 0.0

    # Start 1: Natural order
    sel1 = greedy(list(range(N)))
    sel1, val1 = local_search(sel1, time_limit=10.0)
    if val1 > best_score:
        best_score = val1
        best_result_mask = sel1

    # Start 2: Reverse order
    sel2 = greedy(list(range(N - 1, -1, -1)))
    sel2, val2 = local_search(sel2, time_limit=10.0)
    if val2 > best_score:
        best_score = val2
        best_result_mask = sel2

    # Start 3: Random orderings
    rng = random.Random(42)
    for _ in range(10):
        if time.time() - t0 > DEADLINE - 20:
            break
        order = list(range(N))
        rng.shuffle(order)
        sel = greedy(order)
        sel, val = local_search(sel, time_limit=5.0)
        if val > best_score:
            best_score = val
            best_result_mask = sel

    # Final Polish
    if time.time() - t0 < DEADLINE - 5:
        best_result_mask, best_score = local_search(best_result_mask, time_limit=10.0)
        # One more exhaustive polish to be safe
        best_result_mask, best_score = exhaustive_polish(best_result_mask)

    # Convert to list of actual numbers
    result = [nums[i] for i in range(N) if (best_result_mask >> i) & 1]
    return sorted(result) if result else [2]
# EVOLVE-BLOCK-END
