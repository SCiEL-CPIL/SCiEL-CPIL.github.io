"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _try_add(x, chosen, counts):
    """
    Check if x can be added to chosen without violating B2[2].
    If yes, update counts and return True.
    If no, return False.
    """
    # Check x + x
    s = 2 * x
    if counts.get(s, 0) >= 2:
        return False
    
    # Check x + y for all y in chosen
    # We need to ensure that for all y, counts[x+y] < 2
    for y in chosen:
        s = x + y
        if counts.get(s, 0) >= 2:
            return False
    
    # If we get here, it's safe to add
    # Update counts
    counts[s] = counts.get(s, 0) + 1  # Wait, s is from the last loop iteration. Need to reset.
    
    # Correct update:
    counts[2 * x] = counts.get(2 * x, 0) + 1
    for y in chosen:
        s = x + y
        counts[s] = counts.get(s, 0) + 1
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    N = 1000
    best_result = []
    best_size = 0
    
    # Helper to run a greedy pass
    def run_greedy(order):
        chosen = []
        counts = defaultdict(int)
        for x in order:
            if time.time() > deadline:
                break
            # Check validity
            s_xx = 2 * x
            if counts.get(s_xx, 0) >= 2:
                continue
            
            valid = True
            for y in chosen:
                s_xy = x + y
                if counts.get(s_xy, 0) >= 2:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Add x
            chosen.append(x)
            counts[s_xx] = counts.get(s_xx, 0) + 1
            for y in chosen:
                if y == x:
                    continue
                s_xy = x + y
                counts[s_xy] = counts.get(s_xy, 0) + 1
        
        return chosen

    # Strategy 1: Increasing order
    order1 = list(range(1, N + 1))
    res1 = run_greedy(order1)
    if len(res1) > best_size:
        best_size = len(res1)
        best_result = res1

    # Strategy 2: Decreasing order
    order2 = list(range(N, 0, -1))
    res2 = run_greedy(order2)
    if len(res2) > best_size:
        best_size = len(res2)
        best_result = res2

    # Strategy 3: Random
    order3 = list(range(1, N + 1))
    random.shuffle(order3)
    res3 = run_greedy(order3)
    if len(res3) > best_size:
        best_size = len(res3)
        best_result = res3

    # Strategy 4: Random with bias towards small numbers (often better for packing?)
    # Or bias towards large? Let's try a few random restarts.
    
    # Run multiple random restarts with time budget
    remaining_time = deadline - time.time()
    max_restarts = 10
    time_per_restart = remaining_time / max_restarts if max_restarts > 0 else 0
    
    for i in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random order
        order_r = list(range(1, N + 1))
        random.shuffle(order_r)
        
        # Optional: bias
        # For now, pure random is a good baseline.
        
        res_r = run_greedy(order_r)
        if len(res_r) > best_size:
            best_size = len(res_r)
            best_result = res_r

    # Local improvement: try to fill gaps
    if best_result and time.time() < deadline - 10:
        # Rebuild counts for the best result
        counts_best = defaultdict(int)
        n = len(best_result)
        for i in range(n):
            for j in range(i, n):
                s = best_result[i] + best_result[j]
                counts_best[s] += 1
        
        used = set(best_result)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try adding any missing number
            for x in range(1, N + 1):
                if x in used:
                    continue
                
                # Check if x can be added
                s_xx = 2 * x
                if counts_best.get(s_xx, 0) >= 2:
                    continue
                
                valid = True
                for y in best_result:
                    s_xy = x + y
                    if counts_best.get(s_xy, 0) >= 2:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Add x
                best_result.append(x)
                used.add(x)
                counts_best[s_xx] = counts_best.get(s_xx, 0) + 1
                for y in best_result:
                    if y == x:
                        continue
                    s_xy = x + y
                    counts_best[s_xy] = counts_best.get(s_xy, 0) + 1
                
                improved = True
                best_size = len(best_result)

    return best_result
# EVOLVE-BLOCK-END
