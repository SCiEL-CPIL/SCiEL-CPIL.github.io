"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def _has_4ap_with(S, x):
    """Check if adding x to set S creates any nonconstant 4-AP."""
    # A 4-AP: a, a+d, a+2d, a+3d with d>0
    # x could be in any of the 4 positions
    S_set = S
    # x as a (position 0): need x+d, x+2d, x+3d in S
    for d in range(1, 86):  # max d such that x+3d <= 256
        a = x
        if a + 3*d > 256:
            break
        if (a+d in S_set) and (a+2*d in S_set) and (a+3*d in S_set):
            return True
    # x as a+d (position 1): need a, a+2d, a+3d in S where a = x-d
    for d in range(1, 86):
        a = x - d
        if a < 1:
            break
        if a + 3*d > 256:
            continue
        if (a in S_set) and (a+2*d in S_set) and (a+3*d in S_set):
            return True
    # x as a+2d (position 2): need a, a+d, a+3d in S where a = x-2d
    for d in range(1, 43):
        a = x - 2*d
        if a < 1:
            break
        if a + 3*d > 256:
            continue
        if (a in S_set) and (a+d in S_set) and (a+3*d in S_set):
            return True
    # x as a+3d (position 3): need a, a+d, a+2d in S where a = x-3d
    for d in range(1, 86):
        a = x - 3*d
        if a < 1:
            break
        if (a in S_set) and (a+d in S_set) and (a+2*d in S_set):
            return True
    return False

def _count_3aps(S):
    """Count distinct 3-APs (a, a+d, a+2d) with d>0 in set S."""
    count = 0
    for a in S:
        for d in range(1, 129):
            if a + 2*d > 256:
                break
            if (a+d in S) and (a+2*d in S):
                count += 1
    return count

def _delta_add(S, x):
    """Compute change in 3-AP count when adding x to S. Returns -inf if 4-AP created."""
    if _has_4ap_with(S, x):
        return float('-inf')
    delta = 0
    # x as middle: a, x, x+(x-a) => a + 2*(x-a) = 2x-a
    # 3-AP (a, x, b) where b = 2x-a, a < x < b
    for a in S:
        if a < x:
            b = 2*x - a
            if b > x and b <= 256 and b in S:
                delta += 1
    # x as first: x, x+d, x+2d
    for d in range(1, 129):
        if x + 2*d > 256:
            break
        if (x+d in S) and (x+2*d in S):
            delta += 1
    # x as last: a, a+d, x where a+d = x-d, a = x-2d
    for d in range(1, 86):
        a = x - 2*d
        if a < 1:
            break
        if (a in S) and (a+d in S):
            delta += 1
    return delta

def _delta_remove(S, x):
    """Compute change in 3-AP count when removing x from S. (Negative of what was added.)"""
    delta = 0
    # x as middle
    for a in S:
        if a < x:
            b = 2*x - a
            if b > x and b <= 256 and b in S:
                delta += 1
    # x as first
    for d in range(1, 129):
        if x + 2*d > 256:
            break
        if (x+d in S) and (x+2*d in S):
            delta += 1
    # x as last
    for d in range(1, 86):
        a = x - 2*d
        if a < 1:
            break
        if (a in S) and (a+d in S):
                delta += 1
    return delta

def _greedy_initial():
    """Greedy construction: add elements in order if no 4-AP is created."""
    S = set()
    for x in range(1, 257):
        if not _has_4ap_with(S, x):
            S.add(x)
    return S

def _local_search(S, time_limit=75.0):
    """Local search via element swapping to maximize 3-AP count."""
    start_time = time.time()
    S = set(S)
    current_3aps = _count_3aps(S)
    all_elements = list(range(1, 257))
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        improved = False
        # Shuffle for diversity
        random.shuffle(all_elements)
        
        for x_out in all_elements:
            if x_out not in S:
                continue
            # Try removing x_out and adding some x_in
            delta_remove = _delta_remove(S, x_out)
            S.remove(x_out)
            
            for x_in in all_elements:
                if x_in in S:
                    continue
                if x_in == x_out:
                    continue
                delta_add = _delta_add(S, x_in)
                if delta_add == float('-inf'):
                    continue
                total_delta = delta_add - delta_remove
                if total_delta > 0:
                    S.add(x_in)
                    current_3aps += total_delta
                    improved = True
                    break
            if not improved:
                S.add(x_out)
        
        if not improved:
            # Try single additions (if possible)
            any_add = False
            for x in all_elements:
                if x not in S:
                    delta = _delta_add(S, x)
                    if delta > 0:
                        S.add(x)
                        current_3aps += delta
                        any_add = True
            if not any_add:
                break
    return S, current_3aps

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # Phase 1: Greedy initial construction
    S = _greedy_initial()
    current_3aps = _count_3aps(S)
    best_S = set(S)
    best_3aps = current_3aps
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return sorted(best_S)
    
    # Phase 2: Local search with swaps
    S2, aps2 = _local_search(S, time_limit=deadline - elapsed - 5.0)
    if aps2 > best_3aps:
        best_S = set(S2)
        best_3aps = aps2
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return sorted(best_S)
    
    # Phase 3: More local search iterations with different random seeds
    for seed in range(1, 10):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            break
        random.seed(seed)
        S3, aps3 = _local_search(best_S, time_limit=deadline - elapsed - 5.0)
        if aps3 > best_3aps:
            best_S = set(S3)
            best_3aps = aps3
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return sorted(best_S)
    
    # Final: try greedy with different orderings
    for perm_seed in range(1, 5):
        elapsed = time.time() - start_time
        if elapsed > deadline - 3.0:
            break
        random.seed(perm_seed)
        order = list(range(1, 257))
        random.shuffle(order)
        S4 = set()
        for x in order:
            if not _has_4ap_with(S4, x):
                S4.add(x)
        aps4 = _count_3aps(S4)
        if aps4 > best_3aps:
            best_S = S4
            best_3aps = aps4
            # Local search from this start
            S5, aps5 = _local_search(best_S, time_limit=deadline - time.time() - start_time - 3.0)
            if aps5 > best_3aps:
                best_S = set(S5)
                best_3aps = aps5
    
    return sorted(best_S)
# EVOLVE-BLOCK-END
