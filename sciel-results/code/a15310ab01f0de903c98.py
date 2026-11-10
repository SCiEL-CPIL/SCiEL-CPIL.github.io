"""Weighted greedy + local search for max 3-APs without 4-APs in [1,256]."""
import time
import random

N = 256
DEADLINE = 85.0  # safety margin before 90s kill

def _count_3aps(A_set):
    """Count distinct 3-APs (positive difference) in A_set."""
    count = 0
    for a in A_set:
        for c in A_set:
            if c > a and (c - a) % 2 == 0:
                b = (a + c) // 2
                if b in A_set:
                    count += 1
    return count

def _has_4ap(A_set):
    """Check if A_set contains a nonconstant 4-AP."""
    for a in A_set:
        for d in range(1, (N - a) // 3 + 1):
            if a + d in A_set and a + 2*d in A_set and a + 3*d in A_set:
                return True
    return False

def _build_4ap_conflicts():
    """Precompute for each element x, the list of (a, d) 4-APs containing x."""
    # For each 4-AP (a, a+d, a+2d, a+3d), all four elements conflict.
    # We'll store for each x: list of 4-AP tuples it participates in.
    conflicts = {x: [] for x in range(1, N+1)}
    for a in range(1, N+1):
        for d in range(1, (N - 3*d) // 1 + 1):
            if a + 3*d > N:
                break
            ap = (a, a+d, a+2*d, a+3*d)
            for x in ap:
                conflicts[x].append(ap)
    return conflicts

def _weighted_greedy():
    """
    Greedy construction: at each step, pick the element that maximizes
    (3-APs gained) - (4-APs created), while maintaining 4-AP-free property.
    """
    selected = set()
    conflicts = _build_4ap_conflicts()
    
    # Precompute 3-AP participation: for each x, which pairs (a,c) with x as middle
    # 3-AP: a, x, c where x = (a+c)/2, so a = x-d, c = x+d
    # x as middle: pairs (x-d, x+d) for valid d
    # x as first: (x, x+d, x+2d)
    # x as last: (x-2d, x-d, x)
    
    remaining = set(range(1, N+1))
    
    while remaining:
        best_x = None
        best_score = -1
        for x in remaining:
            # Check if adding x creates any 4-AP
            creates_4ap = False
            for ap in conflicts[x]:
                if all(y in selected for y in ap if y != x):
                    creates_4ap = True
                    break
            if creates_4ap:
                continue
            
            # Count 3-APs gained by adding x
            # x as middle: (x-d, x+d) both in selected
            gain = 0
            for d in range(1, min(x-1, N-x) + 1):
                if (x-d) in selected and (x+d) in selected:
                    gain += 1
            # x as first: (x, x+d, x+2d) both in selected
            for d in range(1, (N - x)//2 + 1):
                if (x+d) in selected and (x+2*d) in selected:
                    gain += 1
            # x as last: (x-2d, x-d, x) both in selected
            for d in range(1, x//2 + 1):
                if (x-2*d) in selected and (x-d) in selected:
                    gain += 1
            
            if gain > best_score:
                best_score = gain
                best_x = x
        
        if best_x is None:
            break
        selected.add(best_x)
        remaining.discard(best_x)
    
    return selected

def _local_search(A_set, time_limit=30.0):
    """
    Local search: try single additions, removals, and swaps.
    Accept moves that increase 3-AP count while maintaining 4-AP-free.
    """
    current = set(A_set)
    current_3aps = _count_3aps(current)
    best = set(current)
    best_3aps = current_3aps
    start_time = time.time()
    
    # Precompute conflicts
    conflicts = _build_4ap_conflicts()
    
    def _check_4ap_free_with_add(s, x):
        for ap in conflicts[x]:
            if all(y in s for y in ap if y != x):
                return False
        return True
    
    def _check_4ap_free_after_swap(s, remove_x, add_x):
        # After removing remove_x and adding add_x
        # Check add_x doesn't form 4-AP with remaining elements
        temp = s - {remove_x} | {add_x}
        for ap in conflicts[add_x]:
            if all(y in temp for y in ap if y != add_x):
                return False
        return True
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try additions
        for x in range(1, N+1):
            if x not in current:
                if _check_4ap_free_with_add(current, x):
                    new_set = current | {x}
                    # Count 3-APs gained
                    gain = 0
                    for d in range(1, min(x-1, N-x) + 1):
                        if (x-d) in current and (x+d) in current:
                            gain += 1
                    for d in range(1, (N - x)//2 + 1):
                        if (x+d) in current and (x+2*d) in current:
                            gain += 1
                    for d in range(1, x//2 + 1):
                        if (x-2*d) in current and (x-d) in current:
                            gain += 1
                    if gain > 0:
                        current = new_set
                        current_3aps += gain
                        improved = True
                        if current_3aps > best_3aps:
                            best_3aps = current_3aps
                            best = set(current)
                        break
        
        if improved:
            no_improve_count = 0
            continue
        
        # Try removals that allow better additions (simplified: just try removing low-value elements)
        # Try swaps
        for remove_x in list(current):
            # Find elements that would be "freed"
            for add_x in range(1, N+1):
                if add_x not in current:
                    if _check_4ap_free_after_swap(current, remove_x, add_x):
                        new_set = (current - {remove_x}) | {add_x}
                        new_3aps = _count_3aps(new_set)
                        if new_3aps > current_3aps:
                            current = new_set
                            current_3aps = new_3aps
                            improved = True
                            if current_3aps > best_3aps:
                                best_3aps = current_3aps
                                best = set(current)
                            break
            if improved:
                no_improve_count = 0
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    return best, best_3aps

def _grasp_tabu(num_iters=50, time_limit=40.0):
    """
    GRASP with Tabu: multiple greedy constructions with different orderings,
    followed by local search.
    """
    start_time = time.time()
    best_set = set()
    best_3aps = 0
    
    for it in range(num_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Diversified greedy: randomize the order of element consideration
        elements = list(range(1, N+1))
        random.shuffle(elements)
        
        selected = set()
        conflicts = _build_4ap_conflicts()
        
        for x in elements:
            # Check 4-AP constraint
            creates_4ap = False
            for ap in conflicts[x]:
                if all(y in selected for y in ap if y != x):
                    creates_4ap = True
                    break
            if creates_4ap:
                continue
            
            # Count 3-APs gained
            gain = 0
            for d in range(1, min(x-1, N-x) + 1):
                if (x-d) in selected and (x+d) in selected:
                    gain += 1
            for d in range(1, (N - x)//2 + 1):
                if (x+d) in selected and (x+2*d) in selected:
                    gain += 1
            for d in range(1, x//2 + 1):
                if (x-2*d) in selected and (x-d) in selected:
                    gain += 1
            
            # Accept with probability based on gain (simulated annealing style)
            if gain > 0 or random.random() < 0.3:
                selected.add(x)
        
        # Local search refinement
        refined, refined_3aps = _local_search(selected, time_limit=5.0)
        if refined_3aps > best_3aps:
            best_3aps = refined_3aps
            best_set = refined
    
    return best_set, best_3aps

def construct():
    global DEADLINE
    start_time = time.time()
    
    # Phase 1: Weighted greedy
    greedy_set = _weighted_greedy()
    greedy_3aps = _count_3aps(greedy_set)
    
    # Verify no 4-AP
    assert not _has_4ap(greedy_set), "Greedy produced a 4-AP!"
    
    best_set = greedy_set
    best_3aps = greedy_3aps
    
    # Phase 2: Local search on greedy
    if time.time() - start_time < DEADLINE - 10:
        refined, refined_3aps = _local_search(greedy_set, time_limit=min(20, DEADLINE - (time.time()-start_time) - 5))
        if refined_3aps > best_3aps:
            best_3aps = refined_3aps
            best_set = refined
    
    # Phase 3: GRASP with Tabu
    remaining_time = DEADLINE - (time.time() - start_time) - 3
    if remaining_time > 5:
        grasp_set, grasp_3aps = _grasp_tabu(num_iters=30, time_limit=remaining_time)
        if grasp_3aps > best_3aps:
            best_3aps = grasp_3aps
            best_set = grasp_set
    
    # Final safety check
    if time.time() - start_time > DEADLINE:
        # Return what we have
        pass
    
    assert not _has_4ap(best_set), "Final set has a 4-AP!"
    assert len(best_set) > 0, "Empty set!"
    
    return sorted(best_set)
