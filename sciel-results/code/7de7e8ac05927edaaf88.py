"""SA + greedy repair for 4-AP-free set maximizing 3-AP count."""
import time
import random
import math

N = 256
TIME_LIMIT = 85.0  # safety margin below 90s

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Precompute all 4-APs and 3-APs (0-indexed)
    four_aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(0, N - 3 * d):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    three_aps = []
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(0, N - 2 * d):
            three_aps.append((a, a + d, a + 2 * d))
    
    # For each element, which 4-APs does it participate in?
    elem_to_4aps = [[] for _ in range(N)]
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(i)
    
    # For each element, count how many 3-APs it participates in (weight)
    elem_3ap_count = [0] * N
    for ap in three_aps:
        for x in ap:
            elem_3ap_count[x] += 1
    
    def count_3aps(sel):
        """Count 3-APs in selected set (sel is a set of 0-indexed elements)."""
        sel_set = sel
        cnt = 0
        for ap in three_aps:
            if ap[0] in sel_set and ap[1] in sel_set and ap[2] in sel_set:
                cnt += 1
        return cnt
    
    def count_4ap_violations(sel):
        """Count 4-APs fully contained in sel."""
        sel_set = sel
        cnt = 0
        for ap in four_aps:
            if ap[0] in sel_set and ap[1] in sel_set and ap[2] in sel_set and ap[3] in sel_set:
                cnt += 1
        return cnt
    
    def greedy_construct():
        """Greedy: add elements in order of decreasing 3-AP weight, skip if creates 4-AP."""
        selected = set()
        # Sort by weight descending, break ties by index
        order = sorted(range(N), key=lambda x: (-elem_3ap_count[x], x))
        for x in order:
            # Check if adding x would complete any 4-AP
            conflict = False
            for ap_idx in elem_to_4aps[x]:
                ap = four_aps[ap_idx]
                others = [y for y in ap if y != x]
                if all(y in selected for y in others):
                    conflict = True
                    break
            if not conflict:
                selected.add(x)
        return selected
    
    def greedy_repair(selected):
        """Remove elements to eliminate all 4-APs, keeping as many as possible."""
        selected = set(selected)
        # Repeatedly find a 4-AP and remove the element with lowest weight
        changed = True
        while changed:
            changed = False
            for ap in four_aps:
                if all(x in selected for x in ap):
                    # Remove the element in this 4-AP with lowest 3-AP count
                    worst = min(ap, key=lambda x: elem_3ap_count[x])
                    selected.discard(worst)
                    changed = True
                    break
        return selected
    
    def local_search(selected, max_iters=100):
        """Hill-climbing: try adds, removes, swaps to improve 3-AP count while maintaining no 4-APs."""
        selected = set(selected)
        current_score = count_3aps(selected)
        
        for _ in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            
            improved = False
            
            # Try adding an element
            candidates = [x for x in range(N) if x not in selected]
            rng.shuffle(candidates)
            for x in candidates:
                # Check if adding x creates a 4-AP
                conflict = False
                for ap_idx in elem_to_4aps[x]:
                    ap = four_aps[ap_idx]
                    others = [y for y in ap if y != x]
                    if all(y in selected for y in others):
                        conflict = True
                        break
                if conflict:
                    continue
                
                new_sel = selected | {x}
                new_score = count_3aps(new_sel)
                if new_score > current_score:
                    selected = new_sel
                    current_score = new_score
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try removing an element
            sel_list = list(selected)
            rng.shuffle(sel_list)
            for x in sel_list:
                new_sel = selected - {x}
                new_score = count_3aps(new_sel)
                if new_score > current_score:
                    selected = new_sel
                    current_score = new_score
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one, add another
            sel_list = list(selected)
            rng.shuffle(sel_list)
            for x_remove in sel_list:
                if time.time() - start_time > TIME_LIMIT - 3:
                    return selected, current_score
                candidates = [x for x in range(N) if x not in selected]
                rng.shuffle(candidates)
                for x_add in candidates:
                    # Check if x_add creates 4-AP after removing x_remove
                    new_sel = (selected - {x_remove}) | {x_add}
                    # Check 4-AP violations
                    has_4ap = False
                    for ap_idx in elem_to_4aps[x_add]:
                        ap = four_aps[ap_idx]
                        others = [y for y in ap if y != x_add]
                        if all(y in new_sel for y in others):
                            has_4ap = True
                            break
                    if has_4ap:
                        continue
                    
                    new_score = count_3aps(new_sel)
                    if new_score > current_score:
                        selected = new_sel
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return selected, current_score
    
    # Phase 1: Greedy construction
    best_solution = greedy_construct()
    best_score = count_3aps(best_solution)
    
    # Phase 2: Local search on greedy solution
    best_solution, best_score = local_search(best_solution, max_iters=200)
    
    # Phase 3: Simulated Annealing with toggle moves
    # SA can temporarily allow 4-AP violations (penalized) to escape local optima
    current_solution = set(best_solution)
    current_score = best_score
    
    T_start = 10.0
    T_end = 0.01
    alpha = 0.97  # cooling rate
    T = T_start
    
    sa_iter = 0
    max_sa_iters = 5000
    
    while T > T_end and sa_iter < max_sa_iters:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        sa_iter += 1
        
        # Generate a toggle move: add, remove, or swap
        move_type = rng.random()
        new_solution = set(current_solution)
        
        if move_type < 0.4:
            # Add an element
            x = rng.randint(0, N - 1)
            if x not in new_solution:
                new_solution.add(x)
        elif move_type < 0.7:
            # Remove an element
            if current_solution:
                x = rng.choice(list(current_solution))
                new_solution.discard(x)
        else:
            # Swap: remove one, add another
            if current_solution:
                x_remove = rng.choice(list(current_solution))
                x_add = rng.randint(0, N - 1)
                if x_add not in current_solution:
                    new_solution.discard(x_remove)
                    new_solution.add(x_add)
        
        # Compute new score with penalty for 4-APs
        new_3ap_count = count_3aps(new_solution)
        new_4ap_count = count_4ap_violations(new_solution)
        new_score = new_3ap_count - 100 * new_4ap_count
        
        old_4ap_count = count_4ap_violations(current_solution)
        old_score = current_3ap_count - 100 * old_4ap_count if 'current_3ap_count' in dir() else current_score - 100 * old_4ap_count
        # Actually let's track properly
        old_3ap = count_3aps(current_solution)
        old_score_val = old_3ap - 100 * old_4ap_count
        
        delta = new_score - old_score_val
        
        if delta >= 0 or rng.random() < math.exp(delta / T):
            current_solution = new_solution
            current_3ap_count = new_3ap_count
            current_score_val = new_score
        
        # Periodic greedy repair every 100 iterations to maintain feasibility
        if sa_iter % 100 == 0:
            current_solution = greedy_repair(current_solution)
            current_3ap_count = count_3aps(current_solution)
            current_score_val = current_3ap_count  # no 4-APs after repair
        
        # Track best feasible solution
        if count_4ap_violations(current_solution) == 0:
            feasible_score = count_3aps(current_solution)
            if feasible_score > best_score:
                best_score = feasible_score
                best_solution = set(current_solution)
        
        T *= alpha
    
    # Final local search on best solution
    if time.time() - start_time < TIME_LIMIT - 3:
        best_solution, best_score = local_search(best_solution, max_iters=200)
    
    # Verify no 4-APs
    assert count_4ap_violations(best_solution) == 0, "4-AP violation found!"
    
    return sorted(x + 1 for x in best_solution)
