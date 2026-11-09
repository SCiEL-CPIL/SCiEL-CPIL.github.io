"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute: for each element x, the list of (other1, other2) pairs such that {x, other1, other2} forms a 3-AP
    # A 3-AP is (a, a+d, a+2d). For element x:
    #   x can be the first term: (x, x+d, x+2d) -> partners (x+d, x+2d)
    #   x can be the middle term: (x-d, x, x+d) -> partners (x-d, x+d)
    #   x can be the last term: (x-2d, x-d, x) -> partners (x-2d, x-d)
    elem_3ap_partners = {x: [] for x in range(1, N + 1)}
    for d in range(1, (N - 1) // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            a1, a2, a3 = a, a + d, a + 2 * d
            elem_3ap_partners[a1].append((a2, a3))
            elem_3ap_partners[a2].append((a1, a3))
            elem_3ap_partners[a3].append((a1, a2))
    
    # Precompute: for each element x, the list of 4-APs containing x
    # A 4-AP is (a, a+d, a+2d, a+3d). For element x:
    #   x can be position 0: (x, x+d, x+2d, x+3d) -> others (x+d, x+2d, x+3d)
    #   x can be position 1: (x-d, x, x+d, x+2d) -> others (x-d, x+d, x+2d)
    #   x can be position 2: (x-2d, x-d, x, x+d) -> others (x-2d, x-d, x+d)
    #   x can be position 3: (x-3d, x-2d, x-d, x) -> others (x-3d, x-2d, x-d)
    elem_4ap_others = {x: [] for x in range(1, N + 1)}
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            a1, a2, a3, a4 = a, a + d, a + 2 * d, a + 3 * d
            elem_4ap_others[a1].append((a2, a3, a4))
            elem_4ap_others[a2].append((a1, a3, a4))
            elem_4ap_others[a3].append((a1, a2, a4))
            elem_4ap_others[a4].append((a1, a2, a3))
    
    def count_3aps(selected_set):
        """Count distinct 3-APs in selected_set. O(|A| * avg_degree)."""
        count = 0
        for x in selected_set:
            for (p1, p2) in elem_3ap_partners[x]:
                # Only count each 3-AP once: require x < p1 < p2 (x is the smallest)
                if x < p1 and p1 < p2 and p1 in selected_set and p2 in selected_set:
                    count += 1
        return count
    
    def marginal_gain(selected_set, x):
        """Count new 3-APs created by adding x to selected_set."""
        gain = 0
        for (p1, p2) in elem_3ap_partners[x]:
            # x is the smallest element in the 3-AP
            if x < p1 and p1 < p2 and p1 in selected_set and p2 in selected_set:
                gain += 1
            # x is the middle element
            elif p1 < x < p2 and p1 in selected_set and p2 in selected_set:
                gain += 1
            # x is the largest element
            elif p1 < p2 < x and p1 in selected_set and p2 in selected_set:
                gain += 1
        return gain
    
    def can_add(selected_set, x):
        """Check if adding x to selected_set creates a 4-AP."""
        for (o1, o2, o3) in elem_4ap_others[x]:
            if o1 in selected_set and o2 in selected_set and o3 in selected_set:
                return False
        return True
    
    def local_search(selected_set, time_limit):
        """Hill-climbing local search with add/remove/swap moves."""
        current_set = set(selected_set)
        current_score = count_3aps(current_set)
        best_set = set(current_set)
        best_score = current_score
        
        while time.time() < time_limit:
            improved = False
            
            # Try all possible additions
            for x in range(1, N + 1):
                if x not in current_set:
                    if can_add(current_set, x):
                        new_set = current_set | {x}
                        new_score = current_score + marginal_gain(current_set, x)
                        if new_score > current_score:
                            current_set = new_set
                            current_score = new_score
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_set = set(current_set)
            
            # Try all possible removals (only if it doesn't decrease score)
            for x in list(current_set):
                # Compute loss from removing x
                loss = 0
                for (p1, p2) in elem_3ap_partners[x]:
                    if x < p1 and p1 < p2 and p1 in current_set and p2 in current_set:
                        loss += 1
                    elif p1 < x < p2 and p1 in current_set and p2 in current_set:
                        loss += 1
                    elif p1 < p2 < x and p1 in current_set and p2 in current_set:
                        loss += 1
                new_score = current_score - loss
                if new_score >= current_score:
                    current_set.discard(x)
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(current_set)
            
            # Try swaps: remove one, add another
            if not improved:
                swapped = False
                for x in list(current_set):
                    if swapped:
                        break
                    for y in range(1, N + 1):
                        if y not in current_set:
                            temp_set = current_set - {x}
                            if can_add(temp_set, y):
                                new_score = count_3aps(temp_set | {y})
                                if new_score > current_score:
                                    current_set = temp_set | {y}
                                    current_score = new_score
                                    improved = True
                                    swapped = True
                                    if current_score > best_score:
                                        best_score = current_score
                                        best_set = set(current_set)
                                    break
            
            if not improved:
                break
        
        return best_set, best_score
    
    def greedy_build(seed_offset=0):
        """Randomized greedy construction."""
        rng = random.Random(seed_offset)
        selected = set()
        elements = list(range(1, N + 1))
        rng.shuffle(elements)
        
        for x in elements:
            if can_add(selected, x):
                gain = marginal_gain(selected, x)
                if gain > 0 or len(selected) < 10:
                    selected.add(x)
        
        return selected
    
    # Multi-restart approach
    best_solution = set()
    best_score = 0
    num_restarts = 12
    
    # Allocate time per restart
    per_restart_time = (deadline - start_time - 3.0) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 3.0:
            break
        
        restart_start = time.time()
        restart_deadline = min(deadline - 3.0, restart_start + per_restart_time * 2.0)
        
        # Phase 1: Greedy construction
        initial_set = greedy_build(seed_offset=r * 1000 + 42)
        initial_score = count_3aps(initial_set)
        
        # Phase 2: Local search
        polished_set, polished_score = local_search(initial_set, restart_deadline)
        
        if polished_score > best_score:
            best_score = polished_score
            best_solution = polished_set
    
    # Final polish on the best solution
    if time.time() < deadline - 1.0 and len(best_solution) > 0:
        final_set, final_score = local_search(best_solution, deadline - 0.5)
        if final_score > best_score:
            best_score = final_score
            best_solution = final_set
    
    return sorted(best_solution)
# EVOLVE-BLOCK-END
