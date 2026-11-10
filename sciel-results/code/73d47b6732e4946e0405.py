"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

def construct():
    n = 256
    
    # Precompute all 4-term APs
    four_aps = []
    for d in range(1, (n-1)//3 + 1):
        for a in range(1, n - 3*d + 1):
            four_aps.append((a, a+d, a+2*d, a+3*d))
    
    # Map vertex to list of indices of 4-APs it belongs to
    vertex_to_aps = {i: [] for i in range(1, n+1)}
    for idx, ap in enumerate(four_aps):
        for v in ap:
            vertex_to_aps[v].append(idx)
    
    # Weight of vertex: number of 3-APs it participates in
    weights = {}
    for x in range(1, n+1):
        w = 0
        w += (n - x) // 2
        w += min(x - 1, n - x)
        w += (x - 1) // 2
        weights[x] = w
        
    # Sort vertices by weight descending for better pruning
    sorted_vertices = sorted(range(1, n+1), key=lambda v: -weights[v])
    
    # Initial greedy solution to set a strong lower bound
    selected_greedy = set()
    for v in sorted_vertices:
        can_add = True
        for ap_idx in vertex_to_aps[v]:
            ap = four_aps[ap_idx]
            others = [u for u in ap if u != v]
            if all(u in selected_greedy for u in others):
                can_add = False
                break
        if can_add:
            selected_greedy.add(v)
            
    # Objective is sum of weights (which is 3 * number of 3-APs)
    best_weight = sum(weights[v] for v in selected_greedy)
    best_set = list(selected_greedy)
    
    # B&B State
    def branch_and_bound(idx, current_set, forbidden, current_weight):
        nonlocal best_weight, best_set
        
        if idx == n:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = list(current_set)
            return
        
        # Calculate Upper Bound
        # Sum of weights of remaining vertices not in forbidden
        ub = current_weight
        for i in range(idx, n):
            v = sorted_vertices[i]
            if v not in forbidden:
                ub += weights[v]
                
        if ub <= best_weight:
            return
            
        v = sorted_vertices[idx]
        
        # Option 1: Exclude v
        branch_and_bound(idx + 1, current_set, forbidden, current_weight)
        
        # Option 2: Include v
        if v not in forbidden:
            # Check if adding v is valid (no 4-AP formed with existing 3)
            valid = True
            new_forbidden = set(forbidden)
            
            for ap_idx in vertex_to_aps[v]:
                ap = four_aps[ap_idx]
                others = [u for u in ap if u != v]
                count_in_set = 0
                remaining_others = []
                for u in others:
                    if u in current_set:
                        count_in_set += 1
                    else:
                        remaining_others.append(u)
                        
                if count_in_set == 3:
                    valid = False
                    break
                elif count_in_set == 2:
                    # The remaining one(s) must be forbidden
                    new_forbidden.update(remaining_others)
            
            if valid:
                new_set = current_set | {v}
                branch_and_bound(idx + 1, new_set, new_forbidden, current_weight + weights[v])
                
    branch_and_bound(0, set(), set(), 0)
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
