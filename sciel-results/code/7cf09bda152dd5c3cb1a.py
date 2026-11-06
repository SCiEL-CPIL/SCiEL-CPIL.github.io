import math
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    n = 256
    max_sum = 2 * n  # 512

    # Perfect squares in [2, 512]
    squares = set()
    i = 1
    while i * i <= max_sum:
        if i * i >= 2:
            squares.add(i * i)
        i += 1

    # Diagonal constraint: 2a not in squares
    candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    num_cand = len(candidates)

    # Build conflict graph using bitmasks
    # conflict_mask[i] = bitmask of candidates j (j != i) such that candidates[i]+candidates[j] is a square
    conflict_mask = [0] * num_cand
    for i in range(num_cand):
        a = candidates[i]
        for j in range(i + 1, num_cand):
            b = candidates[j]
            if a + b in squares:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # Precompute degree for ordering
    degree = [bin(conflict_mask[i]).count('1') for i in range(num_cand)]

    # Greedy maximum clique on a set of vertices (represented as bitmask)
    # Returns size of the clique found
    def greedy_clique_size(vertices_mask):
        """Find a greedy clique in the conflict graph restricted to vertices_mask.
        Returns the size of the clique found."""
        if vertices_mask == 0:
            return 0
        
        clique_size = 0
        remaining = vertices_mask
        
        while remaining:
            # Pick vertex with highest degree in remaining (greedy heuristic)
            best_v = None
            best_deg = -1
            temp = remaining
            while temp:
                v = temp & (-temp)
                vi = v.bit_length() - 1
                deg = bin(conflict_mask[vi] & remaining).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                    if deg == 0:
                        break
                temp &= ~v
            
            if best_v is None:
                break
            
            vi = best_v.bit_length() - 1
            clique_size += 1
            # Next vertex must be adjacent to all current clique vertices
            # For simplicity in greedy clique, we just restrict to neighbors of the last added vertex
            # This is a valid greedy heuristic for finding *a* clique (not necessarily max)
            remaining &= conflict_mask[vi]
        
        return clique_size

    # Better greedy clique: at each step, intersect with neighbors of ALL clique vertices
    def greedy_clique_size_v2(vertices_mask):
        """Find a greedy clique. At each step, candidate set = intersection of neighbors of all clique members."""
        if vertices_mask == 0:
            return 0
        
        clique_size = 0
        # Common neighbors of all vertices in the clique
        # Start with all vertices
        candidates_for_next = vertices_mask
        
        while candidates_for_next:
            # Pick vertex with highest degree in candidates_for_next
            best_v = None
            best_deg = -1
            temp = candidates_for_next
            while temp:
                v = temp & (-temp)
                vi = v.bit_length() - 1
                deg = bin(conflict_mask[vi] & candidates_for_next).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                    if deg == 0:
                        break
                temp &= ~v
            
            if best_v is None:
                break
            
            vi = best_v.bit_length() - 1
            clique_size += 1
            # Intersect with neighbors of vi
            candidates_for_next &= conflict_mask[vi]
        
        return clique_size

    # Upper bound for maximum independent set in vertices_mask
    # Uses: if we find a clique of size k in the conflict graph induced by vertices_mask,
    # then at most floor(|vertices_mask| / k) vertices can be in an independent set.
    # Also use: |vertices_mask| itself as a trivial bound.
    def upper_bound(vertices_mask):
        if vertices_mask == 0:
            return 0
        
        num_vertices = bin(vertices_mask).count('1')
        
        # Trivial bound
        ub = num_vertices
        
        # Clique-based bound
        # Try a few different greedy orderings to find a good clique
        # For speed, just do one greedy clique
        clique_sz = greedy_clique_size_v2(vertices_mask)
        if clique_sz > 0:
            clique_ub = (num_vertices + clique_sz - 1) // clique_sz  # ceil division
            ub = min(ub, clique_ub)
        
        return ub

    # Branch and bound for maximum independent set
    best_size = [0]
    best_mask = [0]

    def branch_and_bound(vertices_mask, current_mask, current_size):
        if time.time() - start_time > deadline - 2.0:
            return
        
        if vertices_mask == 0:
            if current_size > best_size[0]:
                best_size[0] = current_size
                best_mask[0] = current_mask
            return
        
        # Upper bound
        ub = current_size + upper_bound(vertices_mask)
        if ub <= best_size[0]:
            return
        
        # Pick a vertex to branch on
        # Heuristic: pick vertex with highest degree in vertices_mask
        v = None
        best_deg = -1
        temp = vertices_mask
        while temp:
            vv = temp & (-temp)
            vi = vv.bit_length() - 1
            deg = bin(conflict_mask[vi] & vertices_mask).count('1')
            if deg > best_deg:
                best_deg = deg
                v = vv
                if deg == 0:
                    break
            temp &= ~vv
        
        if v is None:
            return
        
        vi = v.bit_length() - 1
        
        # Branch 1: include vi
        new_vertices = vertices_mask & ~(v | conflict_mask[vi])
        new_current = current_mask | v
        branch_and_bound(new_vertices, new_current, current_size + 1)
        
        # Branch 2: exclude vi
        new_vertices2 = vertices_mask & ~v
        branch_and_bound(new_vertices2, current_mask, current_size)

    # Greedy initial solution
    def greedy_solution():
        remaining = (1 << num_cand) - 1
        selected = 0
        while remaining:
            best_v = None
            best_deg = float('inf')
            temp = remaining
            while temp:
                v = temp & (-temp)
                vi = v.bit_length() - 1
                deg = bin(conflict_mask[vi] & remaining).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                    if deg == 0:
                        break
                temp &= ~v
            
            if best_v is None:
                break
            
            selected |= best_v
            vi = best_v.bit_length() - 1
            remaining &= ~(best_v | conflict_mask[vi])
        
        return selected

    # Local search
    def local_search(initial_mask):
        current_mask = initial_mask
        current_size = bin(current_mask).count('1')
        best_mask_local = current_mask
        best_size_local = current_size
        
        for iteration in range(500):
            if time.time() - start_time > deadline - 5.0:
                break
            
            improved = False
            
            # Try to add a vertex
            for i in range(num_cand):
                v = 1 << i
                if not (current_mask & v):
                    if not (conflict_mask[i] & current_mask):
                        current_mask |= v
                        current_size += 1
                        improved = True
                        if current_size > best_size_local:
                            best_size_local = current_size
                            best_mask_local = current_mask
                        break
            
            if improved:
                continue
            
            # Try swap: remove one, add multiple
            for i in range(num_cand):
                v = 1 << i
                if current_mask & v:
                    temp_mask = current_mask & ~v
                    temp_size = current_size - 1
                    for j in range(num_cand):
                        w = 1 << j
                        if not (temp_mask & w) and not (conflict_mask[j] & temp_mask):
                            temp_mask |= w
                            temp_size += 1
                    
                    if temp_size > current_size:
                        current_mask = temp_mask
                        current_size = temp_size
                        improved = True
                        if current_size > best_size_local:
                            best_size_local = current_size
                            best_mask_local = current_mask
                        break
            
            if not improved:
                break
        
        return best_mask_local, best_size_local

    # Get initial solution
    greedy_mask = greedy_solution()
    greedy_size = bin(greedy_mask).count('1')
    best_size[0] = greedy_size
    best_mask[0] = greedy_mask

    # Randomized restarts to improve initial solution
    for seed in range(100):
        if time.time() - start_time > deadline - 20.0:
            break
        random.seed(seed)
        remaining = (1 << num_cand) - 1
        selected = 0
        while remaining:
            min_deg = float('inf')
            min_nodes = []
            temp = remaining
            while temp:
                v = temp & (-temp)
                vi = v.bit_length() - 1
                deg = bin(conflict_mask[vi] & remaining).count('1')
                if deg < min_deg:
                    min_deg = deg
                    min_nodes = [v]
                elif deg == min_deg:
                    min_nodes.append(v)
                temp &= ~v
            
            if not min_nodes:
                break
            
            v = random.choice(min_nodes)
            selected |= v
            vi = v.bit_length() - 1
            remaining &= ~(v | conflict_mask[vi])
        
        sel_size = bin(selected).count('1')
        if sel_size > best_size[0]:
            best_size[0] = sel_size
            best_mask[0] = selected

    # Local search on best
    best_mask[0], best_size[0] = local_search(best_mask[0])

    # Run B&B with time limit
    remaining_time = deadline - 5.0 - (time.time() - start_time)
    if remaining_time > 3.0:
        all_nodes = (1 << num_cand) - 1
        branch_and_bound(all_nodes, 0, 0)

    # Final local search
    best_mask[0], final_size = local_search(best_mask[0])
    best_size[0] = final_size

    # Convert bitmask to set of values
    result_set = set()
    mask = best_mask[0]
    i = 0
    while mask:
        if mask & 1:
            result_set.add(candidates[i])
        mask >>= 1
        i += 1

    # Final verification
    for a in result_set:
        if 2 * a in squares:
            return []
    for i, a in enumerate(result_set):
        for b in result_set:
            if a < b and a + b in squares:
                return []

    if not result_set:
        return [1]

    return list(result_set)
