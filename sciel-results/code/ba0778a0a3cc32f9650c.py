import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 256
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1
    
    # conflict_mask[i] has bit j set if i and j conflict (i+j is square)
    conflict_mask = [0] * (N + 1)
    
    for a in range(1, N + 1):
        mask = 0
        for sq in squares:
            b = sq - a
            if 1 <= b <= N:
                mask |= (1 << b)
        conflict_mask[a] = mask
        
    # Pre-filter: nodes where 2a is a square are invalid
    valid_nodes = []
    valid_mask_all = 0
    for a in range(1, N + 1):
        if 2 * a not in squares:
            valid_nodes.append(a)
            valid_mask_all |= (1 << a)
            
    for a in range(1, N + 1):
        conflict_mask[a] &= valid_mask_all
    
    def popcount(x):
        return bin(x).count('1')
    
    def get_vertices(mask):
        verts = []
        temp = mask
        while temp:
            v = (temp & -temp).bit_length() - 1
            if v >= 1:
                verts.append(v)
            temp &= temp - 1
        return verts
    
    def greedy_min_degree_init(seed):
        """Greedy independent set using min-degree heuristic."""
        rng = random.Random(seed)
        remaining = set(valid_nodes)
        chosen = []
        chosen_mask = 0
        
        while remaining:
            min_deg = float('inf')
            candidates_at_min = []
            for a in remaining:
                deg = popcount(conflict_mask[a] & (1 << 0))  # placeholder
                # Compute degree properly
                deg = 0
                cm = conflict_mask[a]
                for b in remaining:
                    if b != a and (cm >> b) & 1:
                        deg += 1
                if deg < min_deg:
                    min_deg = deg
                    candidates_at_min = [a]
                elif deg == min_deg:
                    candidates_at_min.append(a)
            
            if candidates_at_min:
                min_node = rng.choice(candidates_at_min)
                chosen.append(min_node)
                chosen_mask |= (1 << min_node)
                remaining.remove(min_node)
                to_remove = [b for b in remaining if (conflict_mask[min_node] >> b) & 1]
                for b in to_remove:
                    remaining.discard(b)
        
        return chosen_mask, chosen
    
    def greedy_max_independent_set(candidate_mask):
        """Compute greedy max independent set size (lower bound)."""
        if candidate_mask == 0:
            return 0
        count = 0
        remaining = candidate_mask
        while remaining:
            verts = get_vertices(remaining)
            if not verts:
                break
            best_v = None
            best_deg = float('inf')
            for v in verts:
                deg = popcount(conflict_mask[v] & remaining)
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            count += 1
            remaining &= ~(1 << best_v)
            remaining &= ~conflict_mask[best_v]
        return count
    
    def greedy_upper_bound_tight(candidate_mask):
        """
        Tighter upper bound: use greedy coloring / clique cover approximation.
        We use a simple approach: repeatedly find a vertex, remove it and all neighbors,
        count iterations. This gives an upper bound on MIS size.
        """
        if candidate_mask == 0:
            return 0
        count = 0
        remaining = candidate_mask
        while remaining:
            verts = get_vertices(remaining)
            if not verts:
                break
            # Pick vertex with max degree to minimize remaining
            best_v = None
            max_deg = -1
            for v in verts:
                deg = popcount(conflict_mask[v] & remaining)
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
            count += 1
            remaining &= ~(1 << best_v)
            remaining &= ~conflict_mask[best_v]
        return count
    
    def local_search_improve(best_mask, best_size):
        """Try to add vertices to the current solution via local search."""
        current_mask = best_mask
        current_size = best_size
        improved = True
        while improved:
            improved = False
            # Try to add any vertex that doesn't conflict
            for a in valid_nodes:
                if (current_mask >> a) & 1:
                    continue
                # Check if a conflicts with any in current_mask
                if (conflict_mask[a] & current_mask) == 0:
                    current_mask |= (1 << a)
                    current_size += 1
                    improved = True
                    break
            # Try swap: remove one, add two
            if not improved:
                verts_in = get_vertices(current_mask)
                for a in verts_in:
                    trial_mask = current_mask & ~(1 << a)
                    # Try to add two vertices
                    added = 0
                    for b in valid_nodes:
                        if (trial_mask >> b) & 1:
                            continue
                        if (conflict_mask[b] & trial_mask) == 0:
                            trial_mask |= (1 << b)
                            added += 1
                            if added == 2:
                                break
                    if added == 2:
                        current_mask = trial_mask
                        current_size += 1
                        improved = True
                        break
        return current_mask, current_size
    
    def branch_and_bound_v2(candidate_mask, current_set_mask, current_size, best_ref):
        """
        Improved B&B with better pruning and vertex ordering.
        best_ref: list [best_size, best_mask]
        """
        if time.time() - start_time > deadline - 3:
            return
        
        if candidate_mask == 0:
            if current_size > best_ref[0]:
                best_ref[0] = current_size
                best_ref[1] = current_set_mask
            return
        
        # Quick upper bound
        ub = current_size + greedy_upper_bound_tight(candidate_mask)
        if ub <= best_ref[0]:
            return
        
        # If we can potentially beat best, continue
        verts = get_vertices(candidate_mask)
        if not verts:
            if current_size > best_ref[0]:
                best_ref[0] = current_size
                best_ref[1] = current_set_mask
            return
        
        # Choose pivot: vertex with max degree in candidate subgraph
        pivot = None
        max_deg = -1
        for v in verts:
            deg = popcount(conflict_mask[v] & candidate_mask)
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        if pivot is None:
            pivot = verts[0]
        
        # Branch 1: Include pivot (usually better to try inclusion first for larger sets)
        new_mask_include = candidate_mask & ~(1 << pivot) & ~conflict_mask[pivot]
        branch_and_bound_v2(new_mask_include, current_set_mask | (1 << pivot), current_size + 1, best_ref)
        
        # Branch 2: Exclude pivot
        new_mask_exclude = candidate_mask & ~(1 << pivot)
        branch_and_bound_v2(new_mask_exclude, current_set_mask, current_size, best_ref)
    
    # Get initial solution with multiple seeds
    best_size = 0
    best_mask = 0
    for seed in range(10):
        init_mask, init_list = greedy_min_degree_init(seed)
        if len(init_list) > best_size:
            best_size = len(init_list)
            best_mask = init_mask
    
    # Local search to improve
    best_mask, best_size = local_search_improve(best_mask, best_size)
    
    # B&B to try to find larger solution
    best_ref = [best_size, best_mask]
    
    if time.time() - start_time < deadline - 5:
        branch_and_bound_v2(valid_mask_all, 0, 0, best_ref)
    
    best_size = best_ref[0]
    best_mask = best_ref[1]
    
    # Final local search
    best_mask, best_size = local_search_improve(best_mask, best_size)
    
    # Extract the list from best_mask
    best_list = get_vertices(best_mask)
    
    # Final verification and cleanup
    result = set(best_list)
    to_remove = set()
    for a in result:
        for b in result:
            if a <= b:
                s = a + b
                r = math.isqrt(s)
                if r * r == s:
                    to_remove.add(b)
                    break
    result -= to_remove
    
    return list(result)
