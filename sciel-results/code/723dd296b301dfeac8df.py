import time
import random
import sys

def construct():
    t0 = time.time()
    deadline = t0 + 55  # safety margin
    
    N = 256
    all_mask = (1 << N) - 1
    
    # Precompute conflict masks (0-indexed: bit i corresponds to number i+1)
    conflict_mask = [0] * N
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if (a * b) % (a + b) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    # Helper functions
    def popcount(x):
        return bin(x).count('1')
    
    def mask_to_list(mask):
        return [i + 1 for i in range(N) if (mask >> i) & 1]
    
    # Phase 1: Greedy construction - start with odds, add compatible evens
    S = 0
    for i in range(N):
        if (i + 1) % 2 == 1:
            S |= (1 << i)
    
    # Greedy add: try to add remaining numbers
    improved = True
    while improved:
        improved = False
        for v in range(N):
            if (S >> v) & 1:
                continue
            if (conflict_mask[v] & S) == 0:
                S |= (1 << v)
                improved = True
    
    best_size = popcount(S)
    best_mask = S
    
    # Phase 2: Local search with 1-for-2 moves
    def local_search_1for2(mask, time_limit):
        nonlocal best_size, best_mask
        cur = mask
        cur_size = popcount(cur)
        
        while time.time() < time_limit:
            improved = False
            S_list = mask_to_list(cur)
            random.shuffle(S_list)
            
            for u in S_list:
                if time.time() > time_limit:
                    break
                u_idx = u - 1
                S_without_u = cur & ~(1 << u_idx)
                
                # Find candidates not in cur compatible with S_without_u
                cands = []
                for v in range(N):
                    if (cur >> v) & 1:
                        continue
                    if (conflict_mask[v] & S_without_u) == 0:
                        cands.append(v)
                
                if len(cands) >= 2:
                    c_len = len(cands)
                    for i in range(c_len):
                        v1 = cands[i]
                        for j in range(i + 1, c_len):
                            v2 = cands[j]
                            if (conflict_mask[v1] & (1 << v2)) == 0:
                                new_mask = S_without_u | (1 << v1) | (1 << v2)
                                new_size = cur_size + 1
                                if new_size > best_size:
                                    best_size = new_size
                                    best_mask = new_mask
                                cur = new_mask
                                cur_size = new_size
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                # Try a random swap to escape local minimum
                S_list = mask_to_list(cur)
                if not S_list:
                    break
                u = random.choice(S_list)
                u_idx = u - 1
                S_without_u = cur & ~(1 << u_idx)
                
                candidates_swap = []
                for v in range(N):
                    if (cur >> v) & 1:
                        continue
                    if (conflict_mask[v] & S_without_u) == 0:
                        candidates_swap.append(v)
                
                if candidates_swap:
                    v = random.choice(candidates_swap)
                    cur = S_without_u | (1 << v)
                    # size unchanged, continue searching
                else:
                    break
        
        return cur
    
    local_search_1for2(S, deadline - 20)
    
    # Phase 3: Branch and Bound with greedy coloring (Tomita-style)
    # We'll use the B&B to try to prove optimality or find larger sets
    
    def greedy_color_bound(candidates_mask):
        """Compute greedy coloring upper bound for the induced subgraph on candidates_mask."""
        if candidates_mask == 0:
            return 0
        
        # Get list of candidate indices
        cands = []
        cm = candidates_mask
        while cm:
            lsb = cm & (-cm)
            idx = lsb.bit_length() - 1
            cands.append(idx)
            cm ^= lsb
        
        n_cands = len(cands)
        if n_cands == 0:
            return 0
        
        # Greedy coloring: assign colors to vertices
        # color[i] = color class index for candidate i
        color = [-1] * n_cands
        color_classes = []  # list of lists of candidate indices per color
        
        for i in range(n_cands):
            ci = cands[i]
            # Find a color that doesn't conflict with already-colored neighbors in this color class
            placed = False
            for c in range(len(color_classes)):
                # Check if ci conflicts with any vertex already in color_classes[c]
                compatible = True
                for j in color_classes[c]:
                    cj = cands[j]
                    if (conflict_mask[ci] >> cj) & 1:
                        compatible = False
                        break
                if compatible:
                    color_classes[c].append(i)
                    color[i] = c
                    placed = True
                    break
            if not placed:
                color_classes.append([i])
                color[i] = len(color_classes) - 1
        
        return len(color_classes)
    
    def branch_and_bound(candidates_mask, current_size, current_set, time_limit):
        """B&B to find maximum independent set within candidates_mask."""
        nonlocal best_size, best_mask
        
        if time.time() > time_limit:
            return current_size, current_set
        
        if candidates_mask == 0:
            return current_size, current_set
        
        # Upper bound: current_size + number of candidates (optimistic)
        # Better bound: current_size + greedy_color_bound(candidates_mask)
        ub = current_size + greedy_color_bound(candidates_mask)
        if ub <= best_size:
            return current_size, current_set
        
        # Select a vertex to branch on
        # Choose vertex with highest degree in the candidate subgraph
        cm = candidates_mask
        best_v = -1
        best_deg = -1
        while cm:
            lsb = cm & (-cm)
            idx = lsb.bit_length() - 1
            cm ^= lsb
            deg = popcount(conflict_mask[idx] & candidates_mask)
            if deg > best_deg:
                best_deg = deg
                best_v = idx
        
        if best_v == -1:
            return current_size, current_set
        
        # Branch 1: Include best_v
        new_set = current_set | (1 << best_v)
        # Remove best_v and its neighbors from candidates
        new_cands = candidates_mask & ~(1 << best_v) & ~conflict_mask[best_v]
        sz1, set1 = branch_and_bound(new_cands, current_size + 1, new_set, time_limit)
        
        # Branch 2: Exclude best_v
        new_cands2 = candidates_mask & ~(1 << best_v)
        sz2, set2 = branch_and_bound(new_cands2, current_size, current_set, time_limit)
        
        if sz1 >= sz2:
            return sz1, set1
        else:
            return sz2, set2
    
    # Run B&B with the full candidate set
    # Start with empty set and all vertices as candidates
    # But we already have a good solution, so use it as a lower bound
    
    # Try B&B from scratch with time limit
    if time.time() < deadline - 5:
        # Use a simplified B&B that works on the full graph
        # We'll try multiple times with different vertex orderings
        
        # First, let's try a direct B&B approach
        # We need to be careful about recursion depth - use iterative approach
        
        def bb_iterative(time_limit):
            nonlocal best_size, best_mask
            
            # We'll use a stack-based approach
            # State: (candidates_mask, current_size, current_set)
            # Start with all vertices as candidates, empty set
            
            # But this could be too large. Let's use the existing solution as a starting point.
            # Actually, for B&B to work well, we need a good upper bound.
            # Let's try a different approach: use the B&B on a reduced problem.
            
            # For now, let's try a simple recursive B&B with depth limit
            def bb_recursive(cands, cur_size, cur_set, depth, time_limit):
                nonlocal best_size, best_mask
                
                if time.time() > time_limit:
                    return cur_size, cur_set
                
                if depth > 30 or cands == 0:
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur_set
                    return cur_size, cur_set
                
                # Upper bound
                ub = cur_size + popcount(cands)
                if ub <= best_size:
                    return cur_size, cur_set
                
                # Greedy coloring bound
                color_ub = greedy_color_bound(cands)
                if cur_size + color_ub <= best_size:
                    return cur_size, cur_set
                
                # Select vertex
                cm = cands
                best_v = -1
                best_deg = -1
                while cm:
                    lsb = cm & (-cm)
                    idx = lsb.bit_length() - 1
                    cm ^= lsb
                    deg = popcount(conflict_mask[idx] & cands)
                    if deg > best_deg:
                        best_deg = deg
                        best_v = idx
                
                if best_v == -1:
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur_set
                    return cur_size, cur_set
                
                # Branch: include best_v
                new_set = cur_set | (1 << best_v)
                new_cands = cands & ~(1 << best_v) & ~conflict_mask[best_v]
                sz1, set1 = bb_recursive(new_cands, cur_size + 1, new_set, depth + 1, time_limit)
                
                # Branch: exclude best_v
                new_cands2 = cands & ~(1 << best_v)
                sz2, set2 = bb_recursive(new_cands2, cur_size, cur_set, depth + 1, time_limit)
                
                if sz1 >= sz2:
                    return sz1, set1
                else:
                    return sz2, set2
            
            # Run B&B with full candidate set
            bb_recursive(all_mask, 0, 0, 0, time_limit)
            
            return best_size, best_mask
        
        if time.time() < deadline - 5:
            bb_iterative(deadline - 2)
    
    # Final: return the best solution found
    result = mask_to_list(best_mask)
    
    # Ensure nonempty
    if not result:
        result = [1]
    
    return result

if __name__ == "__main__":
    print(construct())
