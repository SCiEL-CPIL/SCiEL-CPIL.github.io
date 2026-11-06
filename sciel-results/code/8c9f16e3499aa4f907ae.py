import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s deadline for safety
    
    n = 18
    all_vertices_mask = (1 << n) - 1
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute triangle masks: for each pair (u,v), mask of w such that u-v-w is a triangle
    # If (u,v) is an edge, then any common neighbor w creates a triangle.
    # So for each edge (u,v), the "forbidden" set for triangle-free is the common neighbors.
    # But we need to check if the induced subgraph is triangle-free.
    # A set S is triangle-free iff for every edge (u,v) in S, u and v have no common neighbor in S.
    
    # Precompute: for each pair (u,v), the mask of common neighbors (excluding u,v)
    # This depends on the graph, so we can't precompute it once.
    # But we can precompute the structure.
    
    def count_bits(x):
        return bin(x).count('1')
    
    def is_k4_free(adj_mask):
        """Check if graph is K4-free."""
        for combo in itertools.combinations(range(n), 4):
            u, v, w, x = combo
            if (adj_mask[u] & (1 << v)) and (adj_mask[u] & (1 << w)) and (adj_mask[u] & (1 << x)) and \
               (adj_mask[v] & (1 << w)) and (adj_mask[v] & (1 << x)) and (adj_mask[w] & (1 << x)):
                return False
        return True
    
    def can_add_edge(adj_mask, u, v):
        """Check if adding edge (u,v) keeps graph K4-free."""
        if adj_mask[u] & (1 << v):
            return False
        # Adding (u,v) creates K4 if there exists an edge (w,x) where w,x are common neighbors of u and v
        common = adj_mask[u] & adj_mask[v] & ~((1 << u) | (1 << v))
        # Check if any pair in common has an edge
        temp = common
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            # For each w in common, check neighbors of w in common
            if adj_mask[w] & common:
                return False
            temp -= lsb
        return True
    
    def largest_triangle_free(adj_mask):
        """Find largest triangle-free induced subgraph using branch and bound."""
        best = 0
        
        # Precompute for this graph: for each vertex, its adjacency mask
        # For triangle-free check: a set S is triangle-free iff for all u in S, 
        # the neighbors of u in S form an independent set.
        
        def is_triangle_free_mask(mask):
            # Check if induced subgraph on mask is triangle-free
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                neigh = adj_mask[i] & mask
                # Check if neigh is independent
                temp = neigh
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    if adj_mask[v] & neigh:
                        return False
                    temp -= lsb
            return True
        
        def branch_and_bound(remaining, current_mask, current_size):
            nonlocal best
            # Upper bound: current_size + count_bits(remaining)
            if current_size + count_bits(remaining) <= best:
                return
            if remaining == 0:
                if current_size > best:
                    best = current_size
                return
            
            # Heuristic: pick vertex with maximum degree in remaining to fail fast
            max_deg = -1
            v = None
            for i in range(n):
                if remaining & (1 << i):
                    deg = count_bits(adj_mask[i] & remaining)
                    if deg > max_deg:
                        max_deg = deg
                        v = i
                        
            if v is None:
                return
            
            bit_v = 1 << v
            
            # Option 1: Exclude v
            branch_and_bound(remaining & ~bit_v, current_mask, current_size)
            
            # Option 2: Include v
            # Check if adding v to current_mask creates a triangle
            # v creates a triangle if there is an edge (a,b) in current_mask 
            # such that a and b are both neighbors of v.
            neigh_in_current = adj_mask[v] & current_mask
            creates_triangle = False
            temp = neigh_in_current
            while temp:
                lsb = temp & -temp
                a = lsb.bit_length() - 1
                if adj_mask[a] & neigh_in_current:
                    creates_triangle = True
                    break
                temp -= lsb
            
            if not creates_triangle:
                branch_and_bound(remaining & ~bit_v, current_mask | bit_v, current_size + 1)
        
        branch_and_bound(all_vertices_mask, 0, 0)
        return best
    
    def find_max_triangle_free_set(adj_mask, target_size):
        """Find a triangle-free set of size target_size."""
        for combo in itertools.combinations(range(n), target_size):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            # Check triangle-free
            is_tf = True
            for i in range(n):
                if not (mask & (1 << i)):
                    continue
                neigh = adj_mask[i] & mask
                temp = neigh
                while temp:
                    lsb = temp & -temp
                    v = lsb.bit_length() - 1
                    if adj_mask[v] & neigh:
                        is_tf = False
                        break
                    temp -= lsb
                if not is_tf:
                    break
            if is_tf:
                return list(combo)
        return None
    
    best_score = n
    best_adj_mask = [0] * n
    
    # Initialize with random K4-free graphs
    for restart in range(30):
        if time.time() > deadline:
            break
        
        # Random graph with density 0.5, ensuring K4-free
        adj_mask = [0] * n
        for u, v in all_pairs:
            if random.random() < 0.5:
                if can_add_edge(adj_mask, u, v):
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        
        score = largest_triangle_free(adj_mask)
        if score < best_score:
            best_score = score
            best_adj_mask = adj_mask[:]
        
        # Local search: try to add edges to reduce score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = largest_triangle_free(adj_mask)
            
            # If current_score is already 1, we can't do better (min is 1)
            if current_score <= 1:
                break
                
            # Find a max triangle-free set
            target_set = find_max_triangle_free_set(adj_mask, current_score)
            if target_set is None:
                break
            
            # Try adding edges within target_set
            found_improvement = False
            for i in range(len(target_set)):
                for j in range(i + 1, len(target_set)):
                    u = target_set[i]
                    v = target_set[j]
                    if not (adj_mask[u] & (1 << v)):
                        if can_add_edge(adj_mask, u, v):
                            # Add edge
                            adj_mask[u] |= (1 << v)
                            adj_mask[v] |= (1 << u)
                            
                            # Check new score
                            new_score = largest_triangle_free(adj_mask)
                            if new_score < current_score:
                                current_score = new_score
                                improved = True
                                found_improvement = True
                                break
                            else:
                                # Revert
                                adj_mask[u] &= ~(1 << v)
                                adj_mask[v] &= ~(1 << u)
                if found_improvement:
                    break
                    
            if current_score < best_score:
                best_score = current_score
                best_adj_mask = adj_mask[:]
                
            # If no improvement, try perturbation
            if not improved:
                # Remove 1-3 random edges
                adj_mask_perturbed = adj_mask[:]
                edges_present = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj_mask_perturbed[u] & (1 << v):
                            edges_present.append((u, v))
                
                if edges_present:
                    num_remove = random.randint(1, 3)
                    for _ in range(num_remove):
                        if not edges_present:
                            break
                        u, v = random.choice(edges_present)
                        edges_present.remove((u, v))
                        adj_mask_perturbed[u] &= ~(1 << v)
                        adj_mask_perturbed[v] &= ~(1 << u)
                    
                    # Run local search on perturbed
                    improved2 = True
                    while improved2 and time.time() < deadline:
                        improved2 = False
                        current_score2 = largest_triangle_free(adj_mask_perturbed)
                        
                        target_set2 = find_max_triangle_free_set(adj_mask_perturbed, current_score2)
                        if target_set2 is None:
                            break
                        
                        for i in range(len(target_set2)):
                            for j in range(i + 1, len(target_set2)):
                                u = target_set2[i]
                                v = target_set2[j]
                                if not (adj_mask_perturbed[u] & (1 << v)):
                                    if can_add_edge(adj_mask_perturbed, u, v):
                                        adj_mask_perturbed[u] |= (1 << v)
                                        adj_mask_perturbed[v] |= (1 << u)
                                        
                                        new_score2 = largest_triangle_free(adj_mask_perturbed)
                                        if new_score2 < current_score2:
                                            current_score2 = new_score2
                                            improved2 = True
                                            break
                                        else:
                                            adj_mask_perturbed[u] &= ~(1 << v)
                                            adj_mask_perturbed[v] &= ~(1 << u)
                            if improved2:
                                break
                            
                        if current_score2 < best_score:
                            best_score = current_score2
                            best_adj_mask = adj_mask_perturbed[:]
                            
                        if not improved2:
                            # Update adj_mask to perturbed for next iteration?
                            adj_mask = adj_mask_perturbed[:]
                            break
                            
    # Final polish: try to improve best_adj_mask with more aggressive local search
    # Specifically, try all pairs of vertices, not just those in a specific triangle-free set
    if time.time() < deadline:
        adj_mask = best_adj_mask[:]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = largest_triangle_free(adj_mask)
            
            # Try adding any edge that reduces score
            for u, v in all_pairs:
                if not (adj_mask[u] & (1 << v)):
                    if can_add_edge(adj_mask, u, v):
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        
                        new_score = largest_triangle_free(adj_mask)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            break
                        else:
                            adj_mask[u] &= ~(1 << v)
                            adj_mask[v] &= ~(1 << u)
                    
                    if current_score < best_score:
                        best_score = current_score
                        best_adj_mask = adj_mask[:]
                        
            if not improved:
                # Try removing edges to escape local minimum
                edges_present = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj_mask[u] & (1 << v):
                            edges_present.append((u, v))
                
                if edges_present:
                    u, v = random.choice(edges_present)
                    adj_mask[u] &= ~(1 << v)
                    adj_mask[v] &= ~(1 << u)
                    
                    # Run local search again
                    improved2 = True
                    while improved2 and time.time() < deadline:
                        improved2 = False
                        current_score2 = largest_triangle_free(adj_mask)
                        
                        for u2, v2 in all_pairs:
                            if not (adj_mask[u2] & (1 << v2)):
                                if can_add_edge(adj_mask, u2, v2):
                                    adj_mask[u2] |= (1 << v2)
                                    adj_mask[v2] |= (1 << u2)
                                    
                                    new_score2 = largest_triangle_free(adj_mask)
                                    if new_score2 < current_score2:
                                        current_score2 = new_score2
                                        improved2 = True
                                        break
                                    else:
                                        adj_mask[u2] &= ~(1 << v2)
                                        adj_mask[v2] &= ~(1 << u2)
                                
                                if current_score2 < best_score:
                                    best_score = current_score2
                                    best_adj_mask = adj_mask[:]
                                    
                        if not improved2:
                            break
    
    # Convert best_adj_mask to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj_mask[i] & (1 << j):
                edges.append([i, j])
    
    return edges
