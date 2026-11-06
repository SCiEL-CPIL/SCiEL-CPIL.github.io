import time
from random import Random

def construct():
    deadline = time.time() + 140
    
    def instance(n, p, seed):
        rng = Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = instance(n, p, seed)
        if not edges:
            return []
        
        rem_mask = [0] * n
        for u, v in edges:
            rem_mask[u] |= (1 << v)
            rem_mask[v] |= (1 << u)
            
        remaining_count = len(edges)
        cliques = []
        
        # Helper to count bits
        def bit_count(x):
            return bin(x).count('1')
            
        # Helper to get list of vertices from mask
        def mask_to_list(mask):
            res = []
            while mask:
                lsb = mask & (-mask)
                res.append(lsb.bit_length() - 1)
                mask ^= lsb
            return res

        def find_max_clique():
            """
            Finds a maximum clique in the graph defined by rem_mask.
            Uses Branch and Bound with bitmasks.
            Returns a list of vertices.
            """
            # Get all vertices with remaining edges
            all_vertices = [v for v in range(n) if rem_mask[v] != 0]
            if not all_vertices:
                return []
            
            # Sort by degree descending to help pruning
            all_vertices.sort(key=lambda v: bit_count(rem_mask[v]), reverse=True)
            
            best_clique = []
            best_size = 0
            node_count = 0
            max_nodes = 50000 # Limit to prevent hanging
            
            # P_mask is the set of candidate vertices
            # We start with all vertices
            # But B&B usually works on a list. Let's use the list approach 
            # but with bitmask intersection for speed.
            
            # Actually, let's use the recursive function on the mask.
            # The order of processing vertices in P affects performance.
            # We can't easily sort the bits in the mask dynamically without converting to list.
            # Let's convert P_mask to a sorted list at the start of each expand call? 
            # That might be slow.
            
            # Alternative: 
            # expand(P_list, clique, depth)
            # P_list is sorted by degree.
            # When we pick v, new_P_list = [u for u in P_list if u != v and (rem_mask[v] >> u) & 1]
            # This is O(|P|) per step.
            
            def expand(P_list, clique, depth):
                nonlocal best_clique, best_size, node_count
                
                if node_count > max_nodes:
                    return
                
                if not P_list:
                    if depth > best_size:
                        best_size = depth
                        best_clique = clique.copy()
                    return
                    
                # Pruning: if depth + len(P_list) <= best_size, no better solution possible
                if depth + len(P_list) <= best_size:
                    return
                
                while P_list:
                    node_count += 1
                    if node_count > max_nodes:
                        return
                        
                    # Pick a vertex. 
                    # Heuristic: Pick the one with highest degree in P_list?
                    # Or just the first one?
                    # Tomita's algorithm typically processes in the order they were sorted.
                    # Let's take the first one.
                    v = P_list[0]
                    P_list = P_list[1:]
                    
                    # Add v to clique
                    clique.append(v)
                    
                    # New candidates are neighbors of v in P_list
                    # Note: P_list currently contains vertices that were candidates 
                    # and are not v. We need to filter P_list to only those adjacent to v.
                    # Since P_list is a list, we can filter it.
                    new_P_list = [u for u in P_list if (rem_mask[v] >> u) & 1]
                    
                    expand(new_P_list, clique, depth + 1)
                    
                    clique.pop()
                    
                    # If we found a clique of size n, we can stop
                    if best_size == n:
                        return

            # Initial call
            # We need to pass a copy of the sorted list
            initial_P = all_vertices[:]
            expand(initial_P, [], 0)
            
            return best_clique

        while remaining_count > 0:
            if time.time() > deadline - 10:
                # Emergency fallback: just take edges
                break
                
            clique = find_max_clique()
            
            if not clique or len(clique) < 2:
                # Fallback: take a single edge
                found = False
                for u in range(n):
                    mask = rem_mask[u]
                    while mask:
                        lsb = mask & (-mask)
                        v = lsb.bit_length() - 1
                        if u < v:
                            cliques.append([u, v])
                            rem_mask[u] &= ~(1 << v)
                            rem_mask[v] &= ~(1 << u)
                            remaining_count -= 1
                            found = True
                            break
                        mask ^= lsb
                    if found:
                        break
                if not found:
                    break
                continue

            # Remove all edges within clique from remaining
            removed = 0
            clique_set = set(clique)
            for i, u in enumerate(clique):
                for v in clique[i + 1:]:
                    if (rem_mask[u] >> v) & 1:
                        rem_mask[u] &= ~(1 << v)
                        rem_mask[v] &= ~(1 << u)
                        removed += 1

            if removed == 0:
                # Shouldn't happen, but fallback
                e = (clique[0], clique[1])
                cliques.append(list(e))
                rem_mask[e[0]] &= ~(1 << e[1])
                rem_mask[e[1]] &= ~(1 << e[0])
                remaining_count -= 1
                continue

            cliques.append(clique)
            remaining_count -= removed

        # Handle any remaining edges as individual edge-cliques
        for u in range(n):
            mask = rem_mask[u]
            while mask:
                lsb = mask & (-mask)
                v = lsb.bit_length() - 1
                if u < v:
                    cliques.append([u, v])
                    rem_mask[u] &= ~(1 << v)
                    rem_mask[v] &= ~(1 << u)
                mask ^= lsb

        return cliques

    result = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5:
            # Emergency fallback
            edges = instance(n, p, seed)
            result.append([list(e) for e in edges])
        else:
            result.append(solve_instance(n, p, seed))

    return result
