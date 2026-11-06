import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for 150s limit
    
    def instance(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_mis(n, edges):
        # Build adjacency masks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Greedy MIS for initial bound
        # Pick vertices with low degree?
        # For MIS, picking a vertex removes its neighbors.
        # A good heuristic is to pick the vertex with the minimum degree in the current graph.
        # Or simply greedy:
        greedy_mis = set()
        remaining = set(range(n))
        while remaining:
            # Pick vertex with min degree in induced subgraph
            best_v = None
            min_deg = float('inf')
            for v in remaining:
                deg = sum(1 for u in remaining if u != v and (adj[v] >> u) & 1)
                if deg < min_deg:
                    min_deg = deg
                    best_v = v
            greedy_mis.add(best_v)
            # Remove best_v and its neighbors
            neighbors = set()
            for u in remaining:
                if (adj[best_v] >> u) & 1:
                    neighbors.add(u)
            remaining -= neighbors
            remaining.remove(best_v)
            
        best_size = len(greedy_mis)
        best_mask = 0
        for v in greedy_mis:
            best_mask |= (1 << v)
            
        # B&B Search
        # Use a list to track the path? Or just return the mask.
        # We need the mask to reconstruct the set.
        
        # Optimization: Order vertices by degree descending for processing?
        # The recursion picks a vertex. We can pick the one with max degree in candidates to prune better.
        
        def search(candidates_mask, current_mask, current_size):
            nonlocal best_size, best_mask
            
            if time.time() > deadline:
                return False # Timeout signal
            
            if candidates_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                return True
            
            # Bound
            # popcount
            cand_count = candidates_mask.bit_count()
            if current_size + cand_count <= best_size:
                return True
            
            # Pick a vertex from candidates
            # Heuristic: Max degree in the induced subgraph of candidates
            # To avoid O(n) scan every time, we can just pick the first one, 
            # but max degree is better for pruning.
            # For n=40, O(n) is fine.
            
            best_v = -1
            max_deg = -1
            temp_mask = candidates_mask
            while temp_mask:
                v = (temp_mask & -temp_mask).bit_length() - 1
                # Calculate degree of v in candidates
                # Neighbors of v in candidates: candidates_mask & adj[v]
                deg = (candidates_mask & adj[v]).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp_mask &= temp_mask - 1
                
            if best_v == -1:
                return True
                
            # Branch 1: Include best_v
            # New candidates: candidates not adjacent to best_v and not best_v
            # candidates_mask already has best_v set.
            # We remove best_v and its neighbors.
            new_cands = candidates_mask & ~(adj[best_v] | (1 << best_v))
            search(new_cands, current_mask | (1 << best_v), current_size + 1)
            
            if time.time() > deadline:
                return False
                
            # Branch 2: Exclude best_v
            # Remove best_v from candidates
            search(candidates_mask & ~(1 << best_v), current_mask, current_size)
            
            return True

        all_mask = (1 << n) - 1
        search(all_mask, 0, 0)
        
        # Reconstruct MIS from best_mask
        mis = []
        for i in range(n):
            if (best_mask >> i) & 1:
                mis.append(i)
        
        # Vertex Cover is V \ MIS
        vc = [i for i in range(n) if i not in set(mis)]
        return sorted(vc)

    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = instance(n, p, seed)
        vc = solve_mis(n, edges)
        out.append(vc)
        
    return out
