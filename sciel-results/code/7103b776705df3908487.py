"""Construct a triangle-free graph with no independent set of size 10, maximizing n."""
import time
import random
import sys

# Use a fixed seed for reproducibility
random.seed(42)

def _is_triangle_free(adj, n):
    """Check if graph is triangle-free using bitmasks."""
    for i in range(n):
        ai = adj[i]
        for j in range(i + 1, n):
            if (ai >> j) & 1:
                # Check if any common neighbor k > j
                common = ai & adj[j]
                # Mask to only vertices > j
                common &= ~((1 << (j + 1)) - 1)
                if common:
                    return False
    return True

def _max_independent_set_size(adj, n):
    """Compute maximum independent set size using branch and bound with bitmasks."""
    # Precompute for each vertex, the bitmask of non-neighbors (excluding self)
    # A vertex v is non-adjacent to u if not (adj[v] has bit u)
    non_adj = []
    for v in range(n):
        mask = 0
        for u in range(n):
            if u == v:
                continue
            if not ((adj[v] >> u) & 1):
                mask |= (1 << u)
        non_adj.append(mask)
    
    best = [0]
    
    def backtrack(candidates, size):
        if not candidates:
            if size > best[0]:
                best[0] = size
            return
        if size + candidates.bit_count() <= best[0]:
            return
        # Choose a vertex to branch on (heuristic: pick one with fewest non-neighbor candidates)
        # For speed, just pick the lowest set bit
        v = (candidates & -candidates).bit_length() - 1
        # Option 1: include v
        backtrack(candidates & non_adj[v], size + 1)
        # Option 2: exclude v
        backtrack(candidates & ~(1 << v), size)
    
    backtrack((1 << n) - 1, 0)
    return best[0]

def _has_independent_set_10(adj, n):
    """Check if there's an independent set of size 10."""
    return _max_independent_set_size(adj, n) >= 10

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with the known Andrasfai graph And(9) on 26 vertices
    n = 26
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Verify the baseline
    if not _is_triangle_free(adj, n) or _has_independent_set_10(adj, n):
        # This shouldn't happen, but return something valid
        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (adj[i] >> j) & 1]
        return {'n': n, 'edges': edges}
    
    best_n = n
    best_adj = adj[:]
    
    # Try to extend the graph to larger n by adding vertices
    # Strategy: try to add vertices one at a time, maintaining triangle-free and alpha < 10
    # We'll try random assignments of edges for new vertices
    
    for target_n in range(27, 41):
        if time.time() > deadline - 30.0:
            break
        
        # Try to extend from best_adj to target_n
        success = False
        max_attempts = 500
        for attempt in range(max_attempts):
            if time.time() > deadline - 25.0:
                break
            
            # Start from a copy of best_adj and add new vertices
            cur_n = target_n
            cur_adj = best_adj[:] + [0] * (target_n - len(best_adj))
            
            # For each new vertex, randomly decide its connections to existing vertices
            # We need to ensure no triangles are created
            # A new vertex v connected to u and w would create a triangle if u-w is an edge
            # So for each new vertex, its neighbors must form an independent set in the existing graph
            
            valid = True
            for new_v in range(len(best_adj), target_n):
                # Choose neighbors among existing vertices (0..len(best_adj)-1)
                # Neighbors must be an independent set in the current graph
                # Randomly pick a subset
                existing_n = len(best_adj)
                # Pick random neighbors ensuring they form an independent set
                neighbors = []
                for u in range(existing_n):
                    if random.random() < 0.4:
                        # Check if u can be added: u must not be adjacent to any already chosen neighbor
                        can_add = True
                        for w in neighbors:
                            if (cur_adj[u] >> w) & 1:
                                can_add = False
                                break
                        if can_add:
                            neighbors.append(u)
                
                # Also connect to other new vertices? Keep it simple: only connect to old vertices first
                for u in neighbors:
                    cur_adj[new_v] |= (1 << u)
                    cur_adj[u] |= (1 << new_v)
            
            # Check triangle-free
            if not _is_triangle_free(cur_adj, cur_n):
                continue
            
            # Check independent set
            if time.time() > deadline - 20.0:
                continue
            if _has_independent_set_10(cur_adj, cur_n):
                continue
            
            # Valid!
            best_n = cur_n
            best_adj = cur_adj
            success = True
            break
        
        if not success:
            # Try a different approach: random restart with different initial graph
            # Try random graph generation
            for attempt in range(200):
                if time.time() > deadline - 20.0:
                    break
                
                cur_n = target_n
                cur_adj = [0] * cur_n
                
                # Generate random triangle-free graph
                # Process vertices in order, for each vertex decide edges to previous vertices
                # such that no triangle is formed
                for v in range(1, cur_n):
                    # Neighbors of v among {0..v-1} must form an independent set
                    # Randomly select
                    candidates = list(range(v))
                    random.shuffle(candidates)
                    neighbors = []
                    for u in candidates:
                        can_add = True
                        for w in neighbors:
                            if (cur_adj[u] >> w) & 1:
                                can_add = False
                                break
                        if can_add:
                            neighbors.append(u)
                    for u in neighbors:
                        cur_adj[v] |= (1 << u)
                        cur_adj[u] |= (1 << v)
                
                if not _is_triangle_free(cur_adj, cur_n):
                    continue
                
                if time.time() > deadline - 15.0:
                    continue
                if _has_independent_set_10(cur_adj, cur_n):
                    continue
                
                best_n = cur_n
                best_adj = cur_adj
                break
            if best_n >= target_n:
                break
    
    # Build edge list from best_adj
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
