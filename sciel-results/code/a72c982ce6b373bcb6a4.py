import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def is_triangle_free(adj, n):
        for i in range(n):
            # adj[i] has bits for neighbors
            # For each neighbor j of i, check if adj[i] & adj[j] != 0
            # Actually, just check if adj[i] & adj[j] is non-zero for j in neighbors
            # Faster: for each i, for each j > i where adj[i] has bit j, check (adj[i] & adj[j]) != 0
            # But we maintain triangle-free invariant, so we only need to check after modifications.
            pass # Handled by construction
    
    def max_independent_set(adj, n):
        # Branch and bound
        # Returns size of max independent set
        best = 0
        
        def dfs(candidates, current_size):
            nonlocal best
            if current_size + len(candidates) <= best:
                return
            if not candidates:
                if current_size > best:
                    best = current_size
                return
            
            # Heuristic: pick vertex with highest degree in induced subgraph?
            # Or just first.
            # To speed up, we can use a simple bound: 
            # If we pick v, we can only pick non-neighbors of v from rest.
            
            v = candidates[0]
            rest = candidates[1:]
            
            # Branch 1: Include v
            # Neighbors of v in candidates
            v_mask = adj[v]
            new_cands = []
            for u in rest:
                if not (v_mask >> u) & 1:
                    new_cands.append(u)
            
            dfs(new_cands, current_size + 1)
            
            # Branch 2: Exclude v
            dfs(rest, current_size)
            
            if best >= 10:
                return
        
        dfs(list(range(n)), 0)
        return best

    def remove_triangles(adj, n):
        # Find all triangles and remove one edge from each
        # This is O(n^3) if done naively, but we can be smarter
        # Iterate until no triangles found
        changed = True
        while changed:
            changed = False
            for i in range(n):
                # Get neighbors of i
                neigh_i = []
                mask = adj[i]
                for j in range(i+1, n):
                    if (mask >> j) & 1:
                        neigh_i.append(j)
                
                # For each pair of neighbors j, k, check if (j,k) is an edge
                # If yes, triangle (i, j, k)
                for idx, j in enumerate(neigh_i):
                    for k in neigh_i[idx+1:]:
                        if (adj[j] >> k) & 1:
                            # Triangle found: i, j, k
                            # Remove a random edge
                            # Prefer removing edge not incident to i? 
                            # Or just random.
                            choice = random.randint(0, 2)
                            if choice == 0:
                                adj[i] &= ~(1 << j)
                                adj[j] &= ~(1 << i)
                            elif choice == 1:
                                adj[i] &= ~(1 << k)
                                adj[k] &= ~(1 << i)
                            else:
                                adj[j] &= ~(1 << k)
                                adj[k] &= ~(1 << j)
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break

    def find_large_independent_set(adj, n, target_size):
        # Returns a list of vertices forming an independent set of size >= target_size, or None
        found = None
        
        def dfs(candidates, current_set):
            nonlocal found
            if found:
                return
            if len(current_set) >= target_size:
                found = current_set[:]
                return
            if not candidates:
                return
            
            v = candidates[0]
            rest = candidates[1:]
            
            # Include v
            new_cands = []
            v_mask = adj[v]
            for u in rest:
                if not (v_mask >> u) & 1:
                    new_cands.append(u)
            
            dfs(new_cands, current_set + [v])
            
            if not found:
                # Exclude v
                dfs(rest, current_set)
        
        dfs(list(range(n)), [])
        return found

    # Main Logic
    n = 40
    adj = [0] * n
    
    # Initialize with random graph
    random.seed(42)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # Remove triangles to make it valid
    remove_triangles(adj, n)
    
    # Check alpha
    alpha = max_independent_set(adj, n)
    
    # If alpha < 10, we are done
    if alpha < 10:
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    # Local Search
    # We want to reduce alpha.
    # Move: Add an edge between two non-adjacent vertices in a large independent set,
    # then fix resulting triangles by removing other edges.
    
    best_adj = adj[:]
    best_alpha = alpha
    temperature = 1.0
    cooling = 0.999
    
    steps = 0
    max_steps = 5000
    
    while steps < max_steps and time.time() < deadline - 5:
        steps += 1
        
        # Find a large independent set
        large_is = find_large_independent_set(adj, n, 10)
        
        if not large_is:
            # Should not happen if alpha >= 10, but just in case
            alpha = max_independent_set(adj, n)
            if alpha < 10:
                break
            continue
            
        # Pick two vertices u, v from the independent set
        u, v = random.sample(large_is, 2)
        
        # Try to add edge (u, v)
        if (adj[u] >> v) & 1:
            continue # Already connected
        
        # Calculate conflicts: vertices w such that (u,w) and (v,w) are edges
        conflicts = []
        common_mask = adj[u] & adj[v]
        w_mask = common_mask
        while w_mask:
            # Get lowest set bit
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            conflicts.append(w)
            w_mask ^= lsb
            
        if not conflicts:
            # Safe to add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            new_alpha = max_independent_set(adj, n)
            
            # Since we added an edge, alpha cannot increase? 
            # Actually, adding an edge can increase the independence number? No.
            # Independent sets in G are independent sets in G+e.
            # So alpha(G+e) <= alpha(G).
            # So new_alpha <= alpha.
            
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_adj = adj[:]
                if new_alpha < 10:
                    break
        else:
            # We have conflicts.
            # We need to remove edges to break triangles.
            # Let's try to remove edges (u, w) or (v, w) for w in conflicts.
            # We want to keep (u, v) if possible.
            
            # Save state
            old_adj = adj[:]
            
            # Add (u, v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Remove edges to break triangles
            for w in conflicts:
                # Remove either (u, w) or (v, w)
                if random.random() < 0.5:
                    adj[u] &= ~(1 << w)
                    adj[w] &= ~(1 << u)
                else:
                    adj[v] &= ~(1 << w)
                    adj[w] &= ~(1 << v)
            
            # Check if we broke the independent set?
            # The independent set `large_is` contained u and v.
            # We added (u, v), so `large_is` is no longer independent.
            # But we removed edges from u and v to other vertices.
            # This might have created new independent sets.
            
            new_alpha = max_independent_set(adj, n)
            
            # Acceptance
            delta = new_alpha - alpha
            if delta < 0:
                alpha = new_alpha
                if new_alpha < best_alpha:
                    best_alpha = new_alpha
                    best_adj = adj[:]
                    if new_alpha < 10:
                        break
            else:
                # Accept with probability exp(-delta / temperature)
                if random.random() < (2.718281828 ** (-delta / temperature)):
                    alpha = new_alpha
                    if new_alpha < best_alpha:
                        best_alpha = new_alpha
                        best_adj = adj[:]
                        if new_alpha < 10:
                            break
                else:
                    # Revert
                    adj = old_adj[:]
            
        temperature *= cooling

    # Use best found
    if best_alpha < 10:
        adj = best_adj
    else:
        # Fallback to best_alpha if not < 10? 
        # The task requires a valid graph. If we don't find one, we should return a smaller valid one.
        # But the knowledge says we CAN find one.
        # If we didn't, maybe try n=39?
        pass

    # If we still don't have a valid graph on 40, try to construct a valid one on smaller n
    # Or just return the best we have if it's valid?
    # If best_alpha >= 10, the graph is INVALID.
    # We must return a valid graph.
    
    # Let's try a quick check on best_adj
    if max_independent_set(best_adj, n) < 10:
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (best_adj[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
        
    # Fallback: Construct a known valid graph on smaller n
    # e.g. n=30 random search
    for n_try in [39, 38, 30]:
        if time.time() > deadline - 5:
            break
        adj_try = [0] * n_try
        for i in range(n_try):
            for j in range(i+1, n_try):
                if random.random() < 0.5:
                    adj_try[i] |= (1 << j)
                    adj_try[j] |= (1 << i)
        remove_triangles(adj_try, n_try)
        if max_independent_set(adj_try, n_try) < 10:
            edges = []
            for i in range(n_try):
                for j in range(i+1, n_try):
                    if (adj_try[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n_try, 'edges': edges}
            
    # Absolute fallback
    return {'n': 1, 'edges': []}
