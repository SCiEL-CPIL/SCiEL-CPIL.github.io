import time
import random

def construct():
        start_time = time.time()
        deadline = start_time + 140.0
        
        n = 39
        best_graph = None
        
        # Helper functions using bitmasks
        # adj is a list of ints, adj[i] is bitmask of neighbors of i
        
        def can_add(adj, u, v):
            # Check if adding (u, v) creates a C4
            # Condition: For all x in N(u), N(x) and N(v) must be disjoint.
            # And for all y in N(v), N(u) and N(y) must be disjoint.
            # Note: u and v are not neighbors yet.
            mask_u = adj[u]
            mask_v = adj[v]
            
            # Check x in N(u)
            x = mask_u
            while x:
                lsb = x & -x
                xi = lsb.bit_length() - 1
                # Check if N(xi) and N(v) intersect
                if adj[xi] & mask_v:
                    return False
                x -= lsb
            
            # Check y in N(v)
            y = mask_v
            while y:
                lsb = y & -y
                yi = lsb.bit_length() - 1
                # Check if N(u) and N(yi) intersect
                if mask_u & adj[yi]:
                    return False
                y -= lsb
                
            return True

        def has_independent_set_11(adj, n):
            # Returns True if there is an independent set of size 11
            # Branch and bound
            # We want to find if alpha >= 11
            
            # Precompute degrees?
            # Use a recursive function
            # state: current_set_mask, available_mask, current_size
            # available_mask: vertices that can still be added (not adjacent to current_set)
            
            best_found = 0
            
            # Order vertices by degree to prune?
            # For now, simple B&B
            
            def dfs(available, current_size):
                nonlocal best_found
                if current_size >= 11:
                    return True
                if current_size + bin(available).count('1') < 11:
                    return False
                
                # Pick a vertex
                # Heuristic: pick vertex with highest degree in available?
                # Or just lowest index?
                # Let's pick the one with lowest index for simplicity, or highest degree to reduce branching?
                # High degree in the original graph means it blocks many others, so picking it might reduce available quickly.
                # But we are looking for an independent set, so we want to pick vertices that are NOT connected.
                # In the subgraph induced by `available`, we want to find a large independent set.
                # This is equivalent to finding a large clique in the complement.
                
                # Let's just pick the first available vertex.
                if available == 0:
                    return False
                    
                lsb = available & -available
                v = lsb.bit_length() - 1
                
                # Option 1: Include v
                # New available: available & ~adj[v] & ~lsb (remove v and its neighbors)
                new_avail = available & ~adj[v] & ~lsb
                if dfs(new_avail, current_size + 1):
                    return True
                
                # Option 2: Exclude v
                new_avail = available & ~lsb
                if dfs(new_avail, current_size):
                    return True
                
                return False

            return dfs((1 << n) - 1, 0)

        # Main loop
        for i in range(100000):
            if time.time() > deadline:
                break
            
            # Randomize edge order
            edges = [(u, v) for u in range(n) for v in range(u+1, n)]
            random.shuffle(edges)
            
            adj = [0] * n
            
            for u, v in edges:
                if can_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Check alpha
            if not has_independent_set_11(adj, n):
                # Convert adj to edge list
                edge_list = []
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            edge_list.append([u, v])
                        mask -= lsb
                return {'n': n, 'edges': edge_list}
            
            # If not found, try again
            # Maybe try n=40?
            # But 39 is safer.
        
        # Fallback: return the seed solution
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
