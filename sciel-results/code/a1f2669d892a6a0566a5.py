import time
import random

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    def count_paths_len_5(u, v, adj_c, n):
        """
        Count simple paths of length 5 from u to v in graph adj_c.
        adj_c is a list of bitmasks.
        """
        if u == v:
            return 0
        
        count = 0
        
        def dfs(curr, depth, visited):
            nonlocal count
            if depth == 5:
                if curr == v:
                    count += 1
                return
            
            mask = adj_c[curr]
            # Optimization: prune if not enough vertices left?
            # For now, just basic pruning
            
            # We cannot visit v before depth 5
            if depth < 4:
                mask &= ~(1 << v)
            
            while mask:
                lsb = mask & -mask
                nxt = lsb.bit_length() - 1
                mask -= lsb
                
                if (visited >> nxt) & 1:
                    continue
                
                dfs(nxt, depth + 1, visited | (1 << nxt))
        
        dfs(u, 0, 1 << u)
        return count

    def get_total_cost(adj, n):
        """
        Calculate total number of 'potential' C6s (sum over edges of path counts).
        """
        total = 0
        for c in range(4):
            for i in range(n):
                mask = adj[c][i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    mask -= lsb
                    if j > i:
                        total += count_paths_len_5(i, j, adj[c], n)
        return total

    def local_search(n, time_limit):
        """
        Run local search to find a valid coloring for K_n.
        Returns (n, colors) if found, else None.
        """
        end_time = start_time + time_limit
        
        # Initialize random coloring
        adj = [[0] * n for _ in range(4)]
        colors = [0] * (n * (n - 1) // 2)
        
        # Assign random colors
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = random.randint(0, 3)
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                colors[idx] = c
                idx += 1
        
        current_cost = get_total_cost(adj, n)
        
        best_cost = current_cost
        best_adj = [row[:] for row in adj]
        best_colors = colors[:]
        
        if current_cost == 0:
            return {'n': n, 'colors': colors}
        
        while time.time() < end_time:
            improved = False
            
            # Try random moves
            # We can iterate over all edges or pick random ones
            # Picking random ones is faster to implement and often works
            
            # Shuffle edge order for diversity?
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(edges)
            
            for u, v in edges:
                if time.time() > end_time:
                    break
                
                # Find current color of (u, v)
                c_old = -1
                for c in range(4):
                    if (adj[c][u] >> v) & 1:
                        c_old = c
                        break
                
                # Try other colors
                for c_new in range(4):
                    if c_new == c_old:
                        continue
                    
                    # Calculate delta cost
                    # Cost contribution of (u,v) in c_old: count_paths_len_5(u, v, adj[c_old])
                    # Cost contribution of (u,v) in c_new: count_paths_len_5(u, v, adj[c_new])
                    
                    # Note: count_paths_len_5 does not depend on the edge (u,v) being present 
                    # in the graph for the path count, as established.
                    
                    cost_old = count_paths_len_5(u, v, adj[c_old], n)
                    cost_new = count_paths_len_5(u, v, adj[c_new], n)
                    
                    delta = cost_new - cost_old
                    
                    if delta < 0:
                        # Accept move
                        # Remove (u,v) from c_old
                        adj[c_old][u] &= ~(1 << v)
                        adj[c_old][v] &= ~(1 << u)
                        
                        # Add (u,v) to c_new
                        adj[c_new][u] |= (1 << v)
                        adj[c_new][v] |= (1 << u)
                        
                        # Update colors array
                        # Find index of (u,v)
                        # idx = u * n - u * (u + 1) // 2 + (v - u - 1) ?
                        # Simpler: precompute indices or just update colors at the end if valid
                        # For now, we track adj. We can extract colors at the end.
                        
                        current_cost += delta
                        improved = True
                        
                        if current_cost == 0:
                            # Valid coloring found
                            colors_out = []
                            for i in range(n):
                                for j in range(i + 1, n):
                                    for c in range(4):
                                        if (adj[c][i] >> j) & 1:
                                            colors_out.append(c)
                                            break
                            return {'n': n, 'colors': colors_out}
                        
                        break # Move made, next edge
            
            if not improved:
                # Stuck, do a random kick
                # Change K random edges to random colors
                K = 5
                for _ in range(K):
                    u, v = random.choice(edges)
                    c_old = -1
                    for c in range(4):
                        if (adj[c][u] >> v) & 1:
                            c_old = c
                            break
                    c_new = random.choice([c for c in range(4) if c != c_old])
                    
                    # Update adj
                    adj[c_old][u] &= ~(1 << v)
                    adj[c_old][v] &= ~(1 << u)
                    adj[c_new][u] |= (1 << v)
                    adj[c_new][v] |= (1 << u)
                    
                    # Update cost roughly? 
                    # We just recalculate total cost for simplicity if stuck?
                    # Or just continue and let it recover.
                    # Recalculating is O(E * Count). E=171, Count=1000. 171k ops. Fast.
                    current_cost = get_total_cost(adj, n)
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = [row[:] for row in adj]
                    best_colors = [0] * (n * (n - 1) // 2) # Placeholder, will extract if needed

        # If no valid coloring found, return best effort? 
        # The task requires a VALID coloring.
        # So we return None if not valid.
        return None

    # Try different n
    best_result = None
    
    # Try 19
    res = local_search(19, 30)
    if res:
        best_result = res
    
    # Try 18
    if time.time() < deadline:
        res = local_search(18, 30)
        if res:
            if not best_result or res['n'] > best_result['n']:
                best_result = res
                
    # Try 17
    if time.time() < deadline:
        res = local_search(17, 30)
        if res:
            if not best_result or res['n'] > best_result['n']:
                best_result = res

    if best_result:
        return best_result
    
    # Fallback to small n
    # K5 is always valid
    return {'n': 5, 'colors': [0] * 10}
