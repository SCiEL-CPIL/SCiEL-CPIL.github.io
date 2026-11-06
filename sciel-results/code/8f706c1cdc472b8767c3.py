import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s buffer
    
    best_n = 1
    best_colors = []
    
    # Try n from 18 down to 10
    # We prioritize higher n
    for n in range(18, 9, -1):
        if time.time() > deadline - 5:
            break
            
        # Allocate time: more time for higher n
        # n=18: 40s, n=17: 50s, others: 10s
        if n >= 17:
            time_limit = 50.0
        else:
            time_limit = 10.0
            
        target_end = time.time() + time_limit
        
        result = _local_search(n, target_end)
        if result:
            return result
            
        # If we found a valid solution in _local_search, it returns immediately.
        # If not, we continue to next n.
        
    # Fallback: return a small valid solution
    # We can generate a trivial one for n=1, or try a very short search for n=10
    if time.time() < deadline - 1:
        res = _local_search(10, time.time() + 1.0)
        if res:
            return res
            
    return {'n': 1, 'colors': []}

def _local_search(n, deadline):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
        
    # State: colors array of length num_edges
    # adj[c][u] is a bitmask of neighbors of u in color c
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    adj = [[0] * n for _ in range(4)]
    
    def build_adj():
        for c in range(4):
            for u in range(n):
                adj[c][u] = 0
        for idx, c in enumerate(colors):
            u, v = edges[idx]
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            
    build_adj()
    
    def count_paths_c6(u, v, c, exclude_edge_idx=None):
        """
        Count simple paths of length 5 between u and v in G_c.
        If exclude_edge_idx is provided, that edge is not in G_c.
        """
        # We use DFS with visited mask
        # Path: u - x1 - x2 - x3 - x4 - v
        # We start at u, need to reach v in 5 steps.
        # visited mask includes u initially.
        
        count = 0
        
        def dfs(current, depth, visited):
            nonlocal count
            if depth == 5:
                if current == v:
                    count += 1
                return
                
            # Get neighbors of current in color c
            # If exclude_edge_idx is not None, we must ensure we don't traverse that edge.
            # But since we are checking if adding (u,v) creates a C6, the edge (u,v) is NOT in the graph yet.
            # So we don't need to exclude it from the graph structure, it's just not there.
            # Wait, the function is called to check if adding (u,v) creates a C6.
            # So the graph G_c does NOT contain (u,v).
            # So we just traverse G_c.
            
            neighbors = adj[c][current]
            # We cannot go back to visited nodes
            neighbors &= ~visited
            
            # Optimization: if v is not reachable in remaining steps?
            # Simple check: if distance > remaining, return.
            # But we don't have distance precomputed.
            
            while neighbors:
                lsb = neighbors & -neighbors
                next_node = lsb.bit_length() - 1
                neighbors ^= lsb
                
                # Pruning: if next_node is v and depth < 4, we can't finish in 5 steps?
                # If depth == 4 and next_node == v, then we have a path of length 5?
                # No, depth counts edges.
                # Start depth 0 at u.
                # Step 1: depth 1.
                # ...
                # Step 5: depth 5.
                # So if depth == 4 and next_node == v, then path length is 5?
                # u (0) -> x1 (1) -> x2 (2) -> x3 (3) -> x4 (4) -> v (5).
                # Yes.
                # So if depth == 4 and next_node == v, count += 1 and return (don't go deeper).
                # Actually, the base case handles depth==5.
                # If we go to v at depth 4, we call dfs(v, 5, visited|v).
                # Then depth==5, current==v, count++.
                
                dfs(next_node, depth + 1, visited | (1 << next_node))
                
        dfs(u, 0, 1 << u)
        return count

    def get_cost():
        total = 0
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            # Count C6s containing this edge
            # This is equivalent to counting paths of length 5 between u and v in G_c \ {(u,v)}
            # Since (u,v) is in G_c, we must exclude it.
            # Our count_paths_c6 uses adj[c] which INCLUDES (u,v).
            # So we need to temporarily remove it or pass a flag.
            # Let's just calculate paths in G_c and subtract those using (u,v)?
            # No, paths of length 5 between u and v in G_c might use the edge (u,v) as part of the path?
            # No, a path between u and v doesn't use the edge (u,v) unless it's a cycle?
            # A simple path from u to v does not contain the edge (u,v) because that would mean u-v is an edge in the path, but the path starts at u and ends at v.
            # Wait, if the path is u-v-..., it uses the edge (u,v).
            # But a simple path from u to v cannot visit u again, so it cannot use the edge (u,v) to leave u?
            # Actually, the edge (u,v) connects u and v.
            # If the path is u -> v -> ..., it uses (u,v).
            # But we want a path of length 5 *between* u and v.
            # If the path uses (u,v), it is u-v-...-v? No, simple path.
            # So a simple path from u to v of length 5 does NOT use the edge (u,v) as an internal edge.
            # It might use it as the first or last edge?
            # If it uses (u,v) as the first edge: u-v-x2-x3-x4-v? No, v is visited twice.
            # So a simple path from u to v does not contain the edge (u,v).
            # Therefore, we can just count paths in G_c.
            total += count_paths_c6(u, v, c)
        return total

    # Simulated Annealing
    current_cost = get_cost()
    best_cost = current_cost
    best_colors = colors.copy()
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    steps = 0
    while time.time() < deadline:
        # Pick random edge
        idx = random.randint(0, num_edges - 1)
        u, v = edges[idx]
        old_c = colors[idx]
        
        # Pick new color
        new_c = random.randint(0, 3)
        if new_c == old_c:
            continue
            
        # Calculate delta cost
        # Remove (u,v) from old_c: removes count_paths_c6(u, v, old_c)
        # Add (u,v) to new_c: adds count_paths_c6(u, v, new_c)
        # Note: count_paths_c6 uses the current adj.
        # We must calculate these BEFORE updating adj.
        
        # But wait, count_paths_c6(u, v, c) counts paths in G_c.
        # If we remove (u,v) from old_c, the paths that *use* (u,v) are removed.
        # But as established, simple paths between u and v don't use (u,v).
        # So the count of paths between u and v in G_c \ {(u,v)} is the same as in G_c.
        # So delta = paths_new - paths_old.
        
        paths_old = count_paths_c6(u, v, old_c)
        paths_new = count_paths_c6(u, v, new_c)
        
        delta = paths_new - paths_old
        
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            # Accept
            # Update state
            colors[idx] = new_c
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                
            if best_cost == 0:
                return {'n': n, 'colors': best_colors}
                
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart with higher T?
            T = 1.0
            
    if best_cost == 0:
        return {'n': n, 'colors': best_colors}
        
    return None
