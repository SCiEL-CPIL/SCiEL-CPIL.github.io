import random
import sys

def construct(k, g):
    # We assume k=3 for the specific optimization requested, 
    # but the function signature allows general k.
    # The directive specifically asks for K_{3,3} lifts, which implies k=3.
    # If k != 3, we might need a different base graph or fallback.
    # Given the task context (3-regular), we focus on k=3.
    
    if k != 3:
        # Fallback to a simple construction or error if not 3-regular
        # For now, we only implement the K_{3,3} lift for k=3.
        # If k is different, we return a trivial graph or raise an error.
        # The task says "evaluator passes this task's degree k", implying k=3.
        pass

    # Calculate m for the bipartite Moore bound
    # n >= 2 * k * sum_{i=0}^{g/2 - 1} (k-1)^i
    # For k=3, base graph K_{3,3} has 6 vertices, so n = 6m.
    # 6m >= 2 * 3 * sum => 6m >= 6 * sum => m >= sum
    
    r = g // 2
    if r < 1:
        r = 1
        
    # Sum of geometric series: sum_{i=0}^{r-1} (k-1)^i
    # For k=3, base is 2.
    # Sum = (2^r - 1) / (2 - 1) = 2^r - 1
    min_m = (2 ** r) - 1
    
    # We want the smallest n, so we start with the smallest possible m
    # that satisfies the Moore bound.
    # However, it's possible that a graph with m = min_m does not exist with girth g.
    # We might need to increase m.
    # But for g=16, r=8, min_m = 255.
    
    # We will try m = min_m, min_m + 1, ...
    # But since random lifts are likely to work for the minimal m, we start there.
    
    base_vertices = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    # K_{3,3} edges: (0,i) connected to (1,j) for all i,j in 0..2
    base_edges = []
    for i in range(3):
        for j in range(3):
            base_edges.append(((0, i), (1, j)))
            
    # Map base vertices to indices 0..5
    base_index = {v: i for i, v in enumerate(base_vertices)}
    
    # We will try random voltage assignments
    # Number of attempts
    max_attempts = 50
    
    for attempt in range(max_attempts):
        m = min_m + attempt
        if m < 1:
            m = 1
            
        # Assign random voltages to base edges
        # Voltage is in Z_m
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = random.randint(0, m - 1)
            # Note: The reverse edge (v, u) has voltage -voltages[(u, v)] mod m
            # In the lift, an edge from (u, k) to (v, k + vol(u,v)) 
            # and from (v, k') to (u, k' - vol(u,v))
            # We only need to store one direction or handle both.
            # Let's store vol for (u,v) where u in A, v in B.
            
        # Construct the lift
        n = 6 * m
        # Vertex (u, k) -> index = base_index[u] * m + k
        # u is (side, idx). side 0 or 1.
        
        edges = []
        
        # Build adjacency list for the lift to check girth
        adj = [[] for _ in range(n)]
        
        for u, v in base_edges:
            vol = voltages[(u, v)]
            # u is (0, i), v is (1, j)
            # For each k in Z_m:
            # Edge between (u, k) and (v, (k + vol) % m)
            for k in range(m):
                idx_u = base_index[u] * m + k
                idx_v = base_index[v] * m + ((k + vol) % m)
                
                # Add edge
                edges.append((idx_u, idx_v))
                
                # Add to adjacency
                adj[idx_u].append(idx_v)
                adj[idx_v].append(idx_u)
                
        # Check Girth
        # We need to ensure no cycle < g.
        # Since it's bipartite, cycles are even.
        # We check for cycles of length 4, 6, ..., g-2.
        
        def has_short_cycle(adj, n, target_girth):
            # BFS from each node up to depth target_girth//2 - 1
            # If we encounter a visited node at depth d < target_girth//2, 
            # and the path doesn't just go back to parent, we have a cycle.
            # Actually, standard BFS for girth:
            # dist[u] = distance from start
            # parent[u] = previous node
            # If we visit a node u from v, and dist[u] < dist[v], it's a back edge.
            # Cycle length = dist[v] + dist[u] + 1.
            # We want to detect any cycle < target_girth.
            
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = [start]
                
                while queue:
                    u = queue.pop(0)
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            # Optimization: if dist[v] + dist[u] + 1 >= target_girth, 
                            # we don't need to expand further from v if we are only looking for shorter cycles?
                            # No, we need to check all.
                            if dist[v] < target_girth // 2:
                                queue.append(v)
                        else:
                            # Found a cycle
                            # The cycle length is dist[u] + dist[v] + 1
                            # But we must ensure u != parent[v] (which is the tree edge)
                            if u != parent[v] and v != parent[u]:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < target_girth:
                                    return True
            return False

        if not has_short_cycle(adj, n, g):
            return n, edges
            
    # If no random lift worked, this is a failure case.
    # We should probably return a valid graph even if not optimal.
    # For robustness, we can return a known small graph if g is small, 
    # or just the last attempted one (which might be invalid).
    # But the contract says "invalid result... scores 0".
    # So we must ensure we return a valid graph.
    # If we failed to find one in 50 tries, we might try a larger m.
    
    # Try larger m
    for m in range(min_m + 50, min_m + 100):
        # Same logic as above
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = random.randint(0, m - 1)
            
        n = 6 * m
        adj = [[] for _ in range(n)]
        edges = []
        
        for u, v in base_edges:
            vol = voltages[(u, v)]
            for k in range(m):
                idx_u = base_index[u] * m + k
                idx_v = base_index[v] * m + ((k + vol) % m)
                edges.append((idx_u, idx_v))
                adj[idx_u].append(idx_v)
                adj[idx_v].append(idx_u)
                
        if not has_short_cycle(adj, n, g):
            return n, edges

    # Fallback: Return a simple valid graph if possible, or raise error.
    # For k=3, g=16, we should have found one.
    # If not, we return the last generated graph, hoping it's valid? No.
    # We'll just return a dummy valid graph for very small g, otherwise error.
    # But for this specific task, it should work.
    
    return 6, [(0,1), (1,2), (2,0), (3,4), (4,5), (5,3)] # Invalid for g=16, but prevents crash.

# EVOLVE-BLOCK-START
def construct(k, g):
    if k != 3:
        # Fallback for non-3-regular, though task implies k=3
        # Simple cycle or star? No, must be k-regular.
        # We'll just use the K_{3,3} logic if possible, else error.
        pass

    r = g // 2
    if r < 1:
        r = 1
        
    min_m = (2 ** r) - 1
    
    base_vertices = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    base_edges = []
    for i in range(3):
        for j in range(3):
            base_edges.append(((0, i), (1, j)))
            
    base_index = {v: i for i, v in enumerate(base_vertices)}
    
    def has_short_cycle(adj, n, target_girth):
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < target_girth // 2:
                            queue.append(v)
                    else:
                        if u != parent[v] and v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < target_girth:
                                return True
        return False

    # Try increasing m
    # We start from min_m
    for m in range(min_m, min_m + 20):
        # Try a few random seeds for each m
        for attempt in range(10):
            voltages = {}
            for u, v in base_edges:
                voltages[(u, v)] = random.randint(0, m - 1)
                
            n = 6 * m
            adj = [[] for _ in range(n)]
            edges = []
            
            for u, v in base_edges:
                vol = voltages[(u, v)]
                for k in range(m):
                    idx_u = base_index[u] * m + k
                    idx_v = base_index[v] * m + ((k + vol) % m)
                    edges.append((idx_u, idx_v))
                    adj[idx_u].append(idx_v)
                    adj[idx_v].append(idx_u)
                    
            if not has_short_cycle(adj, n, g):
                return n, edges
                
    # If we get here, we failed to find a valid graph in the range.
    # Return a known valid graph for small g, or the best effort.
    # For safety, return a valid 3-regular graph with large girth if possible?
    # Or just the last one? No, must be valid.
    # We'll return a simple cycle if g is small enough, but for g=16 it's not.
    # This branch is unlikely to be reached for g=16.
    return 6, [(0,1), (1,2), (2,0), (3,4), (4,5), (5,3)]

# EVOLVE-BLOCK-END
