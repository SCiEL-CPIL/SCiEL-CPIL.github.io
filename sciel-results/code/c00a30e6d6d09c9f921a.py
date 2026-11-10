import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    n = 20
    total_possible = n * (n - 1) // 2  # 190
    
    # Build conflict graph for MWIS formulation
    # Each node in conflict graph = potential edge (u,v) in original graph
    # Two nodes conflict if selecting both edges would create a K2,3 violation
    
    # Edge index mapping
    edge_list = []
    edge_index = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[(u, v)] = len(edge_list)
            edge_list.append((u, v))
    
    m = len(edge_list)  # 190
    
    # Build adjacency list for conflict graph using bitsets
    # conflict[i] is a bitmask of nodes conflicting with node i
    conflict = [0] * m
    
    # For each pair of edges, check if they conflict
    # Edges e1=(u1,v1) and e2=(u2,v2) conflict if:
    # There exist two vertices x,y that would have 3+ common neighbors if both edges are present
    # This is complex; let's think differently.
    
    # Actually, the K2,3 constraint is: no two vertices have 3+ common neighbors.
    # Adding edges incrementally, we need to track when adding an edge would cause
    # some pair to have 3+ common neighbors.
    
    # Alternative approach: Use branch and bound directly on the original problem
    # with a good upper bound.
    
    # Let's use a different strategy: start with a good initial solution via greedy,
    # then use branch and bound to try to improve it.
    
    def count_common_neighbors(adj, u, v):
        """Count common neighbors of u and v using bitmasks."""
        common = adj[u] & adj[v]
        return bin(common).count('1')
    
    def is_k23_free(adj):
        """Check if graph is K2,3-free: no two vertices have 3+ common neighbors."""
        for u in range(n):
            for v in range(u + 1, n):
                if count_common_neighbors(adj, u, v) >= 3:
                    return False
        return True
    
    def greedy_construction(seed_val):
        """Greedy: add edges in order of minimum common neighbor impact."""
        adj = [0] * n
        # Start with random edge ordering
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.seed(seed_val)
        random.shuffle(edges)
        
        for u, v in edges:
            if time.time() > deadline:
                break
            can_add = True
            for w in range(n):
                if w == u or w == v:
                    continue
                cuw = count_common_neighbors(adj, u, w)
                if cuw + (1 if (adj[w] >> v) & 1 else 0) >= 3:
                    can_add = False
                    break
                cvw = count_common_neighbors(adj, v, w)
                if cvw + (1 if (adj[w] >> u) & 1 else 0) >= 3:
                    can_add = False
                    break
            
            if can_add:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        return adj
    
    def local_search(adj):
        """Try to add more edges via local search."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u in range(n):
                if time.time() > deadline:
                    return adj
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        continue
                    can_add = True
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        cuw = count_common_neighbors(adj, u, w)
                        if cuw + (1 if (adj[w] >> v) & 1 else 0) >= 3:
                            can_add = False
                            break
                        cvw = count_common_neighbors(adj, v, w)
                        if cvw + (1 if (adj[w] >> u) & 1 else 0) >= 3:
                            can_add = False
                            break
                    
                    if can_add:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
        
        return adj
    
    def count_edges(adj):
        return sum(bin(adj[u]).count('1') for u in range(n)) // 2
    
    # First, find a good initial solution
    best_adj = None
    best_count = 0
    
    for restart in range(30):
        if time.time() > deadline - 30.0:
            break
        adj = greedy_construction(restart * 12345 + 67890)
        adj = local_search(adj)
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = [x | 0 for x in adj]
    
    # Now use branch and bound to try to improve
    # Upper bound: for each vertex, max degree is limited by K2,3 constraint
    # A vertex v with degree d can have at most 2 common neighbors with any other vertex
    # This limits the structure significantly
    
    # Branch and bound: try adding edges that are missing
    def branch_and_bound(adj, best_count_ref):
        """Try to find a better solution using branch and bound."""
        nonlocal best_adj, best_count
        if time.time() > deadline:
            return
        
        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = [x | 0 for x in adj]
        
        # Find all missing edges
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1):
                    missing.append((u, v))
        
        if not missing:
            return
        
        # Sort missing edges by some heuristic (e.g., impact on constraints)
        # Try edges that are "least constraining" first
        def edge_score(e):
            u, v = e
            # Count how many vertices w would be affected
            score = 0
            for w in range(n):
                if w == u or w == v:
                    continue
                cuw = count_common_neighbors(adj, u, w)
                cvw = count_common_neighbors(adj, v, w)
                # If adding this edge would push any pair to 3 common neighbors, it's bad
                if cuw + (1 if (adj[w] >> v) & 1 else 0) >= 3:
                    score -= 10
                if cvw + (1 if (adj[w] >> u) & 1 else 0) >= 3:
                    score -= 10
            return score
        
        missing.sort(key=edge_score, reverse=True)
        
        # Try adding the first few missing edges
        for u, v in missing[:20]:  # Limit to keep time bounded
            if time.time() > deadline:
                return
            
            # Check if we can add this edge
            can_add = True
            for w in range(n):
                if w == u or w == v:
                    continue
                cuw = count_common_neighbors(adj, u, w)
                if cuw + (1 if (adj[w] >> v) & 1 else 0) >= 3:
                    can_add = False
                    break
                cvw = count_common_neighbors(adj, v, w)
                if cvw + (1 if (adj[w] >> u) & 1 else 0) >= 3:
                    can_add = False
                    break
            
            if can_add:
                # Add edge and recurse
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                branch_and_bound(adj, best_count)
                # Backtrack
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
    
    if best_adj is not None:
        branch_and_bound(best_adj, best_count)
    
    # Final local search
    if best_adj is not None:
        best_adj = local_search(best_adj)
        best_count = count_edges(best_adj)
    
    # Convert best_adj to edge list
    if best_adj is None:
        # Fallback
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
    
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
