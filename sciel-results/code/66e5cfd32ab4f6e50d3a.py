"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _get_degrees(n, edges):
    """Compute degree of each vertex."""
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return deg

def _is_valid(n, edges):
    """Check if no 3-edge path has endpoints of equal degree."""
    if not edges:
        return True
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deg = [0] * n
    for u in range(n):
        deg[u] = len(adj[u])
    
    # Check all 3-edge paths: s-u-v-t where s != t
    # A 3-edge path is s-u-v-t with edges (s,u), (u,v), (v,t)
    # We need to check that deg(s) != deg(t) for all such paths
    for v in range(n):
        for u in adj[v]:
            for t in adj[v]:
                if t == u:
                    continue
                # s is a neighbor of u (s != v to avoid backtracking, but s can be t? No, s != t)
                # Actually, s is any neighbor of u except v (to make it a simple path)
                # But the problem says "simple three-edge path", so s, u, v, t must be distinct
                # Wait, re-reading: "every path s-u-v-t on edge (u, v) with s != t"
                # So s is adjacent to u, t is adjacent to v, s != t, and the path is simple
                # For simplicity, let's check: for each edge (u,v), for each s in adj[u] (s != v), 
                # for each t in adj[v] (t != u), if s != t, then deg[s] != deg[t]
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u:
                            continue
                        if s == t:
                            continue
                        # Check if s-u-v-t is a simple path (all vertices distinct)
                        # s != u (since s is neighbor of u and no loops), s != v (we skipped), 
                        # u != v (no loops), s != t (checked), u != t (t != u checked), v != t (t is neighbor of v, no loops)
                        # So we just need deg[s] != deg[t]
                        if deg[s] == deg[t]:
                            return False
    return True

def _count_edges(edges):
    return len(edges)

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # All possible edges
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i, j))
    
    best_edges = []
    best_count = 0
    
    # Strategy 1: Start from complete graph and remove edges
    # Strategy 2: Greedy addition
    # Strategy 3: Random restarts with local search
    
    # First, try a greedy approach: add edges in random order, keep if valid
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        random.shuffle(all_edges)
        current_edges = []
        adj = [set() for _ in range(n)]
        
        for u, v in all_edges:
            # Try adding edge (u, v)
            new_edges = current_edges + [(u, v)]
            if _is_valid(n, new_edges):
                current_edges = new_edges
                adj[u].add(v)
                adj[v].add(u)
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
        
        if best_count >= 100:  # K15 has 105 edges, so 100 is very good
            break
    
    # Local search: try to add back removed edges or swap
    if best_edges and time.time() < deadline - 10:
        current = list(best_edges)
        adj = [set() for _ in range(n)]
        for u, v in current:
            adj[u].add(v)
            adj[v].add(u)
        
        # Try adding each missing edge
        missing = set(all_edges) - set(current)
        for u, v in list(missing):
            if time.time() > deadline - 5:
                break
            new_edges = current + [(u, v)]
            if _is_valid(n, new_edges):
                current = new_edges
                adj[u].add(v)
                adj[v].add(u)
        
        if len(current) > best_count:
            best_count = len(current)
            best_edges = current
    
    # Convert to list of lists
    result = [[u, v] for u, v in best_edges]
    
    # Ensure types are correct
    return result

# EVOLVE-BLOCK-END
