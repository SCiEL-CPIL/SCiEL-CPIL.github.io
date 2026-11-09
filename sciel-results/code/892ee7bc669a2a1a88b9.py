import time
import random
import numpy as np

def construct():
    start = time.time()
    deadline = start + 135.0
    N = 40

    def _creates_c6(adj, u, v):
        """Check if adding edge (u,v) to current adj (without u,v) creates a C6.
        adj is a list of lists of ints (neighbors). u,v are not connected in adj."""
        # We need a path of length 5 from u to v in adj.
        # BFS with depth limit, tracking visited.
        # For speed, use a queue with (node, depth).
        # But we also need to avoid cycles within the path.
        # Actually, a simple path of length 5.
        # Given small N, we can do DFS with backtracking.
        # But for speed, let's use a different approach:
        # Count paths of length 5? No, we just need existence.
        # Let's use BFS up to depth 5, but we need simple paths.
        # Alternative: use the fact that N=40, so we can enumerate.
        # Actually, a faster method: for each common neighbor of u, do BFS of depth 4 to v?
        # No, that's for paths of length 3.
        
        # Let's just do a DFS with memoization of (node, depth, visited_set).
        # But visited_set is expensive.
        
        # Better: since we're checking if adding (u,v) creates a C6,
        # we need a path u -> a -> b -> c -> d -> v of length 5 in adj.
        # We can iterate over pairs of neighbors of u and v and check paths of length 3 between them?
        # No, that's for C4.
        
        # Let's do a simple BFS/DFS. With N=40 and max depth 5, it's manageable.
        # We'll use a queue: (current_node, depth, visited_tuple)
        # But visited_tuple is a frozenset, which is expensive.
        
        # Alternative: use a recursive DFS with a list for the path.
        def dfs(node, depth, path):
            if depth == 5:
                return node == v
            for nxt in adj[node]:
                if nxt not in path:
                    if nxt == v and depth == 4:
                        return True
                    if nxt != v:  # don't go to v before depth 4
                        path.append(nxt)
                        if dfs(nxt, depth + 1, path):
                            return True
                        path.pop()
            return False
        
        path = [u]
        return dfs(u, 0, path)
    
    def _creates_c6_fast(adj, u, v):
        """Faster C6 check using BFS with depth tracking."""
        # BFS from u, tracking shortest path length to each node.
        # If we reach v at depth 5, return True.
        # But we need simple paths, so BFS gives shortest path.
        # If shortest path from u to v (without edge uv) is exactly 5, then C6 exists.
        # But also if there's a non-shortest path of length 5? 
        # Actually, for C6, we need a path of length exactly 5 (simple).
        # BFS gives shortest path. If shortest path < 5, no C6 via this edge? 
        # No, there could be a longer simple path.
        # Hmm, this is tricky. Let's just use DFS for correctness.
        # Given N=40, DFS with depth 5 should be fast enough.
        # Let's optimize the DFS.
        
        # Actually, let's try a different approach:
        # A C6 containing edge (u,v) is equivalent to a path of length 5 from u to v.
        # We can use dynamic programming or just enumerate.
        
        # For now, let's use a simple BFS that tracks all paths up to depth 5.
        # Use a list of sets: level[i] = set of nodes reachable by a simple path of length i from u.
        # But "simple" means no repeated nodes, which is hard to track with just sets.
        
        # Let's just use the DFS approach but with some pruning.
        def dfs_fast(node, depth, visited):
            if depth == 5:
                return node == v
            for nxt in adj[node]:
                if not visited[nxt]:
                    if nxt == v and depth == 4:
                        return True
                    if nxt != v:
                        visited[nxt] = True
                        if dfs_fast(nxt, depth + 1, visited):
                            return True
                        visited[nxt] = False
            return False
        
        visited = [False] * N
        visited[u] = True
        return dfs_fast(u, 0, visited)
    
    def _edge_count(adj):
        return sum(len(nei) for nei in adj) // 2
    
    def _greedy_build(seed_adj, time_budget):
        """Try to add edges greedily to seed_adj."""
        adj = [list(nei) for nei in seed_adj]
        m = _edge_count(adj)
        
        # Try all non-edges in random order
        non_edges = [(u, v) for u in range(N) for v in range(u+1, N) if v not in adj[u]]
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > time_budget:
                break
            # Check if adding (u,v) creates a C6
            # Temporarily remove from consideration: check path of length 5 from u to v in current adj
            if not _creates_c6_fast(adj, u, v):
                adj[u].append(v)
                adj[v].append(u)
                m += 1
        
        return adj, m
    
    def _local_search(adj, m, time_budget):
        """Local search: try to add one edge, or swap."""
        best_adj = [list(nei) for nei in adj]
        best_m = m
        
        while time.time() < time_budget:
            # Try to add a random non-edge
            non_edges = [(u, v) for u in range(N) for v in range(u+1, N) if v not in adj[u]]
            if not non_edges:
                break
            
            improved = False
            for u, v in non_edges:
                if time.time() > time_budget:
                    break
                if not _creates_c6_fast(adj, u, v):
                    adj[u].append(v)
                    adj[v].append(u)
                    m += 1
                    improved = True
                    if m > best_m:
                        best_m = m
                        best_adj = [list(nei) for nei in adj]
            
            if not improved:
                # Try 1-edge removal + 2-edge addition (net +1)
                found_swap = False
                edges = [(u, v) for u in range(N) for v in adj[u] if u < v]
                random.shuffle(edges)
                
                for u_r, v_r in edges[:20]:  # limit swaps to try
                    if time.time() > time_budget:
                        break
                    # Remove (u_r, v_r)
                    adj[u_r].remove(v_r)
                    adj[v_r].remove(u_r)
                    m -= 1
                    
                    # Try to add 2 edges
                    non_edges = [(u, v) for u in range(N) for v in range(u+1, N) if v not in adj[u]]
                    random.shuffle(non_edges)
                    
                    added = 0
                    for u_a, v_a in non_edges:
                        if time.time() > time_budget:
                            break
                        if not _creates_c6_fast(adj, u_a, v_a):
                            adj[u_a].append(v_a)
                            adj[v_a].append(u_a)
                            m += 1
                            added += 1
                            if added == 2:
                                break
                    
                    if m > best_m:
                        best_m = m
                        best_adj = [list(nei) for nei in adj]
                    found_swap = True
                    
                    # Revert
                    for u_a, v_a in non_edges[:added]:
                        if v_a in adj[u_a]:
                            adj[u_a].remove(v_a)
                            adj[v_a].remove(u_a)
                    adj[u_r].append(v_r)
                    adj[v_r].append(u_r)
                    m += 1
                
                if not found_swap:
                    break
        
        return best_adj, best_m
    
    def _random_seed_adj():
        """Create a random sparse graph as seed."""
        adj = [[] for _ in range(N)]
        # Add random edges with probability ~0.15
        for u in range(N):
            for v in range(u+1, N):
                if random.random() < 0.15:
                    adj[u].append(v)
                    adj[v].append(u)
        return adj
    
    def _structured_seed_adj():
        """Create a structured seed: e.g., a random regular-ish graph."""
        adj = [[] for _ in range(N)]
        # Try to make it roughly 5-regular
        for u in range(N):
            targets = [v for v in range(N) if v != u and v not in adj[u]]
            random.shuffle(targets)
            for v in targets[:5]:
                if u not in adj[v]:
                    adj[u].append(v)
                    adj[v].append(u)
        return adj
    
    best_adj = None
    best_m = 0
    
    # Multiple restarts with different seeds
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Choose seed strategy
        if restart % 3 == 0:
            seed = _random_seed_adj()
        elif restart % 3 == 1:
            seed = _structured_seed_adj()
        else:
            seed = _random_seed_adj()
            # Perturb
            for _ in range(5):
                u = random.randint(0, N-1)
                v = random.randint(0, N-1)
                if u != v and v in adj[u]:
                    adj[u].remove(v)
                    adj[v].remove(u)
        
        # Greedy build
        t1 = min(time.time() + 1.0, deadline - 2.0)
        adj, m = _greedy_build(seed, t1)
        
        # Local search
        t2 = min(time.time() + 2.0, deadline - 1.0)
        adj, m = _local_search(adj, m, t2)
        
        if m > best_m:
            best_m = m
            best_adj = [list(nei) for nei in adj]
    
    # Final answer
    if best_adj is None:
        # Fallback: empty graph
        return []
    
    edges = []
    for u in range(N):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    
    return edges
