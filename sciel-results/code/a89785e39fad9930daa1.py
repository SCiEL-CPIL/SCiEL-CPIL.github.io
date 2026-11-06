import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    N = 16
    # Represent graph as adjacency bitmasks
    # adj[i] is a 16-bit integer where bit j is set if edge (i,j) exists
    
    def is_q3_free(adj):
        """Check if graph contains a Q3 subgraph.
        Q3 is the 3-cube: 8 vertices, each degree 3, bipartite with parts of size 4.
        We check all 8-subsets for being a Q3.
        """
        # Q3 structure: vertices can be labeled 0..7 as 3-bit numbers.
        # Two vertices are adjacent iff they differ in exactly one bit.
        # So Q3 has edges between pairs with Hamming distance 1.
        # 
        # A set of 8 vertices forms a Q3 if:
        # - Each vertex has exactly 3 neighbors within the set
        # - The graph is connected
        # - It's bipartite (which follows from 3-regular + 8 vertices + connected for this case)
        # 
        # Actually, a 3-regular bipartite graph on 8 vertices with 4+4 partition IS Q3
        # (since there's only one such graph up to isomorphism).
        #
        # So we need: 8 vertices, each with exactly 3 neighbors in the set, and the induced
        # subgraph is bipartite.
        
        # Precompute: for each 8-subset, check if it induces a Q3
        # But that's C(16,8) = 12870 subsets, each check is O(8*16) = manageable
        
        vertices = list(range(N))
        # Use combinations
        from itertools import combinations
        for subset in combinations(vertices, 8):
            sset = set(subset)
            # Check each vertex in subset has exactly 3 neighbors in subset
            ok = True
            for v in subset:
                neighbors_in_subset = 0
                adj_v = adj[v]
                for u in subset:
                    if u != v and (adj_v >> u) & 1:
                        neighbors_in_subset += 1
                if neighbors_in_subset != 3:
                    ok = False
                    break
            if not ok:
                continue
            # Now check bipartite: try 2-coloring
            # Build adjacency within subset
            color = {}
            from collections import deque
            found = False
            for v in subset:
                if v in color:
                    continue
                color[v] = 0
                q = deque([v])
                while q:
                    cur = q.popleft()
                    adj_cur = adj[cur]
                    for u in subset:
                        if u == cur:
                            continue
                        if (adj_cur >> u) & 1:
                            if u not in color:
                                color[u] = 1 - color[cur]
                                q.append(u)
                            elif color[u] == color[cur]:
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if not found:
                # It's a Q3!
                return False
        return True
    
    def count_edges(adj):
        cnt = 0
        for i in range(N):
            cnt += bin(adj[i]).count('1')
        return cnt // 2
    
    def try_remove_edge(adj, u, v):
        """Remove edge (u,v) and check if still Q3-free. Return new adj or None."""
        new_adj = adj.copy()
        new_adj[u] &= ~(1 << v)
        new_adj[v] &= ~(1 << u)
        if is_q3_free(new_adj):
            return new_adj
        return None
    
    def try_add_edge(adj, u, v):
        """Try adding edge (u,v). Return new adj if Q3-free, else None."""
        if (adj[u] >> v) & 1:
            return adj  # already exists
        new_adj = adj.copy()
        new_adj[u] |= (1 << v)
        new_adj[v] |= (1 << u)
        if is_q3_free(new_adj):
            return new_adj
        return None
    
    # Start with a dense graph: complete graph minus some edges
    # Let's start with K16 and remove edges until Q3-free
    # Better: start with a structured dense graph
    
    # Strategy: Start with random dense graph, then use local search
    # Phase 1: Build initial Q3-free graph greedily
    random.seed(42)
    
    # Start with empty graph and add edges greedily
    adj = [0] * N
    # Try all edges in random order, add if Q3-free
    all_edges = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() > deadline:
            break
        new_adj = try_add_edge(adj, u, v)
        if new_adj is not None:
            adj = new_adj
    
    best_adj = adj.copy()
    best_count = count_edges(adj)
    
    # Phase 2: Local search - try to add edges, and do 1-remove-1-add swaps
    while time.time() < deadline - 5:
        improved = False
        
        # Try adding any missing edge
        for u in range(N):
            for v in range(u+1, N):
                if time.time() > deadline:
                    break
                if not ((adj[u] >> v) & 1):
                    new_adj = try_add_edge(adj, u, v)
                    if new_adj is not None:
                        adj = new_adj
                        best_adj = adj.copy()
                        best_count = count_edges(adj)
                        improved = True
        
        if improved:
            continue
        
        # Try 1-remove-1-add swaps
        found_swap = False
        for u in range(N):
            for v in range(u+1, N):
                if time.time() > deadline:
                    break
                if not ((adj[u] >> v) & 1):
                    continue  # only consider adding this edge
                # Try removing some existing edge and adding (u,v)
                for w in range(N):
                    for x in range(w+1, N):
                        if time.time() > deadline:
                            break
                        if ((adj[w] >> x) & 1) and (w != u and x != u and w != v and x != v):
                            # Try: remove (w,x), add (u,v)
                            new_adj = adj.copy()
                            new_adj[w] &= ~(1 << x)
                            new_adj[x] &= ~(1 << w)
                            new_adj[u] |= (1 << v)
                            new_adj[v] |= (1 << u)
                            if is_q3_free(new_adj):
                                adj = new_adj
                                best_adj = adj.copy()
                                best_count = count_edges(adj)
                                found_swap = True
                                break
                    if found_swap:
                        break
                if found_swap:
                    break
            if found_swap:
                break
        
        if not found_swap and not improved:
            # Restart with random perturbation
            if time.time() > deadline - 10:
                break
            # Random restart: take best, remove a few random edges, try to rebuild
            adj = best_adj.copy()
            # Remove 2-3 random edges
            existing = [(i,j) for i in range(N) for j in range(i+1,N) if (adj[i]>>j)&1]
            if len(existing) > 5:
                to_remove = random.sample(existing, min(3, len(existing)))
                for w, x in to_remove:
                    adj[w] &= ~(1 << x)
                    adj[x] &= ~(1 << w)
    
    # Extract edges
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    
    # Use best if current is worse
    if count_edges(best_adj) > count_edges(adj):
        adj = best_adj
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
    
    return edges
