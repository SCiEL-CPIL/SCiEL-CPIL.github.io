"""Fewest four-cycles in graphs with 31 vertices and 91 edges."""
import time
import random
import numpy as np

N = 31
M = 91
TIME_LIMIT = 115  # safety margin before 120s kill

def count_c4(adj):
    """Count number of C4 subgraphs in graph with adjacency matrix adj."""
    # A C4 is a set of 4 vertices {a,b,c,d} with edges ab, bc, cd, da (in some order)
    # Count via: for each pair of vertices (u,v), count common neighbors.
    # A C4 corresponds to 2 pairs of opposite vertices with 2 common neighbors each.
    # Actually: number of C4 = sum over all pairs {u,v} of C(common_neighbors(u,v), 2) / 2
    # Wait, let me think more carefully.
    # A 4-cycle a-b-c-d-a: the diagonals are (a,c) and (b,d).
    # For diagonal pair (a,c), common neighbors include b and d (and possibly others).
    # Each C4 is counted exactly once by choosing its two diagonal pairs.
    # But each C4 has 2 ways to pick which pair is the "first" diagonal.
    # Actually no: a C4 a-b-c-d-a has diagonals (a,c) and (b,d).
    # For pair (a,c): common neighbors are b, d, and any other vertex connected to both a and c.
    # The number of C4s where (a,c) is a diagonal = C(common(a,c), 2)
    # But this counts each C4 twice (once for each diagonal pair).
    # So total C4 = (1/2) * sum_{u<v} C(common(u,v), 2)
    
    # common(u,v) = number of w such that adj[u,w]=1 and adj[v,w]=1
    # = (adj[u,:] . adj[v,:]) - adj[u,v] (subtract the edge between u and v if present)
    
    c4_count = 0
    for u in range(N):
        for v in range(u+1, N):
            cn = int(adj[u] @ adj[v]) - (1 if adj[u,v] else 0)
            if cn >= 2:
                c4_count += cn * (cn - 1) // 2
    return c4_count // 2

def count_c4_fast(adj):
    """Faster C4 count using matrix operations."""
    # adj is NxN numpy array of 0s and 1s
    # common(u,v) = (adj^2)[u,v] - adj[u,v]
    # Actually (adj @ adj)[u,v] = sum_w adj[u,w]*adj[w,v] = common neighbors of u and v (including v itself if adj[u,v]=1? No)
    # (adj @ adj)[u,v] counts paths of length 2 from u to v, which equals number of common neighbors.
    # This does NOT include u->v directly, it's intermediate vertices.
    # So common(u,v) = (adj @ adj)[u,v]
    # Wait: (adj @ adj)[u,v] = sum_w adj[u,w] * adj[w,v]. If w=v, adj[u,v]*adj[v,v]=0 (no self-loops).
    # If w=u, adj[u,u]*adj[u,v]=0.
    # So (adj@adj)[u,v] = number of w != u,v with adj[u,w] and adj[w,v] = common neighbors.
    
    A2 = adj @ adj  # NxN matrix
    # For each pair (u,v), common = A2[u,v]
    # C4 count = (1/2) * sum_{u<v} C(A2[u,v], 2)
    # C(x,2) = x*(x-1)/2
    # We need sum over u<v of A2[u,v]*(A2[u,v]-1)/2, then divide by 2.
    
    # Get upper triangle
    triu = np.triu(A2, 1)
    vals = triu[triu > 0].astype(np.float64)
    if len(vals) == 0:
        return 0
    c4 = np.sum(vals * (vals - 1)) / 4.0
    return int(round(c4))

def construct():
    start_time = time.time()
    
    # Strategy: Start with a random graph with 91 edges, then use local search
    # to minimize C4 count. Also try to find a near-C4-free subgraph of 90 edges
    # and add the 91st edge optimally.
    
    best_edges = None
    best_c4 = float('inf')
    
    # Helper to convert edge list to adjacency matrix
    def edges_to_adj(edges):
        adj = np.zeros((N, N), dtype=np.int32)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
        return adj
    
    def adj_to_edges(adj):
        edges = []
        for u in range(N):
            for v in range(u+1, N):
                if adj[u, v]:
                    edges.append([u, v])
        return edges
    
    # Strategy 1: Random init + greedy improvement
    # Try multiple random starts
    rng = np.random.RandomState(42)
    
    for trial in range(50):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Random graph with 91 edges
        all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
        rng.shuffle(all_pairs)
        edges = all_pairs[:M]
        adj = edges_to_adj(edges)
        
        c4 = count_c4_fast(adj)
        if c4 < best_c4:
            best_c4 = c4
            best_edges = [list(e) for e in edges]
        
        # Local search: try edge swaps to reduce C4
        improved = True
        while improved and time.time() - start_time < TIME_LIMIT - 5:
            improved = False
            # Try random edge swaps
            for _ in range(200):
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                # Pick a random edge to remove and a random non-edge to add
                edge_list = adj_to_edges(adj)
                if len(edge_list) != M:
                    break
                rem_idx = rng.randint(0, len(edge_list)-1)
                u1, v1 = edge_list[rem_idx]
                
                # Find a non-edge to add
                # Sample a random pair until we find a non-edge
                for _ in range(50):
                    u2 = rng.randint(0, N-1)
                    v2 = rng.randint(0, N-1)
                    if u2 == v2:
                        continue
                    if u2 > v2:
                        u2, v2 = v2, u2
                    if adj[u2, v2] == 0:
                        break
                else:
                    continue
                
                # Perform swap
                adj[u1, v1] = 0
                adj[v1, u1] = 0
                adj[u2, v2] = 1
                adj[v2, u2] = 1
                
                new_c4 = count_c4_fast(adj)
                if new_c4 < c4:
                    c4 = new_c4
                    improved = True
                    if c4 < best_c4:
                        best_c4 = c4
                        best_edges = adj_to_edges(adj)
                else:
                    # Revert
                    adj[u1, v1] = 1
                    adj[v1, u1] = 1
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0
        
        if best_c4 <= 1:
            break
    
    # Strategy 2: Try to build a C4-free (or near C4-free) 90-edge graph, then add 91st edge
    if time.time() - start_time < TIME_LIMIT - 10:
        # Use greedy construction to build a C4-free graph with 90 edges
        # Start empty, add edges that don't create C4s
        for attempt in range(10):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            adj = np.zeros((N, N), dtype=np.int32)
            edge_count = 0
            # Random order of edges
            all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
            rng.shuffle(all_pairs)
            
            for u, v in all_pairs:
                if edge_count >= 90:
                    break
                if time.time() - start_time > TIME_LIMIT - 10:
                    break
                # Check if adding (u,v) creates a C4
                # A C4 is created if there exists w1, w2 such that
                # u-w1-v-w2-u or similar. Specifically, if there's a vertex w
                # adjacent to both u and v, and another vertex w' adjacent to both u and v,
                # then u-w-v-w'-u is a C4 (if w != w').
                # More precisely: adding edge (u,v) creates a C4 iff there are at least 2
                # common neighbors of u and v in the current graph.
                cn = int(adj[u] @ adj[v])
                if cn < 2:
                    adj[u, v] = 1
                    adj[v, u] = 1
                    edge_count += 1
            
            if edge_count >= 90:
                # We have a C4-free graph with 90 edges
                # Now add the 91st edge to minimize added C4s
                # Adding edge (u,v) adds C(common(u,v), 2) new C4s
                best_add_c4 = float('inf')
                best_add_edge = None
                for u, v in all_pairs:
                    if adj[u, v] == 0:
                        cn = int(adj[u] @ adj[v])
                        added = cn * (cn - 1) // 2
                        if added < best_add_c4:
                            best_add_c4 = added
                            best_add_edge = (u, v)
                
                if best_add_edge is not None:
                    u, v = best_add_edge
                    adj[u, v] = 1
                    adj[v, u] = 1
                    
                    c4 = count_c4_fast(adj)
                    if c4 < best_c4:
                        best_c4 = c4
                        best_edges = adj_to_edges(adj)
                    
                    if best_c4 <= 1:
                        break
    
    # Strategy 3: If we still have time, do more local search from best solution
    if best_edges is not None and time.time() - start_time < TIME_LIMIT - 10:
        adj = edges_to_adj(best_edges)
        c4 = count_c4_fast(adj)
        
        # More aggressive local search
        for iteration in range(200):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            if c4 == 0:
                break
            
            improved = False
            # Try all possible single edge swaps (more systematic)
            edge_list = adj_to_edges(adj)
            non_edges = [(u, v) for u in range(N) for v in range(u+1, N) if adj[u, v] == 0]
            
            # Sample a subset of swaps for speed
            n_swaps = min(500, len(edge_list) * len(non_edges) // 10)
            for _ in range(n_swaps):
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                rem = edge_list[rng.randint(0, len(edge_list)-1)]
                add = non_edges[rng.randint(0, len(non_edges)-1)]
                u1, v1 = rem
                u2, v2 = add
                
                # Quick check: if u2,v2 is same as u1,v1 skip
                if (u1 == u2 and v1 == v2):
                    continue
                
                adj[u1, v1] = 0
                adj[v1, u1] = 0
                adj[u2, v2] = 1
                adj[v2, u2] = 1
                
                new_c4 = count_c4_fast(adj)
                if new_c4 < c4:
                    c4 = new_c4
                    improved = True
                    if c4 < best_c4:
                        best_c4 = c4
                        best_edges = adj_to_edges(adj)
                        # Rebuild non_edges
                        non_edges = [(u, v) for u in range(N) for v in range(u+1, N) if adj[u, v] == 0]
                        edge_list = adj_to_edges(adj)
                else:
                    adj[u1, v1] = 1
                    adj[v1, u1] = 1
                    adj[u2, v2] = 0
                    adj[v2, u2] = 0
            
            if not improved:
                # Try a random restart perturbation
                for _ in range(10):
                    rem = edge_list[rng.randint(0, len(edge_list)-1)]
                    add = non_edges[rng.randint(0, len(non_edges)-1)]
                    if (rem[0], rem[1]) == (add[0], add[1]):
                        continue
                    adj[rem[0], rem[1]] = 0
                    adj[rem[1], rem[0]] = 0
                    adj[add[0], add[1]] = 1
                    adj[add[1], add[0]] = 1
                    c4 = count_c4_fast(adj)
                    edge_list = adj_to_edges(adj)
                    non_edges = [(u, v) for u in range(N) for v in range(u+1, N) if adj[u, v] == 0]
                    if c4 < best_c4:
                        best_c4 = c4
                        best_edges = adj_to_edges(adj)
                    break
    
    # Final: ensure we return exactly 91 distinct edges
    if best_edges is None:
        # Fallback: first 91 lexicographic pairs
        best_edges = [[i, j] for i in range(N) for j in range(i+1, N)][:M]
    
    # Verify
    assert len(best_edges) == M
    assert len(set(tuple(sorted(e)) for e in best_edges)) == M
    for e in best_edges:
        assert 0 <= e[0] < N and 0 <= e[1] < N and e[0] != e[1]
    
    return best_edges

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
