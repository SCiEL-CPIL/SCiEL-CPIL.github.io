import time
import random
import itertools
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    n = 16
    all_edges = list(itertools.combinations(range(n), 2))
    m = len(all_edges)  # 120 edges

    # Edge index
    edge_index = {}
    for i, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    # Q3 structure: 8 vertices, 12 edges
    q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    q3_adj = {
        0: [1, 2, 4],
        1: [0, 3, 5],
        2: [0, 3, 6],
        3: [1, 2, 7],
        4: [0, 5, 6],
        5: [1, 4, 7],
        6: [2, 4, 7],
        7: [3, 5, 6]
    }

    # Precompute all Q3 embeddings: for each edge pair (e1, e2) in the graph,
    # determine if they can be part of the same Q3.
    # A pair of edges (u,v) and (x,y) conflict if there exists a Q3 embedding
    # where both edges are present and the other 10 edges can also be present.
    
    # Better: precompute for each pair of edges whether they are "Q3-compatible"
    # i.e., whether adding both edges to an empty graph could be part of a Q3.
    
    # Actually, we need: two edges conflict if there's a Q3 containing both.
    # For each Q3 embedding (injective map from {0..7} to {0..15}), the 12 edges
    # of Q3 all mutually conflict.
    
    # Number of embeddings: 16! / 8! = 518,918,400 -- too many to enumerate all.
    # But we only need to know which pairs of graph edges appear together in some Q3.
    
    # Alternative: for each pair of graph edges (e1, e2), check if they can coexist in a Q3.
    # e1 = (u1, v1), e2 = (u2, v2). They can be in the same Q3 iff there's an injective
    # map f: {0..7} -> {0..15} such that {f(0),f(1),...,f(7)} contains {u1,v1} and {u2,v2},
    # and the Q3 edge structure is preserved.
    
    # This is still complex. Let's use a different approach:
    # Precompute the conflict matrix using the fact that Q3 has specific structure.
    
    # For each pair of edges, check if they form a path of length 1, 2, or 3 in Q3,
    # or are disjoint, etc. Actually, let's just precompute by checking all 24 pairs
    # of Q3 edges and seeing which graph edge pairs can realize them.
    
    # Simpler approach: use the current creates_q3_fast logic but precompute a
    # "conflict" bitset for each edge: conflict[i] is a bitmask of edges that conflict with edge i.
    # Two edges conflict if adding both to some graph would create a Q3 where both are present.
    
    # Actually, the correct formulation for maximum independent set:
    # We want the largest set of edges S such that S contains no Q3.
    # This is NOT the same as no two edges in S conflict pairwise.
    # A Q3 requires 12 specific edges. So the "conflict" is not pairwise but 12-wise.
    
    # The branch-and-bound on independent set won't work directly because the constraint
    # is not pairwise. We need a different approach.
    
    # Let's stick with a smarter greedy + local search, but with much faster Q3 checking.
    
    # Precompute for each edge (u,v) and each Q3 edge position, the set of "witness" embeddings.
    
    # Fast Q3 check: when considering adding edge (u,v), check if there's a Q3 where
    # (u,v) is one of the 12 edges. Precompute for each (u,v) the list of partial embeddings.
    
    # For speed, use bitmasks and precomputed neighbor lists.
    
    # Let's use a different strategy: start from a known good structure and optimize.
    # Known: K_{3,13} + matching gives 42 edges. Can we do better?
    # Try K_{2,14} + internal edges in both parts, or other structures.
    
    # Strategy: Use simulated annealing / hill climbing with fast Q3 detection.
    
    def q3_check_bitset(adj_bits, u, v):
        """Check if adding edge (u,v) creates a Q3. Returns True if Q3 is created."""
        # For each of the 12 Q3 edges, try to embed it as (u,v)
        for i, j in q3_edges:
            # Map Q3 vertex i -> u, Q3 vertex j -> v
            if _embed_check(i, j, u, v, adj_bits, q3_adj, n):
                return True
            # Map Q3 vertex i -> v, Q3 vertex j -> u  
            if _embed_check(j, i, v, u, adj_bits, q3_adj, n):
                return True
        return False

    def _embed_check(q_i, q_j, g_u, g_v, adj_bits, q3_adj, n):
        """Check if Q3 can be embedded with q_i->g_u, q_j->g_v."""
        mapping = [-1] * 8
        mapping[q_i] = g_u
        mapping[q_j] = g_v
        used = [False] * n
        used[g_u] = True
        used[g_v] = True
        
        # Order remaining Q3 vertices by number of already-mapped neighbors (MRV heuristic)
        remaining = [x for x in range(8) if x != q_i and x != q_j]
        
        def get_mapped_neighbors(qv):
            mapped = []
            for nb in q3_adj[qv]:
                if mapping[nb] != -1:
                    mapped.append(mapping[nb])
            return mapped
        
        # Sort by descending number of mapped neighbors
        remaining.sort(key=lambda x: -len(get_mapped_neighbors(x)))
        
        def backtrack(idx):
            if idx == len(remaining):
                # Verify all Q3 edges
                for a, b in q3_edges:
                    ga = mapping[a]
                    gb = mapping[b]
                    if (ga == g_u and gb == g_v) or (ga == g_v and gb == g_u):
                        continue
                    if not (adj_bits[ga] & (1 << gb)):
                        return False
                return True
            
            qv = remaining[idx]
            mapped_nbs = get_mapped_neighbors(qv)
            
            # Candidates: unused graph vertices adjacent to all mapped neighbors
            candidates = []
            for gc in range(n):
                if used[gc]:
                    continue
                ok = True
                for gn in mapped_nbs:
                    if (gc == g_u and gn == g_v) or (gc == g_v and gn == g_u):
                        continue
                    if not (adj_bits[gc] & (1 << gn)):
                        ok = False
                        break
                if ok:
                    candidates.append(gc)
            
            for gc in candidates:
                mapping[qv] = gc
                used[gc] = True
                if backtrack(idx + 1):
                    return True
                mapping[qv] = -1
                used[gc] = False
            
            return False
        
        return backtrack(0)

    # Simulated annealing / hill climbing
    best_count = 0
    best_adj = [0] * n
    
    def count_edges(adj_bits):
        return sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2

    # Initialize with K_{3,13}
    def init_k313():
        adj = [0] * n
        for u in range(3):
            for v in range(3, 16):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    # Try multiple random initializations and hill climb
    for trial in range(50):
        if time.time() > deadline:
            break
        
        random.seed(trial * 104729 + 7)
        
        if trial % 5 == 0:
            adj = init_k313()
        else:
            # Random sparse graph
            adj = [0] * n
            for u, v in all_edges:
                if random.random() < 0.3:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Greedy add
        edges_to_try = all_edges[:]
        random.shuffle(edges_to_try)
        for u, v in edges_to_try:
            if time.time() > deadline:
                break
            if not (adj[u] & (1 << v)):
                if not q3_check_bitset(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        # Hill climbing: try adding edges, and 1-swap (remove one, add another)
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding missing edges
            for u, v in all_edges:
                if not (adj[u] & (1 << v)):
                    if not q3_check_bitset(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        improved = True
            
            if not improved:
                # Try 1-swap: remove edge e1, add edge e2
                current_count = count_edges(adj)
                existing = [(u, v) for u, v in all_edges if adj[u] & (1 << v)]
                random.shuffle(existing)
                
                for e1 in existing:
                    if time.time() > deadline:
                        break
                    u1, v1 = e1
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    
                    for u2, v2 in all_edges:
                        if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                            continue
                        if not (adj[u2] & (1 << v2)):
                            if not q3_check_bitset(adj, u2, v2):
                                adj[u2] |= (1 << v2)
                                adj[v2] |= (1 << u2)
                                new_count = count_edges(adj)
                                if new_count > current_count:
                                    improved = True
                                    break
                                else:
                                    # Undo
                                    adj[u2] &= ~(1 << v2)
                                    adj[v2] &= ~(1 << u2)
                    
                    # Restore e1
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    
                    if improved:
                        break
            
            if not improved:
                break
        
        count = count_edges(adj)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Convert to edge list
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i] & (1 << j):
                result.append([i, j])
    
    return result
