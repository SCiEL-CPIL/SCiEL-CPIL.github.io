"""Combined approach: 3-partite Turán graph with vertex-swap local search."""
import time
import random
import math
import sys

def construct():
    """Construct a K4-free graph on 18 vertices minimizing the max triangle-free induced subgraph."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N = 18
    full_mask = (1 << N) - 1
    
    # Helper: Check if a bitmask represents an independent set in graph defined by adj (list of bitmasks)
    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True
    
    # Helper: Check if adding edge (u,v) keeps graph K4-free
    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)
    
    # Exact B&B for largest triangle-free induced subgraph
    def largest_triangle_free(adj, time_limit=None):
        best = [0]
        
        # Greedy heuristic to get a good initial lower bound
        for _ in range(10):
            order = list(range(N))
            random.shuffle(order)
            S = 0
            for v in order:
                if is_independent(adj[v] & S, adj):
                    S |= (1 << v)
            sz = bin(S).count('1')
            if sz > best[0]:
                best[0] = sz
        
        if best[0] == N:
            return N
        
        # Branch and bound
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = bin(S_mask).count('1')
            
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return
            
            new_C = 0
            c = C_mask
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                if is_independent(adj[v] & S_mask, adj):
                    new_C |= lsb
                c ^= lsb
            
            if not new_C:
                if sz > best[0]:
                    best[0] = sz
                return
            
            C_mask = new_C
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            lsb = C_mask & -C_mask
            v = lsb.bit_length() - 1
            
            bnb(S_mask | lsb, C_mask ^ lsb)
            bnb(S_mask, C_mask ^ lsb)
        
        bnb(0, full_mask)
        return best[0]
    
    # Get edge list from adjacency bitmasks
    def get_edge_list(adj):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    # Create a 3-partite Turán graph T_3(18)
    # Parts are cliques, edges only between different parts
    def make_turan3_adj(part_sizes):
        """part_sizes: list of 3 ints summing to 18"""
        adj = [0] * N
        parts = []
        v = 0
        for size in part_sizes:
            parts.append(list(range(v, v + size)))
            v += size
        
        # Within each part: complete graph
        for part in parts:
            for i in range(len(part)):
                for j in range(i + 1, len(part)):
                    u, w = part[i], part[j]
                    adj[u] |= (1 << w)
                    adj[w] |= (1 << u)
        
        # Between parts: no edges (that's the Turán structure)
        # Actually wait - Turán graph T_3(18) has edges BETWEEN parts, not within
        # Let me reconsider: T_r(n) is r-partite with edges only between parts
        # So parts are independent sets, edges connect different parts
        
        # Reset and redo
        adj = [0] * N
        parts = []
        v = 0
        for size in part_sizes:
            parts.append(list(range(v, v + size)))
            v += size
        
        # Edges only between different parts
        for i in range(3):
            for j in range(i + 1, 3):
                for u in parts[i]:
                    for w in parts[j]:
                        adj[u] |= (1 << w)
                        adj[w] |= (1 << u)
        
        return adj
    
    # Verify K4-free
    def is_k4_free(adj):
        for i in range(N):
            for j in range(i + 1, N):
                if not ((adj[i] >> j) & 1):
                    continue
                for k in range(j + 1, N):
                    if not ((adj[i] >> k) & 1) or not ((adj[j] >> k) & 1):
                        continue
                    for l in range(k + 1, N):
                        if ((adj[i] >> l) & 1) and ((adj[j] >> l) & 1) and ((adj[k] >> l) & 1):
                            return False
        return True
    
    # For a 3-partite graph with parts A, B, C:
    # A triangle-free induced subgraph can contain at most 2 vertices from any part
    # (since 3 vertices from the same part form a triangle if the part is a clique,
    #  but in Turán graph parts are independent, so no triangle within a part)
    # Wait - in T_3(18), parts are independent sets. So any subset of one part is independent.
    # A triangle requires vertices from at least 2 parts.
    # Actually in a complete 3-partite graph, a triangle exists iff we pick one vertex
    # from each of 3 different parts.
    # So a triangle-free induced subgraph can have:
    # - All vertices from at most 2 parts
    # - Or if from 3 parts, no triple (one from each) can form a triangle
    #   But in complete 3-partite, any such triple IS a triangle.
    # So triangle-free means: vertices come from at most 2 parts.
    # Max triangle-free = max(|A|+|B|, |A|+|C|, |B|+|C|) = 18 - min(part sizes)
    
    # So to minimize the max triangle-free subgraph, we want to maximize the minimum part size!
    # With 18 vertices and 3 parts: minimize (18 - min_size) = maximize min_size
    # Best: parts of size 6,6,6 -> max triangle-free = 12
    # But we can do better by NOT using a complete 3-partite graph!
    
    # The key insight: we don't need a complete 3-partite graph. We can remove some
    # inter-part edges to break triangles while keeping the graph K4-free.
    
    random.seed(42)
    
    # Start with balanced 3-partite Turán: 6+6+6
    best_adj = None
    best_score = 999
    
    # Try different part size distributions
    part_configs = [
        [6, 6, 6],
        [7, 6, 5],
        [7, 7, 4],
        [8, 5, 5],
        [8, 6, 4],
        [9, 5, 4],
        [9, 6, 3],
        [9, 7, 2],
        [10, 4, 4],
        [10, 5, 3],
        [10, 6, 2],
        [11, 4, 3],
        [11, 5, 2],
        [11, 6, 1],
        [12, 3, 3],
        [12, 4, 2],
        [12, 5, 1],
    ]
    
    for config in part_configs:
        if sum(config) != 18:
            continue
        adj = make_turan3_adj(config)
        score = largest_triangle_free(adj, deadline - 5.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
    
    # Local search: try to improve by removing inter-part edges strategically
    # to reduce the largest triangle-free subgraph
    def local_search_turan(adj, best_score, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each possible inter-part edge
            for i in range(N):
                if time.time() > deadline:
                    return best_score
                for j in range(i + 1, N):
                    if time.time() > deadline:
                        return best_score
                    if not ((adj[i] >> j) & 1):
                        continue
                    
                    # Determine if this is an inter-part edge
                    # In our construction, we need to track parts
                    # For now, just try removing any edge
                    
                    old_val = 1
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                    
                    score = largest_triangle_free(adj, deadline - 5.0)
                    
                    if score < best_score:
                        best_score = score
                        improved = True
                    else:
                        # Revert
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
        
        return best_score
    
    # Run local search from the best Turán configuration
    if best_adj is not None:
        best_score = local_search_turan(best_adj, best_score, deadline)
    
    # Also try adding some edges within parts (making parts into cliques)
    # This changes the structure from Turán to something else
    # A graph where each part is a clique and inter-part edges are sparse
    # This is K4-free if no 4 vertices form a K4
    
    # Alternative approach: start with a random K4-free graph and optimize
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        
        # Generate a random K4-free graph
        adj = [0] * N
        all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if can_add_edge(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        if not is_k4_free(adj):
            continue
        
        score = largest_triangle_free(adj, deadline - 5.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
            # Local search
            best_score = local_search_turan(adj, best_score, deadline)
        
        # Try perturbing the best solution
        if attempts % 3 == 0 and best_adj is not None:
            adj2 = best_adj[:]
            # Randomly flip a few edges
            num_flips = random.randint(2, 8)
            for _ in range(num_flips):
                u, v = random.sample(range(N), 2)
                if (adj2[u] >> v) & 1:
                    # Remove edge
                    adj2[u] &= ~(1 << v)
                    adj2[v] &= ~(1 << u)
                else:
                    # Try adding edge
                    if can_add_edge(adj2, u, v):
                        adj2[u] |= (1 << v)
                        adj2[v] |= (1 << u)
            
            if is_k4_free(adj2):
                score2 = largest_triangle_free(adj2, deadline - 5.0)
                if score2 < best_score:
                    best_score = score2
                    best_adj = adj2[:]
                    best_score = local_search_turan(adj2, best_score, deadline)
    
    if best_adj is None:
        # Fallback: balanced Turán
        best_adj = make_turan3_adj([6, 6, 6])
    
    return get_edge_list(best_adj)

# EVOLVE-BLOCK-END
