"""3-partite Turán graph baseline + vertex-swap local search + optimized B&B."""
import time
import random
import math

def construct():
    """Construct a K4-free graph on 18 vertices minimizing the max triangle-free induced subgraph."""
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
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
    
    # Optimized B&B for largest triangle-free induced subgraph
    def largest_triangle_free(adj, time_limit=None):
        best = [0]
        
        # Greedy heuristic: try multiple random orders
        for _ in range(8):
            order = list(range(N))
            random.shuffle(order)
            S = 0
            for v in order:
                if is_independent(adj[v] & S, adj):
                    S |= (1 << v)
            sz = S.bit_count() if hasattr(int, 'bit_count') else bin(S).count('1')
            if sz > best[0]:
                best[0] = sz
        
        if best[0] == N:
            return N
        
        # Precompute: for each vertex, its neighbors
        # Branch and bound with better pruning
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = S_mask.bit_count() if hasattr(int, 'bit_count') else bin(S_mask).count('1')
            c_sz = C_mask.bit_count() if hasattr(int, 'bit_count') else bin(C_mask).count('1')
            
            if sz + c_sz <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return
            
            # Filter candidates: v can be added if adj[v] & S_mask is independent
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
            c_sz = C_mask.bit_count() if hasattr(int, 'bit_count') else bin(C_mask).count('1')
            if sz + c_sz <= best[0]:
                return
            
            # Branch on vertex with highest degree in C_mask (more constraining)
            # Pick first set bit for simplicity, but try a few heuristics
            lsb = C_mask & -C_mask
            v = lsb.bit_length() - 1
            
            # Branch 1: Include v
            bnb(S_mask | lsb, C_mask ^ lsb)
            
            # Branch 2: Exclude v
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
    
    # Create 3-partite Turán graph T_3(18): three cliques of size 6, complete between parts
    # Parts: {0..5}, {6..11}, {12..17}
    # Edges: all pairs within each part (clique) + all pairs between different parts
    def make_turan3_adj():
        adj = [0] * N
        for i in range(N):
            for j in range(i + 1, N):
                pi = i // 6
                pj = j // 6
                # In 3-partite Turán: edges within same part AND between different parts
                # Actually T_3(18) is complete 3-partite: edges BETWEEN parts, no edges within
                # Wait: Turán graph T_k(n) has k parts, no edges within parts, all edges between parts
                # For K4-free: T_3(18) is K4-free since any 4 vertices span at most 3 parts
                if pi != pj:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
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
    
    # For 3-partite graph: triangle-free induced subgraph can have at most 2 vertices from any part
    # (since within a part there are no edges, so any subset of a part is independent)
    # Actually in complete 3-partite: no edges within parts, all edges between parts
    # A triangle needs 3 vertices from 3 different parts
    # Triangle-free: cannot have 1 vertex from each of 3 parts that form a triangle
    # In complete 3-partite, any triple (one from each part) forms a triangle
    # So triangle-free S: for any three parts, can't have all three represented
    # Max triangle-free: pick all from 2 parts = 6+6=12, or all from 1 part = 6
    # Actually: S is triangle-free iff it doesn't contain a triangle
    # In K_{6,6,6}, a triangle is any 3 vertices from 3 distinct parts
    # So S is triangle-free iff S is contained in at most 2 parts
    # Max = 12 (all of two parts)
    
    random.seed(42)
    
    # Start with T_3(18)
    current_adj = make_turan3_adj()
    current_score = largest_triangle_free(current_adj, deadline - 5.0)
    best_adj = current_adj[:]
    best_score = current_score
    
    # For 3-partite graph, local search by swapping vertices between parts
    # This changes the part structure but we need to maintain K4-free
    # Actually, let's do edge-flip local search but with a faster score evaluation
    
    # Fast upper bound: in any graph, triangle-free subgraph <= N
    # For 3-partite: max is 12
    
    # Local search: try edge flips
    def local_search(adj, best_score, deadline):
        improved = True
        iterations = 0
        while improved and time.time() < deadline - 5.0:
            improved = False
            iterations += 1
            # Try all possible single edge flips
            for i in range(N):
                if time.time() > deadline - 5.0:
                    return best_score
                for j in range(i + 1, N):
                    if time.time() > deadline - 5.0:
                        return best_score
                    old_val = (adj[i] >> j) & 1
                    
                    if old_val == 0:
                        if not can_add_edge(adj, i, j):
                            continue
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                    else:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    
                    score = largest_triangle_free(adj, deadline - 10.0)
                    
                    if score < best_score:
                        best_score = score
                        improved = True
                    else:
                        if old_val == 0:
                            adj[i] &= ~(1 << j)
                            adj[j] &= ~(1 << i)
                        else:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
        
        return best_score
    
    # Run local search from T_3(18)
    best_score = local_search(current_adj, best_score, deadline)
    if best_score < current_score:
        best_adj = current_adj[:]
    
    # Random restarts
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        
        # Generate random K4-free graph
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
            best_score = local_search(adj, best_score, deadline)
            if best_score < score:
                best_adj = adj[:]
        
        # Perturb T_3(18)
        if attempts % 3 == 0:
            adj2 = make_turan3_adj()
            num_adds = random.randint(2, 10)
            for _ in range(num_adds):
                u, v = random.sample(range(N), 2)
                if can_add_edge(adj2, u, v):
                    adj2[u] |= (1 << v)
                    adj2[v] |= (1 << u)
            
            if is_k4_free(adj2):
                score2 = largest_triangle_free(adj2, deadline - 5.0)
                if score2 < best_score:
                    best_score = score2
                    best_adj = adj2[:]
                    best_score = local_search(adj2, best_score, deadline)
                    if best_score < score2:
                        best_adj = adj2[:]
    
    return get_edge_list(best_adj)

# EVOLVE-BLOCK-END
