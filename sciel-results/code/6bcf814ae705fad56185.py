"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    TOTAL_DEADLINE = t0 + 140.0  # safety margin before 150s kill

    def gen_edges(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_clique_partition(n, p, seed, budget):
        edges = gen_edges(n, p, seed)
        edges_set = set(edges)
        if not edges:
            return []
        
        # Build adjacency
        G_adj = [set() for _ in range(n)]
        for u, v in edges:
            G_adj[u].add(v)
            G_adj[v].add(u)
        
        remaining = set(edges_set)
        rem_adj = [set() for _ in range(n)]
        for u, v in remaining:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
        
        cliques = []
        
        def get_priority(v):
            return len(rem_adj[v])
        
        # Fast max clique using Bron-Kerbosch with pivot and bitmask optimization for small n
        def max_clique_fast(candidate_set, adj, deadline=None):
            if not candidate_set:
                return []
            
            # Use bitmask for n <= 32 for speed
            use_bitmask = n <= 32 and len(candidate_set) <= 32
            
            if use_bitmask:
                # Map vertices to bit positions
                vert_list = list(candidate_set)
                vert_to_bit = {v: i for i, v in enumerate(vert_list)}
                m = len(vert_list)
                
                # Build bitmask adjacency within candidate_set
                bm_adj = [0] * m
                for i, u in enumerate(vert_list):
                    for j, v in enumerate(vert_list):
                        if i != j and v in adj[u]:
                            bm_adj[i] |= (1 << j)
                
                best = []
                
                def bronk_bm(R, P, X):
                    nonlocal best
                    if deadline is not None and time.time() > deadline:
                        return
                    if P == 0 and X == 0:
                        if len(R) > len(best):
                            best = R[:]
                        return
                    
                    # Choose pivot: vertex in P|X with max neighbors in P
                    PX = P | X
                    if PX == 0:
                        return
                    
                    pivot_bit = -1
                    max_cnt = -1
                    tmp = PX
                    while tmp:
                        lb = tmp & (-tmp)
                        bi = lb.bit_length() - 1
                        cnt = bin(P & bm_adj[bi]).count('1')
                        if cnt > max_cnt:
                            max_cnt = cnt
                            pivot_bit = bi
                        tmp ^= lb
                    
                    # Candidates: P minus neighbors of pivot
                    if pivot_bit >= 0:
                        candidates = P & ~(bm_adj[pivot_bit] | (1 << pivot_bit))
                    else:
                        candidates = P
                    
                    tmp = candidates
                    while tmp:
                        lb = tmp & (-tmp)
                        bi = lb.bit_length() - 1
                        v = vert_list[bi]
                        bronk_bm(R + [v], P & bm_adj[bi], X & bm_adj[bi])
                        P &= ~(1 << bi)
                        X |= (1 << bi)
                        tmp ^= lb
                
                bronk_bm([], (1 << m) - 1, 0)
                return best
            else:
                # Standard Bron-Kerbosch for larger n
                best = []
                
                def bronk(R, P, X):
                    nonlocal best
                    if deadline is not None and time.time() > deadline:
                        return
                    if not P and not X:
                        if len(R) > len(best):
                            best = R[:]
                        return
                    
                    # Choose pivot
                    PX = P | X
                    if not PX:
                        return
                    
                    pivot = None
                    max_adj_count = -1
                    for v in PX:
                        adj_count = len(P & adj[v])
                        if adj_count > max_adj_count:
                            max_adj_count = adj_count
                            pivot = v
                    
                    if pivot is None:
                        candidates = list(P)
                    else:
                        candidates = list(P - adj[pivot])
                    
                    for v in candidates:
                        bronk(R + [v], P & adj[v], X & adj[v])
                        P.discard(v)
                        X.add(v)
                
                bronk([], set(candidate_set), set())
                return best
        
        # Greedy clique partition with multi-candidate selection
        while remaining and time.time() < budget - 2.0:
            # Find active vertices
            active = set(v for v in range(n) if rem_adj[v])
            if not active:
                break
            
            # Try multiple high-degree vertices and pick best clique
            # Sort active vertices by degree, try top K
            sorted_active = sorted(active, key=get_priority, reverse=True)
            K = min(5, len(sorted_active))
            
            best_clique = []
            best_edge_count = 0
            
            for trial_v in sorted_active[:K]:
                candidate_set = {trial_v} | (rem_adj[trial_v] & active)
                C = max_clique_fast(candidate_set, rem_adj, budget - 3.0)
                
                if len(C) < 2:
                    continue
                
                # Count how many remaining edges this clique covers
                C_set = set(C)
                edge_count = 0
                for i in range(len(C)):
                    for j in range(i + 1, len(C)):
                        a, b = C[i], C[j]
                        e = (a, b) if a < b else (b, a)
                        if e in remaining:
                            edge_count += 1
                
                # Prefer cliques with more edges covered, break ties by size
                if edge_count > best_edge_count or (edge_count == best_edge_count and len(C) > len(best_clique)):
                    best_edge_count = edge_count
                    best_clique = C
            
            if not best_clique:
                # Fallback: take one edge
                u, v = next(iter(remaining))
                cliques.append([u, v])
                remaining.discard((u, v))
                rem_adj[u].discard(v)
                rem_adj[v].discard(u)
                continue
            
            # Remove all edges within best_clique
            removed = 0
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    a, b = best_clique[i], best_clique[j]
                    e = (a, b) if a < b else (b, a)
                    if e in remaining:
                        remaining.discard(e)
                        rem_adj[a].discard(b)
                        rem_adj[b].discard(a)
                        removed += 1
            cliques.append(best_clique)
            if removed == 0:
                break
        
        # Handle remaining edges
        for e in remaining:
            cliques.append([e[0], e[1]])
        
        return cliques
    
    def is_valid(partition, edges_set, G_adj):
        seen = set()
        for C in partition:
            if len(C) < 2:
                return False
            s = set(C)
            if len(s) != len(C):
                return False
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    a, b = C[i], C[j]
                    if b not in G_adj[a]:
                        return False
                    e = (a, b) if a < b else (b, a)
                    if e in seen:
                        return False
                    seen.add(e)
        return seen == edges_set
    
    def merge_pass(cliques, G_adj):
        # More aggressive merging: try all pairs, not just overlapping ones
        changed = True
        while changed:
            changed = False
            T = len(cliques)
            if T < 2:
                break
            # Sort by size descending for better merge candidates
            order = sorted(range(T), key=lambda i: -len(cliques[i]))
            for idx_i in order:
                if changed:
                    break
                i = idx_i
                if i >= T:
                    continue
                Ci = set(cliques[i])
                for idx_j in order:
                    if changed:
                        break
                    j = idx_j
                    if j == i or j >= T:
                        continue
                    Cj = set(cliques[j])
                    
                    # Check if union is a clique
                    union = Ci | Cj
                    is_clique = True
                    for u in union:
                        for v in union:
                            if u != v and v not in G_adj[u]:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if not is_clique:
                        continue
                    
                    # Check no other clique shares more than 1 vertex with union
                    ok2 = True
                    for k in range(T):
                        if k == i or k == j:
                            continue
                        if len(union & set(cliques[k])) > 1:
                            ok2 = False
                            break
                    if not ok2:
                        continue
                    
                    cliques[i] = list(union)
                    del cliques[j]
                    changed = True
                    break
        return cliques
    
    result = []
    # First instance: n=32
    t1 = time.time()
    result.append(solve_clique_partition(32, 0.6, 601, t0 + 45.0))
    
    # Apply merge pass to first result
    edges1 = gen_edges(32, 0.6, 601)
    G_adj1 = [set() for _ in range(32)]
    for u, v in edges1:
        G_adj1[u].add(v)
        G_adj1[v].add(u)
    result[0] = merge_pass(result[0], G_adj1)
    
    # Second instance: n=64
    if time.time() > TOTAL_DEADLINE - 3.0:
        edges = gen_edges(64, 0.55, 602)
        result.append([[u, v] for (u, v) in edges])
    else:
        result.append(solve_clique_partition(64, 0.55, 602, TOTAL_DEADLINE - 1.0))
        
        # Apply merge pass to second result
        edges2 = gen_edges(64, 0.55, 602)
        G_adj2 = [set() for _ in range(64)]
        for u, v in edges2:
            G_adj2[u].add(v)
            G_adj2[v].add(u)
        result[1] = merge_pass(result[1], G_adj2)
    
    return result
# EVOLVE-BLOCK-END
