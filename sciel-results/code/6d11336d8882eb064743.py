import time
from random import Random

_DEADLINE = 140.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() > _DEADLINE

def construct():
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_split(n, k, p, seed):
        # Generate graph
        # C = [0, k), I = [k, n)
        # Edges in C are all pairs.
        # Edges in I are none.
        # Edges between C and I are random.
        
        # We need N(v) for each v in I.
        # Instead of generating all edges and filtering, we can regenerate or just use the instance function.
        # The instance function returns sorted edges.
        edges = instance(n, k, p, seed)
        
        # Identify N(v) for v in I
        # adj_I[v] is bitmask of neighbors in C
        adj_I = [0] * n
        for u, v in edges:
            if u >= k and v < k:
                adj_I[u] |= (1 << v)
            elif u < k and v >= k:
                adj_I[v] |= (1 << u)
        
        cliques = []
        
        # 1. Create cliques for I vertices
        # Track which edges in C are covered
        # covered_C[u] is bitmask of w such that edge (u,w) is covered
        covered_C = [0] * k
        
        for v in range(k, n):
            N_v = adj_I[v]
            if N_v == 0:
                continue
            
            # Clique is {v} + list of bits in N_v
            clique = [v]
            s = N_v
            while s:
                u = (s & -s).bit_length() - 1
                clique.append(u)
                s &= s - 1
            
            cliques.append(clique)
            
            # Mark edges in C induced by N_v as covered
            # For each u in N_v, for each w in N_v, u < w:
            # covered_C[u] |= (1 << w)
            # covered_C[w] |= (1 << u)
            # Optimization: 
            # The set of covered edges is E(N_v).
            # We can just compute the union of all E(N_v) for all v.
            # But we need to do it incrementally or at the end?
            # At the end is easier.
            # Let's store N_v in a list for later processing.
            pass # We'll process coverage after the loop for clarity, or just accumulate.
        
        # Actually, let's just compute the uncovered edges in C directly.
        # Edge (u, w) in C is uncovered if for ALL v in I, NOT (u in N(v) AND w in N(v)).
        # i.e., u and w do not share any neighbor in I.
        
        # Let's compute a mask for each pair? No, k is small.
        # For each u in C, let's find which v's are neighbors.
        # Actually, we have adj_I.
        # For a fixed pair (u, w), we check if there is any v such that adj_I[v] has bits u and w.
        
        # This is O(k^2 * (n-k)). k=30, n-k=60. 900 * 60 = 54000 ops. Very fast.
        
        uncovered_adj = [0] * k
        
        for u in range(k):
            for w in range(u + 1, k):
                # Check if edge (u, w) is covered
                is_covered = False
                for v in range(k, n):
                    if (adj_I[v] >> u) & 1 and (adj_I[v] >> w) & 1:
                        is_covered = True
                        break
                if not is_covered:
                    uncovered_adj[u] |= (1 << w)
                    uncovered_adj[w] |= (1 << u)
        
        # Now we have the graph G_rem on C with adjacency uncovered_adj.
        # We need to partition the edges of G_rem into cliques.
        
        # Greedy algorithm:
        # While there are edges:
        #   Find a maximal clique in G_rem.
        #   Add it to cliques.
        #   Remove its edges from G_rem.
        
        # To find a maximal clique efficiently:
        # We can use a simple greedy:
        # Pick a vertex with max degree.
        # Add it to clique.
        # Restrict to its neighbors.
        # Repeat.
        
        # To improve, we can try to find the *maximum* clique in the remaining graph?
        # For k=30, finding max clique is feasible with a good heuristic or even simple branch and bound if sparse.
        # But since G_rem is likely sparse or empty, maximal is probably enough.
        # Let's use a heuristic: Pick the maximal clique that has the most edges.
        # We can generate a few candidate maximal cliques and pick the best.
        
        def _find_best_max_clique(adj):
            # adj is list of bitmasks
            # Returns a bitmask of the best maximal clique
            # Heuristic: Try starting from each vertex, build a maximal clique, keep the largest.
            # With k=30, this is 30 * (greedy cost).
            
            best_clique_mask = 0
            best_size = 0
            
            # Sort vertices by degree descending to start from "dense" areas
            vertices = list(range(k))
            vertices.sort(key=lambda x: bin(adj[x]).count('1'), reverse=True)
            
            for start_v in vertices:
                if adj[start_v] == 0:
                    continue
                
                S = 1 << start_v
                cand = adj[start_v]
                
                while cand:
                    # Find vertex in cand with max degree in cand
                    best_u = -1
                    best_deg = -1
                    c = cand
                    while c:
                        u = (c & -c).bit_length() - 1
                        c &= c - 1
                        deg = bin(adj[u] & cand).count('1')
                        if deg > best_deg:
                            best_deg = deg
                            best_u = u
                    
                    if best_u == -1:
                        break
                    
                    S |= (1 << best_u)
                    cand = cand & adj[best_u]
                
                # S is a maximal clique
                size = bin(S).count('1')
                if size > best_size:
                    best_size = size
                    best_clique_mask = S
            
            return best_clique_mask

        # Main loop for G_rem
        # We need to be careful with the deadline
        while True:
            if _deadline_hit():
                # Fallback: cover remaining edges with 2-cliques
                for u in range(k):
                    v_mask = uncovered_adj[u]
                    while v_mask:
                        v = (v_mask & -v_mask).bit_length() - 1
                        v_mask &= v_mask - 1
                        if u < v:
                            cliques.append([u, v])
                            uncovered_adj[u] &= ~(1 << v)
                            uncovered_adj[v] &= ~(1 << u)
                break
            
            # Check if any edges remain
            has_edges = False
            for u in range(k):
                if uncovered_adj[u]:
                    has_edges = True
                    break
            
            if not has_edges:
                break
            
            # Find best maximal clique
            S = _find_best_max_clique(uncovered_adj)
            
            if S == 0:
                # Should not happen if has_edges
                break
            
            # Add clique
            S_list = []
            s = S
            while s:
                u = (s & -s).bit_length() - 1
                S_list.append(u)
                s &= s - 1
            
            cliques.append(S_list)
            
            # Remove edges in S from uncovered_adj
            s = S
            while s:
                u = (s & -s).bit_length() - 1
                s &= s - 1
                # Remove edges between u and other vertices in S
                others = S & ~(1 << u)
                uncovered_adj[u] &= ~others
                # Note: We also need to update the other vertices' adjacency to u
                # But since we are removing the whole clique, we can just clear the bits.
                # Actually, the loop `for u in S` clears `uncovered_adj[u] & S`.
                # This is symmetric.
            
            # Ensure symmetry is maintained?
            # When we do `uncovered_adj[u] &= ~others`, we clear bits in u's mask.
            # We don't explicitly clear u's bit in others' masks in this iteration.
            # But in the next iteration of the while s loop, we process 'others' vertices.
            # So it will be cleared.
            # Wait, if we process u first, we clear u's bits for v.
            # Then we process v, we clear v's bits for u.
            # So yes, it works.

        # Convert to lists of Python ints
        result = []
        for c in cliques:
            result.append([int(x) for x in c])
        return result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    out = []
    for spec in specs:
        if _deadline_hit():
            # Fallback: seed-like
            edges = instance(*spec)
            out.append([list(e) for e in edges])
        else:
            out.append(_solve_split(*spec))
    return out
