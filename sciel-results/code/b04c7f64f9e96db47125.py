"""Rewritten program: clique partition of split graphs via bitmask-accelerated greedy."""
import time
from random import Random

_DEADLINE = 140.0  # safety margin under 150s wall-clock
_START = time.time()

def _elapsed():
    return time.time() - _START

def _deadline_hit():
    return _elapsed() > _DEADLINE

def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _popcount(x):
        return bin(x).count('1')

    def _solve_split(n, k, p, seed):
        """Return a list of cliques (lists of int) partitioning all edges of the split graph."""
        edges = instance(n, k, p, seed)
        
        # Bitmasks for clique part C = [0, k)
        # adj_C[u] is a bitmask of neighbors of u in C
        # Initially, all pairs in C are edges.
        # We maintain `uncovered_adj` for the subgraph of uncovered edges in C.
        # uncovered_adj[u] has bit v set if (u, v) is an uncovered edge in C.
        
        # Initialize uncovered_adj: full clique
        full_mask = (1 << k) - 1
        uncovered_adj = [0] * k
        for u in range(k):
            # All other vertices are neighbors
            uncovered_adj[u] = full_mask & ~(1 << u)
        
        # Build adjacency for independent vertices V_I = [k, n)
        # adj_I[v] is a bitmask of neighbors in C
        adj_I = [0] * n
        for (u, v) in edges:
            if u < k and v < k:
                # Already handled in uncovered_adj initialization
                pass
            elif u >= k and v < k:
                adj_I[u] |= (1 << v)
            elif u < k and v >= k:
                adj_I[v] |= (1 << u)
            # No edges within V_I
        
        cliques = []
        
        # Helper to find a maximal clique in the graph defined by uncovered_adj,
        # restricted to a candidate mask `cand_mask`.
        def _find_max_clique(cand_mask):
            if cand_mask == 0:
                return 0
            S = 0
            cand = cand_mask
            while cand:
                # Find vertex u in cand with max degree in G_uncovered induced by cand
                # Degree of u is popcount(uncovered_adj[u] & cand)
                best_u = -1
                best_deg = -1
                # Iterate through bits in cand
                c = cand
                while c:
                    u = (c & -c).bit_length() - 1
                    c &= c - 1
                    deg = _popcount(uncovered_adj[u] & cand)
                    if deg > best_deg:
                        best_deg = deg
                        best_u = u
                if best_u == -1:
                    break
                S |= (1 << best_u)
                # Update cand: keep only neighbors of best_u in current cand
                cand = cand & uncovered_adj[best_u]
            return S

        # Process independent vertices
        independent = list(range(k, n))
        # Sort by degree in C (descending)
        independent.sort(key=lambda v: _popcount(adj_I[v]), reverse=True)
        
        for v in independent:
            N_v = adj_I[v]
            remaining = N_v
            while remaining:
                if _deadline_hit():
                    # Fallback: cover remaining cross edges with 2-cliques
                    rem = remaining
                    while rem:
                        u = (rem & -rem).bit_length() - 1
                        rem &= rem - 1
                        cliques.append([v, u])
                        # Remove edge (v, u) - cross edge, doesn't affect uncovered_adj
                        # But we must ensure we don't double cover. 
                        # Since we are in fallback, we assume we just need to finish.
                        # However, we must remove (u, w) from uncovered_adj if we form a clique?
                        # No, 2-clique {v, u} only covers (v, u). It does not cover (u, w).
                        # So uncovered_adj is unchanged.
                        # But wait, if we just add [v, u], we cover edge (v, u).
                        # We must ensure (v, u) is not covered again.
                        # The loop structure ensures we process each u once.
                        # But we need to update `remaining` to remove u.
                        # The while loop does that.
                        pass
                    break
                
                # Find a large clique S in remaining such that all edges in S are uncovered
                S = _find_max_clique(remaining)
                if S == 0:
                    # Should not happen if remaining > 0, but guard
                    u = (remaining & -remaining).bit_length() - 1
                    S = 1 << u
                
                # Add clique [v] + list(S)
                S_list = []
                s = S
                while s:
                    u = (s & -s).bit_length() - 1
                    s &= s - 1
                    S_list.append(u)
                cliques.append([v] + S_list)
                
                # Update uncovered_adj: remove edges within S
                # For each u in S, clear bits for other vertices in S
                s = S
                while s:
                    u = (s & -s).bit_length() - 1
                    s &= s - 1
                    # Clear bits in uncovered_adj[u] corresponding to S \ {u}
                    uncovered_adj[u] &= ~(S & ~(1 << u))
                
                # Update remaining: remove S
                remaining &= ~S
        
        # Cover remaining edges in C
        # While there are edges in uncovered_adj
        # Check if any edge exists: if any u has uncovered_adj[u] != 0
        has_edges = False
        for u in range(k):
            if uncovered_adj[u]:
                has_edges = True
                break
        
        while has_edges:
            if _deadline_hit():
                # Fallback: cover remaining edges with 2-cliques
                # Iterate through all u, v
                for u in range(k):
                    v_mask = uncovered_adj[u]
                    while v_mask:
                        v = (v_mask & -v_mask).bit_length() - 1
                        v_mask &= v_mask - 1
                        if u < v:
                            cliques.append([u, v])
                            # Remove edge (u, v)
                            uncovered_adj[u] &= ~(1 << v)
                            uncovered_adj[v] &= ~(1 << u)
                break
            
            # Find a maximal clique in the full C using current uncovered_adj
            S = _find_max_clique(full_mask)
            if S == 0:
                break # Should not happen if has_edges
            
            S_list = []
            s = S
            while s:
                u = (s & -s).bit_length() - 1
                s &= s - 1
                S_list.append(u)
            
            if len(S_list) < 2:
                # If S is a single vertex, it means no edges?
                # But _find_max_clique returns a clique. If it's size 1, no edges within it.
                # This implies no edges incident to that vertex?
                # Actually, if there are edges, the clique size should be at least 2.
                # Unless the graph is empty.
                # But we check has_edges.
                # If S is size 1, it means the vertex is isolated in the uncovered graph?
                # No, _find_max_clique picks a vertex with max degree. If deg > 0, it will pick neighbors.
                # So if S is size 1, it means deg was 0 for all?
                # Then has_edges should be False.
                # So this is a safety break.
                break

            cliques.append(S_list)
            
            # Update uncovered_adj: remove edges within S
            s = S
            while s:
                u = (s & -s).bit_length() - 1
                s &= s - 1
                uncovered_adj[u] &= ~(S & ~(1 << u))
            
            # Check if more edges
            has_edges = False
            for u in range(k):
                if uncovered_adj[u]:
                    has_edges = True
                    break

        # Convert to lists of Python ints
        # Already done in construction mostly, but ensure types
        result = []
        for c in cliques:
            result.append([int(x) for x in c])
        return result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    out = []
    for spec in specs:
        if _deadline_hit():
            # Fallback: seed-like (each edge as 2-clique)
            edges = instance(*spec)
            out.append([list(e) for e in edges])
        else:
            out.append(_solve_split(*spec))
    return out
