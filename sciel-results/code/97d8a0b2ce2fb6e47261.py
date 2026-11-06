# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _build_graph(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _is_clique(vertices, adj):
    """Check if the set of vertices forms a clique in the graph."""
    v_list = list(vertices)
    for i in range(len(v_list)):
        for j in range(i + 1, len(v_list)):
            if v_list[j] not in adj[v_list[i]]:
                return False
    return True

def _max_clique_bron_kerbosch(adj, deadline, rng=None):
    """
    Find a maximum clique in the graph using Bron-Kerbosch algorithm.
    For dense graphs, this is efficient.
    """
    n = len(adj)
    best_clique = []
    
    # Use bitmasks for speed if n <= 64
    if n <= 64:
        adj_mask = [0] * n
        for u in range(n):
            for v in adj[u]:
                adj_mask[u] |= (1 << v)
        
        def expand(R_mask, P_mask, X_mask):
            nonlocal best_clique
            if time.time() > deadline:
                return
            if P_mask == 0 and X_mask == 0:
                # Convert mask to list
                clique = []
                m = R_mask
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    clique.append(v)
                    m ^= lsb
                if len(clique) > len(best_clique):
                    best_clique = clique
                return
            
            if P_mask == 0:
                return
            
            # Pivot selection: choose u in P \ X with max degree in P
            # To simplify, just pick any vertex in P
            # Better: pick u in P with max |P \ N(u)| ? No, standard is max |P \cap N(u)|?
            # Actually, Bron-Kerbosch with pivot:
            # Choose u in P \ X.
            # For each v in P \ N(u):
            #   R' = R \cup {v}
            #   P' = P \cap N(v)
            #   X' = X \cap N(v)
            
            # We need to iterate over P \ N(u).
            # Let's just pick u = first set bit in P_mask
            # This is a simplification, but for dense graphs, pivot helps a lot.
            
            # Find a pivot u in P_mask
            if P_mask == 0:
                return
            
            # Simple pivot: pick the vertex in P with highest degree in P
            # This requires counting bits in P_mask & adj_mask[v]
            # For n=64, this is fast.
            
            best_pivot = -1
            max_deg = -1
            temp_p = P_mask
            while temp_p:
                lsb = temp_p & -temp_p
                v = lsb.bit_length() - 1
                deg = bin(P_mask & adj_mask[v]).count("1")
                if deg > max_deg:
                    max_deg = deg
                    best_pivot = v
                temp_p ^= lsb
            
            if best_pivot == -1:
                return

            # Iterate over P \ N(best_pivot)
            # N(best_pivot) is adj_mask[best_pivot]
            # We want vertices in P_mask that are NOT in adj_mask[best_pivot]
            # Note: best_pivot itself is in P, but is it in N(best_pivot)? No (no self loops).
            # So best_pivot is in P \ N(best_pivot).
            
            candidates = P_mask & ~adj_mask[best_pivot]
            
            while candidates:
                if time.time() > deadline:
                    return
                lsb = candidates & -candidates
                v = lsb.bit_length() - 1
                candidates ^= lsb
                
                R_new = R_mask | (1 << v)
                P_new = P_mask & adj_mask[v]
                X_new = X_mask & adj_mask[v]
                
                expand(R_new, P_new, X_new)
                
                P_mask ^= (1 << v)
                X_mask |= (1 << v)
                
                if P_mask == 0:
                    break

        expand(0, (1 << n) - 1, 0)
        return best_clique
    else:
        # Fallback for larger n (not expected here)
        return list(range(n))

def _greedy_clique_partition(adj, deadline, rng):
    n = len(adj)
    # Remaining edges as a set of (u, v) tuples
    remaining = set()
    for u in range(n):
        for v in adj[u]:
            if v > u:
                remaining.add((u, v))
    
    if not remaining:
        return []
    
    cliques = []
    
    while remaining and time.time() < deadline:
        # Build subgraph of remaining edges
        # Vertices involved
        active_vertices = set()
        for u, v in remaining:
            active_vertices.add(u)
            active_vertices.add(v)
        
        if not active_vertices:
            break
            
        # Find max clique in the subgraph induced by remaining edges
        # We can use Bron-Kerbosch on the full graph, but we must ensure the clique
        # only uses edges from 'remaining'.
        # Actually, Bron-Kerbosch finds a clique in G.
        # We need a clique in G such that ALL its edges are in 'remaining'.
        # This is equivalent to finding a max clique in the subgraph H where H has edges 'remaining'.
        
        # Let's build adj for H
        adj_H = [set() for _ in range(n)]
        for u, v in remaining:
            adj_H[u].add(v)
            adj_H[v].add(u)
            
        max_c = _max_clique_bron_kerbosch(adj_H, deadline, rng)
        
        if not max_c or len(max_c) < 2:
            # Fallback: take any edge
            # This should not happen if remaining is not empty, unless Bron-Kerbosch failed
            # But if remaining has edges, there is at least a K2.
            # If Bron-Kerbosch returns empty, something is wrong or time limit hit.
            if time.time() > deadline:
                break
            # Take first edge
            u, v = next(iter(remaining))
            cliques.append([u, v])
            remaining.discard((u, v))
            continue
            
        # Verify max_c is a valid clique in H
        # (Bron-Kerbosch guarantees it's a clique in G, and we ran it on H, so edges are in H)
        
        # Remove edges of max_c from remaining
        for u, v in combinations(max_c, 2):
            edge = (min(u, v), max(u, v))
            remaining.discard(edge)
            
        cliques.append(max_c)
        
    # Cover any remaining edges as pairs
    for edge in list(remaining):
        if time.time() > deadline:
            break
        u, v = edge
        if (u, v) in remaining:
            cliques.append([u, v])
            remaining.discard((u, v))
            
    return cliques

def _local_search_merge(cliques, adj, deadline):
    """
    Try to merge any two cliques if their union is a clique.
    """
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_cliques = []
        i = 0
        while i < len(cliques):
            c1 = cliques[i]
            merged = False
            for j in range(i + 1, len(cliques)):
                c2 = cliques[j]
                # Check if union is a clique
                union_verts = set(c1) | set(c2)
                if _is_clique(union_verts, adj):
                    # Merge
                    new_cliques.append(list(union_verts))
                    # Remove c2 from consideration
                    # We need to skip c2 in the outer loop
                    # This logic is tricky with indices.
                    # Let's use a list and a flag.
                    cliques[j] = None
                    i = -1 # Restart loop
                    changed = True
                    break
            if not merged and cliques[i] is not None:
                new_cliques.append(cliques[i])
            i += 1
        
        if changed:
            cliques = [c for c in new_cliques if c is not None]
    
    return cliques

def construct():
    start = time.time()
    deadline = start + 140  # Safety margin
    
    results = []
    
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline:
            # Fallback: trivial partition
            adj = _build_graph(n, p, seed)
            cliques = []
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        cliques.append([u, v])
            results.append(cliques)
            continue
        
        adj = _build_graph(n, p, seed)
        
        best_cliques = None
        best_count = float('inf')
        
        # Multi-start greedy
        # Number of restarts depends on time
        # For n=32, we can do many. For n=64, fewer.
        num_restarts = 50 if n <= 32 else 10
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            rng = random.Random(seed + r)
            
            # Run greedy
            cliques = _greedy_clique_partition(adj, deadline, rng)
            
            # Local search merge
            cliques = _local_search_merge(cliques, adj, deadline)
            
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
                
            # Early stop if we found a perfect partition? 
            # Minimum is 1 (if graph is complete).
            # If best_count == 1, break.
            if best_count == 1:
                break

        if best_cliques is None:
            # Fallback
            best_cliques = []
            for u in range(n):
                for v in adj[u]:
                    if v > u:
                        best_cliques.append([u, v])
                        
        results.append(best_cliques)
        
    return results
# EVOLVE-BLOCK-END
