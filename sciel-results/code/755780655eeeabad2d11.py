"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import sys

_START = time.time()
_LIMIT = 145.0  # safety margin under 150s wall-clock

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _generate_maximal_cliques(n, adj):
    """
    Generate all maximal cliques in the graph using Bron-Kerbosch with pivoting.
    adj is a list of bitmasks (int) where adj[i] has bit j set if (i,j) is an edge.
    Returns a list of lists of vertices.
    """
    cliques = []
    
    def bron_kerbosch(R, P, X):
        if not P and not X:
            # R is a maximal clique
            if len(R) >= 2:
                cliques.append(list(R))
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        # Heuristic: pick vertex with max degree in P ∪ X
        union_px = P | X
        if union_px == 0:
            return
            
        # Find pivot
        # Iterate through bits to find one
        # For speed, we can just pick the first one or try to find a good one.
        # Let's pick the one with highest degree in P.
        best_u = -1
        best_deg = -1
        temp = union_px
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            u = lsb.bit_length() - 1
            # Degree of u in P
            deg = (adj[u] & P).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_u = u
            temp ^= lsb
        
        if best_u == -1:
            return
            
        # Recurse on P \ N(u)
        # We iterate over vertices in P that are NOT neighbors of u
        # To do this efficiently, we can iterate over P and check if (adj[u] & (1<<v)) == 0
        # Or iterate over P & ~adj[u]
        
        candidates = P & ~adj[best_u]
        
        while candidates:
            lsb = candidates & -candidates
            v = lsb.bit_length() - 1
            
            # R' = R ∪ {v}
            R.append(v)
            # P' = P ∩ N(v)
            P_new = P & adj[v]
            # X' = X ∩ N(v)
            X_new = X & adj[v]
            
            bron_kerbosch(R, P_new, X_new)
            
            R.pop()
            # Move v from P to X
            P ^= lsb
            X |= lsb
            # Remove v from candidates
            candidates ^= lsb

    # Initial call
    # R = [], P = all vertices, X = []
    all_vertices_mask = (1 << n) - 1
    bron_kerbosch([], all_vertices_mask, 0)
    
    return cliques

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
        
    # Build adjacency bitmasks
    adj = [0] * n
    edge_set = set()
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        edge_set.add((u, v) if u < v else (v, u))
    
    # Generate all maximal cliques
    max_cliques = _generate_maximal_cliques(n, adj)
    
    if not max_cliques:
        return []
        
    # We want to select a subset of these cliques such that:
    # 1. No two cliques share an edge.
    # 2. The union of edges in the selected cliques covers as many edges as possible.
    # 3. Minimize the number of cliques used.
    # Since every edge must be covered, and we can't share edges, this is a packing problem.
    # Actually, the task says "every edge must lie in exactly one listed clique".
    # This implies we need to partition the edge set.
    # If we pick a set of edge-disjoint cliques that cover ALL edges, that's a valid partition.
    # But maximal cliques might not tile the edge space perfectly.
    # If we just pick a maximal set of edge-disjoint cliques, we might leave some edges uncovered.
    # The remaining edges must be covered by cliques of size 2 (single edges) or larger if they form cliques among themselves.
    
    # Strategy:
    # 1. Use a greedy set packing to select a large number of edges covered by few cliques.
    # 2. For the remaining uncovered edges, form cliques (possibly just edges).
    
    # Precompute edges for each clique
    clique_edges = []
    for clique in max_cliques:
        es = set()
        for u, v in combinations(clique, 2):
            es.add((u, v) if u < v else (v, u))
        clique_edges.append(es)
        
    # Randomized Greedy Set Packing with Multiple Restarts
    best_partition = None
    best_count = float('inf')
    
    # We will run several iterations of greedy packing with random tie-breaking
    # to find a good packing.
    
    # To speed up, we can sort cliques by size descending initially, but randomize ties.
    # Or just randomize the order of cliques completely in each restart.
    
    num_restarts = 50
    rng = random.Random(12345) # Fixed seed for reproducibility within the function call
    
    for restart in range(num_restarts):
        if not _deadline_ok():
            break
            
        # Shuffle cliques
        indices = list(range(len(max_cliques)))
        rng.shuffle(indices)
        
        # We want to prioritize larger cliques? 
        # Let's sort by size descending, but add noise to break ties or randomize order slightly?
        # Actually, pure random shuffle might pick small cliques first, blocking large ones.
        # Better: Sort by size descending, but randomize the order of same-size cliques.
        # Or simply: Iterate through cliques in random order, but check if they fit.
        # If we iterate random order, we might pick a small clique that uses edges needed for a big one.
        # So, sorting by size descending is a strong heuristic.
        
        sorted_indices = sorted(indices, key=lambda i: len(max_cliques[i]), reverse=True)
        
        used_edges = set()
        selected_cliques = []
        
        for idx in sorted_indices:
            if not _deadline_ok():
                break
                
            c_edges = clique_edges[idx]
            # Check if any edge in c_edges is already used
            # Optimization: if len(c_edges) is small, this is fast.
            # For large cliques, it might be slow.
            # But n=64, max clique size probably < 20, so edges < 190.
            # used_edges grows.
            
            # Early exit if intersection exists
            # We can just check if c_edges & used_edges is empty
            if c_edges & used_edges:
                continue
                
            # If no conflict, select this clique
            selected_cliques.append(max_cliques[idx])
            used_edges.update(c_edges)
            
        # After greedy selection, we have selected_cliques covering used_edges.
        # Remaining edges = edge_set - used_edges
        remaining_edges = edge_set - used_edges
        
        # Now we need to cover remaining_edges with cliques.
        # Since remaining_edges are disjoint from the large cliques, we can form cliques from them.
        # We can try to find maximal cliques in the subgraph induced by remaining_edges.
        # Or simply, since the remaining graph is sparser, we can use a similar greedy peeling on the remaining graph.
        
        # Build adjacency for remaining graph
        rem_adj = [0] * n
        for u, v in remaining_edges:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
            
        # Find maximal cliques in remaining graph?
        # Or just use a simple greedy: pick an edge, expand to maximal clique in rem graph.
        # Since remaining edges are few, this is fast.
        
        rem_cliques = []
        rem_edges_set = set(remaining_edges)
        
        # Simple greedy peeling on remaining edges
        while rem_edges_set:
            if not _deadline_ok():
                # Fallback: just add all remaining edges as individual cliques
                for e in rem_edges_set:
                    rem_cliques.append(list(e))
                break
                
            # Pick first edge (deterministic for stability within this restart? or random?)
            # Let's pick min for determinism
            u, v = min(rem_edges_set)
            
            # Expand {u, v} to maximal clique in rem graph
            clique = [u, v]
            # Candidates: neighbors of u and v in rem graph
            cand_mask = (rem_adj[u] & rem_adj[v])
            # Remove u and v from candidates
            cand_mask &= ~(1 << u)
            cand_mask &= ~(1 << v)
            
            # Greedy expansion
            while cand_mask:
                # Pick vertex with highest degree in cand_mask induced subgraph?
                # Or just pick first?
                # Let's pick first for speed
                lsb = cand_mask & -cand_mask
                w = lsb.bit_length() - 1
                clique.append(w)
                # Update candidates: must be neighbor of w in rem graph
                cand_mask &= rem_adj[w]
                cand_mask &= ~(1 << w)
                
            rem_cliques.append(clique)
            
            # Remove edges in this clique from rem_edges_set
            for a, b in combinations(clique, 2):
                e = (a, b) if a < b else (b, a)
                if e in rem_edges_set:
                    rem_edges_set.remove(e)
                    
        # Total partition
        total_partition = selected_cliques + rem_cliques
        total_count = len(total_partition)
        
        if total_count < best_count:
            best_count = total_count
            best_partition = total_partition
            
    # If no partition found (shouldn't happen), fallback to edges
    if best_partition is None:
        return [[u, v] for u, v in edges]
        
    return best_partition

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if not _deadline_ok():
            # Critical timeout: return empty or minimal valid?
            # If we can't solve, we must return something valid.
            # But if we time out, we might not have a valid partition.
            # Let's try to return a quick valid partition (edges) if we are very close to limit.
            edges = _random_edges(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
            
        results.append(_solve_instance(n, p, seed))
        
    return results
# EVOLVE-BLOCK-END
