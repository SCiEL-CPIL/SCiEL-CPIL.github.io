"""Clique partition of split graphs with structure-aware decomposition."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # 1. Separate edges into C-internal and Cross (I-C)
        # C vertices: 0..k-1
        # I vertices: k..n-1
        
        c_edges = set()
        cross_edges = []
        
        for u, v in edges:
            if u < k and v < k:
                c_edges.add((min(u, v), max(u, v)))
            else:
                # One is in C, one in I
                # Since it's a split graph, there are no I-I edges.
                # But let's be safe: if both >= k, it's an error in generator or input, 
                # but problem states split graph.
                # We store as (c_vert, i_vert)
                if u < k:
                    cross_edges.append((u, v))
                elif v < k:
                    cross_edges.append((v, u))
                else:
                    # Should not happen for valid split graph input from generator
                    # If it does, treat as generic edge for now
                    c_edges.add((min(u, v), max(u, v)))
        
        cliques = []
        
        # 2. Cover Cross Edges
        # For each i in I, let N_C(i) be the set of C-neighbors.
        # The set {i} U N_C(i) is a clique.
        # All edges incident to i are covered by this clique.
        # We must ensure we don't create cliques of size 1.
        # If N_C(i) is empty, i is isolated in cross edges (no cross edges incident).
        
        # Group cross edges by I-vertex
        i_neighbors = {i: set() for i in range(k, n)}
        for c_v, i_v in cross_edges:
            i_neighbors[i_v].add(c_v)
        
        for i_v in range(k, n):
            neigh = i_neighbors[i_v]
            if neigh:
                # Create clique {i_v} U neigh
                clique = [i_v] + list(neigh)
                # Ensure size >= 2 (guaranteed if neigh is non-empty)
                cliques.append(clique)
        
        # 3. Cover Remaining C-Edges
        # We need to partition the set of remaining edges in C (which is a clique graph on k vertices)
        # into the minimum number of cliques (which are just subsets of C).
        # This is equivalent to edge coloring of the complement graph? 
        # No, it's the clique cover number of the graph G_C_rem.
        # Since G_C_rem is a subgraph of a complete graph K_k, any subset of vertices is a clique.
        # So we just need to cover the edges of G_C_rem with minimum number of subsets.
        # This is the minimum edge clique cover problem.
        # For small k (10, 20, 30), we can use a greedy heuristic or exact DP if k is small enough.
        # k=30 is too large for 2^k DP.
        # Greedy approach: 
        # While edges remain:
        #   Find a maximal clique in G_C_rem? No, any subset is a clique.
        #   We want to pick a subset of vertices that covers the most remaining edges.
        #   This is the Maximum Edge Cover by a Subset problem.
        #   Heuristic: Pick the pair (u,v) with edge, then expand to include all vertices w 
        #   such that (u,w) and (v,w) are in remaining edges? 
        #   Actually, since any subset is a clique, we can just pick a set of vertices S.
        #   The edges covered are all edges in G_C_rem with both endpoints in S.
        #   To maximize edges covered, we should pick S such that the induced subgraph on S is dense.
        #   Simple greedy: 
        #   1. Find the edge (u,v) with the highest sum of degrees in remaining graph?
        #   2. Or just pick the vertex with max degree, say u.
        #   3. Let S = {u} U {w | (u,w) in remaining_edges}.
        #   4. This covers all edges incident to u.
        #   5. Remove those edges.
        #   This is a standard greedy for edge cover.
        
        remaining_c_edges = set(c_edges)
        
        while remaining_c_edges:
            if time.time() > deadline - 5:
                # Fallback: list each edge
                for u, v in remaining_c_edges:
                    cliques.append([u, v])
                break
            
            # Build degree map for remaining C edges
            deg = {i: 0 for i in range(k)}
            adj_rem = {i: set() for i in range(k)}
            for u, v in remaining_c_edges:
                deg[u] += 1
                deg[v] += 1
                adj_rem[u].add(v)
                adj_rem[v].add(u)
            
            if not deg or max(deg.values()) == 0:
                break
                
            # Pick vertex with max degree
            max_v = max(range(k), key=lambda x: deg[x])
            
            # Form clique as {max_v} U neighbors(max_v) in remaining graph
            # This covers all edges incident to max_v.
            # Are there other edges we can cover? 
            # If we add a vertex w not adjacent to max_v in remaining graph, 
            # we don't cover new edges incident to max_v, but we might cover edges 
            # between w and other neighbors of max_v?
            # Let S = {max_v} U N_rem(max_v).
            # Edges covered: All (max_v, u) for u in N_rem(max_v).
            # Also edges (u, w) where u, w in N_rem(max_v) and (u,w) in remaining_c_edges.
            # So picking S = {max_v} U N_rem(max_v) is a good choice.
            
            S = {max_v} | adj_rem[max_v]
            
            # Check if S covers at least one edge (it should, since max_v has deg > 0)
            # Create clique list
            clique = list(S)
            if len(clique) < 2:
                # Should not happen if deg[max_v] > 0
                # But if max_v has no neighbors, it's isolated in remaining graph?
                # No, deg[max_v] > 0 implies neighbors exist.
                # However, if the only edge was (max_v, u), S={max_v, u}.
                pass
            
            cliques.append(clique)
            
            # Remove all edges with both endpoints in S
            S_set = set(S)
            new_remaining = set()
            for u, v in remaining_c_edges:
                if u in S_set and v in S_set:
                    continue
                new_remaining.add((u, v))
            remaining_c_edges = new_remaining
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 15:
            # Emergency fallback
            edges = instance(*spec)
            results.append([[int(e[0]), int(e[1])] for e in edges])
            continue
        
        edges = instance(*spec)
        cliques = partition_split_graph(spec[0], spec[1], edges)
        
        # Convert to lists of Python ints
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
        
        results.append(final_cliques)
    
    return results
# EVOLVE-BLOCK-END
