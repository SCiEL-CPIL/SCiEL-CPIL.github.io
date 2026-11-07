"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for 120s limit

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        # Build adjacency sets
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
        
        cliques = []
        remaining_edges = set(edge_set)
        
        # Step 1: Cover cross edges (edges between clique part 0..k-1 and independent set k..n-1)
        # For each vertex w in independent set, let N_w be its neighbors in 0..k-1.
        # Since 0..k-1 is a clique, N_w is a clique. {N_w union {w}} is a clique.
        # We want to choose subsets of N_w to cover cross edges while maximizing internal edge coverage.
        # Actually, if we use the full N_w, we cover all cross edges incident to w.
        # The internal edges covered are all pairs in N_w.
        # To minimize total cliques, we should cover cross edges as efficiently as possible.
        # Using one clique per independent vertex (if it has any cross edges) is optimal for cross edges
        # because each independent vertex's cross edges can only be covered by cliques containing that vertex,
        # and any such clique can include at most one independent vertex.
        # So we MUST use at least one clique per independent vertex with degree > 0.
        # The question is: what subset of N_w to include?
        # If we include a subset S of N_w, we cover cross edges (u,w) for u in S, and internal edges (u,v) for u,v in S.
        # If we include all of N_w, we cover all cross edges for w.
        # Is there any reason NOT to include all of N_w?
        # If we don't include some u in N_w, we need another clique to cover (u,w). But any other clique containing w
        # would also contain only independent vertices? No, a clique can contain at most one independent vertex.
        # So (u,w) must be covered by a clique containing w.
        # If we split N_w into multiple cliques with w, we use more cliques for the same cross edges.
        # So it's best to use ONE clique per independent vertex, containing all its neighbors in the clique part.
        # This covers all cross edges in exactly (n-k) cliques (for those with degree > 0).
        # And it covers many internal edges (all pairs in N_w).
        
        # Let's verify: is it always optimal to take the full neighborhood?
        # Suppose we take a proper subset S of N_w. Then we need another clique to cover (u,w) for u in N_w \ S.
        # That other clique must also contain w (since w is independent).
        # So we'd have two cliques containing w: {S, w} and {N_w\S, w} (or subsets thereof).
        # This uses 2 cliques instead of 1 for the same cross edges.
        # The internal edges covered: pairs in S and pairs in N_w\S.
        # Versus: pairs in N_w.
        # Since N_w is a clique, pairs in N_w = pairs in S + pairs in N_w\S + cross pairs between S and N_w\S.
        # So taking the full N_w covers MORE internal edges with FEWER cliques.
        # Therefore, it is strictly better to take the full neighborhood.
        
        # So the strategy is:
        # 1. For each w in k..n-1 with degree > 0 in the cross edges:
        #    Let N_w = {u in 0..k-1 : (u,w) is an edge}.
        #    Add clique {N_w union {w}}.
        #    Remove all edges in this clique from remaining_edges.
        # 2. Cover remaining internal edges (within 0..k-1) with minimum cliques.
        
        # Step 1 implementation:
        for w in range(k, n):
            if time.time() > deadline:
                break
            neighbors = [u for u in range(k) if (u, w) in remaining_edges]
            if not neighbors:
                continue
            
            clique = neighbors + [w]
            # Remove all edges in this clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    e = (u, v) if u < v else (v, u)
                    remaining_edges.discard(e)
            cliques.append(clique)
        
        # Step 2: Cover remaining edges within 0..k-1
        # The remaining edges form a graph G' on vertices 0..k-1.
        # We need to cover G' with minimum number of cliques.
        # This is the edge clique cover problem on G'.
        # G' is a subgraph of K_k.
        # We can use a greedy algorithm: repeatedly find a large clique in G', remove its edges, repeat.
        
        # To find a large clique, we can use a heuristic:
        # - Start with an edge (u,v) from G'.
        # - Try to expand it to a maximal clique by adding vertices connected to all current members.
        
        # For better results, we can try multiple starting edges and pick the one that gives the largest clique.
        # Or use a simple maximum clique heuristic.
        
        # Let's implement a greedy edge clique cover:
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: return remaining edges as individual cliques
                for e in remaining_edges:
                    cliques.append(list(e))
                break
            
            # Pick an edge to start with
            # Heuristic: pick the edge with the maximum sum of degrees in G'
            # Or just pick the first one for simplicity.
            # Let's pick the edge with the most common neighbors.
            
            best_edge = None
            best_common_count = -1
            
            # To speed up, we can just pick the first edge. But let's try a bit smarter.
            # Since k is small (up to 30), we can afford to check a few edges.
            # Let's just pick the first edge for now, and see if that's good enough.
            # Actually, let's try to find a maximal clique containing a random edge, or the edge with max degree sum.
            
            # Let's compute degrees in G'
            deg = [0] * k
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            # Pick edge with max degree sum
            max_score = -1
            for u, v in remaining_edges:
                score = deg[u] + deg[v]
                if score > max_score:
                    max_score = score
                    best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u0, v0 = best_edge
            
            # Find a maximal clique containing u0 and v0
            # Start with {u0, v0}
            clique = {u0, v0}
            
            # Candidates: vertices in 0..k-1 connected to all in clique
            candidates = [x for x in range(k) if x not in clique]
            
            # Filter candidates to those connected to all in clique
            valid_candidates = []
            for x in candidates:
                if all((min(x, y), max(x, y)) in remaining_edges for y in clique):
                    valid_candidates.append(x)
            
            # Greedily add vertices to clique
            # Sort valid_candidates by degree in G' (descending) to prioritize high-degree vertices
            valid_candidates.sort(key=lambda x: deg[x], reverse=True)
            
            for x in valid_candidates:
                if time.time() > deadline:
                    break
                # Check if x is still connected to all in clique
                if all((min(x, y), max(x, y)) in remaining_edges for y in clique):
                    clique.add(x)
            
            # Convert clique to list
            clique_list = list(clique)
            cliques.append(clique_list)
            
            # Remove all edges in this clique from remaining_edges
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    u, v = clique_list[i], clique_list[j]
                    e = (u, v) if u < v else (v, u)
                    remaining_edges.discard(e)
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # If we run out of time, we have to return something valid.
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            res = solve_split(*spec)
            # Ensure types are correct
            clean_res = []
            for clique in res:
                clean_clique = [int(v) for v in clique]
                clean_res.append(clean_clique)
            results.append(clean_res)
            
    return results
# EVOLVE-BLOCK-END
