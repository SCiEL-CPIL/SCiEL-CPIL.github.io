"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s kill
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def solve_instance(adj):
        n = len(adj)
        if n == 0:
            return []
        
        # Track remaining edges as a set of tuples (u, v) with u < v
        remaining_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    remaining_edges.add((u, v))
        
        if not remaining_edges:
            return []

        # We will greedily select cliques.
        # To find a good clique, we pick a seed vertex and expand it to a maximal clique
        # within the remaining edges.
        # To maximize edges covered per clique, we prioritize seeds that are part of many
        # remaining edges and have high connectivity among their neighbors in the remaining graph.
        
        cliques_found = []
        
        # Active vertices: vertices that have at least one remaining edge incident to them
        # We can maintain a set of active vertices, or just check degrees on the fly.
        # Given n <= 64, on-the-fly checks are cheap.
        
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: cover remaining edges individually to ensure validity
                # This is suboptimal but valid.
                for e in remaining_edges:
                    cliques_found.append(list(e))
                return cliques_found

            # 1. Select a seed vertex
            # Heuristic: Pick the vertex with the highest degree in the remaining graph.
            # To break ties, pick the one whose neighbors are most densely connected?
            # For now, simple degree is a strong baseline.
            max_deg = -1
            seed = -1
            for u in range(n):
                deg = sum(1 for v in adj[u] if (min(u,v), max(u,v)) in remaining_edges)
                if deg > max_deg:
                    max_deg = deg
                    seed = u
            
            if seed == -1:
                # No edges left? Should be handled by while loop.
                break

            # 2. Expand seed to a maximal clique in the remaining graph
            # The "remaining graph" induced by a set of vertices S consists of edges
            # in remaining_edges where both endpoints are in S.
            # We want a clique C such that all edges of C are in remaining_edges.
            # This is equivalent to finding a clique in the graph G_rem where G_rem(u,v)=1 iff (u,v) in remaining_edges.
            
            # Greedy expansion:
            # Start with C = {seed}
            # Candidates = neighbors of seed in G_rem
            # While Candidates is not empty:
            #   Pick v from Candidates that maximizes |Candidates ∩ N(v) in G_rem| (or similar heuristic)
            #   Add v to C
            #   Candidates = Candidates ∩ N(v) in G_rem
            
            # To speed up, we represent the "remaining adjacency" for a set of vertices
            # implicitly.
            
            C = {seed}
            # Candidates must be neighbors of all vertices in C in the remaining graph.
            # Initially, neighbors of seed in remaining graph.
            candidates = set()
            for v in adj[seed]:
                e = (min(seed, v), max(seed, v))
                if e in remaining_edges:
                    candidates.add(v)
            
            while candidates:
                # Select next vertex from candidates
                # Heuristic: Pick v that maximizes the number of edges it would add to the clique
                # which are already "available" (i.e., v is connected to all current C, and we check connectivity within candidates)
                # Actually, a standard greedy maximal clique heuristic:
                # Pick v that has the most neighbors within `candidates` in the remaining graph.
                
                best_v = -1
                best_score = -1
                
                # Optimization: If len(candidates) is small, check all.
                # For each v in candidates, count how many other nodes in candidates are connected to v in remaining_edges.
                # This ensures we build a dense clique.
                
                for v in candidates:
                    # Count neighbors of v in candidates within remaining_edges
                    cnt = 0
                    for u in candidates:
                        if u == v:
                            continue
                        e = (min(u, v), max(u, v))
                        if e in remaining_edges:
                            cnt += 1
                    if cnt > best_score:
                        best_score = cnt
                        best_v = v
                
                if best_v == -1:
                    break
                
                # Add best_v to clique
                C.add(best_v)
                
                # Update candidates:
                # New candidates must be neighbors of best_v in remaining_edges
                # AND be in the old candidates.
                new_candidates = set()
                for u in candidates:
                    if u == best_v:
                        continue
                    e = (min(u, best_v), max(u, best_v))
                    if e in remaining_edges:
                        new_candidates.add(u)
                
                candidates = new_candidates

            # C is now a clique in the remaining graph.
            # Verify C has at least 2 vertices.
            if len(C) < 2:
                # If we couldn't form a clique of size >= 2, it means the seed had no neighbors in remaining graph?
                # But seed was picked with max degree > 0 (unless max_deg was 0, but then remaining_edges is empty).
                # If max_deg > 0, candidates was not empty initially.
                # If candidates became empty immediately, it means seed had neighbors but they weren't connected to each other?
                # No, candidates are neighbors of seed. If we pick best_v, we add it.
                # The only way C stays size 1 is if candidates was empty at start.
                # This implies seed has degree 0 in remaining graph. Contradiction with max_deg > 0.
                # However, if max_deg == 0, we break.
                # So C should always be size >= 2 if remaining_edges is not empty.
                # Edge case: if remaining_edges has isolated edges?
                # If seed has degree 1, candidates has 1 node. We pick it. C becomes size 2.
                # So C should be valid.
                pass

            # 3. Add C to result and remove its edges from remaining_edges
            cliques_found.append(list(C))
            
            # Remove all edges within C from remaining_edges
            C_list = list(C)
            for i in range(len(C_list)):
                u = C_list[i]
                for j in range(i + 1, len(C_list)):
                    v = C_list[j]
                    e = (min(u, v), max(u, v))
                    if e in remaining_edges:
                        remaining_edges.remove(e)
            
            # Optional: Prune adjacency sets? Not strictly necessary as we check membership in remaining_edges.
            # But `adj` is static. We rely on `remaining_edges` for validity.
            
        return cliques_found

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        if time.time() > deadline - 5.0:
            # If we are running out of time before even starting an instance,
            # we cannot guarantee a valid result for it.
            # However, the task requires a result.
            # If we haven't started, we return empty list which is invalid.
            # But we assume the first instance is fast.
            # If we are here, it means previous instance took too long.
            # We must return *something*.
            # The contract says "Every listed set must be a clique...".
            # An empty list of cliques implies no edges are covered.
            # If the graph has edges, this is invalid.
            # We should try to compute a trivial solution (single edges) if we run out of time?
            # But we don't have the graph generated yet if we skip.
            # Let's generate the graph and do the trivial fallback immediately.
            adj = random_graph(n, p, seed)
            # Trivial fallback: each edge is a clique of size 2
            trivial_cliques = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        trivial_cliques.append([u, v])
            results.append(trivial_cliques)
            continue

        adj = random_graph(n, p, seed)
        cliques = solve_instance(adj)
        results.append(cliques)
        
    return results

# EVOLVE-BLOCK-END
