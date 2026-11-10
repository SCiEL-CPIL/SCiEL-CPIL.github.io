"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_partition(n, edges, seed_heur):
        # Precompute initial neighbors
        neighbors = [set() for _ in range(n)]
        for u, v in edges:
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        remaining_edges = set(edges)
        cliques = []
        rng_heur = random.Random(seed_heur)
        
        # To speed up, we can track active vertices
        active_vertices = set(range(n))
        
        while remaining_edges:
            # 1. Find the best edge to start a clique
            # Heuristic: Maximize |N(u) ∩ N(v)| among remaining edges
            best_edge = None
            best_score = -1
            # Sample edges if too many, but for N=64, E<2000, so iterate all
            # To add randomness, we can shuffle the list of edges
            edge_list = list(remaining_edges)
            rng_heur.shuffle(edge_list)
            
            # We don't need to check ALL edges if we find a very good one early,
            # but for simplicity and correctness, we check a subset or all.
            # Given N is small, checking all is fine.
            # Optimization: Only check edges where both endpoints are active (they are, by definition of remaining_edges)
            
            # To improve speed, we can limit the search for the best edge to a random sample if N was large,
            # but here N=64 is small.
            for u, v in edge_list:
                # Intersection of neighbors
                # Note: neighbors[u] contains ALL neighbors, we need to intersect with active_vertices
                # Actually, if an edge is in remaining_edges, its endpoints are active.
                # But neighbors might contain inactive vertices.
                # common = neighbors[u] & neighbors[v] & active_vertices
                # However, calculating intersection for every edge is O(E * V).
                # E ~ 1000, V ~ 64. 64000 ops. Very fast.
                
                # We can optimize by precomputing or just doing it.
                common = neighbors[u] & neighbors[v]
                # Filter common to only active vertices? 
                # If a vertex is inactive, it shouldn't be in the clique.
                # But if (u,v) is in remaining_edges, u and v are active.
                # Any w in common that is inactive means the edge (u,w) or (v,w) was removed.
                # If (u,w) was removed, w is still active but not connected to u anymore?
                # No, "remaining_edges" tracks which edges are left.
                # neighbors[u] is updated to remove edges when they are covered.
                # So neighbors[u] only contains active neighbors connected by remaining edges.
                # Thus, neighbors[u] & neighbors[v] is exactly the set of vertices that can be added to clique {u,v}.
                
                score = len(common)
                if score > best_score:
                    best_score = score
                    best_edge = (u, v)
                    # Early termination? If best_score is very high, we might stop.
                    # But let's stick to finding the global max among the shuffled list for better diversity.
                    # Actually, shuffling and taking the first "good" one might be faster but less optimal.
                    # Let's just find the max.
            
            if best_edge is None:
                break
                
            u, v = best_edge
            C = [u, v]
            P = neighbors[u] & neighbors[v]
            
            # 2. Greedily expand the clique
            while P:
                # Find w in P with max degree in P
                max_deg = -1
                candidates = []
                for w in P:
                    deg = len(neighbors[w] & P)
                    if deg > max_deg:
                        max_deg = deg
                        candidates = [w]
                    elif deg == max_deg:
                        candidates.append(w)
                
                if not candidates:
                    break
                    
                # Random tie-breaking
                w = rng_heur.choice(candidates)
                C.append(w)
                P = P & neighbors[w]
            
            # 3. Remove edges of C from remaining graph
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    a, b = C[i], C[j]
                    e = (a, b) if a < b else (b, a)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                        neighbors[a].discard(b)
                        neighbors[b].discard(a)
            
            cliques.append(C)
            
        return cliques

    def solve_instance(n, p, seed, time_limit):
        edges = random_graph(n, p, seed)
        if not edges:
            return []
        
        best_cliques = None
        best_count = float('inf')
        
        attempt = 0
        while time.time() < time_limit:
            heur_seed = seed + attempt * 1000 + 42
            cliques = greedy_partition(n, edges, heur_seed)
            
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
                # Optional: if we reached a lower bound, we could stop, but we don't know the LB easily.
            
            attempt += 1
                
        if best_cliques is None:
            # Fallback: each edge is a clique
            best_cliques = [[u, v] for u, v in edges]
            
        return best_cliques

    # Allocate time: 35s for n=32, rest for n=64
    t1_limit = start_time + 35.0
    res1 = solve_instance(32, 0.6, 601, t1_limit)
    
    t2_limit = deadline
    res2 = solve_instance(64, 0.55, 602, t2_limit)
    
    return [res1, res2]
# EVOLVE-BLOCK-END
