"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    def random_graph(n, p, seed):
        rng = Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_partition(n, edges, seed_heur):
        neighbors = [set() for _ in range(n)]
        for u, v in edges:
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        remaining_edges = set(edges)
        cliques = []
        rng_heur = Random(seed_heur)
        
        while remaining_edges:
            # Find v with max degree in remaining graph
            max_deg = -1
            candidates = []
            for u in range(n):
                deg = len(neighbors[u])
                if deg > max_deg:
                    max_deg = deg
                    candidates = [u]
                elif deg == max_deg and deg > 0:
                    candidates.append(u)
            
            if not candidates:
                break
            v = rng_heur.choice(candidates)
            
            C = [v]
            P = set(neighbors[v])
            
            while P:
                # Find u in P with max degree in induced subgraph P
                max_score = -1
                tie_candidates = []
                for u in P:
                    score = len(neighbors[u] & P)
                    if score > max_score:
                        max_score = score
                        tie_candidates = [u]
                    elif score == max_score:
                        tie_candidates.append(u)
                
                u = rng_heur.choice(tie_candidates)
                C.append(u)
                P = P & neighbors[u]
            
            # Remove edges of C from remaining_edges and neighbors
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, v2 = C[i], C[j]
                    e = (u, v2) if u < v2 else (v2, u)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                    
                    neighbors[u].discard(v2)
                    neighbors[v2].discard(u)
                    
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
            heur_seed = seed + attempt * 1000
            cliques = greedy_partition(n, edges, heur_seed)
            
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
            
            attempt += 1
                
        if best_cliques is None:
            best_cliques = [[u, v] for u, v in edges]
            
        return best_cliques

    # Allocate time: 30s for n=32, rest for n=64
    t1_limit = start_time + 30.0
    res1 = solve_instance(32, 0.6, 601, t1_limit)
    
    t2_limit = deadline
    res2 = solve_instance(64, 0.55, 602, t2_limit)
    
    return [res1, res2]
# EVOLVE-BLOCK-END
