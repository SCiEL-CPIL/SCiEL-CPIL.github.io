"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

    def solve_instance(n, p, seed):
        edges = set(instance(n, p, seed))
        # Normalize edges to (min, max)
        edges = {(min(u, v), max(u, v)) for u, v in edges}
        
        cliques = []
        
        while edges:
            # Calculate degrees
            deg = {}
            for u, v in edges:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            if not deg:
                break

            # Get top K candidates
            # Sort by degree descending
            sorted_vertices = sorted(deg.keys(), key=lambda x: deg[x], reverse=True)
            top_k = sorted_vertices[:min(10, len(sorted_vertices))]
            
            best_clique = []
            best_size = 0
            
            for start_v in top_k:
                clique = greedy_clique(start_v, edges, n)
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
            
            if not best_clique:
                # Fallback: just pick an edge
                u, v = next(iter(edges))
                best_clique = [u, v]
            
            cliques.append(best_clique)
            
            # Remove edges covered by best_clique
            to_remove = set()
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    u, v = best_clique[i], best_clique[j]
                    if u < v:
                        to_remove.add((u, v))
                    else:
                        to_remove.add((v, u))
            
            edges -= to_remove
            
            # Time check
            if time.time() > deadline - 5:
                # If we are running out of time, finish quickly
                # Just add remaining edges as pairs
                for u, v in edges:
                    cliques.append([u, v])
                break

        return cliques

    def greedy_clique(start_v, edges, n):
        clique = [start_v]
        # Candidates must be connected to ALL vertices in clique
        candidates = set()
        for u in range(n):
            if u != start_v:
                if (min(start_v, u), max(start_v, u)) in edges:
                    candidates.add(u)
        
        while candidates:
            # Calculate degrees within candidates subgraph
            cand_deg = {v: 0 for v in candidates}
            cand_list = list(candidates)
            for i in range(len(cand_list)):
                u = cand_list[i]
                for j in range(i + 1, len(cand_list)):
                    v = cand_list[j]
                    if (min(u, v), max(u, v)) in edges:
                        cand_deg[u] += 1
                        cand_deg[v] += 1
            
            if not any(d > 0 for d in cand_deg.values()):
                break
                
            # Pick vertex with max degree in candidate subgraph
            next_v = max(cand_deg, key=cand_deg.get)
            clique.append(next_v)
            
            # Update candidates: intersect with neighbors of next_v
            new_candidates = set()
            for u in candidates:
                if u == next_v:
                    continue
                if (min(next_v, u), max(next_v, u)) in edges:
                    new_candidates.add(u)
            candidates = new_candidates
            
        return clique

    res1 = solve_instance(32, 0.6, 601)
    res2 = solve_instance(64, 0.55, 602)
    
    return [res1, res2]
# EVOLVE-BLOCK-END
