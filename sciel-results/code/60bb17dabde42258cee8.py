"""
Strategy: Maximum-weight edge-coloring heuristic.
Instead of picking arbitrary large cliques, we treat this as a coloring problem of the line graph.
We iteratively select the 'best' clique to cover remaining edges, where 'best' is defined by the
potential to cover many edges relative to the size (efficiency) and density.
We use a weighted greedy expansion for each candidate clique start to ensure we pick cliques
that are not just large, but 'dense' in terms of remaining edge coverage.
"""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def get_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def clique_partition(n, p, seed):
        adj = get_graph(n, p, seed)
        remaining_edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    remaining_edges.add((u, v))
        
        cliques = []
        
        # Pre-compute adjacency for fast lookups
        # adj is list of sets of neighbors
        
        while remaining_edges:
            if time.time() > deadline - 5.0:
                break
            
            # We want to find a clique that covers the most remaining edges.
            # A clique of size k covers k*(k-1)/2 edges.
            # We want to maximize this count among valid cliques formed by remaining edges.
            
            # Heuristic: 
            # 1. Identify vertices with high degree in the *remaining* graph.
            # 2. For each such vertex, try to expand into a clique using only remaining edges.
            # 3. Score the clique by the number of remaining edges it covers.
            
            # Calculate degrees in remaining graph
            deg = defaultdict(int)
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            # Get top candidates to start from
            # We only care about vertices that are part of remaining edges
            candidates = [v for v in range(n) if deg[v] > 0]
            
            # Sort by degree descending to try "central" vertices first
            candidates.sort(key=lambda x: deg[x], reverse=True)
            
            best_clique = None
            best_edges_covered = 0
            
            # Try a limited number of starting vertices to save time
            # For n=64, trying top 20-30 is usually enough to find a good clique
            max_starts = min(30, len(candidates))
            
            for start in candidates[:max_starts]:
                # Greedy expansion
                clique = [start]
                # Candidates for next vertex: neighbors of start that have edge to start in remaining
                potential = set()
                for v in adj[start]:
                    e = (min(start, v), max(start, v))
                    if e in remaining_edges:
                        potential.add(v)
                
                # Order potential vertices by how many edges they share with current clique
                # This helps build a dense clique quickly
                while potential:
                    best_next = None
                    best_score = -1
                    
                    # Optimization: if clique is small, just check all. 
                    # If large, we might prune, but for n<=64 it's fine.
                    for v in potential:
                        # Check if v connects to all in clique
                        valid = True
                        edges_incl = 0
                        for u in clique:
                            if u == v: continue
                            e = (min(u, v), max(u, v))
                            if e not in remaining_edges:
                                valid = False
                                break
                            else:
                                edges_incl += 1
                        if not valid:
                            continue
                        
                        # Score: number of new edges added by adding v
                        # Also prefer vertices with high remaining degree to keep expansion potential
                        score = edges_incl + deg[v] * 0.1 
                        
                        if score > best_score:
                            best_score = score
                            best_next = v
                    
                    if best_next is None:
                        break
                    
                    clique.append(best_next)
                    potential.remove(best_next)
                    
                    # Update potential: remove vertices that don't connect to best_next
                    # Actually, we re-filter in the next iteration's validity check, 
                    # but removing non-neighbors from 'potential' set speeds it up.
                    # We need 'potential' to contain only vertices connected to ALL in clique.
                    # The loop above checks validity against ALL in clique.
                    # So we can just keep 'potential' as is, but we must ensure 
                    # we don't pick a vertex that is not connected to the new one.
                    # The validity check handles this.
                    
                    # Wait, the `potential` set initially contains neighbors of `start`.
                    # When we add `best_next`, any vertex in `potential` that is not connected 
                    # to `best_next` will fail the validity check in the next iteration.
                    # To speed up, we can intersect `potential` with `adj[best_next]` 
                    # and also with the set of vertices connected via remaining edges.
                    
                    new_potential = set()
                    for v in potential:
                        if v in adj[best_next]:
                            e = (min(v, best_next), max(v, best_next))
                            if e in remaining_edges:
                                new_potential.add(v)
                    potential = new_potential

                if len(clique) >= 2:
                    # Calculate edges covered
                    edges_covered = 0
                    for i in range(len(clique)):
                        for j in range(i+1, len(clique)):
                            e = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                            if e in remaining_edges:
                                edges_covered += 1
                    
                    if edges_covered > best_edges_covered:
                        best_edges_covered = edges_covered
                        best_clique = clique
            
            if best_clique is None or best_edges_covered == 0:
                # Fallback: pick any remaining edge
                e = next(iter(remaining_edges))
                cliques.append([e[0], e[1]])
                remaining_edges.remove(e)
            else:
                # Remove edges of the best clique
                for i in range(len(best_clique)):
                    for j in range(i+1, len(best_clique)):
                        e = (min(best_clique[i], best_clique[j]), max(best_clique[i], best_clique[j]))
                        if e in remaining_edges:
                            remaining_edges.remove(e)
                cliques.append(best_clique)
        
        # Handle any remaining edges if we timed out
        if remaining_edges:
            for e in remaining_edges:
                cliques.append([e[0], e[1]])
        
        return cliques

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        cliques = clique_partition(n, p, seed)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
