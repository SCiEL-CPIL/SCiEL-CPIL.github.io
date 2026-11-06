import time
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin before 150s kill
    
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        
        # Build adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Precompute triangle counts for each edge
        # An edge (u, v) is in a triangle if there exists w such that (u, w) and (v, w) are edges.
        # triangle_count[(u, v)] = |adj[u] & adj[v]|
        triangle_count = {}
        for u, v in edges:
            common = adj[u].intersection(adj[v])
            triangle_count[(u, v)] = len(common)
        
        best_cliques = [list(e) for e in edges] # Fallback: each edge is a clique
        best_count = len(best_cliques)
        
        end_time = start_time + time_budget
        
        # We will run multiple randomized greedy restarts
        attempts = 0
        while time.time() < end_time:
            attempts += 1
            attempt_rng = random.Random(seed + attempts * 12345 + 67890)
            
            remaining_edges = set(edges)
            cliques = []
            
            # Precompute a list of edges sorted by triangle count for initial selection?
            # No, triangle counts change as edges are removed? 
            # Actually, if we remove edges, the "potential" of remaining edges to form larger cliques 
            # might change, but the original triangle count is a good static proxy for density.
            # However, we need to pick from remaining_edges.
            
            while remaining_edges:
                # Pick a seed edge. 
                # Strategy: Pick the edge in remaining_edges with the highest triangle count.
                # This biases towards dense regions.
                # To avoid O(E) scan every time, we could maintain a heap, but E is small (<= 2000).
                # For n=64, E ~ 1100. Scanning is fast.
                
                best_seed_edge = None
                max_tri = -1
                # Add a bit of randomness to tie-break or diversify if many have same count
                # We can just pick the max. If tie, random.
                
                # Optimization: If the set is large, sampling might be better, but for n=64, full scan is fine.
                for e in remaining_edges:
                    tc = triangle_count.get(e, 0)
                    if tc > max_tri:
                        max_tri = tc
                        best_seed_edge = e
                    elif tc == max_tri and best_seed_edge is not None:
                        # Random tie break
                        if attempt_rng.random() < 0.5:
                            best_seed_edge = e

                if best_seed_edge is None:
                    best_seed_edge = next(iter(remaining_edges))

                u, v = best_seed_edge
                current_clique = [u, v]
                
                # Candidates: vertices w such that (w, c) in remaining for all c in current_clique
                # We will greedily add vertices that maximize the number of edges covered.
                
                cand_set = set(adj[u])
                cand_set.intersection_update(adj[v])
                cand_set.discard(u)
                cand_set.discard(v)
                
                # Filter candidates to those with all edges to current_clique in remaining
                def has_all_edges_to_clique(w, clique, rem_edges):
                    for c in clique:
                        e = (min(w, c), max(w, c))
                        if e not in rem_edges:
                            return False
                    return True

                cand_set = {w for w in cand_set if has_all_edges_to_clique(w, current_clique, remaining_edges)}
                
                while cand_set:
                    # Pick the best candidate
                    # Metric: Number of edges in remaining incident to w that connect to other candidates or clique?
                    # A good heuristic: Count how many edges in remaining_edges would be covered if we added w.
                    # This is simply the number of edges (w, c) where c is in current_clique (which is |current_clique|)
                    # PLUS edges (w, x) where x is in cand_set? No, we only commit w to the clique.
                    # The edges covered by adding w are (w, c) for c in current_clique.
                    # These are already known to be in remaining_edges.
                    # So all candidates cover exactly |current_clique| edges immediately.
                    # However, adding w restricts future candidates.
                    # We want to pick w that allows for the largest future clique.
                    # Heuristic: Pick w that has the most connections to other candidates in cand_set.
                    # This keeps the candidate set "dense".
                    
                    best_w = None
                    max_score = -1
                    for w in cand_set:
                        # Count neighbors of w in cand_set
                        # Also consider triangle count of edges (w, c)?
                        score = 0
                        for x in cand_set:
                            if x == w: continue
                            if x in adj[w]:
                                score += 1
                        # Tie breaker: random
                        if score > max_score:
                            max_score = score
                            best_w = w
                        elif score == max_score and best_w is not None:
                            if attempt_rng.random() < 0.5:
                                best_w = w

                    if best_w is None:
                        break
                        
                    current_clique.append(best_w)
                    cand_set.remove(best_w)
                    
                    # Update candidates: must be neighbor of best_w and have edges to all clique members
                    # New candidates must be in cand_set (so they have edges to old clique) 
                    # AND be neighbor of best_w (and edge in remaining)
                    new_cand_set = set()
                    for w in cand_set:
                        if w in adj[best_w]:
                            e_bw = (min(w, best_w), max(w, best_w))
                            if e_bw in remaining_edges:
                                new_cand_set.add(w)
                    cand_set = new_cand_set

                # Commit this clique
                clique_edges = set()
                for i in range(len(current_clique)):
                    for j in range(i+1, len(current_clique)):
                        clique_edges.add((min(current_clique[i], current_clique[j]), max(current_clique[i], current_clique[j])))
                
                cliques.append(current_clique)
                for e in clique_edges:
                    remaining_edges.discard(e)
            
            # Check if this solution is better
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = cliques
                
            if time.time() > end_time - 2:
                break

        return best_cliques

    # Solve both instances
    t1 = time.time()
    # Allocate time based on complexity. n=64 is much harder.
    # Give n=32 a smaller budget, n=64 the rest.
    budget1 = 10  # n=32 is fast, 10s should be plenty for good solutions
    cliques1 = solve_instance(32, 0.6, 601, budget1)
    
    t2 = time.time()
    remaining_time = DEADLINE - (t2 - start_time)
    budget2 = max(0, remaining_time)
    cliques2 = solve_instance(64, 0.55, 602, budget2)
    
    return [cliques1, cliques2]
