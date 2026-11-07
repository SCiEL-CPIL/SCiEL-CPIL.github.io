import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    results = []
    start_time = time.time()
    deadline = start_time + 115.0

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        
        # Build adjacency and edge list
        neighbors = [set() for _ in range(n)]
        edge_set = set()
        
        # C vertices: 0..k-1
        for u in range(k):
            for v in range(u + 1, k):
                neighbors[u].add(v)
                neighbors[v].add(u)
                edge_set.add((u, v))
                
        # I vertices: k..n-1
        rng = random.Random(seed)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edge_set.add((min(u, v), max(u, v)))

        # Precompute all maximal cliques? No, too many.
        # Strategy: Greedy Set Cover on edges.
        # State: set of uncovered edges.
        # Step: Find clique that covers max uncovered edges.
        
        uncovered_edges = set(edge_set)
        cliques = []
        
        # Helper to check if a set is a clique
        def is_clique(vertices):
            vlist = list(vertices)
            for i in range(len(vlist)):
                for j in range(i+1, len(vlist)):
                    if vlist[j] not in neighbors[vlist[i]]:
                        return False
            return True

        # To speed up clique expansion, we can use Bron-Kerbosch or simple intersection.
        # Given small n, simple intersection is fine.
        
        while uncovered_edges:
            if time.time() > deadline:
                # Fallback: cover remaining edges individually
                for e in list(uncovered_edges):
                    cliques.append([e[0], e[1]])
                break
            
            best_clique = None
            best_count = -1
            best_vertices = None
            
            # We can restrict candidate seeds to edges in uncovered_edges
            # But expanding from every edge might be slow if we do it naively.
            # Optimization: Only consider edges whose endpoints are 'active'
            
            # A faster heuristic: 
            # Pick an uncovered edge (u,v).
            # Expand to maximal clique containing (u,v).
            # Calculate score.
            # Do this for a sample of edges if too slow, but n=90 is small.
            
            # Let's iterate over all uncovered edges
            # To avoid duplicate work, we can pick one, find its maximal clique, 
            # and compare with a few others.
            
            # Actually, finding the global max clique covering max uncovered edges is hard.
            # Let's stick to: For each uncovered edge, find the maximal clique containing it.
            # This is O(E * n^2). 3000 * 8100 = 24M ops. Too slow in Python if done every iteration.
            
            # Better:
            # 1. Pick a random uncovered edge (u,v).
            # 2. Expand to maximal clique M.
            # 3. Count uncovered edges in M.
            # 4. Try a few other random edges to see if we get a better score.
            # 5. Pick the best found.
            
            candidates = []
            edge_list = list(uncovered_edges)
            
            # Sample up to 50 edges to find a good clique
            sample_size = min(50, len(edge_list))
            sampled_edges = random.sample(edge_list, sample_size)
            
            for u, v in sampled_edges:
                # Expand (u, v) to maximal clique
                # Start with {u, v}
                clique = {u, v}
                # Candidates: common neighbors of u and v
                common = neighbors[u] & neighbors[v]
                # Remove u, v from common? No, they are in clique.
                # We want to add x to clique if x is connected to all in clique.
                
                # Simple expansion:
                # Iterate through common neighbors
                # If x is connected to all current clique members, add it.
                # This finds A maximal clique, but not necessarily THE maximal one containing u,v 
                # in terms of size, but usually good enough for covering.
                
                # Better expansion:
                # Candidates = common neighbors
                # While candidates:
                #   pick x
                #   if x connected to all in clique: add x, update candidates (intersect with neighbors[x])
                #   else: remove x
                
                curr_clique = {u, v}
                candidates = set(neighbors[u] & neighbors[v])
                candidates -= curr_clique
                
                while candidates:
                    # Pick one candidate
                    x = next(iter(candidates))
                    # Check if x connects to all in curr_clique
                    can_add = True
                    for y in curr_clique:
                        if y not in neighbors[x]:
                            can_add = False
                            break
                    if can_add:
                        curr_clique.add(x)
                        # Update candidates: must be connected to x
                        candidates &= neighbors[x]
                        candidates -= curr_clique
                    else:
                        candidates.discard(x)
                
                # Calculate score: number of uncovered edges in curr_clique
                score = 0
                for e in combinations(sorted(curr_clique), 2):
                    if e in uncovered_edges:
                        score += 1
                
                if score > best_count:
                    best_count = score
                    best_clique = curr_clique
                    best_vertices = sorted(curr_clique)
            
            if best_clique:
                # Add this clique
                cliques.append(best_vertices)
                # Remove covered edges
                for e in combinations(best_vertices, 2):
                    uncovered_edges.discard((min(e[0], e[1]), max(e[0], e[1])))
            else:
                # Should not happen if uncovered_edges not empty
                u, v = next(iter(uncovered_edges))
                cliques.append([u, v])
                uncovered_edges.discard((min(u, v), max(u, v)))

        # Local Optimization: Merge cliques if possible
        # Repeat until no more merges
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            new_cliques = []
            # Sort cliques by size descending to try merging larger ones first?
            # Or just pairwise check.
            # To be efficient, we can check if union of any two is a clique.
            
            # Simple O(M^2) check
            i = 0
            while i < len(cliques):
                merged = False
                for j in range(i + 1, len(cliques)):
                    c1 = set(cliques[i])
                    c2 = set(cliques[j])
                    union = c1 | c2
                    if is_clique(union):
                        # Merge
                        cliques[i] = sorted(union)
                        # Remove j
                        cliques.pop(j)
                        merged = True
                        changed = True
                        break
                if not merged:
                    i += 1
            
            # Also try to remove vertices from cliques if they are not needed?
            # No, every vertex in a clique contributes to covering edges.
            # However, if a vertex in a clique is not connected to any other vertex in that clique 
            # that is in another clique... 
            # Actually, minimal clique partition usually implies minimal cliques.
            # But we can try to "shrink" a clique if removing a vertex still allows 
            # the edges to be covered by other cliques? 
            # This is complex. Stick to merging.

        results.append([[int(x) for x in cl] for cl in cliques])

    return results
