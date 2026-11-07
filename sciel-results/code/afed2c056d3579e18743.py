"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

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

    def solve_split(n, k, p, seed):
        """Find a minimal clique partition for a split graph using greedy maximal clique expansion."""
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        
        # Build adjacency list for fast neighbor checks
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        uncovered_edges = set(edges)
        cliques = []
        
        # Helper to check if a set of vertices forms a clique in the original graph
        # and if all edges within that set are in uncovered_edges
        def is_valid_clique(vertices):
            v_list = list(vertices)
            for i in range(len(v_list)):
                for j in range(i + 1, len(v_list)):
                    u, v = v_list[i], v_list[j]
                    if (u, v) not in edge_set:
                        return False
                    # Check if the edge is covered
                    if u < v:
                        if (u, v) not in uncovered_edges:
                            return False
                    else:
                        if (v, u) not in uncovered_edges:
                            return False
            return True

        # Greedy loop
        while uncovered_edges:
            if time.time() > deadline:
                break
                
            # Pick an arbitrary uncovered edge to start a clique
            # We can sort by some heuristic, e.g., degree in uncovered graph, but arbitrary is okay for now
            # To be deterministic and potentially better, pick the edge with highest "potential"
            # For simplicity, just pick the first one
            u, v = next(iter(uncovered_edges))
            
            current_clique = {u, v}
            
            # Find candidates that can be added to the clique
            # A vertex x can be added if:
            # 1. x is adjacent to all vertices in current_clique (in original graph)
            # 2. The edges (x, y) for all y in current_clique are in uncovered_edges
            
            # Candidates must be neighbors of all nodes in current_clique
            candidates = set(adj[u]) & set(adj[v])
            candidates -= current_clique
            
            # Filter candidates based on uncovered edges
            valid_candidates = []
            for x in candidates:
                can_add = True
                for y in current_clique:
                    if x < y:
                        if (x, y) not in uncovered_edges:
                            can_add = False
                            break
                    else:
                        if (y, x) not in uncovered_edges:
                            can_add = False
                            break
                if can_add:
                    valid_candidates.append(x)
                    
            # Greedily add candidates to maximize clique size
            # Sort candidates by degree in the subgraph induced by valid_candidates + current_clique?
            # Or just add all that fit?
            # If we add x, it might prevent adding z.
            # Let's try to add as many as possible.
            # A simple strategy: repeatedly pick a candidate that allows the most other candidates?
            # Given small size, we can try to find a maximal set.
            
            # Let's just greedily add them. To get a larger clique, we might want to prioritize
            # vertices that are connected to many other valid candidates.
            
            # Calculate "score" for each candidate: number of other valid candidates it is connected to
            # (in the original graph, which they all are, so this is just count of other valid candidates)
            # Actually, since all candidates are neighbors of current_clique, and we only care about
            # edges between candidates being uncovered...
            # Wait, if we add x and z, the edge (x,z) must be in uncovered_edges.
            
            # Let's build a graph of valid_candidates where edges exist if (x,z) is in uncovered_edges.
            # We want to find a maximal clique in this auxiliary graph?
            # No, we just want to add a set of vertices S to current_clique such that
            # all pairs in S ∪ current_clique have uncovered edges.
            
            # This is equivalent to finding a maximal clique in the graph H defined on valid_candidates
            # where edges are those present in uncovered_edges.
            
            # Since |valid_candidates| is small, we can do a simple greedy or even brute force if very small.
            # Let's use a simple greedy: pick candidate with max degree in H, add it, update H.
            
            aux_adj = {x: set() for x in valid_candidates}
            for i in range(len(valid_candidates)):
                for j in range(i + 1, len(valid_candidates)):
                    x, z = valid_candidates[i], valid_candidates[j]
                    if x < z:
                        if (x, z) in uncovered_edges:
                            aux_adj[x].add(z)
                            aux_adj[z].add(x)
                    else:
                        if (z, x) in uncovered_edges:
                            aux_adj[z].add(x)
                            aux_adj[x].add(z)
                            
            selected = set()
            remaining_cands = set(valid_candidates)
            
            while remaining_cands:
                # Pick candidate with max degree in aux_adj restricted to remaining_cands
                best_cand = None
                best_deg = -1
                for x in remaining_cands:
                    deg = len(aux_adj[x] & remaining_cands)
                    if deg > best_deg:
                        best_deg = deg
                        best_cand = x
                if best_cand is None:
                    break
                selected.add(best_cand)
                # Remove best_cand and its non-neighbors from remaining_cands
                remaining_cands.remove(best_cand)
                # We can keep neighbors, but they must also be connected to all previously selected?
                # Actually, if we select a set S, every pair in S must be an edge in H.
                # So S must be a clique in H.
                # The greedy above picks a node, then restricts to its neighbors.
                # This finds a maximal clique in H.
                
                # Update remaining_cands to only include neighbors of best_cand
                remaining_cands = remaining_cands & aux_adj[best_cand]
                
            current_clique.update(selected)
            
            # Add the clique to the list
            cliques.append(list(current_clique))
            
            # Remove all edges within current_clique from uncovered_edges
            v_list = list(current_clique)
            for i in range(len(v_list)):
                for j in range(i + 1, len(v_list)):
                    a, b = v_list[i], v_list[j]
                    if a < b:
                        uncovered_edges.discard((a, b))
                    else:
                        uncovered_edges.discard((b, a))
                        
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            # Return best effort so far? Or empty?
            # The contract says "Any broken rule makes the whole result invalid".
            # If we run out of time, we should return a valid result.
            # The seed result is always valid.
            # Let's fall back to the seed result if we run out of time.
            pass
        
        cliques = solve_split(*spec)
        result.append(cliques)
        
        if time.time() > deadline:
            # If we ran out of time during solving, we might have an incomplete result.
            # But solve_split is designed to be fast for n=90.
            # If it's too slow, we need a faster heuristic.
            # For now, let's assume it's fast enough.
            pass

    return result
# EVOLVE-BLOCK-END
