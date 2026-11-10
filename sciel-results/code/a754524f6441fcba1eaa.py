import time
import random
from itertools import combinations

def construct():
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def find_max_clique(vertices, adj_matrix, time_limit=0.5):
        """
        Find a large clique in the subgraph induced by 'vertices'.
        adj_matrix is a set of edges (u, v) with u < v.
        Returns a list of vertices forming the clique.
        """
        if not vertices:
            return []
        
        # Sort vertices by degree in the subgraph, descending
        deg = {v: 0 for v in vertices}
        for u in vertices:
            for v in vertices:
                if u < v:
                    if (u, v) in adj_matrix:
                        deg[u] += 1
                        deg[v] += 1
                elif v < u:
                    if (v, u) in adj_matrix:
                        deg[u] += 1
                        deg[v] += 1
        
        sorted_vertices = sorted(vertices, key=lambda x: -deg[x])
        
        # Greedy heuristic to find a good initial clique
        clique = []
        candidates = set(sorted_vertices)
        while candidates:
            # Pick vertex with max degree in candidates
            best_v = None
            best_deg = -1
            for v in candidates:
                if deg[v] > best_deg:
                    best_deg = deg[v]
                    best_v = v
            
            # Check if best_v is connected to all in clique
            is_connected = True
            for u in clique:
                if u < best_v:
                    if (u, best_v) not in adj_matrix:
                        is_connected = False
                        break
                else:
                    if (best_v, u) not in adj_matrix:
                        is_connected = False
                        break
            
            if is_connected:
                clique.append(best_v)
                candidates.remove(best_v)
                # Update candidates: remove those not connected to best_v
                to_remove = set()
                for v in candidates:
                    if v < best_v:
                        if (v, best_v) not in adj_matrix:
                            to_remove.add(v)
                    else:
                        if (best_v, v) not in adj_matrix:
                            to_remove.add(v)
                candidates -= to_remove
            else:
                # If not connected, just remove it? Or try next best?
                # For a simple heuristic, we might just stop or skip.
                # Let's try a simple backtracking for small sets if greedy fails to be optimal?
                # Given the size, let's stick to a simple greedy + local search or just greedy.
                # Actually, let's use a simple recursive max clique for small N.
                pass

        # A better approach for small N: Branch and Bound
        def max_clique_bb(candidates, current_clique):
            if not candidates:
                return current_clique[:]
            
            # Heuristic: pick vertex with max degree in candidates
            best_v = None
            best_deg = -1
            cand_list = list(candidates)
            for v in cand_list:
                # Count neighbors in candidates
                d = 0
                for u in cand_list:
                    if u == v: continue
                    if u < v:
                        if (u, v) in adj_matrix: d += 1
                    else:
                        if (v, u) in adj_matrix: d += 1
                if d > best_deg:
                    best_deg = d
                    best_v = v
            
            # Branch 1: Include best_v
            new_candidates = set()
            for u in candidates:
                if u == best_v: continue
                if u < best_v:
                    if (u, best_v) in adj_matrix:
                        new_candidates.add(u)
                else:
                    if (best_v, u) in adj_matrix:
                        new_candidates.add(u)
            
            c1 = max_clique_bb(new_candidates, current_clique + [best_v])
            
            # Branch 2: Exclude best_v
            candidates_without = set(candidates)
            candidates_without.remove(best_v)
            c2 = max_clique_bb(candidates_without, current_clique)
            
            return c1 if len(c1) > len(c2) else c2

        # Use BB if size is small, else greedy
        if len(vertices) <= 25:
             return max_clique_bb(set(vertices), [])
        else:
             # Fallback to greedy for larger sets
             clique = []
             candidates = set(sorted_vertices)
             while candidates:
                 best_v = None
                 best_deg = -1
                 for v in candidates:
                     if deg[v] > best_deg:
                         best_deg = deg[v]
                         best_v = v
                 
                 is_connected = True
                 for u in clique:
                     if u < best_v:
                         if (u, best_v) not in adj_matrix:
                             is_connected = False
                             break
                     else:
                         if (best_v, u) not in adj_matrix:
                             is_connected = False
                             break
                 
                 if is_connected:
                     clique.append(best_v)
                     candidates.remove(best_v)
                     to_remove = set()
                     for v in candidates:
                         if v < best_v:
                             if (v, best_v) not in adj_matrix:
                                 to_remove.add(v)
                         else:
                             if (best_v, v) not in adj_matrix:
                                 to_remove.add(v)
                     candidates -= to_remove
                 else:
                     # If we can't add the best, try removing it and continuing?
                     # This greedy might get stuck. Let's just return what we have.
                     break
             return clique

    def solve_instance(n, p, seed):
        edges = generate_graph(n, p, seed)
        remaining_edges = set(edges)
        cliques = []
        
        # Precompute adjacency for fast lookup
        adj = set(edges)
        
        while remaining_edges:
            # Find edge with max common neighbors in remaining_edges
            best_edge = None
            max_common = -1
            
            # To speed up, we can sample or just iterate if E is small.
            # E is ~1000. Iterating is fine.
            for u, v in remaining_edges:
                common = 0
                # Count w such that (u,w) and (v,w) in remaining_edges
                # This is O(N) per edge. Total O(E*N) ~ 1000*64 = 64000. Fine.
                for w in range(n):
                    if w == u or w == v: continue
                    e1 = (u, w) if u < w else (w, u)
                    e2 = (v, w) if v < w else (w, v)
                    if e1 in remaining_edges and e2 in remaining_edges:
                        common += 1
                
                if common > max_common:
                    max_common = common
                    best_edge = (u, v)
            
            if best_edge is None:
                break
                
            u, v = best_edge
            
            # Find common neighbors in remaining_edges
            common_neighbors = []
            for w in range(n):
                if w == u or w == v: continue
                e1 = (u, w) if u < w else (w, u)
                e2 = (v, w) if v < w else (w, v)
                if e1 in remaining_edges and e2 in remaining_edges:
                    common_neighbors.append(w)
            
            # Find max clique in subgraph induced by common_neighbors
            # The clique will be C = {u, v} U clique_in_common
            # We need the edges of the clique to be in remaining_edges.
            # Since common_neighbors are defined by edges in remaining_edges,
            # and we find a clique in the induced subgraph, all edges within common_neighbors
            # that are part of the clique are in remaining_edges.
            # Edges from u/v to common_neighbors are in remaining_edges by definition.
            
            if common_neighbors:
                # Construct adj_matrix for the subgraph
                sub_adj = set()
                for i in range(len(common_neighbors)):
                    w1 = common_neighbors[i]
                    for j in range(i+1, len(common_neighbors)):
                        w2 = common_neighbors[j]
                        e = (w1, w2) if w1 < w2 else (w2, w1)
                        if e in remaining_edges:
                            sub_adj.add(e)
                
                clique_common = find_max_clique(common_neighbors, sub_adj)
                clique = [u, v] + clique_common
            else:
                clique = [u, v]
            
            cliques.append(clique)
            
            # Remove edges of this clique from remaining_edges
            for i in range(len(clique)):
                for j in range(i+1, len(clique)):
                    u_c, v_c = clique[i], clique[j]
                    e = (u_c, v_c) if u_c < v_c else (v_c, u_c)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                        
        return cliques

    start_time = time.time()
    timeout = 110 # Leave 10s margin
    
    result1 = solve_instance(32, 0.6, 601)
    if time.time() - start_time > timeout:
         # Fallback: return edges as cliques (valid but bad score)
         # Just return what we have? No, must be complete.
         # If we timeout, we should return a valid solution.
         # The greedy loop is fast. Unlikely to timeout.
         pass
         
    result2 = solve_instance(64, 0.55, 602)
    
    return [result1, result2]

construct()
