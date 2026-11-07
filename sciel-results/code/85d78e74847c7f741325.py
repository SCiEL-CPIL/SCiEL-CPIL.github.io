import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    edges = [tuple(sorted(e)) for e in P['edges']]
    edge_set = set(edges)
    n = P['n']
    
    # Adjacency list for fast lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def is_biclique(edge_list):
        """Check if a set of edges forms a complete bipartite graph."""
        if not edge_list:
            return True, set(), set()
        
        # Build graph of the edge set
        verts = set()
        for u, v in edge_list:
            verts.add(u)
            verts.add(v)
        
        if len(verts) == 0:
            return True, set(), set()
        
        # Check bipartiteness and completeness
        # We need to find L, R such that edge_list == L x R
        
        # 1. Find connected components. If >1, not a single biclique (unless isolated vertices, but edges define vertices)
        # Actually, a biclique is connected if it has edges.
        # Let's just try to find a bipartition.
        
        # Pick an edge (u, v). Assume u in L, v in R.
        # BFS/DFS to assign sides.
        side = {}
        u0, v0 = edge_list[0]
        side[u0] = 0
        side[v0] = 1
        
        # Build local adjacency for the subgraph
        local_adj = {v: set() for v in verts}
        for u, v in edge_list:
            local_adj[u].add(v)
            local_adj[v].add(u)
            
        queue = [u0]
        valid = True
        while queue and valid:
            curr = queue.pop()
            curr_side = side[curr]
            for nei in local_adj[curr]:
                if nei not in side:
                    side[nei] = 1 - curr_side
                    queue.append(nei)
                else:
                    if side[nei] == curr_side:
                        valid = False
                        break
        
        if not valid:
            return False, set(), set()
            
        # Check if all vertices are assigned (connected)
        if len(side) != len(verts):
            # Disconnected. A biclique must be connected (if edges exist).
            # Wait, if the edge set is empty, it's trivial.
            # If the edge set is non-empty and disconnected, it's not a biclique.
            return False, set(), set()

        L = set(v for v, s in side.items() if s == 0)
        R = set(v for v, s in side.items() if s == 1)
        
        # Check completeness: |E| == |L| * |R|
        if len(edge_list) != len(L) * len(R):
            return False, set(), set()
            
        return True, L, R

    def get_biclique_vertices(edge_list):
        """Returns (L, R) for a known biclique edge set."""
        _, L, R = is_biclique(edge_list)
        return L, R

    # Greedy Initialization
    def greedy_partition():
        remaining = set(edges)
        bicliques = []
        while remaining:
            # Find largest biclique in remaining
            # Heuristic: Pick a random edge, expand to maximal biclique
            # To be deterministic and fast, pick the edge with highest potential?
            # Let's just pick the first edge, then greedily add vertices.
            
            # Better: Iterate through all edges, pick one, try to grow it.
            # For speed, just pick a random edge from remaining.
            e = random.choice(list(remaining))
            u, v = e
            L = {u}
            R = {v}
            
            # Grow L and R
            # A vertex x can be added to L if it is connected to all vertices in R.
            # A vertex y can be added to R if it is connected to all vertices in L.
            
            changed = True
            while changed:
                changed = False
                # Try adding to L
                candidates_L = set()
                for x in range(n):
                    if x in L: continue
                    if all(adj[x].superset(R)): # x connected to all in R
                        # And x must be connected to v (which is in R)
                        # And we must ensure that adding x doesn't break the biclique property with existing L?
                        # No, if x is connected to all in R, and existing L is connected to all in R,
                        # then L U {x} is connected to all in R.
                        # We also need to ensure that we don't include edges NOT in 'remaining'.
                        # Wait, the biclique edges must be a subset of 'remaining'.
                        # So we can only add x if ALL edges (x, y) for y in R are in 'remaining'.
                        if all((min(x,y), max(x,y)) in remaining for y in R):
                            candidates_L.add(x)
                if candidates_L:
                    # Pick the one with most connections to remaining? Or just first?
                    # Pick one that maximizes size?
                    # Let's just pick one.
                    new_l = random.choice(list(candidates_L))
                    L.add(new_l)
                    changed = True
                    
                # Try adding to R
                candidates_R = set()
                for y in range(n):
                    if y in R: continue
                    if all(adj[y].superset(L)):
                        if all((min(x,y), max(x,y)) in remaining for x in L):
                            candidates_R.add(y)
                if candidates_R:
                    new_r = random.choice(list(candidates_R))
                    R.add(new_r)
                    changed = True
            
            # Extract edges
            biclique_edges = set((min(u,v), max(u,v)) for u in L for v in R)
            # Ensure all are in remaining
            biclique_edges = biclique_edges & remaining
            
            if biclique_edges:
                bicliques.append(biclique_edges)
                remaining -= biclique_edges
            else:
                # Should not happen
                break
                
        return bicliques

    # Local Search
    def local_search(bicliques, max_iters=5000):
        best = bicliques
        best_len = len(bicliques)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            
            # 1. Try Merges
            for i in range(len(best)):
                for j in range(i+1, len(best)):
                    union_edges = best[i] | best[j]
                    is_bic, L, R = is_biclique(union_edges)
                    if is_bic:
                        new_bicliques = best[:i] + best[i+1:j] + best[j+1:] + [union_edges]
                        # Remove the merged ones
                        # Wait, list manipulation
                        new_bicliques = []
                        for k, b in enumerate(best):
                            if k != i and k != j:
                                new_bicliques.append(b)
                        new_bicliques.append(union_edges)
                        
                        if len(new_bicliques) < best_len:
                            best = new_bicliques
                            best_len = len(new_bicliques)
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                continue
                
            # 2. Try Edge Swaps (to enable future merges)
            # Pick a random biclique, pick a random edge, try to move it to another biclique
            if best_len > 1:
                i = random.randint(0, len(best)-1)
                if len(best[i]) > 1:
                    e = random.choice(list(best[i]))
                    # Try to move e to j
                    for j in range(len(best)):
                        if i == j: continue
                        # Remove e from i
                        b_i_new = best[i] - {e}
                        # Add e to j
                        b_j_new = best[j] | {e}
                        
                        is_bic_i, _, _ = is_biclique(b_i_new)
                        is_bic_j, _, _ = is_biclique(b_j_new)
                        
                        if is_bic_i and is_bic_j:
                            # Valid swap. Does it help?
                            # We can't know if it helps immediately.
                            # Accept with some probability (Simulated Annealing) or just accept if it doesn't increase size (it doesn't).
                            # To avoid loops, we can just do it.
                            new_bicliques = list(best)
                            new_bicliques[i] = b_i_new
                            new_bicliques[j] = b_j_new
                            # Remove empty bicliques
                            new_bicliques = [b for b in new_bicliques if b]
                            best = new_bicliques
                            best_len = len(best)
                            improved = True
                            break
                    if improved:
                        continue
                        
            if not improved:
                # Stuck. Try a random restart or more aggressive moves?
                # For now, break and rely on multiple restarts.
                break
                
        return best

    # Main Loop
    best_result = None
    best_count = float('inf')
    
    # Run multiple restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        random.seed(restarts) # Change seed for diversity
        
        init_bicliques = greedy_partition()
        if not init_bicliques:
             # Fallback to star partition
             init_bicliques = [set([e for e in edges if e[0]==u or e[1]==u]) for u in range(n) if any(e[0]==u or e[1]==u for e in edges)]
             
        result = local_search(init_bicliques)
        
        if len(result) < best_count:
            best_count = len(result)
            best_result = result
            
        # If we hit a very low number (e.g., < 10), we can stop early?
        # Theoretical lower bound: Max degree is 14. So at least 14 bicliques?
        # No, a biclique can cover high degree vertices.
        # But each biclique is a complete bipartite graph.
        # If we have 184 edges, and max biclique size is ~50, we need at least 4.
        # But structural constraints matter.
        
        if best_count <= 8: # Heuristic stop
             break

    if best_result is None:
        # Fallback
        best_result = [set([e for e in edges if e[0]==u or e[1]==u]) for u in range(n) if any(e[0]==u or e[1]==u for e in edges)]

    # Convert to required format
    final_bicliques = []
    for b in best_result:
        _, L, R = is_biclique(b)
        # Ensure lists are sorted and types are int
        L_list = sorted([int(x) for x in L])
        R_list = sorted([int(x) for x in R])
        final_bicliques.append([L_list, R_list])
        
    return final_bicliques

