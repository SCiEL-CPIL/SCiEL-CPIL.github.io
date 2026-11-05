import random
from collections import defaultdict

def construct():
    def random_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def bron_kerbosch(adj, R, P, X, cliques):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(list(R))
            return
        
        # Pivot selection: choose vertex in P union X with max degree in P
        pivot = None
        max_deg = -1
        for v in P.union(X):
            deg = len(adj[v].intersection(P))
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        if pivot is None:
            return
            
        # Iterate over P \ N(pivot)
        for v in list(P):
            if v not in adj[pivot]:
                bron_kerbosch(adj, R | {v}, P & adj[v], X & adj[v], cliques)
                P.remove(v)
                X.add(v)

    def solve_instance(n, p, seed):
        adj = random_graph(n, p, seed)
        
        # Find maximal cliques
        cliques = []
        bron_kerbosch(adj, set(), set(range(n)), set(), cliques)
        
        # Filter cliques of size >= 2
        valid_cliques = [c for c in cliques if len(c) >= 2]
        
        if not valid_cliques:
            return []
            
        M = len(valid_cliques)
        
        # Map each vertex to a bitmask of cliques it belongs to
        masks = [0] * n
        for i, clique in enumerate(valid_cliques):
            bit = 1 << i
            for v in clique:
                masks[v] |= bit
                
        all_cliques_mask = (1 << M) - 1
        
        best_sol = None
        best_size = float('inf')
        
        # Greedy Set Cover with Local Search
        for _ in range(50):
            uncovered = all_cliques_mask
            selected = []
            
            # 1. Greedy Construction
            while uncovered:
                best_v = -1
                best_count = -1
                for v in range(n):
                    if not (uncovered & masks[v]):
                        continue
                    count = (uncovered & masks[v]).bit_count()
                    if count > best_count:
                        best_count = count
                        best_v = v
                    elif count == best_count:
                        if random.random() < 0.5:
                            best_v = v
                            
                if best_v == -1:
                    break
                    
                selected.append(best_v)
                uncovered &= ~masks[best_v]
                
            # 2. Local Search: Redundancy Removal and 2-for-1 Replacement
            improved = True
            while improved:
                improved = False
                
                # 2a. Remove redundant vertices
                for i in range(len(selected) - 1, -1, -1):
                    v = selected[i]
                    other_mask = 0
                    for j in range(len(selected)):
                        if j != i:
                            other_mask |= masks[selected[j]]
                            
                    if not (masks[v] & ~other_mask):
                        selected.pop(i)
                        improved = True
                        break # Restart inner loop as list changed

                if not selected:
                    break

                # 2b. Try to replace two vertices with one
                # We iterate over pairs in the current solution
                # To avoid index issues, we work on a copy or careful indices
                # Since n is small (40), O(|S|^2 * n) is fine
                found_replacement = False
                for i in range(len(selected)):
                    for j in range(i + 1, len(selected)):
                        u = selected[i]
                        w = selected[j]
                        
                        # Calculate coverage of S \ {u, w}
                        rest_mask = 0
                        for k in range(len(selected)):
                            if k != i and k != j:
                                rest_mask |= masks[selected[k]]
                                
                        needed = all_cliques_mask & ~rest_mask
                        
                        # If needed is 0, it means u and w were redundant together? 
                        # Actually if needed is 0, we could just remove them? No, we need to cover 'needed'.
                        # If needed is 0, it implies the rest of the set covers everything, so u and w are redundant.
                        # But we checked single redundancy. If single redundancy didn't catch it, 
                        # it means each covers something unique, but together they might be replaceable.
                        
                        # Find a vertex v not in selected that covers 'needed'
                        for v in range(n):
                            if v in selected:
                                continue
                            if (needed & ~masks[v]) == 0:
                                # v covers all cliques that u and w uniquely cover (relative to rest)
                                # We can replace u, w with v
                                # Remove u and w, add v
                                # Careful with indices: remove larger index first
                                if i > j:
                                    i, j = j, i
                                selected.pop(j)
                                selected.pop(i)
                                selected.append(v)
                                found_replacement = True
                                improved = True
                                break
                        if found_replacement:
                            break
                    if found_replacement:
                        break

            if len(selected) < best_size:
                best_size = len(selected)
                best_sol = selected[:]
                
        return sorted(best_sol)
        
    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        res = solve_instance(n, p, seed)
        results.append(res)
        
    return results
