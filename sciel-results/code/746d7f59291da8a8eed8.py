import time
import sys
from itertools import combinations

def construct():
    start = time.time()
    DEADLINE = 145.0  # safety margin before 150s

    def build_graph(n, p, seed):
        from random import Random
        rng = Random(seed)
        adj = [0]*n
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def bron_kerbosch(adj, n):
        # Returns list of maximal cliques (as lists of vertices)
        cliques = []
        
        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(list(R))
                return
            
            # Pivot selection: choose vertex in P ∪ X with max degree in P
            # This reduces the number of recursive calls
            if P or X:
                # Pick pivot u from P ∪ X
                # For simplicity and speed in Python, we can just pick the first one or one with high degree
                # Let's pick one from P if available, else X
                # Better: pick u that maximizes |P ∩ N(u)|
                u = -1
                max_deg = -1
                for v in P:
                    deg = bin(adj[v] & P).count('1')
                    if deg > max_deg:
                        max_deg = deg
                        u = v
                if u == -1:
                    for v in X:
                        deg = bin(adj[v] & P).count('1')
                        if deg > max_deg:
                            max_deg = deg
                            u = v
                
                # Branch on P \ N(u)
                # If u is -1, we branch on all P
                if u == -1:
                    candidates = list(P)
                else:
                    candidates = [v for v in P if not (adj[v] & (1 << u))]
                
                for v in candidates:
                    R.append(v)
                    P_new = P & adj[v]
                    X_new = X & adj[v]
                    bk(R, P_new, X_new)
                    R.pop()
                    P &= ~(1 << v)
                    X |= (1 << v)

        bk([], (1 << n) - 1, 0)
        return cliques

    def solve_instance(adj, n):
        cliques = bron_kerbosch(adj, n)
        
        if not cliques:
            return []
        
        # Map clique index to bitmask of vertices? No, map vertex to set of clique indices it covers.
        # This is a bipartite graph: Vertices on left, Cliques on right.
        # We want to select min Vertices to cover all Cliques.
        
        num_cliques = len(cliques)
        # cover[v] = bitmask of cliques containing v
        cover = [0] * n
        for i, clique in enumerate(cliques):
            bit = 1 << i
            for v in clique:
                cover[v] |= bit
        
        # Greedy Set Cover
        selected = [False] * n
        covered_cliques = 0
        # Bitmask of covered cliques
        covered_mask = 0
        target_mask = (1 << num_cliques) - 1
        
        # To speed up greedy, we can use a heap or just linear scan since n is small (<=40)
        # But we need to track uncovered count per vertex dynamically.
        # Let's just do iterative greedy.
        
        # Initial greedy
        while covered_mask != target_mask:
            best_v = -1
            best_gain = 0
            for v in range(n):
                if not selected[v]:
                    gain = bin(cover[v] & ~covered_mask).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
            if best_v == -1:
                break # Should not happen if graph is valid
            selected[best_v] = True
            covered_mask |= cover[best_v]
            
        # Local Search: 1-for-0 (Removal)
        # Check if any selected vertex is redundant
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if selected[v]:
                    # Try removing v
                    selected[v] = False
                    new_mask = 0
                    for u in range(n):
                        if selected[u]:
                            new_mask |= cover[u]
                    if new_mask == target_mask:
                        # Keep it removed
                        changed = True
                        break
                    else:
                        # Restore
                        selected[v] = True
            
        # Local Search: 2-for-1 (Swap)
        # Try to replace 2 selected vertices with 1 unselected vertex
        # If (cover[v1] | cover[v2]) is subset of (cover[u] | (target_mask & ~cover[v1] & ~cover[v2]))?
        # Actually, simpler:
        # Current cover C.
        # Remove v1, v2 from C. Let C' = C \ {v1, v2}.
        # Let M' be the mask covered by C'.
        # Let M_missing = target_mask & ~M'.
        # We need a vertex u not in C such that cover[u] covers M_missing.
        # If such u exists, we replace {v1, v2} with {u}. Size decreases by 1.
        
        improved = True
        while improved:
            improved = False
            # Iterate over pairs of selected vertices
            sel_list = [v for v in range(n) if selected[v]]
            for i in range(len(sel_list)):
                for j in range(i+1, len(sel_list)):
                    v1 = sel_list[i]
                    v2 = sel_list[j]
                    if not selected[v1] or not selected[v2]:
                        continue
                    
                    # Calculate mask covered by C \ {v1, v2}
                    mask_without = 0
                    for u in range(n):
                        if selected[u] and u != v1 and u != v2:
                            mask_without |= cover[u]
                    
                    missing = target_mask & ~mask_without
                    
                    # Find u not in selected that covers missing
                    # Optimization: only check u if (cover[u] & missing) == missing
                    for u in range(n):
                        if not selected[u]:
                            if (cover[u] & missing) == missing:
                                # Perform swap
                                selected[v1] = False
                                selected[v2] = False
                                selected[u] = True
                                improved = True
                                break
                    if improved:
                        # Restart search as structure changed
                        sel_list = [v for v in range(n) if selected[v]]
                        break
                if improved:
                    break
            
            # After a 2-for-1, check 1-for-0 again
            if improved:
                for v in range(n):
                    if selected[v]:
                        selected[v] = False
                        new_mask = 0
                        for u in range(n):
                            if selected[u]:
                                new_mask |= cover[u]
                        if new_mask == target_mask:
                            continue # Keep removed
                        else:
                            selected[v] = True
                            break # Restart 2-for-1 loop if we restored? Or just continue?
                            # Let's just break out of 1-for-0 and let 2-for-1 run again if needed.
                            # Actually, just break the inner loop and let the outer 'improved' logic handle it.
                            # But 'improved' is already True.
                            break

        return [v for v in range(n) if selected[v]]

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            # Fallback: just return empty or greedy vertex cover? 
            # Better to return a valid cover if possible, but if we are out of time, 
            # we should return the best we have. Since we solve one by one, 
            # if we are out of time, we might not have results for all.
            # We must return a list of 3 lists.
            # If we fail, we can return empty lists? No, must be valid.
            # Let's assume we finish. The graphs are small.
            pass
        
        adj = build_graph(n, p, seed)
        sol = solve_instance(adj, n)
        results.append(sorted(sol))
        
    return results
