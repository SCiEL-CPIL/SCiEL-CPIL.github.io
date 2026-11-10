"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Precompute pair indices for Union-Find
    # We will use a dictionary mapping (c1, c2) to an index, or just a 2D array if k is small
    # Since k is small, we can just use a dict of lists or a 2D list of parents
    
    def find(parent, x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(parent, x, y):
        px, py = find(parent, x), find(parent, y)
        if px == py:
            return False
        parent[px] = py
        return True

    # Backtracking search
    # colors: list of length n, -1 if uncolored
    # ufs: dict mapping (c1, c2) to parent list for that pair
    
    def solve(k):
        colors = [-1] * n
        # Initialize UF structures for all pairs (i, j) with i < j
        # We only need to initialize them when we first use a pair, or we can pre-initialize
        # Pre-initializing is safer.
        ufs = {}
        for i in range(k):
            for j in range(i + 1, k):
                ufs[(i, j)] = list(range(n))
                
        # Order vertices by degree (descending) to prune earlier?
        # All degrees are 4, so order doesn't matter much.
        # But we can use a static order 0..n-1
        
        def dfs(idx):
            if idx == n:
                return True
            
            v = idx
            # Try each color
            for c in range(k):
                # Check 1: Neighbor conflict
                valid = True
                for u in adj[v]:
                    if colors[u] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check 2: Bicolored cycle constraint
                # For each neighbor u with color d != c, edge (v,u) is added to pair (c,d)
                # We need to check if v and u are already connected in the UF for (c,d)
                # If they are, adding the edge creates a cycle.
                # We must also ensure that we don't create a cycle with *other* edges of the same pair.
                # The UF for (c,d) currently contains all edges between vertices colored c and d that have been processed.
                # If v and u are connected, it means there is a path between v and u using edges of colors c and d.
                # Adding (v,u) closes the loop.
                
                # We need to simulate the unions to see if they are valid, then commit.
                # Or, we can just check connectivity first.
                # If valid, we commit the unions.
                
                # Let's check connectivity for all neighbors with color d != c
                # Note: We haven't committed the unions for v yet, so v is isolated in the UF (unless it was connected via other edges? No, v is being colored now, so it has no edges in the UF yet).
                # Wait, v is uncolored, so it has no edges in any UF.
                # So find(parent, v) is v.
                # We check if find(parent, u) == v? No, u is colored, so u might be connected to other nodes.
                # If find(parent, u) == v, that's impossible since v is not in the UF yet (parent[v]=v, but no other node points to v).
                # Actually, we just need to check if v and u are in the same component.
                # Since v is isolated, find(parent, v) is v.
                # So we check if find(parent, u) == v? No.
                # We check if they are connected.
                # If we add edge (v,u), v becomes connected to u's component.
                # A cycle is created if v and u were ALREADY connected.
                # Since v is new, it cannot be connected to u unless there is a path from v to u.
                # But v has no edges yet. So v and u are NOT connected.
                # So adding the first edge for v to any neighbor is always safe regarding cycles?
                # NO. The cycle is formed by the path between v and u in the *existing* graph of pair (c,d).
                # But v is not in the graph yet.
                # So there is no path between v and u.
                # So the first edge added involving v is always safe?
                # Yes, because a cycle requires a path between the endpoints of the new edge.
                # Since v is new, it has no path to anyone.
                
                # Wait, this logic applies to the *first* edge incident to v.
                # What if v connects to two neighbors u1 and u2 of the same color d?
                # Edge (v, u1) is added. v is now in u1's component.
                # Edge (v, u2) is added.
                # If u1 and u2 are in the same component (which they are, because v is now connected to u1, and if u1 and u2 were already connected, then v-u1-...-u2-v is a cycle? No.
                # If u1 and u2 are in the same component BEFORE adding v, then adding (v,u1) connects v to that component.
                # Then adding (v,u2) connects v to the same component.
                # This creates a cycle: v - u1 - ... - u2 - v.
                
                # So, we must check:
                # For a given color c for v:
                # 1. For each neighbor u with color d:
                #    Let comp_u be the component of u in UF(c,d).
                #    If we add edge (v,u), v joins comp_u.
                #    If we then add another edge (v, u') where u' has the same color d, and u' is in the SAME component as u (in the original UF, before v was added), then a cycle is formed.
                #    Also, if u and u' are in different components, adding both edges merges them, which is fine (no cycle).
                
                # So the check is:
                # For each color d used by neighbors of v:
                #    Collect all neighbors u with color d.
                #    Check if any two of these neighbors are in the SAME component in UF(c,d).
                #    If yes, then adding v with color c will create a cycle.
                
                # Let's implement this check.
                valid = True
                # Group neighbors by their color
                neighbors_by_color = {}
                for u in adj[v]:
                    d = colors[u]
                    if d == -1:
                        continue
                    if d == c:
                        # Already handled by neighbor conflict check? 
                        # Yes, "if colors[u] == c: valid=False"
                        # So d != c here.
                        pass
                    if d not in neighbors_by_color:
                        neighbors_by_color[d] = []
                    neighbors_by_color[d].append(u)
                    
                for d, us in neighbors_by_color.items():
                    if len(us) > 1:
                        # Check if any two in us are in the same component in UF(c,d)
                        # We need to find the component ID for each u
                        parent = ufs[(min(c,d), max(c,d))]
                        comp_ids = set()
                        for u in us:
                            cid = find(parent, u)
                            if cid in comp_ids:
                                valid = False
                                break
                            comp_ids.add(cid)
                        if not valid:
                            break
                
                if not valid:
                    continue
                
                # If valid, commit
                # Save state
                old_ufs = {key: val.copy() for key, val in ufs.items()}
                old_color = colors[v]
                
                colors[v] = c
                
                # Update UFs
                for u in adj[v]:
                    d = colors[u]
                    if d == -1:
                        continue
                    if d == c:
                        continue # Should not happen
                    pair = (min(c, d), max(c, d))
                    if union(ufs[pair], v, u):
                        pass # Success
                    else:
                        # This should not happen if our check was correct
                        # But if it does, rollback
                        colors[v] = old_color
                        ufs.clear()
                        ufs.update(old_ufs)
                        valid = False
                        break
                
                if not valid:
                    continue
                
                if dfs(idx + 1):
                    return True
                
                # Rollback
                colors[v] = old_color
                ufs.clear()
                ufs.update(old_ufs)
                
            return False
        
        return dfs(0)

    # Try increasing k
    for k in range(3, 11):
        if solve(k):
            # Extract coloring
            # The solve function modifies the global 'colors' and 'ufs' inside the closure?
            # No, 'colors' is local to solve.
            # We need to return it.
            # Let's modify solve to return the coloring.
            pass
            
    # Fallback
    return list(range(n))

# EVOLVE-BLOCK-END
