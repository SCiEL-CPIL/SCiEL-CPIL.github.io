import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_result = {'n': 39, 'edges': []}
    best_n = 39
    
    # Try to build C4-free graphs with min degree >= n-39 for n=47, 46, 45, ...
    # We work with the graph G directly.
    # G is C4-free, min_degree(G) >= n - 39.
    # Equivalent: complement H has max_degree(H) <= 38, and G is C4-free.
    # C4 in G <=> C4 in complement? No.
    # Let's stick to G.
    
    # Strategy:
    # 1. Start with a random graph or a structured one.
    # 2. Ensure min degree constraint: if deg(v) < n-39, add edges.
    # 3. Ensure C4-free: if C4 exists, remove an edge from it.
    # 4. Repeat until valid or stuck.
    
    # Efficient C4 detection/removal:
    # Maintain common neighbor counts.
    # A C4 exists if there is a pair of non-adjacent vertices (u, v) with >= 2 common neighbors.
    # Actually, C4 is u-v-w-x-u. This means u and w have common neighbors v and x.
    # So we need to check for pairs of vertices at distance 2 with multiple paths.
    
    # Better: For each pair of vertices (u, v), let c(u, v) be the number of common neighbors.
    # If u and v are NOT adjacent, and c(u, v) >= 2, then there is a C4 (u, x, v, y, u).
    # If u and v ARE adjacent, c(u, v) >= 1 is fine (triangle), but c(u, v) >= 2 implies a diamond or C4?
    # If u-v is an edge, and u, v have 2 common neighbors x, y, then u-x-v-y-u is a C4?
    # No, u-x-v is a path of length 2. u-y-v is a path of length 2.
    # If x != y, then u-x-v-y-u is a 4-cycle.
    # So: A C4 exists iff there is a pair of vertices (u, v) such that:
    # 1. u and v are NOT adjacent, and c(u, v) >= 2.
    # 2. u and v ARE adjacent, and c(u, v) >= 2. (Wait, if u-v is an edge, and x, y are common neighbors, then u-x-v-y-u is a cycle of length 4. Yes.)
    # So in ALL cases, if c(u, v) >= 2, there is a C4.
    # Exception: If the "cycle" is just a triangle? No, u-x-v-y-u has 4 distinct vertices if x != y and x, y != u, v.
    # So the condition for NO C4 is: for all pairs (u, v), c(u, v) <= 1.
    # This is the definition of a C4-free graph: no two vertices have more than one common neighbor.
    
    # So the constraint is: max_{u != v} c(u, v) <= 1.
    
    # This is a very strong constraint.
    # Let's try to construct such a graph.
    
    # For n=46, min degree >= 46-39 = 7.
    # For n=47, min degree >= 47-39 = 8.
    
    # Let's use a greedy construction with repair.
    
    def is_valid(n, adj):
        # Check min degree
        for i in range(n):
            if adj[i].sum() < n - 39:
                return False
        # Check C4-free: for all u < v, common neighbors <= 1
        # We can check this by looking at the square of the adjacency matrix?
        # Or just iterate pairs.
        # n is small (<=47), so O(n^3) is fine.
        for u in range(n):
            for v in range(u+1, n):
                common = 0
                for w in range(n):
                    if adj[u][w] and adj[v][w]:
                        common += 1
                        if common > 1:
                            return False
        return True
    
    def count_common(adj, u, v):
        return int((adj[u] & adj[v]).sum())
    
    def try_construct(n, seed):
        random.seed(seed)
        # Initialize adjacency matrix
        adj = np.zeros((n, n), dtype=np.int8)
        
        # Start with a random graph with density p
        # Min degree needed: n-39.
        # For n=46, deg >= 7. Density ~ 7/45 ~ 0.15
        # For n=47, deg >= 8. Density ~ 8/46 ~ 0.17
        p = (n - 39) / (n - 1) * 1.5 # Slightly higher to start
        
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i][j] = 1
                    adj[j][i] = 1
        
        # Repair loop
        for step in range(1000):
            if time.time() > deadline:
                return None
            
            # Check validity
            valid = True
            
            # 1. Fix min degree
            for i in range(n):
                deg = int(adj[i].sum())
                if deg < n - 39:
                    # Add random edges
                    missing = n - 1 - deg
                    candidates = [j for j in range(n) if j != i and not adj[i][j]]
                    random.shuffle(candidates)
                    for j in candidates[:missing]:
                        # Check if adding (i, j) creates a C4
                        # Adding (i, j) creates a C4 iff c(i, j) >= 1?
                        # No. If c(i, j) = k.
                        # If k=0, no C4 involving (i, j) as a chord?
                        # Wait. Adding edge (i, j).
                        # Any new C4 must use edge (i, j).
                        # A C4 using (i, j) is i-j-x-y-i.
                        # This requires x, y to be common neighbors of i and j?
                        # No. i-j is an edge. i-x, x-y, y-i.
                        # So x is neighbor of i, y is neighbor of i.
                        # j is neighbor of x? No, j-x is not necessarily an edge.
                        # The cycle is i-j-x-y-i.
                        # Edges: (i, j), (j, x), (x, y), (y, i).
                        # So x is a neighbor of j and y.
                        # y is a neighbor of x and i.
                        # This is getting complicated.
                        
                        # Simpler check:
                        # Before adding (i, j), c(i, j) = k.
                        # After adding, c(i, j) is still k (common neighbors don't change when adding an edge between them, unless we count the edge itself? No, common neighbors are vertices w such that (i, w) and (j, w) are edges).
                        # Does adding (i, j) create a C4?
                        # Yes, if there is a path of length 3 between i and j that does not use (i, j).
                        # i-x-y-j.
                        # This means x is neighbor of i, y is neighbor of j, and (x, y) is an edge.
                        # And x != y, x != j, y != i.
                        
                        # Let's just check the global condition: max common neighbors <= 1.
                        # Adding (i, j) increases the degree of i and j.
                        # It does NOT change the common neighbor count of any pair (u, v) except potentially pairs involving i or j?
                        # c(u, v) = sum_w adj[u][w] adj[v][w].
                        # Adding (i, j) sets adj[i][j] = 1 and adj[j][i] = 1.
                        # This affects c(u, v) only if u=i or v=i or u=j or v=j.
                        # Specifically:
                        # c(i, v) increases by 1 if adj[j][v] is 1. (Because w=j is now a common neighbor of i and v).
                        # c(j, v) increases by 1 if adj[i][v] is 1.
                        # c(i, j) is not defined in the "common neighbor" sense for the pair itself?
                        # c(i, j) = sum_w adj[i][w] adj[j][w]. The edge (i, j) is not a common neighbor.
                        # So c(i, j) is unchanged.
                        
                        # So we need to check:
                        # For all v != i, j:
                        #   If adj[j][v] == 1, then c(i, v) increases by 1. Must remain <= 1. So c(i, v) was 0.
                        #   If adj[i][v] == 1, then c(j, v) increases by 1. Must remain <= 1. So c(j, v) was 0.
                        
                        ok = True
                        for v in range(n):
                            if v == i or v == j: continue
                            if adj[j][v]:
                                if count_common(adj, i, v) >= 1:
                                    ok = False
                                    break
                            if adj[i][v]:
                                if count_common(adj, j, v) >= 1:
                                    ok = False
                                    break
                        if ok:
                            adj[i][j] = 1
                            adj[j][i] = 1
                    valid = False
                    break
            if not valid:
                continue
            
            # 2. Fix C4s
            # Find a pair (u, v) with c(u, v) >= 2.
            found_c4 = False
            for u in range(n):
                if found_c4: break
                for v in range(u+1, n):
                    c = count_common(adj, u, v)
                    if c >= 2:
                        # Remove an edge from the C4.
                        # Get common neighbors
                        commons = [w for w in range(n) if w != u and w != v and adj[u][w] and adj[v][w]]
                        # We have at least 2 commons.
                        # Remove an edge incident to u or v?
                        # Or just remove a random edge from the cycle.
                        # Let's remove edge (u, w) for a random common neighbor w.
                        w = random.choice(commons)
                        adj[u][w] = 0
                        adj[w][u] = 0
                        found_c4 = True
                        valid = False
                        break
            if not valid:
                continue
            
            if valid:
                # Check min degree again (removing edges might break it)
                min_deg = min(int(adj[i].sum()) for i in range(n))
                if min_deg >= n - 39:
                    # Success
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i][j]:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
        
        return None
    
    # Try n from 47 down to 40
    for n in range(47, 39, -1):
        found = False
        # Try multiple seeds
        for seed in range(100):
            if time.time() > deadline:
                break
            res = try_construct(n, seed)
            if res is not None:
                if n > best_n:
                    best_n = n
                    best_result = res
                found = True
                break # Found one for this n, go to next n (smaller) or stop if n=47?
                # Actually, we want to maximize n. So if we found n=47, we can stop.
                if n == 47:
                    return best_result
        
        if found:
            continue
        else:
            # If we couldn't find for n, try smaller n?
            # The loop does that.
            pass
    
    return best_result
