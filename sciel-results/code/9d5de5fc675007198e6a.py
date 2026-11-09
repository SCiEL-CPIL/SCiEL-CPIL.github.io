import time
import sys
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s kill

    def _time_left():
        return deadline - time.time()

    def _check_c6(n, colors):
        """
        Check if any color class contains a C6.
        colors is a 2D list colors[i][j] for i<j.
        Returns True if NO C6 exists (valid), False if C6 found.
        """
        # Build adjacency lists for each color
        adj = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[i][j]
                adj[c][i].add(j)
                adj[c][j].add(i)
        
        # For each color, check for C6
        # A C6 is a cycle of length 6.
        # We can look for a path of length 5 between two adjacent vertices.
        # Or simpler: for each edge (u,v) in color c, check if there is a path of length 5 from u to v in color c.
        # Path of length 5: u-x1-x2-x3-x4-v.
        # We can compute "distance" or just use BFS/DFS limited to depth 5.
        # Since n <= 20, we can be a bit generous.
        
        for c in range(4):
            # For each pair (u,v) with edge in color c, check for path of length 5
            # Actually, we just need to check if there is ANY cycle of length 6.
            # Let's do: for each u, for each neighbor v of u in color c,
            # check if there is a path of length 5 from u to v avoiding the edge (u,v)?
            # No, a cycle of length 6 uses 6 distinct vertices.
            # Let's find all paths of length 5.
            # Path: v0-v1-v2-v3-v4-v5.
            # If v0 == v5 and all vertices distinct, it's a C6.
            
            # Efficient check:
            # For each color c:
            #   For each vertex u:
            #     For each pair of neighbors v, w of u in color c (v != w):
            #       Check if there is a path of length 3 from v to w in color c,
            #       using only vertices not in {u, v, w}.
            #       If yes, then u-v-...-w-u is a C6.
            
            # Let's precompute paths of length 3 for each color.
            # path3[v][w] = True if there is a path of length 3 from v to w in color c.
            # Path v-x-y-w.
            
            # Actually, let's just do a direct search for C6.
            # For each color c:
            #   For each 6-tuple (a,b,c,d,e,f) of distinct vertices:
            #     Check if (a,b),(b,c),(c,d),(d,e),(e,f),(f,a) are all in color c.
            #   This is O(n^6) which is 20^6 = 64 million. Might be slow in Python.
            
            # Better:
            # For each color c:
            #   For each edge (u,v) in color c:
            #     Find all paths of length 5 from u to v in color c.
            #     If any path uses distinct vertices (except u=v at ends), it's a C6.
            #     Actually, if u and v are adjacent in color c, and there is a path of length 5 from u to v
            #     with all intermediate vertices distinct from u and v and each other, then we have a C6.
            
            # Let's use the "path of length 3 between neighbors" approach.
            # If u is connected to v and w in color c, and there is a path of length 3 from v to w
            # using vertices other than u, then u-v-...-w-u is a cycle.
            # The path v-x-y-w has 3 edges. Total cycle: u-v-x-y-w-u (5 edges? No, 6 vertices).
            # u-v (1), v-x (2), x-y (3), y-w (4), w-u (5). That's a 5-cycle.
            # We need a 6-cycle: u-v-x-y-z-w-u? No.
            # C6: v0-v1-v2-v3-v4-v5-v0.
            # Let v0 = u. Neighbors v1, v5.
            # We need a path of length 4 from v1 to v5: v1-x-y-z-v5.
            # So: for each u, for each pair of neighbors (v1, v5) of u in color c,
            # check if there is a path of length 4 from v1 to v5 in color c,
            # using only vertices not in {u, v1, v5}.
            
            # Path of length 4: v1-a-b-c-v5.
            # Let's precompute for each color c and each pair (x,y) whether there is a path of length 4
            # from x to y in color c.
            # Then check: for each u, for each v1, v5 in adj[c][u] with v1 < v5,
            # if path4_c[v1][v5] is True AND the path does not use u (we can't easily check "does not use u"
            # from a boolean matrix, so we might need to be careful).
            # Actually, if path4_c[v1][v5] is True, there exists SOME path of length 4.
            # It might go through u. But if it goes through u, then v1-...-u-...-v5.
            # Since u is connected to v1 and v5, if the path goes through u, it would be v1-u-v5 which is length 2.
            # A path of length 4 going through u would be v1-x-u-y-v5.
            # Then the cycle would be u-v1-x-u-y-v5-u? That's not a simple cycle.
            # We need a SIMPLE cycle of length 6.
            
            # Let's just do a direct check for simplicity, given n is small.
            # For each color c:
            #   For each u in 0..n-1:
            #     For each v1 in adj[c][u]:
            #       For each v5 in adj[c][u] with v5 > v1:
            #         Check if there is a path of length 4 from v1 to v5 in color c
            #         that does not visit u.
            #         We can do a BFS/DFS from v1 to v5 with depth 4, avoiding u.
            
            # This is O(n * d^2 * (n^4)) in worst case, but d is small.
            # Actually, let's just check all 6-cycles directly.
            # For each color c:
            #   For each 6-tuple (a,b,c,d,e,f) of distinct vertices:
            #     if (a,b),(b,c),(c,d),(d,e),(e,f),(f,a) are all in color c: return False
            
            # To avoid permutations, we can fix a=0, b=1, and iterate c,d,e,f > 1?
            # No, that misses cycles not containing 0 or 1.
            # We can iterate over all combinations of 6 vertices, and check all 6-cycles on them.
            # There are C(20,6) = 38760 combinations. For each, there are 60 distinct 6-cycles.
            # 38760 * 60 = 2.3 million checks. Each check is 6 edge lookups.
            # This is fast enough.
            
            from itertools import combinations
            for verts in combinations(range(n), 6):
                # Check all 6-cycles on these 6 vertices
                # A 6-cycle is a permutation (v0..v5) such that (v0,v1)..(v5,v0) are edges.
                # To avoid duplicates, we can fix v0 = min(verts), and consider cyclic permutations.
                # There are 5! / 2 = 60 distinct cycles.
                # Let's just iterate over all permutations of the 6 vertices? 6! = 720.
                # 38760 * 720 = 27 million. Might be a bit slow in Python.
                # Let's optimize: for each combination of 6 vertices,
                # check if the induced subgraph contains a C6.
                # We can build the subgraph and check.
                
                # Simpler: just check all 6-tuples (a,b,c,d,e,f) with a<b<c<d<e<f? No, that doesn't work for cycles.
                
                # Let's stick to the neighbor-path method, but implement the BFS for path of length 4 avoiding u.
                
                pass # Placeholder, will implement below
        
        # Let's implement the neighbor-path method properly.
        for c in range(4):
            for u in range(n):
                neighbors = list(adj[c][u])
                # For each pair of neighbors
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        v1 = neighbors[i]
                        v5 = neighbors[j]
                        # Check if there is a path of length 4 from v1 to v5 in color c, avoiding u.
                        # Path: v1-x-y-z-v5.
                        # x, y, z must be distinct from u, v1, v5, and each other.
                        
                        # BFS from v1, depth 4, avoiding u.
                        # We want to reach v5 in exactly 4 steps.
                        # visited: set of (vertex, depth)
                        # Or just BFS and record distances.
                        
                        # Since we only need depth 4, we can do:
                        # Level 0: {v1}
                        # Level 1: neighbors of v1 (excluding u, v1)
                        # Level 2: neighbors of level 1 (excluding u, v1, level 0, level 1)
                        # Level 3: neighbors of level 2 (excluding u, v1, level 0, level 1, level 2)
                        # Level 4: neighbors of level 3 (excluding u, v1, level 0, level 1, level 2, level 3)
                        # If v5 is in Level 4, we have a path.
                        
                        level0 = {v1}
                        level1 = set()
                        for x in adj[c][v1]:
                            if x != u and x != v1:
                                level1.add(x)
                        if not level1:
                            continue
                        level2 = set()
                        for x in level1:
                            for y in adj[c][x]:
                                if y != u and y != v1 and y not in level1:
                                    level2.add(y)
                        if not level2:
                            continue
                        level3 = set()
                        for y in level2:
                            for z in adj[c][y]:
                                if z != u and z != v1 and z not in level1 and z not in level2:
                                    level3.add(z)
                        if not level3:
                            continue
                        level4 = set()
                        for z in level3:
                            for w in adj[c][z]:
                                if w != u and w != v1 and w not in level1 and w not in level2 and w not in level3:
                                    level4.add(w)
                        if v5 in level4:
                            return False # Found a C6
        
        return True # No C6 found

    def _try_construct(n_target, seed):
        """
        Try to construct a valid 4-coloring of K_{n_target}.
        Uses backtracking with dynamic C6 checking.
        """
        if _time_left() < 10:
            return None
            
        rng = np.random.RandomState(seed)
        
        # colors[i][j] for i < j
        colors = [[-1] * n_target for _ in range(n_target)]
        
        # We will assign colors to edges in lexicographic order: (0,1), (0,2), ..., (0,n-1), (1,2), ...
        # Total edges: n_target * (n_target - 1) // 2
        
        def backtrack(edge_idx):
            if _time_left() < 5:
                return False
            if edge_idx == n_target * (n_target - 1) // 2:
                return True
                
            # Determine (i, j) from edge_idx
            # edge_idx = 0 -> (0,1)
            # edge_idx = 1 -> (0,2)
            # ...
            # Formula: i = floor((-1 + sqrt(1 + 8 * edge_idx)) / 2)
            # j = edge_idx - i * (n_target - 1) + i + 1
            # Let's use a precomputed list of edges.
            
            i, j = edges[edge_idx]
            
            # Try each color
            color_order = [0, 1, 2, 3]
            rng.shuffle(color_order) # Randomize to escape local minima
            
            for c in color_order:
                colors[i][j] = c
                colors[j][i] = c # For symmetry in adj lists
                
                # Check if this assignment creates a C6
                # We only need to check cycles involving the new edge (i, j).
                # A cycle involving (i, j) in color c must use (i, j) and 4 other edges in color c.
                # The cycle is i-j-...-i.
                # So we need a path of length 5 from i to j in color c, using the new edge (i, j) as one part?
                # No, the new edge (i, j) is part of the cycle.
                # So the cycle is i-j-x-y-z-w-i.
                # This requires a path of length 4 from j to i in color c, not using the edge (i, j).
                # Wait, the cycle is a sequence of 6 distinct vertices.
                # If we add edge (i, j) in color c, a new C6 could be formed if there is already a path of length 5
                # from i to j in color c using only previously assigned edges.
                # Path: i-v1-v2-v3-v4-j.
                # Then i-v1-v2-v3-v4-j-i is a C6.
                
                # So we need to check: is there a path of length 5 from i to j in color c
                # using only edges that were already assigned (i.e., not the new edge (i, j))?
                
                # Let's check for a path of length 5 from i to j in color c, avoiding the direct edge (i, j).
                # Since we just assigned (i, j), we can check for a path of length 5 in the graph
                # that includes (i, j) but we must ensure the path doesn't use (i, j) as an intermediate edge?
                # No, the path i-v1-v2-v3-v4-j does not use (i, j).
                # So we check for a path of length 5 from i to j in color c in the graph BEFORE adding (i, j).
                # But we already added (i, j) to the colors array.
                # So we need to check for a path of length 5 from i to j in color c
                # that does NOT use the edge (i, j).
                
                # Let's do a BFS/DFS from i to j in color c with depth 5, avoiding the edge (i, j).
                
                # BFS from i, depth 5, avoiding edge (i, j).
                # Level 0: {i}
                # Level 1: neighbors of i in color c, excluding j (to avoid using edge (i, j) directly)
                # ...
                # Level 5: if j is in Level 5, we have a path of length 5.
                
                # But wait, the path could go i-...-j where the first edge is not (i, j).
                # So we just need to check if j is reachable from i in 5 steps in color c,
                # without using the edge (i, j).
                
                # Let's implement this check.
                
                # Build adjacency for color c, excluding edge (i, j)
                # We can just do BFS and skip the edge (i, j).
                
                # BFS:
                visited = [set() for _ in range(6)] # visited[depth]
                visited[0].add(i)
                
                found_path = False
                for depth in range(5):
                    for u in visited[depth]:
                        for v in adj_temp[c][u]:
                            # Skip if this is the edge (i, j) or (j, i)
                            if (u == i and v == j) or (u == j and v == i):
                                continue
                            if v not in visited[depth + 1]:
                                visited[depth + 1].add(v)
                                if depth + 1 == 5 and v == j:
                                    found_path = True
                                    break
                        if found_path:
                            break
                    if found_path:
                        break
                
                if not found_path:
                    # No C6 created by this assignment
                    if backtrack(edge_idx + 1):
                        return True
                
                # Undo assignment
                colors[i][j] = -1
                colors[j][i] = -1
                adj_temp[c][i].discard(j)
                adj_temp[c][j].discard(i)
                
            return False
        
        # Precompute edges list
        edges = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                edges.append((i, j))
        
        # adj_temp is for the current partial coloring
        adj_temp = [[set() for _ in range(n_target)] for _ in range(4)]
        
        if backtrack(0):
            # Extract colors in lexicographic order
            result_colors = []
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    result_colors.append(int(colors[i][j]))
            return {'n': n_target, 'colors': result_colors}
        else:
            return None

    # Try to construct for n=18, 17, 16, ...
    # Start with n=18 (if it works, great; if not, try 17)
    # Use multiple seeds to increase chances of finding a solution.
    
    best_result = None
    best_n = 0
    
    # Try n=18 with several seeds
    for seed in range(5):
        if _time_left() < 20:
            break
        result = _try_construct(18, seed)
        if result is not None:
            best_result = result
            best_n = 18
            break
    
    # If n=18 failed, try n=17
    if best_n < 17:
        for seed in range(10):
            if _time_left() < 20:
                break
            result = _try_construct(17, seed)
            if result is not None:
                best_result = result
                best_n = 17
                break
    
    # If n=17 failed, try n=16
    if best_n < 16:
        for seed in range(10):
            if _time_left() < 20:
                break
            result = _try_construct(16, seed)
            if result is not None:
                best_result = result
                best_n = 16
                break
    
    # If all failed, return a trivial coloring for n=1
    if best_result is None:
        best_result = {'n': 1, 'colors': []}
    
    return best_result

