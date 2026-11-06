import time
import random
import sys
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # We want to maximize n. Best known is 17.
    # We will try to find a valid coloring for n=17, then 16, etc.
    # If we find one, we return it.
    
    # To make it fast, we use a randomized backtracking search.
    # We represent the coloring as a dictionary of edges (i, j) -> color.
    # We build the graph edge by edge in lexicographic order.
    
    def check_c6_completion(color, edges_of_color, u, v):
        # Check if adding (u, v) to edges_of_color creates a C6.
        # This is true if there is a path of length 5 from u to v in edges_of_color.
        # Since we are adding (u, v), the cycle would be u - ... - v - u.
        # The path ... must be length 5.
        # So we look for a path of length 5 from u to v in the current graph.
        
        # BFS from u up to depth 5.
        # State: (node, depth, path) -- path is not needed for existence, just depth and node.
        # But we need to avoid using (u, v) in the path, which is not in the graph yet.
        # So we just search in the current graph.
        
        # Optimization: If degree is low, this is fast.
        # We can also use bitmasks for neighbors.
        
        # Let's use a simple BFS with pruning.
        # visited[node][depth]
        # Actually, we just need to know if v is reachable from u in exactly 5 steps.
        
        # Use bitsets for neighbors for speed?
        # For n <= 20, we can use Python integers as bitmasks.
        
        # This function is called many times, so it must be fast.
        # Let's precompute neighbor bitmasks for each color.
        # But the graph changes.
        
        # Alternative: Just do a DFS with depth limit 5.
        # Max branching factor is n-1. 16^5 is ~ 1 million. Too slow if called 136 times.
        # But we prune heavily.
        
        # Better: Use the fact that we are checking for a path of length 5.
        # We can do BFS from u for 5 steps and see if v is in the final set.
        
        # Let's maintain neighbor lists for each color.
        # This is handled by the caller.
        
        pass

    # The above check is too slow if done naively in Python for n=17.
    # We need a faster representation.
    
    # Let's use a different strategy: Local Search (Simulated Annealing).
    # It is often faster to find a valid coloring for a fixed n than to backtrack.
    
    def solve_n(n, time_limit):
        # Initialize random coloring
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        m = len(edges)
        colors = [random.randint(0, 3) for _ in range(m)]
        
        # Precompute all C6 cycles? 
        # Number of C6s is 60 * C(n, 6).
        # For n=17, ~740k cycles.
        # Storing them is memory heavy but doable.
        # Checking cost: For each move (flip edge e), update cost.
        # Flipping e affects all C6s containing e.
        # How many C6s contain a specific edge?
        # Fix edge (u, v). We need a path of length 5 from u to v.
        # This is equivalent to choosing 4 intermediate vertices from the remaining n-2.
        # Number of such paths is roughly (n-2)^4.
        # For n=17, 15^4 = 50k.
        # So each flip affects ~50k cycles.
        # If we do 10k flips, that's 500 million operations. Too slow in Python.
        
        # We need a smarter cost update or a different algorithm.
        
        # Let's go back to Backtracking but with a very strong heuristic:
        # 1. Use bitmasks for adjacency.
        # 2. Check C6 completion using bit operations.
        
        # We will try to solve for n=17 first.
        
        # Representation:
        # adj[c][u] is a bitmask of neighbors of u using color c.
        
        # To check if adding (u, v) with color c creates a C6:
        # We need to know if there is a path of length 5 from u to v in adj[c].
        # This can be checked by:
        # path_len_1 = adj[c][u]
        # path_len_2 = union of adj[c][x] for x in path_len_1
        # ...
        # path_len_5 = union of adj[c][x] for x in path_len_4
        # Check if v is in path_len_5.
        
        # This is efficient with bitmasks!
        
        def has_path_len_5(u, v, adj_c):
            # adj_c is a list of bitmasks.
            # We want to know if v is reachable from u in exactly 5 steps.
            # Actually, "exactly 5" is tricky with simple union.
            # We need to ensure the path doesn't revisit nodes? 
            # A C6 is a simple cycle. The path u-...-v must be simple.
            # However, if there is a walk of length 5, does it imply a C6?
            # Not necessarily. But if the graph is sparse, walks are likely simple.
            # For strict correctness, we must ensure simplicity.
            # But for a heuristic search, allowing walks might lead to false positives (rejecting valid moves) or false negatives (accepting invalid moves).
            # If we reject valid moves, we might fail to find a solution.
            # If we accept invalid moves, we return an invalid solution.
            # We MUST be correct.
            
            # So we need to check for a simple path of length 5.
            # With bitmasks, we can do DP:
            # S_k[v] = bitmask of nodes reachable from u in exactly k steps via a simple path?
            # This is hard to do with just bitmasks because we lose the "visited" info.
            
            # However, for n=17, we can just do a DFS with pruning.
            # But we need it to be fast.
            
            # Let's use a hybrid: 
            # If the degree is low, DFS is fast.
            
            # For now, let's assume the bitmask walk check is a good filter.
            # If it says "no path", we are safe (no C6).
            # If it says "path exists", we do a DFS to confirm it's a simple path.
            
            # Step 1: Bitmask reachability for length 5.
            # R1 = adj_c[u]
            # R2 = 0; for x in R1: R2 |= adj_c[x]
            # ...
            # R5
            # if (R5 & (1 << v)) == 0: return False
            
            # Step 2: DFS for simple path.
            
            return False # Placeholder

        # This is getting complex. Let's use a known valid coloring for n=17 if we can find one in memory or generate it via a known construction.
        
        # Construction for n=17:
        # The vertices are elements of Z_17.
        # Colors are determined by the difference d = j - i mod 17.
        # We need to partition the 8 quadratic residues and 8 non-residues into 4 sets of size 4?
        # No, the graph is 4-colored. Each color class is a 4-regular graph?
        # Sum of degrees = 16. So each vertex has degree 4 in each color class?
        # If each color class is 4-regular, does it contain a C6?
        
        # Let's try to construct a 4-regular graph on 17 vertices with no C6.
        # The Paley graph is 8-regular.
        
        # Actually, there is a specific known coloring.
        # Let's try to search online knowledge: "R_4(C_6) 17 coloring".
        # The coloring is often based on the group Z_3 x Z_3 x Z_3? No, 27 vertices.
        
        # Let's try a different approach: **Randomized Restart Local Search with Bitmask Walk Check**.
        # Even if the walk check is not strictly simple, it's a good filter.
        # We will verify the final solution with a strict check.
        
        def strict_verify(colors, n):
            # Build adj lists
            adj = {c: [[] for _ in range(n)] for c in range(4)}
            for idx, (i, j) in enumerate(edges):
                c = colors[idx]
                adj[c][i].append(j)
                adj[c][j].append(i)
            
            # Check for C6 in each color
            for c in range(4):
                # Use BFS from each edge? Or just check all C6s?
                # Number of C6s is small enough to enumerate if we are clever?
                # No, 740k is too many to check naively in Python.
                
                # Use the bitmask walk check as a filter, then DFS.
                # Or just use the DFS check for each potential C6?
                
                # Let's use the property: A C6 exists iff there is a path of length 5 between endpoints of an edge.
                # For each edge (u, v) in color c, check if there is a simple path of length 5 from u to v in color c.
                
                for u in range(n):
                    for v in adj[c][u]:
                        if v <= u: continue
                        # Check path u -> ... -> v of length 5
                        # DFS
                        def dfs(node, target, depth, visited):
                            if depth == 0:
                                return node == target
                            for nb in adj[c][node]:
                                if nb not in visited:
                                    visited.add(nb)
                                    if dfs(nb, target, depth-1, visited):
                                        return True
                                    visited.remove(nb)
                            return False
                            
                        if dfs(u, v, 5, {u}):
                            return False
            return True

        # This verification is O(E * N^5) which is too slow.
        
        # We need a faster verification or a guaranteed construction.
        
        # Let's try to use the **Paley Graph** construction.
        # For n=17, the Paley graph is strongly regular.
        # It is known that the Paley graph of order 17 does NOT contain a C6?
        # Let's check.
        # If Paley(17) has no C6, then its complement also has no C6?
        # If both are C6-free, then we can color edges of Paley with 0,1 and Comp with 2,3?
        # No, we need 4 colors.
        
        # Actually, if we can find a 2-coloring of K_17 where both color classes are C6-free, we are done?
        # No, we need 4 colors.
        
        # Let's assume the task allows us to return n=17 if we find it.
        # Given the complexity, I will implement a **Randomized Backtracking** that is optimized for n=17.
        
        # Optimizations:
        # 1. Symmetry breaking: Fix colors of (0,1), (0,2), etc.
        # 2. Use bitmasks for adjacency.
        # 3. Check C6 completion using "Path of length 5" heuristic (walk) + strict check on conflict.
        
        # I will write a solver that tries to find a valid coloring for n=17.
        # If it fails in 60s, it tries n=16, etc.
        
        pass

    # Given the constraints and my inability to run code, I will provide a robust implementation of the **Bitmask Backtracking** solver.
    # It will attempt to solve for n=17.
    
    def solve(n):
        # Bitmask representation
        # adj[c][u] -> int bitmask
        
        # We will use a stack for backtracking.
        # State: (current_edge_index, adj_list)
        
        # To save memory, we copy adj_list on each branch? 
        # For n=17, 136 edges. Depth 136. 
        # Copying 4*17 integers is fast.
        
        # Heuristic: MRV (Minimum Remaining Values) is hard to implement dynamically here.
        # We use fixed order.
        
        # Symmetry breaking:
        # Color(0,1) = 0
        # Color(0,2) = 1
        # Color(0,3) = 2
        # Color(0,4) = 3
        # This breaks a lot of symmetry.
        
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        m = len(edges)
        
        # Precompute for each edge (u, v) and color c, the "neighbors" that would form a C6?
        # No, we check dynamically.
        
        def check_conflict(c, u, v, adj):
            # Check if adding (u, v) to color c creates a C6.
            # This requires a path of length 5 from u to v in adj[c].
            
            # Bitmask walk check (necessary condition for simple path)
            # R1 = adj[c][u]
            # R2 = 0
            # for x in bits(R1): R2 |= adj[c][x]
            # ...
            # R5
            
            # If v not in R5, return False (no conflict)
            
            # If v in R5, do DFS to confirm simple path.
            
            # To speed up, we can precompute the "5-step reachability" for each pair?
            # No, it changes.
            
            # Let's just do the DFS. It's called only when the bitmask check passes.
            
            # Bitmask check:
            R = [0] * 6
            R[0] = 1 << u
            for k in range(1, 6):
                mask = 0
                prev = R[k-1]
                while prev:
                    lsb = prev & -prev
                    x = lsb.bit_length() - 1
                    mask |= adj[c][x]
                    prev ^= lsb
                R[k] = mask
                if R[k] == 0: break
                
            if not (R[5] & (1 << v)):
                return False
                
            # DFS for simple path
            visited = {u}
            def dfs(node, depth):
                if depth == 5:
                    return node == v
                for nb in range(n):
                    if (adj[c][node] >> nb) & 1:
                        if nb not in visited:
                            visited.add(nb)
                            if dfs(nb, depth + 1):
                                return True
                            visited.remove(nb)
                return False
            return dfs(u, 0)
            
        # Backtracking
        # We use a stack to avoid recursion depth issues.
        # But recursion depth 136 is fine in Python if we increase limit?
        # sys.setrecursionlimit(10000)
        
        def backtrack(idx, adj):
            if time.time() > deadline:
                return None
            
            if idx == m:
                return adj # Found solution
                
            u, v = edges[idx]
            
            # Try colors 0..3
            for c in range(4):
                # Symmetry breaking for first few edges
                if idx == 0 and c != 0: continue
                if idx == 1 and c != 1: continue # (0,2)
                if idx == 2 and c != 2: continue # (0,3)
                if idx == 3 and c != 3: continue # (0,4)
                
                if not check_conflict(c, u, v, adj):
                    # Add edge
                    new_adj = [row[:] for row in adj]
                    new_adj[c][u] |= (1 << v)
                    new_adj[c][v] |= (1 << u)
                    
                    res = backtrack(idx + 1, new_adj)
                    if res is not None:
                        return res
            
            return None
            
        # Initialize adj
        adj = [[0] * n for _ in range(4)]
        
        # Randomize order of colors?
        # We fixed the first 4.
        
        # To add randomness, we can shuffle the remaining edges?
        # No, lexicographic is good for symmetry breaking.
        
        # We will run this for n=17.
        # If it fails, we return the best found (which might be None, so we fallback).
        
        # Actually, if it takes too long, we need to return a valid smaller n.
        # We can run the solver for n=17 with a time limit.
        # If it doesn't finish, we try n=16, etc.
        
        # But the problem says "Maximize n".
        
        # Let's try n=17.
        res = backtrack(0, adj)
        if res:
            # Convert to colors list
            colors = []
            for idx, (i, j) in enumerate(edges):
                # Find color
                for c in range(4):
                    if (res[c][i] >> j) & 1:
                        colors.append(c)
                        break
            return {'n': n, 'colors': colors}
            
        return None

    # Try n from 17 down to 10
    for n in range(17, 9, -1):
        if time.time() > deadline - 5:
            break
        res = solve(n)
        if res:
            return res
            
    # Fallback: return seed
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}

# EVOLVE-BLOCK-END
