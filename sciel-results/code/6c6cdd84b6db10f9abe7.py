import numpy as np
import time
import random
import sys

def construct():
    n = 512
    start_time = time.time()
    # Safety margin: evaluator kills at 90s. We stop at 85s to be safe.
    deadline = start_time + 85.0

    # 1. Build Conflict Graph
    # Vertices are 1..n. Map to indices 0..n-1.
    # G[i] is a bitset of vertices that conflict with vertex i+1.
    # Two vertices x, y conflict if they are part of a 4-AP.
    # Actually, the constraint is "no 4-AP in the set".
    # This is a hypergraph independence problem (4-uniform).
    # However, we can reduce it to a graph problem if we define edges differently?
    # No, standard MIS is for graphs (2-uniform).
    # A 4-AP is {a, a+d, a+2d, a+3d}.
    # If we pick any 3 of these, we can still potentially pick the 4th? No, picking all 4 is forbidden.
    # So it's a 4-uniform hypergraph.
    # MIS on hypergraphs is harder.
    # BUT, we can use a different approach:
    # If we view this as a SAT/MaxSAT or use a specific branching strategy for hypergraphs.
    
    # Alternative: 
    # The prompt suggests "Model the problem as a Maximum Independent Set (MIS) in a conflict graph where vertices are integers 1-512 and edges connect numbers that would form a 4-AP if both were present".
    # Wait, "edges connect numbers that would form a 4-AP if both were present" is slightly ambiguous.
    # If x and y are in the set, do they "form" a 4-AP? No, two numbers don't form a 4-AP.
    # Usually, for k-AP free sets, the conflict is hypergraph.
    # However, if we define an edge between x and y if there EXISTS a 4-AP containing both x and y, then an Independent Set in this graph is NOT necessarily 4-AP free.
    # Example: AP {1, 2, 3, 4}. Edges: (1,2), (1,3), (1,4), (2,3), (2,4), (3,4).
    # If we pick {1, 3}, they are connected by an edge (since 1,3 are in a 4-AP with 2,4). So {1,3} is not independent.
    # If we pick {1, 4}, they are connected.
    # If we pick {1, 2}, connected.
    # So any pair from a 4-AP is an edge.
    # If we find an Independent Set in this graph, does it contain a 4-AP?
    # Suppose S is independent. If S contains a 4-AP {a, b, c, d}, then a and b are connected by an edge (part of that AP). Contradiction.
    # So YES, an Independent Set in the graph where edges connect pairs that share a 4-AP is a valid 4-AP-free set.
    # Is it optimal?
    # Any 4-AP-free set is an Independent Set in this graph.
    # So Max 4-AP-free set == Max Independent Set in this specific conflict graph.
    
    # Let's build this graph.
    # G[i] (bitset) contains j if i+1 and j+1 are in some 4-AP.
    
    G = [0] * n
    
    # Precompute 4-APs
    # a, a+d, a+2d, a+3d <= n
    # For each d:
    #   a from 1 to n-3d
    #     Mark edges between all pairs in {a, a+d, a+2d, a+3d}
    
    # This will be dense. 
    # Number of 4-APs is roughly n^2/6. 
    # Each AP adds 6 edges.
    # Total edges ~ n^2.
    # For n=512, edges ~ 260k.
    # Bitsets are efficient.
    
    for d in range(1, (n-3)//3 + 1):
        for a in range(1, n - 3*d + 1):
            # AP: a, a+d, a+2d, a+3d
            # Indices: a-1, a+d-1, a+2d-1, a+3d-1
            v0 = a - 1
            v1 = a + d - 1
            v2 = a + 2*d - 1
            v3 = a + 3*d - 1
            
            # Add edges between all pairs
            # v0-v1, v0-v2, v0-v3
            G[v0] |= (1 << v1) | (1 << v2) | (1 << v3)
            G[v1] |= (1 << v0) | (1 << v2) | (1 << v3)
            G[v2] |= (1 << v0) | (1 << v1) | (1 << v3)
            G[v3] |= (1 << v0) | (1 << v1) | (1 << v2)
            
            # Symmetry
            G[v1] |= (1 << v0)
            G[v2] |= (1 << v0)
            G[v3] |= (1 << v0)
            
            G[v2] |= (1 << v1)
            G[v3] |= (1 << v1)
            
            G[v3] |= (1 << v2)

    # 2. Solve MIS using Branch and Bound with Coloring (Tomita)
    # We use a recursive approach.
    # State:
    #   R: set of vertices already chosen (not needed for bound, just for result)
    #   P: set of vertices available to be added (candidates)
    #   B: set of vertices excluded (conflicts with R)
    
    # To make it fast, we use bitsets for P and B? Or just P.
    # If we pick v from P, new P = P & ~G[v] & ~(1<<v).
    
    # Coloring for Upper Bound:
    # We need a coloring of the subgraph induced by P.
    # The number of colors is an upper bound on the size of the MIS in P.
    # Simple greedy coloring is fast enough for n=512.
    
    best_size = 0
    best_set = [False] * n
    
    # Initial P: all vertices
    P_init = (1 << n) - 1
    
    # We need a coloring function that takes a bitset P and returns a color for each vertex in P?
    # Or just the max color used?
    # Standard Tomita: 
    # 1. Order vertices in P by degree (or just natural order for simplicity/speed in Python, though degree is better).
    # 2. Greedily color them.
    # 3. The last color assigned is the upper bound.
    
    # To speed up coloring, we can pre-compute a static ordering or use dynamic.
    # Given the time limit, a simple greedy coloring on the bitset might be slow if not optimized.
    # Let's try a simpler bound first: 
    # Upper bound = |P|? No, too loose.
    # Upper bound = sum of 1/(1+deg(v))? No.
    
    # Let's implement a reasonably fast greedy coloring.
    # Since n=512, we can represent P as a Python int (big int).
    # Iterating bits of a big int in Python is okay.
    
    def color_bound(P):
        # Returns (max_color, list_of_colors)
        # Greedy coloring of subgraph induced by P
        colors = {}
        max_c = 0
        # Get list of vertices in P
        # To make it faster, we can sort by degree in the induced subgraph?
        # For now, just iterate in increasing index order.
        
        # Optimization: If P is empty, return 0
        if P == 0:
            return 0, colors
        
        # Extract vertices
        verts = []
        temp = P
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp ^= lsb
            
        # Sort by degree in induced subgraph (descending) often gives tighter bounds
        # Degree of v in P is bit_count(P & G[v])
        # This is O(|P| * cost_of_bitcount).
        # For n=512, this is fine.
        
        # To save time in early branches, maybe just use natural order?
        # Let's try natural order first. If it's too slow, we optimize.
        
        for v in verts:
            # Find smallest available color
            # Used colors by neighbors in P
            # Neighbor in P: G[v] & P
            # We need to know which colors are used by neighbors.
            # This requires storing colors for all nodes.
            
            used = set()
            neighbors = G[v] & P
            # Iterate neighbors? That's expensive.
            # Alternative:
            # Since we are coloring sequentially, we only care about neighbors already colored.
            # But "greedy coloring" usually implies processing in a specific order.
            # If we process in index order, we only need to check neighbors with smaller index that are in P.
            
            # Let's stick to a simpler, slightly looser bound if coloring is too complex/slow.
            # Bound: |P| is always valid.
            # Better Bound: Max degree + 1? No.
            
            # Let's try a fast heuristic coloring:
            # Assign color 1 to all, then bump if conflict.
            # This is O(n^2) worst case per node.
            
            # Actually, for n=512, a standard greedy coloring taking ~1ms per call is acceptable for a few hundred nodes in the search tree.
            
            # Let's implement the standard greedy:
            # For v in verts (ordered):
            #   forbidden = { colors[u] for u in neighbors(v) if u in P and u processed }
            #   color[v] = smallest non-negative int not in forbidden
            
            # To make this fast, we maintain an array `color_arr` of length n.
            # And a set of used colors? No, just check 0..max_c.
            
            pass # Placeholder, will implement below

    # Let's write the actual MIS solver
    
    # Global variables for the solver
    best_val = 0
    best_mask = 0
    
    def solve_mis(P, depth, current_mask):
        nonlocal best_val, best_mask
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if depth > best_val:
                best_val = depth
                best_mask = current_mask
            return
            
        # Upper bound: 
        # Simple bound: depth + bit_count(P)
        # If this is <= best_val, prune.
        if depth + P.bit_count() <= best_val:
            return
            
        # Better bound: Greedy Coloring
        # Let's implement a fast greedy coloring for the upper bound.
        # We will color the subgraph induced by P.
        # To speed this up, we use a simple greedy approach.
        
        # Get vertices in P
        verts = []
        temp = P
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp ^= lsb
            
        # Sort by degree in P (descending) to get tighter bound
        # degree(v) = (G[v] & P).bit_count()
        # This is expensive. Let's try without sorting first.
        
        # Greedy Coloring
        color = [-1] * n # Reusing a global array might be tricky with recursion. 
        # Let's use a local dict or list.
        # Since n=512, a list of 512 ints is cheap.
        c = [0] * n
        max_c = 0
        
        # To make coloring fast, we can use a "color class" approach?
        # No, simple greedy:
        for v in verts:
            # Check colors of neighbors in P
            # Neighbors in P: G[v] & P
            # We need to know which colors are used by these neighbors.
            # Since we are coloring in the order of `verts`, 
            # we only care about neighbors that are ALSO in `verts` and have been colored.
            # Wait, `verts` is the set of vertices in P.
            # So we check all u in verts such that u is a neighbor of v and c[u] is set.
            
            # Optimization: 
            # Instead of iterating all neighbors, we can just check the colors of previously colored vertices in verts that are neighbors.
            # This is O(|verts|^2) worst case.
            
            forbidden = set()
            # Iterate through previously colored vertices in this specific coloring pass
            # This is inefficient if we do it for every v.
            
            # Alternative: 
            # Just use the bound: |P| / (1 + avg_degree)? No.
            # Let's use a simpler bound: 
            # Upper Bound = depth + bit_count(P)
            # And rely on good branching.
            
            # Actually, let's try a very fast coloring:
            # Assign color 1 to first vertex.
            # For next, if conflict with prev, color 2, etc.
            # This is not standard greedy.
            
            # Let's stick to:
            # UB = depth + bit_count(P)
            # If UB <= best_val: return
            
            # Branching:
            # Pick a vertex v from P.
            # Which one? 
            # Heuristic: Pick v with max degree in P? Or min?
            # Max degree usually prunes more.
            
            # Let's pick the first vertex in P for now (LSB).
            # v = (P & -P).bit_length() - 1
            # This is O(1).
            
            # To improve, we can pick v with max degree.
            # But calculating degrees is O(|P|).
            
            # Let's try: Pick v = LSB of P.
            v_lsb = P & -P
            v = v_lsb.bit_length() - 1
            
            # Branch 1: Include v
            # New P = P & ~G[v] & ~v_lsb
            P_new = P & ~G[v] & ~v_lsb
            solve_mis(P_new, depth + 1, current_mask | v_lsb)
            
            # Branch 2: Exclude v
            # New P = P & ~v_lsb
            P_new2 = P & ~v_lsb
            solve_mis(P_new2, depth, current_mask)
            
            # Check time
            if time.time() > deadline:
                return

    # Initial call
    # P_init = (1 << n) - 1
    # This might be too deep/slow without a good bound.
    
    # Let's add a simple coloring bound to `solve_mis`.
    # I will implement a fast greedy coloring that returns the max color.
    
    def get_color_bound(P):
        if P == 0:
            return 0
        
        verts = []
        temp = P
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp ^= lsb
            
        # Sort by degree in P (descending)
        # degree = (G[v] & P).bit_count()
        # This is O(|P| * 1) for bit_count (C-level).
        # Sorting is O(|P| log |P|).
        # For |P| ~ 100, this is fast.
        
        # To save time, let's just use the count of P as a very loose bound initially, 
        # and only use coloring if P is small enough?
        # No, coloring is the key to MIS.
        
        # Let's implement the coloring.
        c = [0] * n
        max_c = 0
        
        # Process in order of descending degree for better bound
        # degrees = [(v, (G[v] & P).bit_count()) for v in verts]
        # degrees.sort(key=lambda x: x[1], reverse=True)
        # verts_sorted = [v for v, d in degrees]
        
        # For now, natural order to save sort time?
        # Let's try natural order.
        
        for v in verts:
            # Find smallest color not used by neighbors in P
            # Neighbors in P: G[v] & P
            # We need to know colors of neighbors that are in P.
            # Since we are coloring sequentially, we only care about neighbors in `verts` that have been colored.
            # But `verts` is just the list. 
            # We can check `c[u]` for u in neighbors if u is in P.
            
            # Optimization: 
            # Instead of iterating all neighbors, we can just check the colors of the vertices in `verts` that are neighbors.
            # This is still O(|P|^2).
            
            # Let's try a different bound: 
            # UB = depth + bit_count(P)
            # If this doesn't work well, we add coloring.
            
            # Given the time, let's just use bit_count(P) for now and see if it runs.
            # If it's too slow, we need coloring.
            
            # Actually, for n=512, bit_count(P) is a very loose bound (512).
            # Best set is ~250?
            # So we will explore a huge tree.
            # We NEED coloring.
            
            # Let's implement a simple greedy coloring inline.
            
            # To make it fast, we use a bitset for colors?
            # No, n=512, max color ~ 20?
            
            # Let's just do:
            forbidden = set()
            # Iterate through previously colored vertices in this pass
            # This is O(k^2) where k is number of colored so far.
            
            # We can optimize by only checking neighbors.
            # For v, check all u in verts such that u < v (in processing order) and (G[v] >> u) & 1.
            
            # Let's just use the bound:
            # UB = depth + bit_count(P)
            # And hope the search space is small enough with good branching?
            # No, MIS is NP-hard.
            
            # Let's implement the coloring properly but efficiently.
            
            # We'll return the max color used.
            
            pass

    # Revised Plan:
    # Use a standard Tomita algorithm implementation in Python.
    # It might be slow, but n=512 is small enough if optimized with bitsets.
    
    # Let's try a different strategy: 
    # Start with a good initial solution (Cantor set or Greedy).
    # Then use local search (Swap/Move) to improve.
    # The prompt asks for MIS, but if MIS is too slow, local search on the graph is valid?
    # "Model the problem as ... MIS ... solve this using branch-and-bound"
    # I should try to implement B&B.
    
    # Let's implement B&B with a simple coloring bound.
    
    def solve_mis_bnb(P, depth, current_mask):
        nonlocal best_val, best_mask
        
        if time.time() > deadline:
            return
            
        if P == 0:
            if depth > best_val:
                best_val = depth
                best_mask = current_mask
            return
            
        # Bound
        # Simple: depth + bit_count(P)
        if depth + P.bit_count() <= best_val:
            return
            
        # Coloring Bound
        # We compute a greedy coloring of the subgraph induced by P.
        # This gives an upper bound on the MIS size in P.
        
        # Extract vertices
        verts = []
        temp = P
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp ^= lsb
            
        # Sort by degree in P (descending)
        # This significantly improves the bound.
        # degree(v) = (G[v] & P).bit_count()
        # To avoid sorting overhead, we can just use the vertices as is?
        # Sorting 512 items is fast.
        
        # Let's sort.
        # But calculating degree for all is O(|P|).
        
        # Optimization: 
        # If |P| is small, sorting is fine.
        # If |P| is large, the bound is loose anyway.
        
        # Let's try without sorting first to save time.
        
        # Greedy Coloring
        c = [0] * n
        max_c = 0
        
        # We process vertices in `verts`.
        # For each v, find smallest color not used by neighbors in P.
        
        # To speed this up, we can use a "color availability" bitset?
        # No, just a set.
        
        # We need to know which neighbors are in P and have been colored.
        # Since we process in `verts` order, we can just check `c[u]` for u in `verts` if u is neighbor.
        
        # This is O(|P|^2).
        # For |P|=100, 10k ops. Fast.
        # For |P|=500, 250k ops. Maybe 10ms.
        
        # Let's do it.
        
        for v in verts:
            # Check colors of neighbors in P
            # Neighbors in P: G[v] & P
            # We iterate through `verts` to find colored neighbors?
            # No, we can just check `c[u]` for all u in `verts` that are neighbors.
            
            # Optimization: 
            # We can pre-compute the list of neighbors in P for v?
            # No, just check the bitset.
            
            forbidden = set()
            # Iterate through all vertices in P? No, just those in `verts` that are neighbors.
            # Since `verts` is the list of P, we can iterate `verts` and check if neighbor.
            # This is O(|P|) per vertex. Total O(|P|^2).
            
            # We can speed this up by only checking previously processed vertices in `verts`.
            # But `verts` is just a list.
            
            # Let's just iterate `verts` and check.
            for u in verts:
                if u == v:
                    continue
                # Is u a neighbor of v?
                if (G[v] >> u) & 1:
                    if c[u] > 0: # Already colored
                        forbidden.add(c[u])
            
            # Find smallest color not in forbidden
            col = 1
            while col in forbidden:
                col += 1
            c[v] = col
            if col > max_c:
                max_c = col
            
        # UB = depth + max_c
        if depth + max_c <= best_val:
            return
            
        # Branching
        # Pick a vertex v.
        # Heuristic: Pick v with max degree in P?
        # Or just LSB?
        # Let's pick the vertex with max degree in P for better pruning.
        # We already have `verts`.
        # Find v with max (G[v] & P).bit_count()
        
        max_deg = -1
        v_sel = -1
        for v in verts:
            deg = (G[v] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                v_sel = v
                
        if v_sel == -1:
            return
            
        v_lsb = 1 << v_sel
        
        # Branch 1: Include v_sel
        P_new = P & ~G[v_sel] & ~v_lsb
        solve_mis_bnb(P_new, depth + 1, current_mask | v_lsb)
        
        # Branch 2: Exclude v_sel
        P_new2 = P & ~v_lsb
        solve_mis_bnb(P_new2, depth, current_mask)

    # Initialize
    # Get a good initial solution to set best_val
    # Use Greedy
    S_greedy = [False] * n
    # Simple greedy: pick 1, then next that doesn't conflict, etc.
    # But our graph is "conflict if share 4-AP".
    # Greedy MIS:
    P_curr = (1 << n) - 1
    mask_curr = 0
    while P_curr:
        # Pick LSB
        lsb = P_curr & -P_curr
        v = lsb.bit_length() - 1
        mask_curr |= lsb
        S_greedy[v] = True
        P_curr = P_curr & ~G[v] & ~lsb
        
    best_val = mask_curr.bit_count()
    best_mask = mask_curr
    
    # Run B&B
    solve_mis_bnb((1 << n) - 1, 0, 0)
    
    # Extract result
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)
            
    return result
