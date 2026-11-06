import time
import random
from itertools import combinations

def _has_c6_in_color(adj, n):
    """
    Check if the graph defined by adjacency matrix 'adj' of size n x n
    contains a cycle of length 6 (C6).
    adj[i][j] is True if edge (i,j) exists in this color class.
    """
    # Precompute list of edges for faster iteration? 
    # Or just iterate over 6-subsets.
    # For n=17, 12376 subsets is fine.
    
    # Optimization: If any vertex has degree < 2, no C6.
    # If sum of degrees is small, no C6.
    
    # Iterate over all combinations of 6 vertices
    vertices = range(n)
    for combo in combinations(vertices, 6):
        # Check if induced subgraph on combo contains a C6
        # A C6 is a cycle visiting all 6 vertices.
        # There are (6-1)!/2 = 60 such cycles.
        # But we can just check if the induced subgraph has a Hamiltonian cycle.
        # Since it's small, we can check specific patterns or just try to find one.
        
        # Fast check: 
        # Let v = combo
        # We need a permutation p of v such that (p0,p1), (p1,p2), ..., (p5,p0) are edges.
        # We can fix p0 = v[0] to reduce symmetry.
        # Then we need to order the remaining 5.
        
        # Even faster: 
        # Count the number of paths of length 2 between non-adjacent vertices?
        # Actually, for n=17, just checking the 60 cycles is fast enough in Python if optimized?
        # 12376 * 60 = 742,560 checks. Each check is 6 lookups. ~4.4M lookups. 
        # In Python, this might take ~0.1s - 0.5s. 
        # We do this for 4 colors. ~2s.
        # If we do this in a loop for local search, it might be too slow if we do many iterations.
        
        # Let's optimize the C6 check.
        # We can precompute for each pair of vertices (u,v) the common neighbors.
        # But the graph changes.
        
        # Alternative: Use a bitset representation for adjacency.
        # adj_bit[i] is an integer bitmask of neighbors of i.
        # For a C6, we need v0-v1-v2-v3-v4-v5-v0.
        # v1 in N(v0)
        # v2 in N(v1) \ {v0}
        # v3 in N(v2) \ {v0, v1}
        # v4 in N(v3) \ {v0, v1, v2}
        # v5 in N(v4) \ {v0, v1, v2, v3}
        # v0 in N(v5)
        
        # This backtracking check for each 6-subset might be faster on average if C6s are rare.
        
        # Let's stick to the subset enumeration but optimize the inner loop.
        v = combo
        # Fix v0 = v[0]
        # We need to find a cycle of length 6.
        # This is equivalent to checking if the induced subgraph has a Hamiltonian cycle.
        # For 6 vertices, we can just check the 60 permutations.
        # To speed up, we can prune.
        
        # Let's try a different approach: 
        # For each pair of opposite vertices in the potential C6 (distance 3), 
        # they must be non-adjacent and share at least 2 common neighbors that are disjoint?
        # No, v0 and v3 are distance 3.
        # Path v0-v1-v2-v3 and v0-v5-v4-v3.
        # v1, v2 are common neighbors of v0, v3? No.
        # v1 is neighbor of v0, v2 is neighbor of v1 and v3.
        
        # Let's just use the permutation check for the 6 vertices.
        # We can hardcode the 60 cycles? Or generate them.
        # Actually, just checking if the induced subgraph is a C6 or contains a C6.
        # Since we are checking for *any* C6 in the whole graph, and we are iterating over 6-subsets,
        # if the induced subgraph on the 6 vertices contains a C6, it must be a Hamiltonian cycle of that subgraph
        # (because a C6 uses all 6 vertices).
        
        # So, does the induced subgraph have a Hamiltonian cycle?
        # We can use a simple DFS for each subset.
        
        # For speed, let's precompute the 60 valid orderings (cycles) for indices 0..5.
        # But the vertices are v[0]..v[5].
        # We can just check all 60 permutations of 1..5 (fixing 0).
        
        # This is getting complex. Let's try the simple 60-check first.
        
        # Optimization: If the induced subgraph has < 6 edges, no C6.
        # Count edges in induced subgraph.
        edge_count = 0
        for i in range(6):
            for j in range(i+1, 6):
                if adj[v[i]][v[j]]:
                    edge_count += 1
        if edge_count < 6:
            continue
            
        # If edge_count >= 6, check for C6.
        # We can try to find a C6 by checking permutations.
        # To be faster, we can just check if there is a cycle of length 6.
        
        # Let's use a specific check:
        # For each vertex v0 in the subset, remove it and check if the remaining 5 form a path of length 4 connecting two neighbors of v0?
        
        # Actually, for n=17, the number of subsets is small.
        # Let's just do the 60 checks.
        
        # To avoid generating permutations every time, we can precompute the list of 60 tuples of indices (0..5) that form a cycle.
        # But we can also just do a small DFS.
        
        # Let's try a different fast check:
        # A C6 exists iff there are two non-adjacent vertices u, v in the subset such that
        # they have at least 2 common neighbors in the subset, AND those paths can be combined?
        # No, that's C4.
        
        # Let's just use the permutation check. It's 60 * 6 = 360 lookups per subset.
        # 12376 * 360 = 4.4M lookups.
        # In Python, 4.4M simple operations is ~0.1s - 0.2s.
        # This is acceptable for a single check.
        
        # We will implement a helper to check if 6 specific vertices form a C6 in the given adj matrix.
        
        # To make it even faster, we can precompute the 60 cycles of 6 elements.
        # Cycle is (0, 1, 2, 3, 4, 5).
        # We fix 0 at position 0.
        # Permutations of 1,2,3,4,5 such that (5,0) is an edge? No, the cycle is closed.
        # The edges are (p0,p1), (p1,p2), (p2,p3), (p3,p4), (p4,p5), (p5,p0).
        # We fix p0 = v[0].
        # We iterate over permutations of v[1:6].
        
        # There are 120 permutations of 5 elements. Half are reverse of each other?
        # A cycle and its reverse are the same. So 60 unique cycles.
        
        # We can precompute these 60 permutations of (1,2,3,4,5).
        
        pass # Placeholder for logic

# Precompute the 60 permutations of (1,2,3,4,5) that form a cycle with 0.
# Actually, any permutation (1,2,3,4,5) forms a cycle 0-1-2-3-4-5-0 if the edges exist.
# We just need to check the 6 edges.
# We can just iterate over all 120 permutations and check. It's fast enough.
# Or better, just check the 60 unique ones.

# Let's refine the _has_c6_in_color to be as fast as possible.

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Target n=17
    # If we find n=17, try n=18?
    # The prompt says 18 is new.
    # Let's aim for 17 first.
    
    best_n = 9
    best_colors = [0] * (9 * 8 // 2) # Placeholder
    
    # Known good coloring for n=17?
    # I will use a local search to find one.
    
    def edge_idx(i, j, n):
        if i > j: i, j = j, i
        return i * (n - 1) - i * (i + 1) // 2 + j - i - 1

    def build_adj(colors, n):
        adj = [[[False] * n for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                col = colors[edge_idx(i, j, n)]
                adj[col][i][j] = True
                adj[col][j][i] = True
        return adj

    def count_c6s(adj, n):
        total = 0
        for c in range(4):
            if _has_c6_in_color(adj[c], n):
                # Counting is expensive. We just need to know if > 0 for local search?
                # For local search, we want to minimize the number.
                # So we need to count them.
                total += count_c6s_single(adj[c], n)
        return total

    def count_c6s_single(adj, n):
        count = 0
        vertices = range(n)
        for combo in combinations(vertices, 6):
            v = combo
            # Check if induced subgraph has a C6
            # Fast check: edge count
            ec = 0
            for i in range(6):
                for j in range(i+1, 6):
                    if adj[v[i]][v[j]]:
                        ec += 1
            if ec < 6:
                continue
            
            # Check for C6
            # We check the 60 cycles.
            # To avoid generating permutations, we can hardcode the checks or use a small helper.
            # Let's use a helper that checks if the 6 vertices contain a C6.
            if _check_c6_subset(adj, v):
                count += 1
        return count

    def _check_c6_subset(adj, v):
        # v is a tuple of 6 vertices
        # Check if there is a Hamiltonian cycle in the induced subgraph
        # We can do this by checking all 60 cycles.
        # Precomputed cycles for indices 0..5:
        # We fix 0.
        # Permutations of 1,2,3,4,5.
        # We can just iterate.
        
        # Optimization: 
        # If v[0] has degree < 2 in the induced subgraph, return False.
        deg0 = 0
        for i in range(1, 6):
            if adj[v[0]][v[i]]:
                deg0 += 1
        if deg0 < 2:
            return False
            
        # Try to find a cycle starting with 0.
        # Neighbors of 0 in the subset:
        nbrs = [i for i in range(1, 6) if adj[v[0]][v[i]]]
        if len(nbrs) < 2:
            return False
            
        # Pick two neighbors a, b of 0.
        # We need a path from a to b of length 4 using the remaining 4 vertices.
        # The remaining vertices are v[1:6] \ {a, b}.
        
        # This is getting complex. Let's just do the 60 permutation check.
        # It's only 60 * 6 = 360 ops.
        
        # To speed up, we can precompute the list of 60 permutations of (1,2,3,4,5)
        # that represent unique cycles.
        # Actually, any permutation (p1, p2, p3, p4, p5) defines the cycle 0-p1-p2-p3-p4-p5-0.
        # We just need to check if all 6 edges exist.
        
        # We can generate the 60 permutations once globally.
        
        for p in CYCLES_6:
            # p is a tuple of 5 indices (1..5)
            # Check edges:
            # (0, p[0]), (p[0], p[1]), (p[1], p[2]), (p[2], p[3]), (p[3], p[4]), (p[4], 0)
            if (adj[v[0]][v[p[0]]] and 
                adj[v[p[0]]][v[p[1]]] and 
                adj[v[p[1]]][v[p[2]]] and 
                adj[v[p[2]]][v[p[3]]] and 
                adj[v[p[3]]][v[p[4]]] and 
                adj[v[p[4]]][v[0]]):
                return True
        return False

    # Precompute CYCLES_6
    global CYCLES_6
    if not hasattr(construct, 'CYCLES_6'):
        from itertools import permutations
        perms = list(permutations([1, 2, 3, 4, 5]))
        # To get unique cycles, we can fix the first element to be the smallest?
        # No, the cycle 0-1-2-3-4-5-0 is the same as 0-5-4-3-2-1-0.
        # We can just take half the permutations?
        # Or just use all 120, it's still fast.
        CYCLES_6 = perms
        construct.CYCLES_6 = CYCLES_6

    # Local Search for n=17
    n = 17
    num_edges = n * (n - 1) // 2
    
    found = False
    best_colors_found = None
    
    # Try to find a valid coloring for n=17
    # Use random restarts with local search
    
    max_restarts = 200
    max_steps = 500
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        # Greedy init: assign random colors
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        adj = build_adj(colors, n)
        current_cost = count_c6s(adj, n)
        
        if current_cost == 0:
            best_n = n
            best_colors = colors
            found = True
            break
            
        # Local Search
        for step in range(max_steps):
            if time.time() > deadline:
                break
                
            # Pick a random edge
            e = random.randint(0, num_edges - 1)
            old_col = colors[e]
            new_col = random.randint(0, 3)
            if new_col == old_col:
                continue
                
            # Update colors
            colors[e] = new_col
            
            # Update adj
            # We need to map edge index to (i,j)
            # This is slow. Let's precompute edge_to_pair.
            
            # For speed, let's precompute edge_to_pair for n=17.
            # Or just rebuild adj? Rebuilding is O(n^2) = 289 ops.
            # count_c6s is O(12376 * 360) ~ 4.4M ops.
            # So rebuilding is negligible compared to counting.
            
            adj = build_adj(colors, n)
            new_cost = count_c6s(adj, n)
            
            if new_cost <= current_cost:
                current_cost = new_cost
                if current_cost == 0:
                    best_n = n
                    best_colors = colors
                    found = True
                    break
            else:
                # Revert
                colors[e] = old_col
                
            if found:
                break
                
        if found:
            break

    if found:
        # Try to extend to n=18?
        # This is hard. We'll skip for now to ensure we return n=17.
        # If we have time, we can try.
        
        # Check if we have time to try n=18
        if time.time() < deadline - 20:
            # Try n=18
            # Start with the n=17 solution and add vertex 17.
            # Assign colors to edges (17, i) for i in 0..16.
            # Check if any C6 is formed.
            # This is a constraint satisfaction problem.
            # We can use backtracking or local search on the 17 edges.
            
            n_try = 18
            num_edges_try = n_try * (n_try - 1) // 2
            colors_try = best_colors[:] + [0] * 17 # 17 new edges
            
            # We need to find colors for the 17 new edges such that no C6 is formed.
            # A C6 involving vertex 17 must use 17 and 5 other vertices.
            # Or a C6 not involving 17 (already valid).
            # So we only need to check C6s that include vertex 17.
            
            # This is a specific check.
            # Let's try random assignment for the 17 edges and check.
            
            for attempt in range(100):
                if time.time() > deadline - 5:
                    break
                    
                for i in range(17):
                    colors_try[edge_idx(i, 17, n_try)] = random.randint(0, 3)
                    
                # Check validity
                adj_try = build_adj(colors_try, n_try)
                # Only check C6s involving 17?
                # No, just check all.
                # But we know the old part is valid.
                # So we only need to check C6s that include at least one new edge.
                # This is complex. Let's just check all.
                
                valid = True
                for c in range(4):
                    if _has_c6_in_color(adj_try[c], n_try):
                        valid = False
                        break
                
                if valid:
                    best_n = n_try
                    best_colors = colors_try
                    break

    if not found:
        # Fallback to n=9 if we couldn't find 17?
        # But we should be able to find 17.
        # If not, return the best we have.
        # The seed had n=9.
        
        # Let's ensure we return at least n=9.
        # The current best_n is 9.
        pass

    return {'n': best_n, 'colors': best_colors}

# We need to define _has_c6_in_color and _check_c6_subset properly.
# And CYCLES_6.

# Let's refine the code structure.

CYCLES_6 = None

def _init_cycles():
    global CYCLES_6
    if CYCLES_6 is None:
        from itertools import permutations
        CYCLES_6 = list(permutations([1, 2, 3, 4, 5]))

_init_cycles()

def _check_c6_subset(adj, v):
    # v is a tuple of 6 vertices
    for p in CYCLES_6:
        if (adj[v[0]][v[p[0]]] and 
            adj[v[p[0]]][v[p[1]]] and 
            adj[v[p[1]]][v[p[2]]] and 
            adj[v[p[2]]][v[p[3]]] and 
            adj[v[p[3]]][v[p[4]]] and 
            adj[v[p[4]]][v[0]]):
            return True
    return False

def _has_c6_in_color(adj, n):
    vertices = range(n)
    for combo in combinations(vertices, 6):
        # Fast edge count check
        ec = 0
        for i in range(6):
            for j in range(i+1, 6):
                if adj[combo[i]][combo[j]]:
                    ec += 1
        if ec < 6:
            continue
        if _check_c6_subset(adj, combo):
            return True
    return False

def count_c6s_single(adj, n):
    count = 0
    vertices = range(n)
    for combo in combinations(vertices, 6):
        ec = 0
        for i in range(6):
            for j in range(i+1, 6):
                if adj[combo[i]][combo[j]]:
                    ec += 1
        if ec < 6:
            continue
        if _check_c6_subset(adj, combo):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_n = 9
    best_colors = [0] * (9 * 8 // 2)
    
    # Hardcode a known good n=17 coloring if possible?
    # I will rely on the search.
    
    n = 17
    num_edges = n * (n - 1) // 2
    
    found = False
    
    for restart in range(200):
        if time.time() > deadline:
            break
            
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        adj = [[[False] * n for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                col = colors[edge_idx(i, j, n)]
                adj[col][i][j] = True
                adj[col][j][i] = True
                
        current_cost = 0
        for c in range(4):
            current_cost += count_c6s_single(adj[c], n)
            
        if current_cost == 0:
            best_n = n
            best_colors = colors
            found = True
            break
            
        for step in range(500):
            if time.time() > deadline:
                break
                
            e = random.randint(0, num_edges - 1)
            old_col = colors[e]
            new_col = random.randint(0, 3)
            if new_col == old_col:
                continue
                
            colors[e] = new_col
            
            # Rebuild adj
            adj = [[[False] * n for _ in range(n)] for _ in range(4)]
            for i in range(n):
                for j in range(i + 1, n):
                    col = colors[edge_idx(i, j, n)]
                    adj[col][i][j] = True
                    adj[col][j][i] = True
                    
            new_cost = 0
            for c in range(4):
                new_cost += count_c6s_single(adj[c], n)
                
            if new_cost <= current_cost:
                current_cost = new_cost
                if current_cost == 0:
                    best_n = n
                    best_colors = colors
                    found = True
                    break
            else:
                colors[e] = old_col
                
            if found:
                break
                
        if found:
            break

    if not found:
        # If we didn't find 17, we return 9.
        # But we should try harder or use a better init.
        # Greedy init might be better.
        
        # Let's try a greedy init.
        # Assign colors to edges in order, picking the color that minimizes C6s formed so far?
        # This is hard to check incrementally.
        
        # Let's just return 9 if we fail.
        pass

    return {'n': best_n, 'colors': best_colors}

def edge_idx(i, j, n):
    if i > j: i, j = j, i
    return i * (n - 1) - i * (i + 1) // 2 + j - i - 1

# The code above is a bit slow due to rebuilding adj and counting all C6s.
# We can optimize by only checking C6s that are affected by the change.
# But for n=17, the count is small enough.

# Let's add a check for n=18 if we succeed with 17.

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_n = 9
    best_colors = [0] * (9 * 8 // 2)
    
    n = 17
    num_edges = n * (n - 1) // 2
    
    found = False
    best_colors_17 = None
    
    for restart in range(200):
        if time.time() > deadline - 10:
            break
            
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        adj = [[[False] * n for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                col = colors[edge_idx(i, j, n)]
                adj[col][i][j] = True
                adj[col][j][i] = True
                
        current_cost = 0
        for c in range(4):
            current_cost += count_c6s_single(adj[c], n)
            
        if current_cost == 0:
            best_n = n
            best_colors = colors
            best_colors_17 = colors
            found = True
            break
            
        for step in range(500):
            if time.time() > deadline - 10:
                break
                
            e = random.randint(0, num_edges - 1)
            old_col = colors[e]
            new_col = random.randint(0, 3)
            if new_col == old_col:
                continue
                
            colors[e] = new_col
            
            adj = [[[False] * n for _ in range(n)] for _ in range(4)]
            for i in range(n):
                for j in range(i + 1, n):
                    col = colors[edge_idx(i, j, n)]
                    adj[col][i][j] = True
                    adj[col][j][i] = True
                    
            new_cost = 0
            for c in range(4):
                new_cost += count_c6s_single(adj[c], n)
                
            if new_cost <= current_cost:
                current_cost = new_cost
                if current_cost == 0:
                    best_n = n
                    best_colors = colors
                    best_colors_17 = colors
                    found = True
                    break
            else:
                colors[e] = old_col
                
            if found:
                break
                
        if found:
            break

    if found and time.time() < deadline - 20:
        # Try n=18
        n_try = 18
        num_edges_try = n_try * (n_try - 1) // 2
        colors_try = best_colors_17[:] + [0] * 17
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
                
            for i in range(17):
                colors_try[edge_idx(i, 17, n_try)] = random.randint(0, 3)
                
            adj_try = [[[False] * n_try for _ in range(n_try)] for _ in range(4)]
            for i in range(n_try):
                for j in range(i + 1, n_try):
                    col = colors_try[edge_idx(i, j, n_try)]
                    adj_try[col][i][j] = True
                    adj_try[col][j][i] = True
                    
            valid = True
            for c in range(4):
                if _has_c6_in_color(adj_try[c], n_try):
                    valid = False
                    break
            
            if valid:
                best_n = n_try
                best_colors = colors_try
                break

    return {'n': best_n, 'colors': best_colors}
