import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # Strategy: 
    # 1. Attempt to construct a graph for n=35, 34, ... down to 29 using a 
    #    heuristic that tries to maintain K4-free and alpha <= 5.
    # 2. Use a greedy construction with backtracking/repair if simple greedy fails.
    # 3. If a base graph is found, run Simulated Annealing to optimize edge density 
    #    or validity, though for fixed n, validity is binary. 
    #    Actually, for this task, if we find a valid graph for n, we return it.
    #    The key is finding the largest n.
    #    
    #    Since R(4,6) is conjectured to be 36 (Exoo's 35 is the record), 
    #    we should aggressively try n=35.
    #    
    #    Approach for n=35:
    #    - Start with a random graph or a structured one (e.g., Paley 29 + 6 random).
    #    - Use local search (edge flips) to fix violations (K4 or Indep-6).
    
    def is_valid(adj, n):
        # Check K4
        # Optimization: Only check triangles formed by neighbors
        # For n=35, this is fast enough if we do it smartly.
        # Simple O(n^4) is ~ 1.5M checks, very fast in Python if optimized.
        # But let's use a slightly smarter check.
        
        # Precompute neighbors
        neighbors = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    neighbors[i].add(j)
                    neighbors[j].add(i)
        
        # Check K4
        for i in range(n):
            for j in neighbors[i]:
                if j <= i: continue
                # Common neighbors of i and j
                common = neighbors[i] & neighbors[j]
                for k in common:
                    if k <= j: continue
                    # Check if any neighbor of k is also in common and > k
                    # Actually, just check if there is a common neighbor l > k
                    # such that adj[k][l] and adj[i][l] and adj[j][l]
                    # Wait, if l is in neighbors[i] & neighbors[j] & neighbors[k], 
                    # and l > k, then i,j,k,l form a K4.
                    # We just need to check if (neighbors[i] & neighbors[j] & neighbors[k]) has any element > k.
                    # To be safe and simple for n<=40:
                    for l in neighbors[k]:
                        if l > k and adj[i][l] and adj[j][l]:
                            return False
                            
        # Check Independent Set of 6
        # This is the harder part. 
        # We can use a branching algorithm or just itertools for n=35?
        # 35 choose 6 is 1.6M. In Python, checking independence is O(1) with bitsets or O(6^2) with loops.
        # 1.6M * 36 ops ~ 60M ops. Might be slow in pure Python (1-2s).
        # Let's use a recursive search for independent set of size 6.
        
        def has_indep_set(size, start_idx):
            # Greedy/Backtracking check
            # Use bitsets for speed
            # Represent graph as list of integers (bitmasks)
            # But we have adj matrix. Let's stick to simple logic for clarity, 
            # but optimize the inner loop.
            
            # Better: Use a specific independent set finder.
            # Since n is small, we can just check all combinations if we are careful.
            # But 1.6M iterations in Python loop is risky for time.
            # Let's implement a quick check using recursion with pruning.
            
            # Actually, for n=35, a simple loop over combinations might be too slow if called repeatedly.
            # But we only call it at the end or periodically.
            
            # Let's try a different approach: 
            # If we maintain the graph, we can check incrementally? No, we are doing SA.
            
            # Let's use itertools.combinations but break early.
            # To speed up, we can precompute for each vertex a bitmask of neighbors.
            
            pass # Placeholder for logic inside is_valid
            
        # Let's rewrite is_valid to be faster using bitmasks.
        # Convert adj to list of bitmasks
        masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if adj[i][j]:
                    mask |= (1 << j)
            masks[i] = mask
            
        # Re-check K4 with bitmasks
        for i in range(n):
            for j in range(i+1, n):
                if not (masks[i] & (1 << j)):
                    continue
                # Common neighbors of i and j
                common = masks[i] & masks[j]
                # Remove i and j from common? They are not neighbors of each other in the subgraph?
                # i is in masks[j], j is in masks[i]. 
                # common includes i and j? No, masks[i] has bits for neighbors. 
                # So common has bits for k such that adj[i][k] and adj[j][k].
                # We need to find if there is a triangle in the subgraph induced by common.
                # i.e., is there k, l in common such that adj[k][l] and k < l?
                # Actually, if common contains a triangle, we have K4.
                # Let's just iterate k in common.
                # To be fast, iterate set bits.
                cm = common
                while cm:
                    lsb = cm & -cm
                    k = lsb.bit_length() - 1
                    cm ^= lsb
                    if k <= j: continue
                    # Check if k forms a triangle with i, j? 
                    # We already know i-j, i-k, j-k.
                    # We need l in common such that adj[k][l].
                    # common_k = masks[k] & common
                    # If common_k has any bit set, we have a K4 (i,j,k,l).
                    # Wait, l must be distinct from i,j,k.
                    # common contains i,j? 
                    # masks[i] has j. masks[j] has i. 
                    # So common has i and j.
                    # We should mask out i and j.
                    common_no_ij = common & ~(1 << i) & ~(1 << j)
                    # Now check for edge in common_no_ij
                    # If any vertex k in common_no_ij has a neighbor in common_no_ij, 
                    # and that neighbor l > k, then we have a triangle k,l,? 
                    # Actually, if there is an edge between any two vertices in common_no_ij,
                    # say k and l, then i,j,k,l is a K4?
                    # Yes, because i,j are connected to all in common.
                    # So we just need to check if the subgraph induced by common_no_ij has any edge.
                    
                    # Iterate k in common_no_ij
                    cm2 = common_no_ij
                    while cm2:
                        lsb2 = cm2 & -cm2
                        k = lsb2.bit_length() - 1
                        cm2 ^= lsb2
                        # Check if k has any neighbor in common_no_ij with index > k
                        # to avoid double counting and self loops (none)
                        # Actually, just check if masks[k] & common_no_ij has any bit.
                        # If it has a bit for l, and l != k, then edge exists.
                        # Since common_no_ij doesn't contain k? No, it does.
                        # masks[k] & common_no_ij will include k? No, no self loops.
                        # So if (masks[k] & common_no_ij) != 0, there is an edge.
                        if (masks[k] & common_no_ij) != 0:
                            return False
                            
        # Check Independent Set of 6
        # Independent set in G = Clique in Complement G.
        # Let's work with non-neighbors.
        # non_masks[i] = ~(masks[i]) & ((1 << n) - 1)
        # We want to find if there is a clique of size 6 in the complement graph.
        
        # Heuristic check: 
        # If average degree is high, independent sets are small.
        # For n=35, if avg degree is ~15, alpha is likely small.
        
        # Let's use a simple backtracking for independent set of size 6.
        # This is the bottleneck.
        
        def check_indep_set_6():
            # Use bitmasks for speed
            # non_adj[i] is bitmask of non-neighbors (including self? no, usually exclude self)
            # Let's define non_adj[i] as bitmask of j != i such that not adj[i][j].
            non_adj = [0] * n
            for i in range(n):
                m = 0
                for j in range(n):
                    if i != j and not adj[i][j]:
                        m |= (1 << j)
                non_adj[i] = m
                
            # We want to pick v1, v2, v3, v4, v5, v6 such that they are mutually non-adjacent.
            # v2 must be in non_adj[v1]
            # v3 must be in non_adj[v1] & non_adj[v2]
            # etc.
            
            # Recursive search
            def backtrack(current_set, candidates_mask, depth):
                if depth == 6:
                    return True
                # If not enough candidates, fail
                # Popcount of candidates_mask < 6 - depth?
                # Simple check: if candidates_mask == 0: return False
                
                # Heuristic: pick a vertex from candidates
                # To reduce branching, pick a vertex that has fewest connections to other candidates?
                # Or just pick the first one?
                # Let's pick the vertex with the highest degree in the complement subgraph?
                # Actually, for independent set in G, we want vertices with low degree in G.
                # Which is high degree in Complement.
                
                # Let's just iterate through candidates.
                # Optimization: if depth == 5, check if any candidate is non-adjacent to all in current_set?
                # No, current_set is already independent.
                # We just need to pick 6 - depth more vertices from candidates that are non-adjacent to each other.
                
                # Get list of candidates
                # candidates_mask has bits for valid next vertices.
                
                # Try each candidate
                cm = candidates_mask
                while cm:
                    lsb = cm & -cm
                    v = lsb.bit_length() - 1
                    cm ^= lsb
                    
                    # New candidates are those in candidates_mask that are non-adjacent to v
                    # AND have index > v (to avoid permutations)
                    # Actually, we can just use the mask of non-adj[v] & candidates_mask
                    # But we must ensure we don't pick vertices already picked?
                    # candidates_mask already excludes picked vertices? 
                    # Let's maintain candidates_mask as the set of available vertices.
                    
                    # When we pick v, we remove v from candidates.
                    # New candidates = candidates_mask & non_adj[v]
                    # But we also need to ensure they are > v? 
                    # If we always pick the lowest index available, we can enforce order.
                    # Let's enforce v > last_picked.
                    
                    # Actually, the standard way:
                    # candidates_mask initially is all vertices.
                    # When picking v, new_mask = candidates_mask & non_adj[v] & ~(bits <= v)
                    # This ensures we only pick higher indices.
                    
                    # But wait, non_adj[v] includes v? No, we defined it as j != i.
                    # So v is not in non_adj[v].
                    
                    # Let's refine:
                    # Start with candidates = (1 << n) - 1
                    # Pick v.
                    # Next candidates = candidates & non_adj[v] & ~((1 << (v+1)) - 1)
                    
                    # In backtrack, we pass the mask of available vertices (indices > last picked).
                    # Let's assume candidates_mask only contains indices > last_picked.
                    
                    next_mask = candidates_mask & non_adj[v]
                    # next_mask should only contain indices > v?
                    # If candidates_mask already enforced that, then yes.
                    
                    if backtrack(current_set | (1 << v), next_mask, depth + 1):
                        return True
                        
                return False
                
            # Initial call:
            # We can start with any vertex.
            # To speed up, we can order vertices by degree in G (ascending).
            # Low degree in G = High chance of being in independent set.
            
            # Let's just try from 0 to n-1.
            for start_v in range(n):
                # Candidates must be > start_v and non-adjacent to start_v
                init_mask = non_adj[start_v] & ~((1 << (start_v + 1)) - 1)
                if backtrack(1 << start_v, init_mask, 1):
                    return True
            return False

        if check_indep_set_6():
            return False
            
        return True

    # Main Loop
    # Try n from 35 down to 29
    # For each n, try to construct a valid graph.
    # Method: Randomized Local Search (Simulated Annealing)
    # Objective: Minimize Violations (K4 count + Indep6 count)
    # If Violations == 0, we have a valid graph.
    
    best_n = 29
    best_edges = None
    
    # Pre-compute Paley 29 as a strong seed
    q = 29
    squares = {x * x % q for x in range(1, q)}
    base_adj = [[False] * 29 for _ in range(29)]
    base_edges = []
    for i in range(29):
        for j in range(i + 1, 29):
            if (j - i) % q in squares:
                base_adj[i][j] = True
                base_adj[j][i] = True
                base_edges.append([i, j])
                
    # Verify base
    if is_valid(base_adj, 29):
        best_n = 29
        best_edges = base_edges.copy()
    else:
        # Fallback to smaller
        # This shouldn't happen, but just in case
        best_n = 1
        best_edges = []
        
    # Try to extend to n=30..35
    for target_n in range(30, 36):
        if time.time() > deadline - 10.0:
            break
            
        # Start with best known graph (or Paley 29 if target_n > 29)
        # If target_n > 29, we start with Paley 29 + (target_n - 29) random vertices
        # Or just start with a random graph of target_n?
        # Random graph G(n, p) with p ~ 0.5 is likely to have K4 and Indep6.
        # We need to tune p or use a structured start.
        
        # Let's start with the current best graph (size best_n) and add random vertices?
        # Or just run SA on a random graph of size target_n?
        
        # SA on random graph of size target_n
        # p = 0.5 is too high for K4-free?
        # Expected K4 in G(35, 0.5) is huge.
        # We need a graph with no K4.
        # Triangle-free graphs have no K4.
        # But we need alpha <= 5.
        # Triangle-free graphs on 35 vertices can have large independent sets.
        # Ramsey R(3, 6) = 18. So any triangle-free graph on 18 vertices has Indep 6.
        # So a triangle-free graph on 35 vertices MUST have an independent set of size >= 6.
        # Therefore, the graph CANNOT be triangle-free.
        # It must have triangles, but no K4.
        
        # So we need a graph with triangles but no K4, and alpha <= 5.
        
        # Strategy: Start with a graph that is "close" to valid.
        # Maybe start with Paley 29 and add 6 random vertices with random connections?
        
        start_adj = [[False] * target_n for _ in range(target_n)]
        
        # Copy Paley 29
        for i in range(29):
            for j in range(29):
                start_adj[i][j] = base_adj[i][j]
                
        # Add random edges for new vertices (29..target_n-1)
        # And random edges between new and old?
        # Let's just randomize the whole graph? No, Paley 29 is a good core.
        # Let's randomize the connections of the new vertices.
        
        for i in range(29, target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    start_adj[i][j] = True
                    start_adj[j][i] = True
            for j in range(29):
                if random.random() < 0.5:
                    start_adj[i][j] = True
                    start_adj[j][i] = True
                    
        # Run SA
        # Objective: Count K4 + Count Indep6
        # We need a fast violation counter.
        
        def count_violations(adj, n):
            k4_count = 0
            # Use bitmasks
            masks = [0] * n
            for i in range(n):
                m = 0
                for j in range(n):
                    if adj[i][j]:
                        m |= (1 << j)
                masks[i] = m
                
            # K4 Count
            for i in range(n):
                for j in range(i+1, n):
                    if not (masks[i] & (1 << j)): continue
                    common = masks[i] & masks[j] & ~((1 << (j+1)) - 1) # Only j+1..n-1
                    # For each k in common, check if there is l in common & masks[k] with l > k
                    cm = common
                    while cm:
                        lsb = cm & -cm
                        k = lsb.bit_length() - 1
                        cm ^= lsb
                        # Check edges in common & masks[k]
                        # We just need to know if there is ANY edge.
                        # If there is an edge between two vertices in common, and one of them is k?
                        # No, we need a triangle in the subgraph induced by common.
                        # If common contains a triangle, we have K4.
                        # Actually, if there is an edge between ANY two vertices in common, 
                        # say k and l, then i,j,k,l is a K4?
                        # Yes, i,j connected to k,l. k,l connected.
                        # So we just need to check if the subgraph induced by common has any edge.
                        
                        # To be precise:
                        # We iterate k.
                        # We check if there is an edge between k and any other vertex in common.
                        # If yes, K4 found.
                        
                        # Optimization: 
                        # If common has more than 2 vertices, it's likely to have an edge.
                        # Let's just check:
                        # edge_found = False
                        # for l in common:
                        #    if l != k and adj[k][l]:
                        #       edge_found = True
                        #       break
                        # if edge_found: k4_count += 1
                        
                        # Faster:
                        # neighbors_of_k_in_common = masks[k] & common
                        # If neighbors_of_k_in_common != 0, then there is an edge.
                        # (It might be an edge to i or j? No, i,j are not in common? 
                        # common = masks[i] & masks[j]. 
                        # i is in masks[j], j is in masks[i]. 
                        # So i and j ARE in common.
                        # We must exclude i and j.
                        common_no_ij = common & ~(1 << i) & ~(1 << j)
                        if (masks[k] & common_no_ij) != 0:
                            k4_count += 1
                            
            # Indep 6 Count is hard to count exactly.
            # Just check if ANY exists.
            # For SA, we need a score.
            # Let's use a heuristic score: 
            # Score = K4_count + 10 * (1 if has_indep_6 else 0)
            # But checking has_indep_6 is expensive.
            
            # Alternative: 
            # Use a relaxation. 
            # Or just do local search on "validity" directly.
            # If invalid, try to fix.
            
            # Let's stick to: 
            # If valid, return 0.
            # Else, return 1.
            # This is a 0-1 landscape. SA might get stuck.
            
            # Better: 
            # Count "potential" violations.
            # e.g., number of triangles that are part of a K4?
            # This is getting complex.
            
            # Let's try a different approach for the search:
            # Greedy Repair.
            # 1. Start with a random graph.
            # 2. While invalid:
            #    a. Find a K4. Remove a random edge from it.
            #    b. Find an Indep 6. Add a random edge within it.
            #    c. Repeat.
            
            return k4_count # Placeholder, we will implement repair loop outside

        # Implement Greedy Repair Loop
        current_adj = start_adj
        valid = False
        attempts = 0
        max_attempts = 5000
        
        while not valid and attempts < max_attempts and time.time() < deadline - 5.0:
            attempts += 1
            
            # Check Validity
            # We need a fast is_valid.
            # Let's use the is_valid defined above.
            if is_valid(current_adj, target_n):
                valid = True
                break
                
            # Find a violation
            # 1. Check K4
            k4_found = False
            # Re-use logic from is_valid to find one K4
            masks = [0] * target_n
            for i in range(target_n):
                m = 0
                for j in range(target_n):
                    if current_adj[i][j]:
                        m |= (1 << j)
                masks[i] = m
                
            for i in range(target_n):
                if k4_found: break
                for j in range(i+1, target_n):
                    if k4_found: break
                    if not (masks[i] & (1 << j)): continue
                    common = masks[i] & masks[j] & ~((1 << (j+1)) - 1)
                    cm = common
                    while cm:
                        if k4_found: break
                        lsb = cm & -cm
                        k = lsb.bit_length() - 1
                        cm ^= lsb
                        common_no_ij = common & ~(1 << i) & ~(1 << j)
                        if (masks[k] & common_no_ij) != 0:
                            # Found K4: i, j, k, l
                            # Find l
                            l_mask = masks[k] & common_no_ij
                            lsb_l = l_mask & -l_mask
                            l = lsb_l.bit_length() - 1
                            
                            # Remove a random edge from this K4
                            edges_to_remove = [(i,j), (i,k), (i,l), (j,k), (j,l), (k,l)]
                            u, v = random.choice(edges_to_remove)
                            current_adj[u][v] = False
                            current_adj[v][u] = False
                            k4_found = True
                            break
                            
            if k4_found:
                continue
                
            # 2. Check Indep 6
            # Find one
            # Use the backtrack logic to return the set
            def find_indep_6():
                non_adj = [0] * target_n
                for i in range(target_n):
                    m = 0
                    for j in range(target_n):
                        if i != j and not current_adj[i][j]:
                            m |= (1 << j)
                    non_adj[i] = m
                    
                def backtrack(current_set, candidates_mask, depth, path):
                    if depth == 6:
                        return path[:]
                    cm = candidates_mask
                    while cm:
                        lsb = cm & -cm
                        v = lsb.bit_length() - 1
                        cm ^= lsb
                        next_mask = candidates_mask & non_adj[v]
                        res = backtrack(current_set | (1 << v), next_mask, depth + 1, path + [v])
                        if res:
                            return res
                    return None
                    
                for start_v in range(target_n):
                    init_mask = non_adj[start_v] & ~((1 << (start_v + 1)) - 1)
                    res = backtrack(1 << start_v, init_mask, 1, [start_v])
                    if res:
                        return res
                return None
                
            indep_set = find_indep_6()
            if indep_set:
                # Add a random edge within this set
                u, v = random.sample(indep_set, 2)
                current_adj[u][v] = True
                current_adj[v][u] = True
            else:
                # Should not happen if is_valid returned False due to indep set
                # But if is_valid is buggy?
                break

        if valid:
            # Convert to edge list
            edges = []
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if current_adj[i][j]:
                        edges.append([i, j])
                        
            best_n = target_n
            best_edges = edges
            # We found a larger n, so we can stop or continue?
            # Task says Maximize n.
            # If we found 35, we can stop.
            if target_n == 35:
                break

    return {'n': best_n, 'edges': best_edges}
