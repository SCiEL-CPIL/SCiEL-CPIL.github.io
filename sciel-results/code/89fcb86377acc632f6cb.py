import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 105  # Leave 15s margin

    # Precompute all 6-subsets for n=20 and n=21
    def gen_combinations(n, k=6):
        return list(itertools.combinations(range(n), k))
    
    # Cache for combinations to avoid regeneration if needed, though lists are fine
    # C(20, 6) = 38760
    # C(21, 6) = 54264

    def check_validity_fast(n, adj_np, combs):
        """
        Checks if no subset of size >= 6 induces a regular subgraph.
        We assume if it's valid for 6, it likely is for larger, but we must check all.
        However, checking all subsets for n=20 is expensive.
        Optimization: If a subset of size 6 is regular, it's bad.
        If a subset of size > 6 is regular, does it contain a regular subset of size 6?
        Not necessarily. A 7-vertex regular graph (e.g., K7) contains 6-vertex regular subgraphs (K6).
        In fact, any regular graph on k vertices with k >= 6 contains a regular induced subgraph on 6 vertices?
        Let's check.
        If G is regular on k vertices.
        If k=6, yes.
        If k>6, does it contain a 6-vertex regular induced subgraph?
        Consider a cycle C_7. It is 2-regular.
        Induced subgraphs of C_7 of size 6:
        Removing one vertex from C_7 creates a path P_6.
        Degrees in P_6: 1, 2, 2, 2, 2, 1. Not regular.
        So C_7 does NOT contain a 6-vertex regular induced subgraph.
        Thus, we MUST check all sizes >= 6.
        
        However, for n=20, checking all subsets is:
        C(20,6) + C(20,7) + ... + C(20,20)
        This is 2^20 - (C(20,0)..C(20,5)) ~ 1,000,000 - 21600 ~ 1M checks.
        Each check involves extracting sub-matrix and summing.
        1M * ~1us = 1s? Maybe 10s with Python overhead.
        We need to be careful.
        
        Strategy:
        1. Find a graph with 0 bad 6-subsets using local search.
        2. Verify full validity. If valid, return.
        3. If invalid, try to fix by removing edges from larger regular subgraphs?
           Or just restart.
        """
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                # Fast check: degrees must be all equal
                # Use adjacency matrix bit operations?
                # Or just numpy.
                indices = np.array(comb)
                sub_adj = adj_np[np.ix_(indices, indices)]
                degrees = np.sum(sub_adj, axis=1)
                if np.all(degrees == degrees[0]):
                    return False
        return True

    def count_bad_6(n, adj_np, combs):
        """
        Counts number of 6-subsets inducing regular subgraph.
        Optimized: 
        For a 6-subset, degrees must be equal.
        Possible degrees in a 6-vertex simple graph: 0, 1, 2, 3, 4, 5.
        We can check if all degrees are 0 (empty), all 5 (complete), all 1 (perfect matching), all 2 (C6 or 3K2 or etc), all 3 (K3,3 or K6-K3,3? no, regular 3 on 6 is K3,3 or complement of 2-regular... wait. 3-regular on 6 vertices is K3,3 or K6 minus perfect matching? No, K6 is 5-regular. 3-regular on 6 is K3,3 or disjoint union? No, connected or not.
        Actually, just checking `np.all(degrees == degrees[0])` is fast enough if vectorized?
        Let's try to speed up `count_bad_6`.
        """
        count = 0
        # Pre-convert combs to numpy arrays?
        # That takes memory but saves time in loop.
        comb_arrays = [np.array(c) for c in combs]
        
        for indices in comb_arrays:
            # Extract subgraph degrees
            # adj_np[indices, :] gives rows for these vertices.
            # We only care about columns that are in indices.
            # Masking is expensive.
            # Alternative: 
            # deg_i = sum(adj_np[i, indices] for i in indices)
            # This is O(k^2) per subset.
            # 
            # Faster approach:
            # sub_adj = adj_np[np.ix_(indices, indices)]
            # degs = sub_adj.sum(axis=1)
            # if (degs == degs[0]).all(): count += 1
            
            sub_adj = adj_np[np.ix_(indices, indices)]
            degs = sub_adj.sum(axis=1)
            if degs[0] == degs[1] == degs[2] == degs[3] == degs[4] == degs[5]:
                count += 1
        return count

    def local_search(n, max_iters=1000, combs=None, verbose=False):
        """
        Local search to minimize number of bad 6-subsets.
        """
        if combs is None:
            combs = gen_combinations(n, 6)
        
        # Initialize random graph
        # Start with a slightly biased graph? 
        # Random is usually fine.
        adj_np = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_np[i, j] = 1
                    adj_np[j, i] = 1
        
        # Initial score
        score = count_bad_6(n, adj_np, combs)
        
        if verbose:
            print(f"Initial score for n={n}: {score}")
            
        if score == 0:
            if check_validity_fast(n, adj_np, combs):
                return adj_np

        best_adj = adj_np.copy()
        best_score = score
        current_score = score
        
        # Precompute edge list
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Simulated Annealing or simple hill climbing
        # Simple hill climbing with restarts is often better for this.
        # Let's do aggressive hill climbing.
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Pick a random edge to flip
            u, v = random.choice(edges)
            
            # Flip
            adj_np[u, v] ^= 1
            adj_np[v, u] ^= 1
            
            # Calculate delta in score?
            # Flipping edge (u,v) affects only 6-subsets containing both u and v.
            # How many 6-subsets contain u and v? C(n-2, 4).
            # For n=20, C(18, 4) = 3060.
            # This is a significant fraction of the 38760 total.
            # So we can't just update locally easily without checking those 3060 subsets.
            # Checking 3060 subsets vs checking all 38760.
            # 3060/38760 ~ 8%.
            # So local update is ~12x faster.
            
            # Let's implement delta calculation.
            # Subsets containing u and v:
            # Need to precompute for each pair (u,v) the list of 6-subsets containing it.
            # This requires memory but speeds up search significantly.
            
            # We will generate these maps outside the loop for this specific n.
            # But local_search is called for n=21 and n=20.
            # Let's pass the maps as well.
            
            # If we don't have maps, just recompute all.
            # For simplicity in this iteration, let's assume we have maps.
            
            new_score = score - delta_score(adj_np, u, v, n, combs_map)
            
            # If new_score < current_score, accept.
            # If new_score == current_score, accept with some probability? Or always?
            # If new_score > current_score, reject.
            
            # Wait, I need `combs_map` passed in.
            # Let's restructure.
            pass # Placeholder

    # We need a faster delta calculation.
    # Let's precompute the "neighborhood" of each edge in terms of 6-subsets.
    
    def prepare_search_data(n):
        combs = list(itertools.combinations(range(n), 6))
        combs_set = set(combs)
        
        # Map each edge (u,v) to list of indices of 6-combs containing u and v
        edge_to_combs = {}
        # Index combs
        for idx, comb in enumerate(combs):
            # For each pair in comb, add idx to their map
            for i in range(6):
                for j in range(i+1, 6):
                    u, v = comb[i], comb[j]
                    if u > v:
                        u, v = v, u
                    edge_to_combs.setdefault((u, v), []).append(idx)
        
        # Convert comb indices to numpy arrays for fast access?
        # Or just keep combs as list of tuples and use indexing.
        # Accessing combs[idx] is fast.
        
        return combs, edge_to_combs

    def get_delta(adj_np, u, v, n, combs, edge_to_combs, idxs):
        """
        Calculate change in number of regular 6-subsets when flipping edge (u,v).
        Only subsets in `idxs` (those containing u and v) are affected.
        """
        delta = 0
        # For each subset index in idxs:
        # Check if it was regular before flip.
        # Check if it is regular after flip.
        # 
        # Optimization: 
        # The flip changes the degree of u and v by +/- 1 in the induced subgraph.
        # If the subset was regular with degree d:
        #   If we add edge (u,v): degs become d, ..., d+1, d+1, ..., d.
        #   New degrees: u and v have d+1, others d.
        #   This is regular ONLY if d == d+1 (impossible) or if all were d+1?
        #   Wait. If all were d, and we add edge, u,v become d+1.
        #   So it is regular if and only if the other 4 vertices also had degree d+1?
        #   No, the other 4 vertices' degrees in the induced subgraph DO NOT change.
        #   So if it was regular (all d), after adding edge, u,v are d+1, others d. Not regular (unless d=d+1).
        #   So adding edge to a regular subset makes it NON-regular.
        #   Exception: If it was already NON-regular, it might become regular.
        #
        # Case 1: Subset was Regular (all deg = d).
        #   Flip adds edge (u,v).
        #   New degs: u,v = d+1. Others = d.
        #   Regular? Only if d+1 == d (False).
        #   So it becomes Non-Regular.
        #   Delta = -1 (we removed a bad subset).
        #
        #   Flip removes edge (u,v).
        #   New degs: u,v = d-1. Others = d.
        #   Regular? Only if d-1 == d (False).
        #   So it becomes Non-Regular.
        #   Delta = -1.
        #
        # Conclusion: If a subset is regular and contains the flipped edge, flipping that edge ALWAYS makes it non-regular.
        #
        # Case 2: Subset was Non-Regular.
        #   Flip changes degs of u,v by +/- 1.
        #   It might become regular.
        #   We need to check if the new degrees are all equal.
        #   We need to know the current degrees of all 6 vertices in the subset.
        #
        # So, for each subset idx in idxs:
        #   1. Calculate current degrees (before flip).
        #   2. If regular, delta -= 1.
        #   3. Else, calculate new degrees (after flip) and check if regular. If so, delta += 1.
        #
        # To calculate current degrees, we need to sum the row for u and v restricted to the subset?
        # No, we need the degree of EACH vertex in the induced subgraph.
        # 
        # This requires checking all 6 vertices.
        # So for each of the ~3000 subsets, we do ~6 sums.
        # 3000 * 6 * 2 (before/after) = 36000 operations.
        # vs 38760 * 36 (full check) = 1.4M operations.
        # ~40x speedup.
        
        # Implementation:
        # We need a function to get degrees of a subset quickly.
        # adj_np is global.
        
        def is_reg_subset(comb_tuple, exclude_edge_flip=False, flipped_edge=None):
            # This is getting complex to inline.
            # Let's just extract submatrix and sum.
            # It might be slow in Python.
            pass

        # Let's try a different optimization:
        # Precompute the degrees of u and v in the subgraph?
        # No, degrees of OTHER vertices don't change.
        # Only u and v change.
        
        # Let's store the degrees of the 6 vertices for the "current" state?
        # No, the graph changes.
        
        # Okay, let's just compute.
        
        for idx in idxs:
            comb = combs[idx]
            # Get degrees of all 6 vertices in this induced subgraph BEFORE flip
            # Note: adj_np is currently in the state BEFORE the flip (we haven't applied it yet? 
            # Wait, in the loop I did `adj_np[u,v] ^= 1` BEFORE calling this?
            # Let's structure the loop in local_search to compute delta BEFORE flipping.
            # Then if accepted, flip.
            
            # So `adj_np` is in "before" state.
            
            # Get degrees for comb
            # degs = np.array([np.sum(adj_np[v, comb]) for v in comb]) -- Wrong, this includes edges to outside.
            # Must be induced subgraph degrees.
            # degs[i] = sum(adj_np[comb[i], comb[j]] for j != i)
            
            # This is O(k^2).
            # Let's try to vectorize the check for ALL idxs at once?
            # Hard because they are different subsets.
            
            # Let's just do it simply first and see if it's fast enough.
            # 3000 subsets * 21^2 ops? No.
            # 
            # Optimization: 
            # We only care if the subset is regular.
            # 
            # Let's try to implement the delta check.
            pass

    # Let's stick to a robust, if slightly slower, approach first: 
    # Full re-evaluation is slow. 
    # Let's try to use the property: 
    # "If a subset is regular and contains the edge, flipping makes it non-regular."
    # "If a subset is non-regular, flipping might make it regular."
    
    # We can precompute for each edge (u,v) the list of subsets containing it.
    # And for each subset, we can check if it's currently regular.
    
    # Let's build a bitset of "currently bad" subsets.
    # bad_mask is a boolean array of size C(n,6).
    
    def local_search_v2(n, max_iters=2000, combs=None, edge_to_combs=None):
        if combs is None:
            combs, edge_to_combs = prepare_search_data(n)
        
        adj_np = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_np[i, j] = 1
                    adj_np[j, i] = 1
        
        # Initial bad mask
        bad_mask = np.zeros(len(combs), dtype=bool)
        # Compute initial bads
        # This is slow, but done once.
        for idx, comb in enumerate(combs):
            indices = np.array(comb)
            sub_adj = adj_np[np.ix_(indices, indices)]
            degs = sub_adj.sum(axis=1)
            if np.all(degs == degs[0]):
                bad_mask[idx] = True
        
        current_bad_count = np.sum(bad_mask)
        if current_bad_count == 0:
            if check_validity_fast(n, adj_np, combs):
                return adj_np

        best_adj = adj_np.copy()
        best_count = current_bad_count
        
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            u, v = random.choice(edges)
            idxs = edge_to_combs.get((u, v), [])
            if not idxs:
                continue
                
            # Calculate delta
            # We need to check each idx in idxs.
            # 1. If it was bad (regular), it becomes good. (Delta -1)
            # 2. If it was good (not regular), check if it becomes bad. (Delta +1 if yes)
            
            delta = 0
            
            # Optimize: 
            # We need degrees of u and v in the induced subgraph for each idx.
            # And degrees of the other 4 vertices.
            
            # Let's extract the other 4 vertices.
            # And compute their degrees.
            
            # This is still heavy.
            # Let's try to just re-evaluate the whole thing every 10 steps?
            # No, that defeats the purpose.
            
            # Let's try a heuristic: 
            # Just flip and recompute score fully every K steps?
            # No.
            
            # Let's try the delta calculation with numpy slicing if possible.
            # For a specific comb, we can get the degrees.
            
            # To speed this up, we can precompute the "neighbors" in the comb?
            
            # Let's try to be simple:
            # For each idx in idxs:
            #   comb = combs[idx]
            #   # Get degrees BEFORE
            #   # We can compute degs_before for all 6 vertices.
            #   # Then check if regular.
            #   # If regular, delta -= 1.
            #   # Else, compute degs_after (change u,v) and check if regular.
            #
            # To compute degs:
            #   degs[i] = sum(adj_np[comb[i], comb[j]] for j != i)
            
            # This inner loop is the bottleneck.
            # Let's hope the number of idxs is small enough or we can vectorize.
            #
            # Vectorization idea:
            # Stack all idxs? No, they are different vertices.
            
            # Let's just do it in Python and hope it's fast enough for 2000 iterations.
            # 2000 * 3000 * (small ops) = 6M * small.
            # Maybe 5-10 seconds. Acceptable.
            
            for idx in idxs:
                comb = combs[idx]
                # Check if currently bad
                if bad_mask[idx]:
                    # It was regular. Flipping edge makes it non-regular.
                    delta -= 1
                    # Update mask? We can update mask after accepting.
                    # For now, just count.
                else:
                    # It was NOT regular. Check if it becomes regular.
                    # Compute degrees BEFORE
                    # We need degrees of u and v in the induced subgraph.
                    # And check if after flipping they are all equal.
                    
                    # Get the 6 vertices
                    # degs_u = sum(adj_np[u, x] for x in comb if x != u)
                    # Note: x are in comb.
                    
                    # Let's compute all 6 degrees.
                    # This is O(36) per subset.
                    degs = np.zeros(6, dtype=np.int8)
                    for i in range(6):
                        ui = comb[i]
                        # Sum edges to other 5
                        s = 0
                        for j in range(6):
                            if i != j:
                                s += adj_np[ui, comb[j]]
                        degs[i] = s
                    
                    # Apply flip to degs
                    # Find indices of u and v in comb
                    # comb is a tuple of 6.
                    # u and v are in comb.
                    # 
                    # If we ADD edge (u,v):
                    #   degs_u += 1, degs_v += 1
                    # If we REMOVE edge (u,v):
                    #   degs_u -= 1, degs_v -= 1
                    
                    # Current state: adj_np[u,v] is the state BEFORE flip.
                    # If adj_np[u,v] == 0, we are ADDING.
                    # If adj_np[u,v] == 1, we are REMOVING.
                    
                    if adj_np[u, v] == 0:
                        # Add
                        # Find idx of u and v in comb
                        # Since u, v are in comb, we can just loop
                        for i in range(6):
                            if comb[i] == u: degs[i] += 1
                            if comb[i] == v: degs[i] += 1
                    else:
                        # Remove
                        for i in range(6):
                            if comb[i] == u: degs[i] -= 1
                            if comb[i] == v: degs[i] -= 1
                    
                    # Check if new degs are all equal
                    if np.all(degs == degs[0]):
                        delta += 1

            new_count = current_bad_count + delta
            
            if new_count <= current_bad_count:
                # Accept
                # Flip edge
                adj_np[u, v] ^= 1
                adj_np[v, u] ^= 1
                
                # Update bad_mask
                for idx in idxs:
                    if bad_mask[idx]:
                        bad_mask[idx] = False
                    else:
                        # Need to check if it became bad
                        # We already checked this in the delta calc, but didn't store result.
                        # Re-check? Or store in a temp list?
                        # Let's re-check to be safe. It's cheap.
                        comb = combs[idx]
                        indices = np.array(comb)
                        sub_adj = adj_np[np.ix_(indices, indices)]
                        degs = sub_adj.sum(axis=1)
                        if np.all(degs == degs[0]):
                            bad_mask[idx] = True
                
                current_bad_count = new_count
                
                if current_bad_count == 0:
                    if check_validity_fast(n, adj_np, combs):
                        return adj_np
                
                if current_bad_count < best_count:
                    best_count = current_bad_count
                    best_adj = adj_np.copy()
                    
        return best_adj if best_count == 0 and check_validity_fast(n, best_adj, combs) else None

    # Try n=21
    target_n = 21
    combs_21, edge_to_combs_21 = prepare_search_data(target_n)
    for _ in range(5):
        if time.time() > start_time + 40:
            break
        adj = local_search_v2(target_n, max_iters=500, combs=combs_21, edge_to_combs=edge_to_combs_21)
        if adj is not None:
            return {'n': target_n, 'edges': [[u, v] for u in range(target_n) for v in range(u + 1, target_n) if adj[u, v] == 1]}

    # Try n=20
    target_n = 20
    combs_20, edge_to_combs_20 = prepare_search_data(target_n)
    for _ in range(10):
        if time.time() > start_time + 80:
            break
        adj = local_search_v2(target_n, max_iters=1000, combs=combs_20, edge_to_combs=edge_to_combs_20)
        if adj is not None:
            return {'n': target_n, 'edges': [[u, v] for u in range(target_n) for v in range(u + 1, target_n) if adj[u, v] == 1]}

    # Fallback n=15
    target_n = 15
    combs_15, edge_to_combs_15 = prepare_search_data(target_n)
    for _ in range(5):
        if time.time() > start_time + 95:
            break
        adj = local_search_v2(target_n, max_iters=500, combs=combs_15, edge_to_combs=edge_to_combs_15)
        if adj is not None:
            return {'n': target_n, 'edges': [[u, v] for u in range(target_n) for v in range(u + 1, target_n) if adj[u, v] == 1]}

    # Ultimate fallback: Empty graph on 5 vertices (trivially valid)
    return {'n': 5, 'edges': []}
