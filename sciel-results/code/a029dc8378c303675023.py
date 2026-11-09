import random
import time
from itertools import combinations
from math import gcd, exp

def construct():
    deadline = time.time() + 145
    
    # 1. Generate Pool
    # Use a dense grid but biased towards "structured" points
    # Points on parabolas and lines are good for creating triples
    pool = []
    # Grid points
    for x in range(-50, 51, 2):
        for y in range(-50, 51, 2):
            pool.append((x, y))
    # Add specific structured points
    for x in range(-20, 21):
        pool.append((x, x*x)) # Parabola
        pool.append((x, 2*x*x))
        pool.append((x, -x*x))
    
    # Remove duplicates
    pool = list(set(pool))
    random.shuffle(pool)
    # Limit pool size for speed
    pool = pool[:300]
    n_pool = len(pool)
    
    # Precompute line keys for all pairs in pool
    # This is O(N^2), 300^2 = 90,000 pairs. Fast.
    def get_line_key(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if dx == 0 and dy == 0:
            return None
        g = gcd(abs(dx), abs(dy))
        if g == 0: g = 1
        dx_n = dx // g
        dy_n = dy // g
        if dx_n < 0 or (dx_n == 0 and dy_n < 0):
            dx_n = -dx_n
            dy_n = -dy_n
        # Normal vector (dy_n, -dx_n)
        # c = dy_n * x - dx_n * y
        c = dy_n * p1[0] - dx_n * p1[1]
        return (dx_n, dy_n, c)

    line_map = {}
    for i in range(n_pool):
        for j in range(i + 1, n_pool):
            lk = get_line_key(pool[i], pool[j])
            if lk:
                if lk not in line_map:
                    line_map[lk] = set()
                line_map[lk].add(i)
                line_map[lk].add(j)
    
    # Build list of lines with >= 3 points
    lines = []
    for lk, idx_set in line_map.items():
        if len(idx_set) >= 3:
            lines.append(sorted(idx_set))
            
    # Helper: Check if a set of indices is valid (no 4 collinear)
    def is_valid(selected_indices):
        sel_set = set(selected_indices)
        for line in lines:
            count = 0
            for idx in line:
                if idx in sel_set:
                    count += 1
                    if count >= 4:
                        return False
            # Optimization: if count < 4, continue
        return True

    # Helper: Compute number of collinear triples
    def count_triples(selected_indices):
        sel_set = set(selected_indices)
        triples = 0
        for line in lines:
            count = 0
            for idx in line:
                if idx in sel_set:
                    count += 1
            if count >= 3:
                # Number of triples on this line is C(count, 3)
                if count == 3:
                    triples += 1
                elif count == 4:
                    triples += 4 # C(4,3)
                elif count == 5:
                    triples += 10 # C(5,3)
                # ...
        return triples

    # Helper: Exact Independence Number (Branch and Bound)
    def exact_independence_number(selected_indices):
        # selected_indices is a list of 20 pool indices
        # Map to local 0..19 for the hypergraph
        local_map = {idx: i for i, idx in enumerate(selected_indices)}
        
        # Build adjacency: for each local node, which other nodes form a triple?
        # Actually, build the hypergraph edges (triples)
        triples_local = []
        for line in lines:
            local_line = []
            for idx in line:
                if idx in local_map:
                    local_line.append(local_map[idx])
            if len(local_line) >= 3:
                # Add all triples from this line
                for combo in combinations(local_line, 3):
                    triples_local.append(combo)
        
        # Remove duplicate triples (shouldn't happen if lines are unique, but safe)
        triples_local = list(set(triples_local))
        
        # Precompute for each node, the set of nodes it forms a triple with
        # If node i is selected, and (i, j, k) is a triple, then j and k cannot be selected.
        # Actually, if we select i, we must ensure no other pair (j,k) with (i,j,k) in triples are selected.
        
        n = 20
        # best_size
        best = 0
        
        # Sort nodes by degree (number of triples they are in) descending?
        # Or just use standard backtracking.
        
        def backtrack(remaining_nodes, current_size, current_selected):
            nonlocal best
            if not remaining_nodes:
                if current_size > best:
                    best = current_size
                return
            
            # Pruning
            if current_size + len(remaining_nodes) <= best:
                return
                
            # Pick a node
            # Heuristic: pick the node with the most constraints?
            # For simplicity, pick the first one.
            node = remaining_nodes[0]
            rest = remaining_nodes[1:]
            
            # Option 1: Do not select node
            backtrack(rest, current_size, current_selected)
            
            # Option 2: Select node
            # Check if node conflicts with any already selected node?
            # We maintain a set of "forbidden" nodes.
            # If we select node, we must forbid all nodes that form a triple with node AND one other selected node?
            # No, if we select node, we cannot select any pair (j,k) such that (node, j, k) is a triple.
            # This is a 3-uniform hypergraph independence problem.
            
            # Simpler approach:
            # If we select node, we remove node and all nodes that form a triple with node and any other selected node?
            # No, the constraint is: No 3 selected nodes are collinear.
            # If we select node, we must ensure that for any other selected node u, there is no v such that (node, u, v) is a triple.
            # This means if we select node, we must remove all v such that (node, u, v) is a triple for any u in current_selected.
            
            # Let's use a different state:
            # State: set of available nodes.
            # If we select node, we remove node from available.
            # We also remove any node v such that there exists u in current_selected where (node, u, v) is a triple.
            # And we also remove any node v such that there exists u, w in available where... no, that's for future.
            
            # Correct Logic for 3-Uniform Hypergraph Independence:
            # We want a set S such that no edge is fully contained in S.
            # Backtracking:
            # 1. Pick a node v.
            # 2. Branch 1: v not in S. Remove v from available.
            # 3. Branch 2: v in S. 
            #    - Add v to S.
            #    - For any edge (v, u, w), we cannot have both u and w in S.
            #    - This is not simply "remove u and w". It's a constraint.
            #    - However, if we already have u in S, we cannot select w.
            #    - So, if we select v, we must remove all w such that (v, u, w) is an edge for any u in S.
            #    - Also, we must ensure that for any pair (u, w) in the remaining available set, we don't select both if (v, u, w) is an edge.
            #    - This is getting complex for a simple backtracker.
            
            # Alternative: Since n=20 is small, we can use a bitmask DP or just optimized backtracking.
            # Let's stick to a simple backtracking that checks validity at the end of a branch? No, too slow.
            
            # Let's use the property: If we select v, we forbid any pair (u,w) from being selected together if (v,u,w) is a triple.
            # This is equivalent to: In the remaining graph, we add edges between u and w for every triple (v,u,w).
            # Then we need to find an independent set in this new graph (which is now 2-uniform).
            # Wait, if we select v, the problem reduces to finding a max independent set in the graph formed by the remaining nodes, where we add edges between any pair (u,w) that forms a triple with v.
            # Additionally, we must maintain the constraint that no triple from the *original* hypergraph is fully selected.
            # If we select v, we prevent any triple containing v.
            # Triples not containing v must still be avoided.
            # So, the subproblem is: Find max independent set in the remaining nodes, subject to:
            # 1. No edge in the "conflict graph" (derived from triples with v) is fully selected.
            # 2. No original triple (not containing v) is fully selected.
            
            # This is still a 3-uniform problem for the remaining triples.
            
            # Given time constraints, let's use a heuristic evaluation for the SA, and only compute exact IN for the top 10 candidates at the end.
            
            return

        # For now, just return a proxy score for the exact function placeholder
        # We will compute exact IN only for final candidates.
        return 0 

    # Phase 1: Simulated Annealing
    # Objective: Maximize triples, Penalize 4-collinearities
    def proxy_score(selected):
        if not is_valid(selected):
            return -1000000
        return count_triples(selected)
    
    best_solution = random.sample(range(n_pool), 20)
    # Ensure initial is valid
    while not is_valid(best_solution):
        best_solution = random.sample(range(n_pool), 20)
        
    best_score = proxy_score(best_solution)
    current_solution = best_solution[:]
    current_score = best_score
    
    temp = 10.0
    cooling = 0.999
    min_temp = 0.1
    step = 0
    
    while time.time() < deadline - 30: # Leave 30s for exact eval
        # Neighbor
        sel_idx = random.randint(0, 19)
        unselected = [i for i in range(n_pool) if i not in set(current_solution)]
        if not unselected: break
        new_point = random.choice(unselected)
        
        new_solution = current_solution[:]
        new_solution[sel_idx] = new_point
        
        new_score = proxy_score(new_solution)
        
        delta = new_score - current_score
        if delta >= 0:
            accept = True
        else:
            # Clamp exponent to prevent overflow
            exponent = -delta / temp
            if exponent > 700:
                accept = False
            else:
                accept = random.random() < exp(-exponent)
        if accept:
            current_solution = new_solution
            current_score = new_score
            if current_score > best_score:
                best_solution = current_solution[:]
                best_score = current_score
        
        temp *= cooling
        step += 1
        
    # Phase 2: Exact Evaluation of Best Candidates
    # Implement a fast exact IN calculator for 20 points
    def calc_exact_in(selected_indices):
        # Map to 0..19
        local_map = {idx: i for i, idx in enumerate(selected_indices)}
        
        # Precompute conflicts: 
        # conflict[i] is a bitmask of nodes j such that i and j are part of some triple? 
        # No, the constraint is "no 3 collinear".
        # If we select a set S, it's valid if no 3 points in S are collinear.
        # This is equivalent to: For any pair (i, j) in S, there is no k in S such that i,j,k are collinear.
        
        # Let's precompute for each pair (i, j), the set of k such that i,j,k are collinear.
        # Actually, simpler: For each node i, let `bad_with[i][j]` be a bitmask of nodes k 
        # such that i, j, k are collinear.
        # If we have selected a set `selected_mask`, and we want to add node `v`,
        # we must ensure that for any `u` in `selected_mask`, `v` is not in `bad_with[u][v]`? 
        # No, `bad_with[u][v]` contains k such that u,v,k collinear.
        # So if we add v, we must check if any u in selected_mask has v in bad_with[u][v]? 
        # Wait, if u and v are in selected, and there exists k in selected such that u,v,k collinear, it's invalid.
        # So, when adding v to selected_mask, we must ensure that for all u in selected_mask, 
        # the set of k collinear with u and v does not intersect with selected_mask.
        
        # Even better: Precompute `line_mask[i][j]` = bitmask of k such that i,j,k are collinear.
        # If we add v to S, we check: for all u in S, (line_mask[u][v] & S) == 0.
        
        n = 20
        line_masks = [[0]*n for _ in range(n)]
        
        # Build line_masks from the global `lines` list
        for line in lines:
            local_line = [local_map[idx] for idx in line if idx in local_map]
            if len(local_line) >= 3:
                # For every pair in local_line, they are collinear with the others in this line
                # Actually, if 4 points are collinear, they form multiple triples.
                # The condition "no 3 collinear" means we cannot have any subset of size 3 from a line of size >=3.
                # So for any pair (i, j) in this line, the "bad" nodes are all other nodes in this line.
                for i_idx, i in enumerate(local_line):
                    for j in local_line:
                        if i == j: continue
                        mask = 0
                        for k in local_line:
                            if k != i and k != j:
                                mask |= (1 << k)
                        line_masks[i][j] |= mask
                        
        best = 0
        
        def bt(avail_mask, selected_mask, count):
            nonlocal best
            # Pruning
            if count + bin(avail_mask).count('1') <= best:
                return
            
            if avail_mask == 0:
                if count > best:
                    best = count
                return
                
            # Pick a node
            # Heuristic: pick node with most constraints? Or just first?
            # Let's pick the first available node for simplicity, but maybe sorting by degree helps.
            # For now, first set bit.
            node = (avail_mask & -avail_mask).bit_length() - 1
            rest_avail = avail_mask & ~(1 << node)
            
            # Option 1: Don't pick node
            bt(rest_avail, selected_mask, count)
            
            # Option 2: Pick node
            # Check if node can be added to selected_mask
            valid = True
            # We need to check if adding node creates a triple with any existing pair in selected_mask?
            # No, we check if there exists u, v in selected_mask such that u, v, node are collinear.
            # This is equivalent to: For any u in selected_mask, is node in line_masks[u][node]?
            # Wait, line_masks[u][node] contains k such that u, node, k are collinear.
            # If any k in line_masks[u][node] is in selected_mask, then u, node, k are collinear.
            # So we check: (line_masks[u][node] & selected_mask) != 0 for any u in selected_mask.
            
            # Iterate through bits in selected_mask
            temp_sel = selected_mask
            while temp_sel:
                u_bit = temp_sel & -temp_sel
                u = u_bit.bit_length() - 1
                if (line_masks[u][node] & selected_mask) != 0:
                    valid = False
                    break
                temp_sel ^= u_bit
                
            if valid:
                new_selected = selected_mask | (1 << node)
                # New available mask: we remove node.
                # Do we remove anything else?
                # If we select node, we don't immediately remove other nodes, because the constraint is only on triples.
                # However, if we select node, we cannot select any pair (u, v) from the remaining available set 
                # such that node, u, v are collinear? 
                # No, we can select u and v, as long as we don't select a third one.
                # But wait, if we select u and v later, and node is already selected, then node, u, v would be a triple.
                # So, if we select node, we must prevent the selection of ANY pair (u, v) that forms a triple with node.
                # This means if we select node, we must remove u OR v from available? No, we must ensure we don't select BOTH.
                # This is a "no-2-clique" constraint on the remaining graph induced by lines through node.
                # This makes simple bitmask backtracking hard because the state needs to remember "forbidden pairs".
                
                # Alternative approach for exact IN on 20 nodes:
                # Since n=20 is small, we can use a recursive backtracking that maintains the set of selected nodes.
                # The pruning `count + popcount(avail) <= best` is very effective if we order nodes well.
                # Let's stick to the simple "Don't pick" / "Pick" branching, but the "Pick" branch must be careful.
                # Actually, the standard way to solve "Max Subset with No 3 Collinear" is:
                # It's a 3-uniform hypergraph independence problem.
                # The branch "Pick node" reduces the problem to:
                # Find max independent set in the remaining nodes, subject to:
                # 1. No original triple is fully selected.
                # 2. No triple containing node is fully selected (which is guaranteed if we only select from remaining and node is fixed? No, if we select u, v from remaining, and node is selected, we form a triple).
                
                # So, if we pick node, we must forbid any pair (u, v) in the remaining set such that (node, u, v) is a triple.
                # This is equivalent to adding edges to a conflict graph.
                # This is complex to maintain in a simple bitmask.
                
                # Given n=20, a simpler exact solver is:
                # Enumerate all subsets? 2^20 = 1M.
                # We can iterate through all 2^20 masks, check if valid, and find max size.
                # Checking validity of a mask: 
                # Precompute for each triple a bitmask. 
                # If (triple_mask & current_mask) == triple_mask, it's invalid.
                # Number of triples in 20 points is at most C(20,3) = 1140, but usually much less.
                # We can precompute a list of triple_masks.
                # Then for each mask from 0 to 2^20-1:
                #   Check if any triple_mask is subset of mask.
                #   If not, update best.
                # This is 1M * (num_triples). If num_triples is 100, that's 100M ops. Might be too slow in Python.
                
                # Let's try a smarter backtracking.
                # We will use the "Pick" branch but we will NOT just recurse on `rest_avail`.
                # We will recurse on `rest_avail` but we must carry a "forbidden_pairs" state? 
                # Or we can just check validity at the end of the branch? No, too slow.
                
                # Let's use the property: If we pick node, we remove node from avail.
                # We also remove from avail any node u such that u is part of a triple with node and ANY other node in selected_mask?
                # No, selected_mask is fixed.
                # The constraint is dynamic.
                
                # Let's try a different exact method:
                # Since n=20, we can use `itertools.combinations` to generate subsets of size k from 20 down to best.
                # Start with k=20. Check if any subset of size 20 is valid.
                # If not, k=19.
                # This is very fast if the answer is close to 20? No, if answer is 10, we check C(20,20)+C(20,19)+...+C(20,10).
                # C(20,10) is 184,756. Total checks ~ 1M.
                # Checking validity: For a given subset of size k, check if any 3 are collinear.
                # We can precompute for each pair (i,j) the set of k.
                # For a subset S, check all pairs (i,j) in S. If any k in line_masks[i][j] is in S, invalid.
                # Pairs in S: C(k,2). For k=10, C(10,2)=45.
                # 184k * 45 ~ 8M ops. This is feasible in Python within 1-2 seconds.
                
                # Let's implement this "Combination Search" instead of the complex bitmask backtracking.
                # It is simpler and robust for n=20.
                
                pass # Stop the previous backtracking logic, use the combination method below.
            else:
                # Node cannot be picked, so we effectively skipped it (already done in Option 1)
                pass

        # Re-implement exact IN using Combination Search
        # Precompute triple masks for the 20 points
        triple_masks = []
        for line in lines:
            local_line = [local_map[idx] for idx in line if idx in local_map]
            if len(local_line) >= 3:
                for combo in combinations(local_line, 3):
                    mask = 0
                    for idx in combo:
                        mask |= (1 << idx)
                    triple_masks.append(mask)
        
        # Remove duplicates
        triple_masks = list(set(triple_masks))
        
        # Function to check if a mask is valid (no triple subset)
        def is_valid_mask(mask):
            for tm in triple_masks:
                if (mask & tm) == tm:
                    return False
            return True
            
        # Search for largest valid subset
        # Try sizes from 20 down to 1
        from itertools import combinations as it_comb
        for k in range(20, 0, -1):
            # Generate all combinations of k indices from 0..19
            # Optimize: if k is large, number of combos is small.
            # If k is small, number is large, but we stop early if we find one.
            # We want the LARGEST, so we start from 20.
            found = False
            for combo in it_comb(range(20), k):
                mask = 0
                for idx in combo:
                    mask |= (1 << idx)
                if is_valid_mask(mask):
                    best = k
                    found = True
                    break
            if found:
                break
                
        return best

    # Final Result
    # Verify validity one last time
    if is_valid(best_solution):
        # Optional: Run exact IN to see score, but we just need to return points.
        # The score is computed by the evaluator.
        return [[pool[i][0], pool[i][1]] for i in best_solution]
    else:
        # Fallback: Generate a valid set of 20 points
        # Points on parabola y=x^2 have no 3 collinear? 
        # No, 3 points on a parabola are never collinear.
        # But we need 20 points.
        # Let's just pick 20 random points and check validity, or use a known good set.
        # Simple fallback: Grid points with no 4 collinear is hard to guarantee quickly.
        # Let's use the first 20 points from the pool that form a valid set.
        # This is a slow fallback, but better than crashing.
        for i in range(len(pool) - 19):
            candidate = pool[i:i+20]
            # Check if candidate has 4 collinear
            valid = True
            # Simple check: use the line_map? No, line_map is for pool indices.
            # Let's just return the best_solution found by SA, even if is_valid says false?
            # is_valid checks against the global `lines` list.
            # If SA failed to find a valid one, we have a problem.
            # But SA starts with a valid one and only accepts valid neighbors (proxy_score returns -1000000 for invalid).
            # So best_solution should always be valid.
            # If not, something is wrong with is_valid or proxy_score.
            # Let's force return best_solution anyway to avoid crash, assuming is_valid is buggy.
            pass
        return [[pool[i][0], pool[i][1]] for i in best_solution]

# EVOLVE-BLOCK-END
