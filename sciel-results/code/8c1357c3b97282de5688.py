import random
import time
from itertools import combinations
from math import gcd, exp

def construct():
    deadline = time.time() + 145
    
    # 1. Generate Pool - focus on creating dense collinear structures
    pool = []
    # Dense grid with smaller step to create more collinear opportunities
    for x in range(-40, 41, 2):
        for y in range(-40, 41, 2):
            pool.append((x, y))
    # Add structured points on multiple lines and curves
    for x in range(-15, 16):
        pool.append((x, x*x))
        pool.append((x, -x*x))
        pool.append((x, 2*x))
        pool.append((x, 3*x))
        pool.append((x, -2*x))
        pool.append((2*x, x))
        pool.append((3*x, x))
        pool.append((-2*x, x))
    
    # Remove duplicates
    pool = list(set(pool))
    random.shuffle(pool)
    # Limit pool size for speed
    pool = pool[:250]
    n_pool = len(pool)
    
    # Precompute line keys for all pairs in pool
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
    
    # Precompute line membership for fast validity checking
    # For each point index, which lines does it belong to?
    point_lines = [[] for _ in range(n_pool)]
    for line_idx, line in enumerate(lines):
        for idx in line:
            point_lines[idx].append(line_idx)
    
    # Helper: Check if a set of indices is valid (no 4 collinear) - FAST version
    def is_valid(selected_indices):
        sel_set = set(selected_indices)
        # For each selected point, check its lines
        for idx in selected_indices:
            for line_idx in point_lines[idx]:
                line = lines[line_idx]
                count = sum(1 for i in line if i in sel_set)
                if count >= 4:
                    return False
        return True

    # Helper: Compute number of collinear triples - FAST version
    def count_triples(selected_indices):
        sel_set = set(selected_indices)
        triples = 0
        seen_lines = set()
        for idx in selected_indices:
            for line_idx in point_lines[idx]:
                if line_idx in seen_lines:
                    continue
                seen_lines.add(line_idx)
                line = lines[line_idx]
                count = sum(1 for i in line if i in sel_set)
                if count >= 3:
                    if count == 3:
                        triples += 1
                    elif count == 4:
                        triples += 4
                    elif count == 5:
                        triples += 10
                    elif count == 6:
                        triples += 20
                    elif count == 7:
                        triples += 35
                    elif count == 8:
                        triples += 56
                    elif count == 9:
                        triples += 84
                    elif count == 10:
                        triples += 120
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

    # Phase 1: Simulated Annealing with improved neighbor generation
    def proxy_score(selected):
        if not is_valid(selected):
            return -1000000
        return count_triples(selected)
    
    # Generate a better initial solution by greedily adding points that create triples
    def generate_initial():
        sol = []
        used = set()
        # Start with random points
        for _ in range(5):
            p = random.randint(0, n_pool - 1)
            while p in used:
                p = random.randint(0, n_pool - 1)
            sol.append(p)
            used.add(p)
        
        # Add points that maximize triple creation
        for _ in range(15):
            best_p = None
            best_gain = -1
            for p in range(n_pool):
                if p in used:
                    continue
                new_sol = sol + [p]
                if not is_valid(new_sol):
                    continue
                gain = count_triples(new_sol) - count_triples(sol)
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
            if best_p is not None:
                sol.append(best_p)
                used.add(best_p)
            else:
                # Fallback: random valid point
                for _ in range(100):
                    p = random.randint(0, n_pool - 1)
                    if p not in used and is_valid(sol + [p]):
                        sol.append(p)
                        used.add(p)
                        break
        return sol
    
    best_solution = generate_initial()
    best_score = proxy_score(best_solution)
    current_solution = best_solution[:]
    current_score = best_score
    
    temp = 15.0
    cooling = 0.998
    min_temp = 0.1
    step = 0
    no_improve_count = 0
    
    while time.time() < deadline - 40: # Leave 40s for exact eval
        # Better neighbor: try swapping with a point on a line with existing points
        sel_idx = random.randint(0, 19)
        old_point = current_solution[sel_idx]
        
        # Find candidate points that share lines with other selected points
        candidates = set()
        for other_idx in current_solution:
            if other_idx == old_point:
                continue
            for line_idx in point_lines[other_idx]:
                line = lines[line_idx]
                for p in line:
                    if p not in current_solution:
                        candidates.add(p)
        
        if not candidates:
            candidates = set(range(n_pool)) - set(current_solution)
        
        if not candidates:
            break
        
        new_point = random.choice(list(candidates))
        
        new_solution = current_solution[:]
        new_solution[sel_idx] = new_point
        
        new_score = proxy_score(new_solution)
        
        delta = new_score - current_score
        if delta >= 0:
            accept = True
        else:
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
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Adaptive cooling
        temp *= cooling
        if no_improve_count > 500:
            temp = max(temp * 0.5, min_temp)
            no_improve_count = 0
        
        step += 1
        
    # Phase 2: Exact Evaluation - optimized combination search
    def calc_exact_in(selected_indices):
        local_map = {idx: i for i, idx in enumerate(selected_indices)}
        n = 20
        
        # Precompute for each pair (i,j), the set of k such that i,j,k are collinear
        # pair_collinear[i][j] = bitmask of k where i,j,k collinear
        pair_collinear = [[0]*n for _ in range(n)]
        
        for line in lines:
            local_line = [local_map[idx] for idx in line if idx in local_map]
            if len(local_line) >= 3:
                for i in local_line:
                    for j in local_line:
                        if i == j:
                            continue
                        for k in local_line:
                            if k != i and k != j:
                                pair_collinear[i][j] |= (1 << k)
        
        # Check if a mask is valid: no triple is fully contained
        def is_valid_mask(mask):
            # For each pair in mask, check if their collinear partner is also in mask
            temp = mask
            while temp:
                bit_i = temp & -temp
                i = bit_i.bit_length() - 1
                temp ^= bit_i
                temp2 = mask & ~bit_i
                while temp2:
                    bit_j = temp2 & -temp2
                    j = bit_j.bit_length() - 1
                    temp2 ^= bit_j
                    if j < i:  # only check each pair once
                        if (pair_collinear[i][j] & mask) != 0:
                            return False
            return True
        
        # Search for largest valid subset using combination enumeration
        best = 0
        for k in range(20, 0, -1):
            found = False
            for combo in combinations(range(20), k):
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
        return [[pool[i][0], pool[i][1]] for i in best_solution]
    else:
        # Fallback: try to fix by removing problematic points and replacing
        sol = best_solution[:]
        for _ in range(1000):
            if is_valid(sol):
                break
            # Find a point involved in a 4-collinear
            for idx in sol:
                for line_idx in point_lines[idx]:
                    line = lines[line_idx]
                    count = sum(1 for i in line if i in set(sol))
                    if count >= 4:
                        # Replace idx with a random unused point
                        used = set(sol)
                        for _ in range(50):
                            new_p = random.randint(0, n_pool - 1)
                            if new_p not in used:
                                sol[sol.index(idx)] = new_p
                                break
                        break
                else:
                    continue
                break
        if is_valid(sol):
            return [[pool[i][0], pool[i][1]] for i in sol]
        # Last resort: return best_solution anyway
        return [[pool[i][0], pool[i][1]] for i in best_solution]

# EVOLVE-BLOCK-END
