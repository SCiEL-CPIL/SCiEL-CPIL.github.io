import time
from math import lcm
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    N = 256
    values = list(range(2, N + 1))
    n = len(values)
    
    # Precompute conflict: conflict[i][j] = True if lcm(values[i], values[j]) <= 256
    # Use 0-indexed: values[0]=2, values[1]=3, ..., values[254]=256
    # We represent the graph using bitmasks for fast intersection checks.
    # Since n=255, we can use Python integers as bitsets.
    
    # 1. Build conflict bitsets
    # conflict_mask[i] has bit j set if i conflicts with j
    conflict_mask = [0] * n
    for i in range(n):
        vi = values[i]
        for j in range(i + 1, n):
            vj = values[j]
            if lcm(vi, vj) <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
    
    weights = [1.0 / values[i] for i in range(n)]
    
    # Global best solution tracking
    best_score = 0.0
    best_set_mask = 0
    
    # Helper to update global best
    def update_best(mask, score):
        nonlocal best_score, best_set_mask
        if score > best_score:
            best_score = score
            best_set_mask = mask

    # Branch and Bound Solver
    # State:
    #   candidates: bitmask of vertices that can still be added
    #   current_mask: bitmask of vertices already in the solution
    #   current_score: sum of weights of vertices in current_mask
    #   ub: upper bound on the total score achievable from this branch
    
    # To make the bound tight, we sort vertices by weight (descending) once.
    # However, the recursive structure works better if we process vertices in a fixed order 
    # or use a dynamic ordering. 
    # A common efficient B&B for MWIS:
    # 1. Order vertices by weight descending.
    # 2. At each step, decide to include or exclude the next vertex.
    # But a more powerful approach for sparse graphs is to pick a vertex, branch on including it 
    # (remove its neighbors from candidates) or excluding it.
    
    # Let's use the "branch on a vertex" approach.
    # To improve pruning, we select a vertex from candidates that has the highest weight 
    # or high degree to split the search space effectively.
    
    # Pre-compute a list of vertices sorted by weight descending for potential use in bounds
    # But for the fractional knapsack bound, we need to consider the remaining candidates 
    # and their weights.
    
    # Since n=255, computing the exact fractional knapsack bound at every node is O(n log n).
    # We can approximate or optimize this.
    
    # Let's implement a recursive B&B.
    # To avoid recursion depth issues (though 255 is small enough for Python if optimized, 
    # the branching factor is the issue), we rely on pruning.
    
    # Optimization: 
    # The conflict graph is sparse for small numbers. 
    # Numbers > 128 have no conflicts with each other (lcm > 256).
    # Actually, if a, b > 128, lcm(a,b) >= min(a,b) > 128? No.
    # lcm(a,b) = a*b/gcd(a,b). If a,b are large, lcm is large.
    # Specifically, if a > 128 and b > 128, lcm(a,b) > 128. 
    # Is it always > 256?
    # If a=129, b=130. lcm = 129*130 = 16770 > 256.
    # If a=128, b=129. lcm = 128*129 > 256.
    # If a=2, b=128. lcm=128 <= 256. Conflict.
    # So small numbers conflict with many, large numbers conflict with few (mostly only with their divisors/multiples which are small).
    
    # Strategy:
    # We can decompose the problem.
    # However, a generic B&B with a good bound should work.
    
    # Bound Calculation:
    # Given a set of candidate vertices C, the maximum weight independent set is at most
    # the solution to the fractional knapsack problem where each item has weight w_i and capacity 1? 
    # No, that's for knapsack. For MWIS, a valid upper bound is simply the sum of all weights in C.
    # This is weak.
    # A better bound: 
    # If we assume we can pick all vertices in C (ignoring conflicts), the sum is the bound.
    # We can improve this by noting that if two vertices conflict, we can pick at most one.
    # A standard bound for MWIS B&B is:
    # UB = current_score + sum(weights of candidates that are not adjacent to each other? No.)
    
    # Let's use a simpler but effective bound:
    # Sort candidates by weight descending.
    # Greedily add them to a "relaxed" solution where we don't check conflicts? 
    # No, that's just the sum.
    
    # A tighter bound for MWIS:
    # Consider the candidates. If we partition them into cliques, we can pick at most 1 from each.
    # Or, use the fractional relaxation of the independent set polytope? Too complex.
    
    # Let's stick to a strong heuristic bound:
    # UB = current_score + sum(w_i for i in candidates)
    # This is very loose.
    
    # Alternative: 
    # Since the graph is sparse, we can use a "dominance" rule or simply rely on the fact 
    # that we explore high-weight vertices first.
    
    # Let's try a different structural approach:
    # The problem is small (255 nodes).
    # We can use a Max-SAT style or simply a well-optimized recursive B&B with the following bound:
    # For the remaining candidates, calculate the sum of weights. 
    # If current_score + sum(weights in candidates) <= best_score, prune.
    
    # To make this effective, we must visit nodes in an order that finds a good solution quickly.
    # We should prioritize including high-weight vertices.
    
    # Algorithm:
    # 1. Start with all vertices as candidates.
    # 2. Pick a vertex v from candidates.
    #    To maximize pruning, pick the vertex with the highest weight?
    #    Or pick a vertex that is part of many conflicts?
    #    Let's pick the vertex with the highest weight among candidates.
    # 3. Branch 1: Include v.
    #    - Add v to solution.
    #    - Remove v and all its neighbors from candidates.
    #    - Recurse.
    # 4. Branch 2: Exclude v.
    #    - Remove v from candidates.
    #    - Recurse.
    
    # To improve the bound, we can use the sum of remaining weights as a quick check.
    
    # Let's implement this.
    
    # To avoid Python recursion limits and overhead, we can use a stack or ensure the depth is manageable.
    # Depth is at most n=255. Python default recursion limit is 1000. So it's safe.
    
    # However, the number of nodes can explode.
    # We need a better bound.
    
    # Improved Bound:
    # The sum of weights of candidates is an upper bound.
    # We can also use: current_score + max_weight_in_candidates * (number of candidates) ? No.
    
    # Let's use a "Lagrangian" or simple sum bound first and see if it prunes enough.
    # Given the sparsity, the sum bound might be loose for the first few levels, 
    # but once we commit to a few high-weight vertices, the candidate set shrinks significantly 
    # (removing neighbors), and the sum drops.
    
    # Let's add a time check.
    
    def solve_bnb(candidates_mask, current_mask, current_score):
        nonlocal best_score, best_set_mask
        
        # Time check
        if time.time() - start_time > timeout:
            return
        
        # Base case: no candidates left
        if candidates_mask == 0:
            if current_score > best_score:
                best_score = current_score
                best_set_mask = current_mask
            return
        
        # Bound: Sum of weights of remaining candidates
        # We need to sum weights of bits set in candidates_mask.
        # To speed this up, we can precompute prefix sums if we iterate in a fixed order,
        # but here the mask changes arbitrarily.
        # We can just iterate through the set bits.
        
        # Optimization: If the sum of remaining weights + current_score <= best_score, prune.
        remaining_sum = 0.0
        temp_mask = candidates_mask
        while temp_mask:
            # Get lowest set bit
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            remaining_sum += weights[idx]
            temp_mask ^= lsb
            
        if current_score + remaining_sum <= best_score:
            return
        
        # Select a vertex to branch on
        # Heuristic: Pick the vertex with the highest weight in candidates
        # This helps find a good solution early (Branch 1)
        
        # Find max weight vertex in candidates
        max_w = -1
        max_idx = -1
        temp_mask = candidates_mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            idx = lsb.bit_length() - 1
            if weights[idx] > max_w:
                max_w = weights[idx]
                max_idx = idx
            temp_mask ^= lsb
            
        if max_idx == -1:
            return
            
        v = max_idx
        
        # Branch 1: Include v
        # New candidates = candidates_mask without v and without neighbors of v
        neighbors_mask = conflict_mask[v]
        new_candidates_include = candidates_mask & ~(1 << v) & ~neighbors_mask
        new_score = current_score + weights[v]
        new_mask = current_mask | (1 << v)
        
        # Recursive call
        solve_bnb(new_candidates_include, new_mask, new_score)
        
        # Check time again
        if time.time() - start_time > timeout:
            return
            
        # Branch 2: Exclude v
        new_candidates_exclude = candidates_mask & ~(1 << v)
        solve_bnb(new_candidates_exclude, current_mask, current_score)

    # Initial call
    # All vertices are candidates
    initial_mask = (1 << n) - 1
    # Note: 1 << 255 is a large integer, Python handles it fine.
    
    # To ensure we find a good initial solution quickly for pruning,
    # we can run a greedy heuristic first.
    
    def greedy_initial():
        mask = initial_mask
        score = 0.0
        sol_mask = 0
        while mask:
            # Find max weight vertex
            max_w = -1
            max_idx = -1
            temp = mask
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                if weights[idx] > max_w:
                    max_w = weights[idx]
                    max_idx = idx
                temp ^= lsb
            
            if max_idx == -1:
                break
                
            # Include it
            sol_mask |= (1 << max_idx)
            score += weights[max_idx]
            
            # Remove it and its neighbors
            mask &= ~(1 << max_idx)
            mask &= ~conflict_mask[max_idx]
            
        return sol_mask, score

    init_mask, init_score = greedy_initial()
    update_best(init_mask, init_score)
    
    # Now run B&B
    # To improve B&B performance, we can order the vertices in the mask 
    # such that we process high weights first? 
    # The B&B above picks the max weight vertex dynamically at each node, which is good.
    
    # However, the "sum of remaining weights" bound is O(n) per node.
    # With n=255, this is fast.
    
    # One issue: The search space might still be too large.
    # Let's add a simple improvement: 
    # If the current node's potential (current_score + remaining_sum) is not much better 
    # than best_score, we might prune earlier? No, that's what the bound does.
    
    # Let's run it.
    solve_bnb(initial_mask, 0, 0.0)
    
    # If time remains, we can try to refine or run local search on the B&B result?
    # B&B is exact, so if it finishes, it's optimal.
    # If it times out, it returns the best found so far, which is a valid independent set.
    
    # Convert best_set_mask back to list of values
    result = []
    m = best_set_mask
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result.append(values[idx])
        m ^= lsb
        
    # Ensure non-empty (greedy always produces non-empty)
    if not result:
        result = [values[0]]
        
    return result

# EVOLVE-BLOCK-END
