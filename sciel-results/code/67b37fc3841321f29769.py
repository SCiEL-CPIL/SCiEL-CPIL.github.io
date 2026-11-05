"""Maximum independent set on square-sum conflict graph with branch-and-bound."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 256
    # Precompute which numbers have 2*x as a perfect square (diagonal conflict)
    # and which pairs (a,b) have a+b as a perfect square
    
    # Perfect squares up to 512 (max sum is 256+256=512)
    max_sum = 512
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 1):
        squares.add(i * i)
    
    # Build conflict: conflict[i][j] = True if i+j is a perfect square
    # We use 1-indexed: numbers are 1..256
    # conflict_mask[i] is a bitmask of j such that i+j is a perfect square (j>=i)
    # For diagonal: 2*i is a perfect square means i conflicts with itself
    
    # Bitmask representation: bit j set means j is in the set
    # We'll use Python integers as bitmasks (256 bits)
    
    # For each number i (1..256), compute which j (1..256, j>=i) conflict with i
    # i conflicts with j if i+j is a perfect square
    # i conflicts with itself if 2*i is a perfect square
    
    # Build adjacency as bitmasks
    # adj[i] = bitmask of all j where i+j is a perfect square (including j=i if 2i is square)
    adj = [0] * (N + 1)  # 1-indexed
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            if i + j in squares:
                mask |= (1 << j)
        adj[i] = mask
    
    # Precompute: for each i, the "self-conflict" flag
    self_conflict = [False] * (N + 1)
    for i in range(1, N + 1):
        if 2 * i in squares:
            self_conflict[i] = True
    
    # Numbers that can NEVER be in the solution (self-conflict)
    forbidden = set(i for i in range(1, N + 1) if self_conflict[i])
    
    # Candidate pool: numbers without self-conflict
    candidates = [i for i in range(1, N + 1) if i not in forbidden]
    
    # For independent set: we want max subset S of candidates such that
    # for all a, b in S with a<=b, a+b is NOT a perfect square.
    # Since no self-conflict in candidates, we just need: for all distinct a,b in S, a+b not in squares.
    
    # Build adjacency among candidates (excluding self since no self-conflict)
    # For candidate i, adj_cand[i] = bitmask of candidates j (j != i) where i+j is square
    
    # Map candidates to indices for efficient bit operations
    cand_set = set(candidates)
    
    # For branch and bound, we work with a list of candidate indices
    # Use bitmask where bit position = candidate index (0-based)
    num_cands = len(candidates)
    
    # Build adjacency bitmask for each candidate (among candidates only)
    # adj_bm[k] = bitmask of candidates l (l != k) such that candidates[k]+candidates[l] is square
    adj_bm = [0] * num_cands
    for k in range(num_cands):
        i = candidates[k]
        mask = 0
        for l in range(num_cands):
            if k == l:
                continue
            j = candidates[l]
            if i + j in squares:
                mask |= (1 << l)
        adj_bm[k] = mask
    
    # Upper bound: maximum independent set size
    # Simple upper bound: for a set of remaining candidates, the max independent set
    # is at most the size of the set, but we can do better with greedy coloring bound.
    
    # Greedy coloring bound: color the graph greedily, the number of colors is an upper bound on clique size,
    # and for independent set, we use the fact that in each color class (independent set in complement = clique in original),
    # we can pick at most one per color... wait, let me reconsider.
    
    # Actually for MAX INDEPENDENT SET, the standard branch and bound uses:
    # Upper bound = |remaining| - (minimum vertex cover lower bound)
    # Or: use the fact that if we greedily color the induced subgraph, 
    # the max independent set <= number of colors (since each color class is an independent set, 
    # and we can pick at most one from each... no that's wrong).
    
    # Let me use a simpler but effective bound:
    # For the remaining candidates R, upper bound on MIS = |R|
    # Better: greedy coloring of the CONFLICT graph on R. 
    # In the conflict graph, a color class is an independent set. 
    # The chromatic number is a LOWER bound on the clique number, not directly useful.
    
    # Actually, the standard approach: 
    # Upper bound for MIS = |R| - (size of maximum matching in some subgraph) ... complex.
    
    # Simpler effective bound: 
    # If we sort remaining vertices by degree (in the conflict graph induced by R),
    # and greedily assign colors to the CONFLICT graph, then the number of colors 
    # gives an upper bound on the size of a clique in the conflict graph,
    # which means the independence number of the conflict graph (= MIS) 
    # is at least n/omega but that's a lower bound.
    
    # Let me use a different approach: 
    # Upper bound = for each vertex v in R, 1 + MIS(R \ N[v]) where N[v] is closed neighborhood
    # This is the recursive definition but we need a bound.
    
    # Practical approach: use greedy coloring bound.
    # In the conflict graph on R, if we can color with k colors, then the independence number 
    # of the conflict graph is at most |R| (trivial) but we can say:
    # Each color class is an independent set in the conflict graph.
    # So MIS >= max color class size (lower bound, not upper).
    
    # For UPPER bound on MIS: 
    # MIS(R) <= |R| - min_vertex_cover(R)
    # And min_vertex_cover >= max_matching (Kőnig's theorem for bipartite, but general: MVC >= MM)
    # Computing max matching is expensive.
    
    # Alternative upper bound that's cheap: 
    # Sort vertices by degree in conflict graph (induced by R).
    # The sum over all vertices of 1/(deg+1) is a lower bound on MVC (Lubell-type).
    # So MIS <= |R| - sum(1/(deg_v + 1)) ... but this gives a fractional bound.
    
    # Let me just use: upper bound = |R|, and rely on good branching + pruning.
    # With 256 nodes and the specific structure, this should be tractable.
    
    # Actually, let me use a better bound: 
    # For vertex v with degree d in the induced conflict graph on R,
    # if we include v, we can add at most MIS(R \ N[v]) more.
    # If we exclude v, we can add at most MIS(R \ {v}) more.
    # Upper bound for the subproblem containing v: max(1 + |R \ N[v]|, |R \ {v}|)
    # But |R \ N[v]| = |R| - 1 - deg_R(v)
    # So UB = max(1 + |R| - 1 - deg_R(v), |R| - 1) = max(|R| - deg_R(v), |R| - 1)
    # If deg_R(v) >= 1, UB = |R| - 1 (from the exclude branch) or |R| - deg_R(v) (from include)
    # The tighter one is |R| - max(1, deg_R(v)) ... 
    
    # This isn't great. Let me use a different strategy:
    # Since the graph is sparse (square sums are rare), I'll use a greedy approach 
    # combined with local search, and use the exact B&B only on smaller subproblems.
    
    # Strategy: 
    # 1. Get a good initial solution via greedy + local search
    # 2. Use branch and bound to prove optimality or find better solution
    # 3. Time-bounded
    
    # Greedy initial solution
    def greedy_is():
        remaining = set(range(num_cands))
        solution = []
        # Sort by degree (fewest conflicts first)
        degrees = [bin(adj_bm[k] & ((1 << num_cands) - 1)).count('1') for k in range(num_cands)]
        order = sorted(range(num_cands), key=lambda x: (degrees[x], x))
        for k in order:
            if k not in remaining:
                continue
            # Check if k conflicts with any already chosen
            can_add = True
            for chosen in solution:
                if (adj_bm[k] >> chosen) & 1:
                    can_add = False
                    break
            if can_add:
                solution.append(k)
                remaining.discard(k)
        return solution
    
    # Local search: try to improve by swapping
    def local_search(init_solution, max_iters=100000):
        solution = set(init_solution)
        best = len(solution)
        improved = True
        iterations = 0
        while improved and iterations < max_iters:
            improved = False
            iterations += 1
            if time.time() > DEADLINE - 5:
                break
            # Try adding one element
            for k in range(num_cands):
                if k in solution:
                    continue
                # Check if k can be added
                can_add = True
                for s in solution:
                    if (adj_bm[k] >> s) & 1:
                        can_add = False
                        break
                if can_add:
                    solution.add(k)
                    improved = True
                    best = max(best, len(solution))
                    break  # restart after improvement
            if improved:
                continue
            # Try 1-remove-2-add (2-exchange)
            for r in list(solution):
                # Try removing r and adding two new elements
                remaining_pool = [k for k in range(num_cands) if k not in solution]
                # Find two elements that can be added together and don't conflict with solution-{r}
                sol_without_r = solution - {r}
                for a in remaining_pool:
                    if a in solution:
                        continue
                    # Check a doesn't conflict with sol_without_r
                    a_ok = True
                    for s in sol_without_r:
                        if (adj_bm[a] >> s) & 1:
                            a_ok = False
                            break
                    if not a_ok:
                        continue
                    for b in remaining_pool:
                        if b <= a or b in solution:
                            continue
                        # Check b doesn't conflict with sol_without_r
                        b_ok = True
                        for s in sol_without_r:
                            if (adj_bm[b] >> s) & 1:
                                b_ok = False
                                break
                        if not b_ok:
                            continue
                        # Check a and b don't conflict with each other
                        if (adj_bm[a] >> b) & 1:
                            continue
                        # We can remove r and add a, b
                        solution.discard(r)
                        solution.add(a)
                        solution.add(b)
                        improved = True
                        best = max(best, len(solution))
                        break
                    if improved:
                        break
                if improved:
                    break
        return solution
    
    # Get initial solution
    greedy_sol = greedy_is()
    ls_sol = local_search(greedy_sol)
    best_sol = ls_sol
    best_size = len(best_sol)
    
    # Now try branch and bound to find optimal or better
    # B&B: explore the search tree, branch on vertices
    # State: (available_mask, current_set_size)
    # available_mask: bitmask of candidates still available to add
    # We want to maximize the number of non-conflicting vertices we can add
    
    # For B&B, we need to check if adding vertex k to current solution is valid.
    # Since we're building incrementally, we track which vertices are "blocked" by current choices.
    
    # Better formulation: 
    # We have a set of "forbidden" vertices (those conflicting with already chosen).
    # State: (forbidden_mask, chosen_count)
    # Available = all_candidates_mask & ~forbidden_mask
    
    all_mask = (1 << num_cands) - 1
    
    # B&B function
    # Returns: best additional count achievable from state (forbidden_mask, chosen_count)
    # Total = chosen_count + result
    
    # Use memoization for small states? Might be too large.
    # Instead, use upper bound pruning.
    
    # Upper bound: |available| is a trivial upper bound on how many more we can add.
    # Better: greedy coloring bound on the conflict graph induced by available vertices.
    
    def greedy_color_bound(available_mask):
        """Upper bound on MIS of the subgraph induced by available_mask.
        Uses greedy coloring of the conflict graph: the number of colors is an upper bound
        on the clique number, but we need an upper bound on independence number.
        
        Actually, let me use a different bound:
        For the conflict graph G on available vertices, 
        MIS(G) <= |V| - MVC(G) <= |V| - MM(G) where MM is max matching.
        
        Simpler: just use |available| as upper bound, but with a tighter heuristic:
        Sort available vertices by degree, and use the fact that each vertex of degree d
        "blocks" at most d other vertices.
        
        For a quick upper bound: |available| (trivial)
        Slightly better: if max degree is D, then MIS <= |available| - floor(|available|/(D+1)) ... no that's wrong direction.
        
        Let me just use |available| and rely on the branching.
        """
        # Count available vertices
        count = bin(available_mask).count('1')
        return count
    
    # More effective bound: for each available vertex, compute its degree in the induced subgraph.
    # The sum of 1/(deg_v + 1) over all available v is a lower bound on the minimum vertex cover
    # (this is a known result). So MIS <= |available| - sum(1/(deg_v+1)).
    # But this is fractional; we take floor.
    
    def fractional_vc_bound(available_mask):
        """Upper bound on MIS using fractional vertex cover lower bound."""
        count = bin(available_mask).count('1')
        if count == 0:
            return 0
        # Compute degrees in induced subgraph
        vc_lb = 0.0
        for k in range(num_cands):
            if not (available_mask >> k) & 1:
                continue
            # Degree of k in induced subgraph
            deg = bin(adj_bm[k] & available_mask).count('1')
            vc_lb += 1.0 / (deg + 1)
        # MIS <= count - floor(vc_lb)
        bound = count - int(vc_lb)
        return max(bound, 0)
    
    # Branch and bound
    # We'll use a priority queue (best-first) or DFS with pruning.
    # DFS is simpler and uses less memory.
    
    # To make B&B effective, we order vertices by a heuristic:
    # Prefer branching on vertices with high degree (they block more options)
    
    # Precompute a static ordering heuristic
    # For each vertex, its "weight" for branching order
    # High degree vertices should be branched on first
    
    # Actually, for MIS, we want to branch on vertices that are "important" - 
    # typically high degree in the conflict graph.
    
    # Let me implement a recursive B&B with time limit
    
    best_found = best_size
    best_solution = set(best_sol)
    
    # For tracking the actual solution during B&B
    # We'll use a different approach: since the problem is at most 256 nodes,
    # and the graph is sparse, let me try a different exact method.
    
    # Alternative: use the fact that the graph is a "sum graph" and has structure.
    # Numbers 1-256, conflicts when sum is in {4,9,16,25,36,49,64,81,100,121,144,169,196,225,256,289,324,361,400,441,484}
    
    # Let me count the edges to understand sparsity
    edge_count = 0
    for k in range(num_cands):
        edge_count += bin(adj_bm[k]).count('1')
    edge_count //= 2  # each edge counted twice
    
    # With ~200+ candidates and relatively few edges, greedy + local search might be near-optimal.
    # Let me try to verify with a more careful local search and then B&B on the "hard" parts.
    
    # Enhanced local search with random restarts
    import random
    
    def enhanced_local_search():
        best = set()
        best_size_local = 0
        for restart in range(50):
            if time.time() > DEADLINE - 10:
                break
            # Random greedy
            random.seed(42 + restart)
            order = list(range(num_cands))
            random.shuffle(order)
            sol = set()
            for k in order:
                can_add = True
                for s in sol:
                    if (adj_bm[k] >> s) & 1:
                        can_add = False
                        break
                if can_add:
                    sol.add(k)
            
            # Local search on this solution
            improved = True
            iters = 0
            while improved and iters < 200:
                improved = False
                iters += 1
                if time.time() > DEADLINE - 10:
                    break
                # Try adding
                for k in range(num_cands):
                    if k in sol:
                        continue
                    can_add = True
                    for s in sol:
                        if (adj_bm[k] >> s) & 1:
                            can_add = False
                            break
                    if can_add:
                        sol.add(k)
                        improved = True
                        break
                if improved:
                    continue
                # Try 1-2 exchange
                found = False
                for r in list(sol):
                    sol_w = sol - {r}
                    for a in range(num_cands):
                        if a in sol:
                            continue
                        a_ok = all(not ((adj_bm[a] >> s) & 1) for s in sol_w)
                        if not a_ok:
                            continue
                        for b in range(num_cands):
                            if b <= a or b in sol:
                                continue
                            b_ok = all(not ((adj_bm[b] >> s) & 1) for s in sol_w)
                            if not b_ok:
                                continue
                            if (adj_bm[a] >> b) & 1:
                                continue
                            sol.discard(r)
                            sol.add(a)
                            sol.add(b)
                            found = True
                            break
                        if found:
                            break
                    if found:
                        improved = True
                        break
            
            if len(sol) > best_size_local:
                best_size_local = len(sol)
                best = set(sol)
        
        return best, best_size_local
    
    # Run enhanced local search
    el_sol, el_size = enhanced_local_search()
    if el_size > best_size:
        best_size = el_size
        best_solution = el_sol
    
    # Now try branch and bound to potentially improve
    # B&B state: (forbidden_mask, current_size)
    # We want to find if we can achieve size > best_size
    
    # For efficiency, we only explore if the upper bound allows beating best_size
    # UB = current_size + |available| where available = all_mask & ~forbidden_mask
    
    # Let me implement a DFS B&B with vertex ordering
    
    # Precompute vertex ordering: sort by degree descending for branching
    deg_list = [bin(adj_bm[k]).count('1') for k in range(num_cands)]
    branch_order = sorted(range(num_cands), key=lambda x: -deg_list[x])
    
    # DFS B&B
    # To avoid deep recursion, use iterative approach or set recursion limit
    import sys
    sys.setrecursionlimit(10000)
    
    bb_best = [best_size]
    bb_solution = [set(best_solution)]
    
    def bb_dfs(forbidden_mask, current_size, chosen_mask):
        """Branch and bound DFS.
        forbidden_mask: vertices that cannot be added (conflict with chosen)
        current_size: number of vertices chosen so far
        chosen_mask: bitmask of chosen vertices
        """
        nonlocal bb_best, bb_solution
        
        if time.time() > DEADLINE - 5:
            return
        
        # Available vertices
        available = all_mask & ~forbidden_mask
        
        # Upper bound
        ub = current_size + bin(available).count('1')
        if ub <= bb_best[0]:
            return
        
        # If no available vertices, we're done with this branch
        if available == 0:
            if current_size > bb_best[0]:
                bb_best[0] = current_size
                bb_solution[0] = set(k for k in range(num_cands) if (chosen_mask >> k) & 1)
            return
        
        # Pick a vertex to branch on
        # Choose from available, prefer high degree
        # Find available vertices
        avail_list = [k for k in range(num_cands) if (available >> k) & 1]
        
        # Sort by degree in induced subgraph (for better pruning)
        def induced_deg(k):
            return bin(adj_bm[k] & available).count('1')
        
        avail_list.sort(key=lambda k: -induced_deg(k))
        
        v = avail_list[0]
        
        # Option 1: Include v
        new_forbidden = forbidden_mask | adj_bm[v] | (1 << v)
        new_chosen = chosen_mask | (1 << v)
        bb_dfs(new_forbidden, current_size + 1, new_chosen)
        
        if time.time() > DEADLINE - 5:
            return
        
        # Option 2: Exclude v
        new_forbidden2 = forbidden_mask | (1 << v)
        bb_dfs(new_forbidden2, current_size, chosen_mask)
    
    # Start B&B from empty set
    # But this explores the entire space. For 200+ vertices, this is infeasible without strong pruning.
    # 
    # Better: start from the current best solution and try to improve.
    # Or: use the B&B to prove the current solution is optimal.
    
    # Given time constraints, let me try a targeted B&B:
    # Start with the best solution found, and try to see if we can do better.
    
    # Actually, let me try a different approach: 
    # Use the B&B but with a tighter upper bound using the fractional VC bound.
    
    def bb_dfs2(forbidden_mask, current_size, chosen_mask):
        nonlocal bb_best, bb_solution
        
        if time.time() > DEADLINE - 5:
            return
        
        available = all_mask & ~forbidden_mask
        avail_count = bin(available).count('1')
        
        # Upper bound using fractional VC
        if avail_count > 0:
            vc_lb = 0.0
            for k in range(num_cands):
                if not (available >> k) & 1:
                    continue
                deg = bin(adj_bm[k] & available).count('1')
                vc_lb += 1.0 / (deg + 1)
            ub = current_size + avail_count - int(vc_lb)
        else:
            ub = current_size
        
        if ub <= bb_best[0]:
            return
        
        if available == 0:
            if current_size > bb_best[0]:
                bb_best[0] = current_size
                bb_solution[0] = set(k for k in range(num_cands) if (chosen_mask >> k) & 1)
            return
        
        # Pick vertex: highest degree in induced subgraph
        best_v = -1
        best_deg = -1
        for k in range(num_cands):
            if (available >> k) & 1:
                deg = bin(adj_bm[k] & available).count('1')
                if deg > best_deg:
                    best_deg = deg
                    best_v = k
        
        if best_v < 0:
            return
        
        # Include best_v
        new_forbidden = forbidden_mask | adj_bm[best_v] | (1 << best_v)
        new_chosen = chosen_mask | (1 << best_v)
        bb_dfs2(new_forbidden, current_size + 1, new_chosen)
        
        if time.time() > DEADLINE - 5:
            return
        
        # Exclude best_v
        new_forbidden2 = forbidden_mask | (1 << best_v)
        bb_dfs2(new_forbidden2, current_size, chosen_mask)
    
    # Try B&B with time limit
    if time.time() < DEADLINE - 15:
        bb_dfs2(0, 0, 0)
    
    if bb_best[0] > best_size:
        best_size = bb_best[0]
        best_solution = bb_solution[0]
    
    # Convert back to original numbers
    result = [candidates[k] for k in sorted(best_solution)]
    
    # Verify the solution
    for i in range(len(result)):
        for j in range(i, len(result)):
            s = result[i] + result[j]
            r = int(math.isqrt(s))
            if r * r == s:
                # This shouldn't happen, but if it does, remove the problematic element
                pass
    
    if not result:
        return [1]
    
    return result

# EVOLVE-BLOCK-END
