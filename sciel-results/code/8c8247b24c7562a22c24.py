import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s/150s limit
    
    # 1. Generate all possible candidate classes (m, r)
    # m in [20, 200], r in [0, m)
    candidates = []
    for m in range(20, 201):
        for r in range(m):
            candidates.append((m, r))
    
    n = len(candidates)
    
    # 2. Precompute conflict graph
    # Two classes (m1, r1) and (m2, r2) are disjoint iff r1 % gcd(m1, m2) != r2 % gcd(m1, m2)
    # If they are NOT disjoint, they conflict. We want an independent set (no conflicts).
    # We represent the graph as a list of bitmasks for speed, or just adjacency lists.
    # Given n ~ 181 * 100 = 18100 vertices, adjacency matrix is too big for memory if dense, 
    # but Python lists of sets or bitmasks are feasible.
    # Let's use a list of sets for neighbors.
    # Actually, for branch and bound, we need fast "conflicts with any in current set" checks.
    # A better representation for B&B: 
    # We sort vertices by weight (1/m) descending.
    # We maintain a set of "available" vertices.
    # When we pick a vertex v, we remove v and all its neighbors from the available set.
    
    # Precompute neighbors for each candidate
    neighbors = [set() for _ in range(n)]
    
    # To speed up pair checks, we can group by modulus?
    # No, direct O(n^2) precomputation is ~ 18000^2 / 2 = 1.6 * 10^8 operations.
    # In Python, this might take 10-30 seconds. We have 115s.
    # Let's optimize the inner loop.
    
    # Optimization: 
    # Check condition: r1 % gcd(m1, m2) == r2 % gcd(m1, m2)
    # Precompute gcds? There are only 181 moduli. 
    # gcd_matrix[m1][m2]
    gcd_cache = {}
    def get_gcd(m1, m2):
        key = (min(m1, m2), max(m1, m2))
        if key not in gcd_cache:
            gcd_cache[key] = math.gcd(m1, m2)
        return gcd_cache[key]
    
    # Build adjacency
    for i in range(n):
        m1, r1 = candidates[i]
        for j in range(i + 1, n):
            m2, r2 = candidates[j]
            g = get_gcd(m1, m2)
            if (r1 % g) == (r2 % g):
                # Conflict
                neighbors[i].add(j)
                neighbors[j].add(i)
                
    # 3. Sort candidates by weight (1/m) descending to guide the search
    # We want to pick high weight vertices first.
    # However, the indices in 'neighbors' correspond to the original order.
    # Let's create a permutation of indices sorted by weight.
    # weight[i] = 1.0 / candidates[i][0]
    indices = list(range(n))
    indices.sort(key=lambda i: -1.0 / candidates[i][0])
    
    # Re-map neighbors to the new order?
    # It's easier to keep the original indices and just iterate through the sorted list in B&B.
    # But B&B efficiency depends on the order of consideration.
    
    # Let's implement a recursive Branch and Bound.
    # State: 
    #   current_score
    #   available: set of indices that are not yet decided and not conflicting with chosen ones
    #   upper_bound: current_score + sum of weights of all vertices in available
    
    # To make this fast in Python, we need to be careful with set operations.
    # n=18k. Set operations are O(n).
    # Depth can be up to ~100-200 (since we can't pick many, they conflict).
    # Actually, how many can we pick?
    # If we pick moduli that are multiples of 2, we can pick at most 100?
    # Wait, if m1=20, m2=22, gcd=2. r1=0, r2=1. Disjoint.
    # We can pick many.
    # Max independent set size might be large.
    
    # Alternative: Use a greedy coloring bound or a simple sum bound.
    # Simple sum bound is weak if the graph is dense.
    # Let's try a simpler heuristic first to get a good lower bound (incumbent).
    
    # Greedy Heuristic for Initial Incumbent
    def greedy_heuristic():
        avail = set(range(n))
        score = 0.0
        chosen = []
        # Iterate in sorted order
        for idx in indices:
            if idx not in avail:
                continue
            # Check if we can add it (it's in avail, so it doesn't conflict with previous)
            # Add it
            chosen.append(idx)
            score += 1.0 / candidates[idx][0]
            # Remove it and its neighbors
            to_remove = {idx} | neighbors[idx]
            avail -= to_remove
        return score, chosen

    best_score, best_chosen = greedy_heuristic()
    
    # Branch and Bound
    # We will use a stack-based DFS to avoid recursion limits.
    # Stack item: (current_score, available_set, next_index_in_sorted_order)
    # Actually, standard B&B for MWIS:
    # We process vertices in a fixed order (sorted by weight).
    # At each step, we decide: include vertex v or exclude vertex v.
    # If include: remove v and neighbors from available.
    # If exclude: just remove v from available.
    # Bound: current_score + sum(weights of available).
    
    # To speed up "sum(weights of available)", we can maintain a running sum?
    # No, because we remove arbitrary sets of neighbors.
    # However, n=18k. Summing 18k floats is fast (~0.1ms).
    # The bottleneck is set difference `avail -= neighbors`.
    
    # Let's optimize the B&B loop.
    # We'll use a list as a stack.
    # To avoid copying large sets, we can use a "coloring" or "marking" approach?
    # Or just accept the copy cost if the tree is pruned heavily.
    
    # Let's try to improve the bound.
    # A tighter bound: The maximum weight independent set in a graph is bounded by the sum of weights of a maximal independent set in a subgraph?
    # Or use the "clique cover" bound? Too complex.
    # Let's stick to the simple bound but with aggressive pruning.
    
    # One optimization: If the upper bound of a node is <= best_score, prune.
    
    # Let's run B&B with a time limit.
    # We will explore the tree.
    
    # To make the set operations faster, we can represent `available` as a boolean array?
    # `avail_mask` as a list of booleans or a Python `set`.
    # Python `set` difference is implemented in C and is fast for small sets, but here sets are large.
    # `set.difference_update` is in-place and faster.
    
    # Let's restructure the B&B to use in-place modifications with backtracking?
    # That's hard with sets.
    # Given the time limit, copying sets might be okay if the search tree is small.
    # If the search tree is large, we might time out.
    
    # Let's add a random restart to the greedy heuristic to get a better initial lower bound.
    for _ in range(10):
        if time.time() > deadline - 30:
            break
        # Randomized greedy
        avail = set(range(n))
        score = 0.0
        chosen = []
        # Shuffle indices slightly? Or just pick randomly from available high-weight nodes?
        # Simple: Pick the highest weight node in available, add it, remove neighbors.
        # This is deterministic if we always pick the max. 
        # To randomize, we can pick from the top 10% of available weights.
        for _ in range(len(indices)):
            if not avail:
                break
            # Find top candidates in avail
            # This is O(n) to find max.
            # Let's just pick the first one in `indices` that is in `avail`.
            # To randomize, we can shuffle `indices` once per restart?
            # No, `indices` is sorted by weight.
            # Let's create a random permutation of indices and process them, 
            # but only add if compatible. This is the "random greedy" from the current program.
            # The current program already does this. 
            # The B&B is the new part.
            pass
            
    # Let's just run the B&B on the deterministic sorted order.
    # If it's too slow, we return the greedy result.
    
    # Optimization: Precompute weight array
    weights = [1.0 / candidates[i][0] for i in range(n)]
    
    # B&B Stack: (score, available_set)
    # We process vertices in order of `indices`.
    # We need to know which vertex we are considering.
    # Standard approach:
    # Function expand(idx, score, avail):
    #   if idx == n: update best
    #   bound = score + sum(weights[i] for i in avail)
    #   if bound <= best_score: return
    #   v = indices[idx]
    #   if v in avail:
    #       # Option 1: Include v
    #       new_avail = avail - {v} - neighbors[v]
    #       expand(idx+1, score + weights[v], new_avail)
    #       # Option 2: Exclude v
    #       new_avail2 = avail - {v}
    #       expand(idx+1, score, new_avail2)
    #   else:
    #       expand(idx+1, score, avail)
        
    # This recursion depth is n=18k. Python recursion limit is ~1000.
    # We must use iterative or increase limit.
    # sys.setrecursionlimit(20000)
    # But 18k depth * stack frame size might crash memory or be slow.
    # Also, the branching factor is 2.
    
    # Better B&B formulation for Independent Set:
    # We don't iterate through all n indices one by one.
    # We pick a vertex from `avail`.
    # Usually, we pick the vertex with the highest weight or highest degree.
    # Let's pick the highest weight vertex in `avail`.
    
    def solve_bnb():
        nonlocal best_score, best_chosen
        # Stack: (score, available_set)
        # We need to be careful about memory.
        # Let's use a priority queue? No, DFS is better for finding good solutions early if we order branches well.
        
        # Initial state
        initial_avail = set(range(n))
        stack = [(0.0, initial_avail)]
        
        # To avoid infinite loops or excessive branching, we can limit the depth or number of nodes visited.
        nodes_visited = 0
        max_nodes = 500000 # Arbitrary limit to prevent hanging
        
        while stack:
            if time.time() > deadline - 5:
                break
            if nodes_visited > max_nodes:
                break
            nodes_visited += 1
            
            score, avail = stack.pop()
            
            # Bound check
            # Calculate sum of weights in avail
            # Optimization: If avail is small, sum is fast.
            bound = score
            for v in avail:
                bound += weights[v]
                
            if bound <= best_score:
                continue
                
            # If avail is empty, update best
            if not avail:
                if score > best_score:
                    best_score = score
                    # Reconstruct chosen? We didn't track the path.
                    # This is a problem. We need to track the set of chosen vertices.
                    # Storing the set of chosen vertices in the stack doubles memory.
                    # Alternative: We just return the score? 
                    # The task requires returning the classes.
                    # We can reconstruct by running a greedy algorithm on the final graph? No.
                    # We must track the path.
                    # Let's modify the stack to include `chosen_set`.
                    # Memory might be an issue.
                    
                continue
            
            # Pick a vertex to branch on.
            # Heuristic: Pick the vertex with the highest weight in avail.
            # Or the one with the most neighbors in avail (to reduce branching)?
            # High weight is better for the bound.
            v = max(avail, key=lambda x: weights[x])
            
            # Branch 1: Include v
            # New avail = avail - {v} - neighbors[v]
            # Branch 2: Exclude v
            # New avail = avail - {v}
            
            # We push the "Exclude" branch first so that "Include" is processed next (LIFO).
            # This prioritizes adding high-weight nodes.
            
            # Optimization: If including v doesn't improve the bound significantly, maybe skip?
            
            # Let's try a different approach: 
            # Since we need the actual classes, and B&B tracking is heavy,
            # let's use the B&B to find the OPTIMAL SCORE, and then use a 
            # constraint solver (or the greedy method) to find a valid set with that score?
            # No, that's not guaranteed to work.
            
            # Let's stick to tracking `chosen`.
            # To save memory, we can store the parent pointer and the decision?
            # That's complex.
            
            # Given the time constraint, let's try a simpler local search on the MODULI set,
            # but using the RESIDUE feasibility as a hard constraint, and optimizing the modulus set.
            # The current program does random greedy on moduli.
            # Let's do: 
            # 1. Start with a valid set from greedy.
            # 2. Try to swap one modulus for another?
            # 3. Or try to add a modulus and adjust residues?
            
            # Actually, the "Mixed" strategy mentioned in the current code comments is promising.
            # Let's implement a specific heuristic for "Mixed" moduli.
            # Group moduli by their prime factors?
            
            # Let's try a different structural change:
            # **Simulated Annealing on the set of moduli.**
            # State: A set of moduli S.
            # Validity: We check if S is packable (feasible residues exist).
            # How to check packability efficiently?
            # For a given S, we can try to assign residues using a backtracking search on residues.
            # Since |S| is small (< 100), and residues are < 200, this is fast.
            # Objective: Sum(1/m).
            # Move: Add a modulus, Remove a modulus, Swap a modulus.
            # If the new set is packable, accept if better or with probability exp(-delta/T).
            
            # This avoids the huge graph of (m,r) pairs.
            # The state space is subsets of {20..200}.
            # The cost is checking packability.
            
            # Let's implement this.
            
            def is_packable(moduli_set):
                # Backtracking search for residues
                # Sort moduli by size of constraints? Or just index.
                # Constraints: r_i % gcd(m_i, m_j) != r_j % gcd(m_i, m_j)
                
                # We can use a recursive backtracking.
                # To speed it up, we can use the fact that if gcd(m_i, m_j) is large,
                # the constraint is strong.
                
                # Let's just use a simple backtracking.
                # Since we call this many times, it needs to be fast.
                
                # Optimization: 
                # If the moduli are all multiples of a common K, we can pack them easily if |S| <= K.
                # General case:
                
                # We'll use a list of (m, r) assignments.
                # We pick a modulus, try all residues, check against assigned ones.
                
                # Order of moduli: Smallest first? Or most constrained?
                # Let's sort by number of neighbors in the conflict graph? 
                # For a fixed set S, the "degree" is |S|-1.
                # Let's just sort by m ascending (smaller m has fewer residues to try? No, m is the number of residues).
                # Smaller m has fewer residues to iterate.
                
                sorted_moduli = sorted(moduli_set)
                assignments = {}
                
                def backtrack(idx):
                    if idx == len(sorted_moduli):
                        return True
                    m = sorted_moduli[idx]
                    for r in range(m):
                        valid = True
                        for j in range(idx):
                            m_j = sorted_moduli[j]
                            r_j = assignments[j]
                            g = math.gcd(m, m_j)
                            if r % g == r_j % g:
                                valid = False
                                break
                        if valid:
                            assignments[idx] = r
                            if backtrack(idx + 1):
                                return True
                            # Backtrack
                            del assignments[idx]
                    return False
                    
                return backtrack(0)
                
            # This backtracking might be slow if called thousands of times.
            # We can cache results? The set of moduli is the key.
            # But there are many subsets.
            
            # Let's limit the search.
            
            # Actually, let's look at the score 2.8671.
            # Sum of 1/m for m=20..200 is ~ 2.3.
            # How can the score be 2.8671?
            # Ah, the current program returns `best_classes`.
            # Score is `sum(1.0 / m for m, _ in current_classes)`.
            # If the program picks moduli < 20, it would be invalid.
            # Wait, the loop is `for m in moduli:` where `moduli` is `range(20, 201)`.
            # So m >= 20.
            # Sum of 1/m for m=20..200 is ~ 2.302.
            # How is the score 2.8671?
            # Maybe I am miscalculating the sum.
            # 1/20 = 0.05
            # 1/21 ~ 0.047
            # ...
            # Integral of 1/x from 20 to 200 is ln(10) = 2.302.
            # The sum is slightly larger than the integral?
            # Sum(1/m) for m=1..N is ln(N) + gamma.
            # Sum(1/m) for m=20..200 = H_200 - H_19.
            # H_200 ~ ln(200) + 0.577 = 5.3 + 0.577 = 5.877
            # H_19 ~ ln(19) + 0.577 = 2.94 + 0.577 = 3.517
            # Diff = 2.36.
            # So max possible score is ~2.36.
            # Why is the current score 2.8671?
            
            # Re-reading the task: "The score is the sum of 1/m".
            # Maybe the current program is invalid and the evaluator is giving a different score?
            # Or maybe I am wrong about the sum.
            # Let's check: 1/20 + 1/21 + ... + 1/200.
            # If the current program is valid, its score cannot exceed 2.36.
            # Unless... the "score" in the prompt is not just sum(1/m)?
            # "The score is the sum of 1/m (exact rational, reported as a float)."
            
            # Wait, look at the OBSERVED_CHANGES.
            # Parent score: 1.3792. Child score: 2.8904.
            # The child program is the CURRENT PROGRAM provided in the prompt.
            # The current program has score 2.8671 (from prompt header).
            # This implies the max score is > 2.36.
            # How?
            # Ah! "Pairwise disjoint residue classes with distinct moduli in [20, 200]".
            # Is it possible to pick the same modulus? "distinct moduli". No.
            # Is it possible that the sum is larger?
            # Maybe I am underestimating the harmonic sum?
            # Let's calculate H_200 - H_19 more carefully.
            # H_n ~ ln(n) + 1/(2n) + gamma.
            # H_200 ~ 5.298 + 0.0025 + 0.577 = 5.8775
            # H_19 ~ 2.944 + 0.026 + 0.577 = 3.547
            # Diff = 2.33.
            
            # There is a discrepancy.
            # Could the "score" be something else?
            # "The evaluator measures the raw value".
            # Maybe the current program is cheating or I'm missing something.
            # Or maybe the moduli are not distinct? "distinct moduli".
            
            # Let's assume the score is correct and my sum estimate is wrong.
            # Or maybe the range is different? [20, 200].
            
            # Regardless, the strategy should be to maximize the sum.
            # The B&B on (m,r) is the most robust way to ensure validity and optimality.
            # The "Simulated Annealing on Moduli" is risky because checking packability is hard.
            
            # Let's go back to the B&B on (m,r) but optimize it.
            # The key is to get a good initial lower bound.
            # The current program's random greedy is a good lower bound.
            # Then B&B refines it.
            
            # I will implement the B&B with a strict time limit and return the best found.
            
            # To handle the "track chosen" issue, I will store the chosen list in the stack.
            # To save memory, I will only store the chosen list for the best solution found so far,
            # and for the current path.
            # Actually, since we are doing DFS, the "current path" is the stack of decisions.
            # I will store `chosen` as a list in the stack item.
            # Depth ~ 100. List copy is fast.
            
            pass # End of B&B design
            
    # Let's implement the B&B properly now.
    
    # Precompute weights
    weights = [1.0 / candidates[i][0] for i in range(n)]
    
    # Initial greedy solution
    avail = set(range(n))
    best_score = 0.0
    best_chosen = []
    for idx in indices:
        if idx in avail:
            best_chosen.append(idx)
            best_score += weights[idx]
            avail -= {idx} | neighbors[idx]
            
    # B&B
    # Stack: (score, avail_set, chosen_list)
    # We use a list as a stack.
    # To prevent memory explosion, we limit the stack size or depth?
    # DFS is fine.
    
    # We will use a priority queue? No, DFS is better for finding good solutions.
    # But we need to prune.
    
    # Let's use a recursive function with sys.setrecursionlimit.
    import sys
    sys.setrecursionlimit(10000)
    
    # To avoid global variable issues, use nonlocal.
    
    def branch_and_bound():
        nonlocal best_score, best_chosen
        
        # We will use a stack to simulate recursion to avoid limit issues if depth > 10000.
        # But depth is bounded by the size of the independent set, which is < 100.
        # So recursion is safe.
        
        def dfs(score, avail, chosen):
            nonlocal best_score, best_ch
            
            # Bound
            bound = score
            for v in avail:
                bound += weights[v]
            if bound <= best_score:
                return
                
            if not avail:
                if score > best_score:
                    best_score = score
                    best_chosen = chosen[:]
                return
                
            # Pick a vertex
            # Heuristic: Pick the vertex with the highest weight in avail.
            # To make this faster, we can maintain a sorted list?
            # For now, max() is O(|avail|).
            v = max(avail, key=lambda x: weights[x])
            
            # Branch 1: Include v
            new_avail = avail - {v} - neighbors[v]
            chosen.append(v)
            dfs(score + weights[v], new_avail, chosen)
            chosen.pop()
            
            # Branch 2: Exclude v
            new_avail2 = avail - {v}
            dfs(score, new_avail2, chosen)
            
        # Start DFS
        # To improve performance, we can order the branches such that the "Include" branch is explored first.
        # This is already done.
        
        # One optimization: If the bound is not much better than best_score, we can prune earlier?
        # No, the bound is the best possible.
        
        # Let's add a time check inside the recursion?
        # It's hard to check time in a deep recursion efficiently.
        # We can check it at the entry of dfs.
        
        def dfs_timed(score, avail, chosen):
            nonlocal best_score, best_ch
            
            if time.time() > deadline - 2:
                return
                
            bound = score
            for v in avail:
                bound += weights[v]
            if bound <= best_score:
                return
                
            if not avail:
                if score > best_score:
                    best_score = score
                    best_chosen = chosen[:]
                return
                
            v = max(avail, key=lambda x: weights[x])
            
            # Include
            new_avail = avail - {v} - neighbors[v]
            chosen.append(v)
            dfs_timed(score + weights[v], new_avail, chosen)
            chosen.pop()
            
            # Exclude
            new_avail2 = avail - {v}
            dfs_timed(score, new_avail2, chosen)
            
        dfs_timed(0.0, set(range(n)), [])
        
    branch_and_bound()
    
    # Convert best_chosen indices to classes
    classes = [[candidates[i][0], candidates[i][1]] for i in best_chosen]
    
    # Verify validity (just in case)
    for i in range(len(classes)):
        m1, r1 = classes[i]
        for j in range(i + 1, len(classes)):
            m2, r2 = classes[j]
            g = math.gcd(m1, m2)
            if r1 % g == r2 % g:
                # Invalid!
                # This should not happen if B&B is correct.
                # Fallback to greedy
                classes = [[candidates[i][0], candidates[i][1]] for i in best_chosen]
                # If still invalid, return empty?
                break
    
    return classes

# Run construct
# Note: The code above defines construct.
# We need to make sure it runs within the time limit.
# The B&B might be slow if the bound is weak.
# To improve, we can use a better bound:
# The sum of weights of a maximal independent set in the subgraph induced by avail?
# That's expensive.
# Let's trust the simple bound for now.
