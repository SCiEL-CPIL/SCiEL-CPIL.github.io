"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct(n_rows, k):
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Total differences per row
    diffs_per_row = k * (k + 1) // 2
    
    # Counting bound: M such that M*(M+1)/2 >= n_rows * diffs_per_row
    # M^2/2 >= n_rows * k(k+1)/2 => M >= sqrt(n_rows * k(k+1))
    # Actually total diffs available in 1..M is M(M+1)/2.
    # We need n_rows * (k(k+1)/2) <= M(M+1)/2
    lower_bound_M = 1
    while lower_bound_M * (lower_bound_M + 1) // 2 < n_rows * diffs_per_row:
        lower_bound_M += 1
        
    # Best known is 112 for (7,5). Let's start slightly above lower bound or known bound.
    # For safety and speed, start from max(lower_bound, 100) but don't exceed 120.
    # If we can't find a solution by 120, we return the best found (which might be invalid if we force it, 
    # but the task says "invalid result... scores 0". We must return a VALID result.
    # So we must ensure we find a valid one.
    # Greedy always finds one. Let's use greedy to get an initial valid solution.
    
    def greedy_row(used, temp=0.0):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                if temp > 0 and random.random() < temp and x < 200:
                    continue
                row.append(x)
                used |= new_diffs
        return row, used

    # Get a valid baseline
    random.seed(42)
    used = set()
    baseline_rows = []
    for _ in range(n_rows):
        row, used = greedy_row(used, temp=0.0)
        baseline_rows.append(row)
    best_rows = list(baseline_rows)
    best_max = max(max(r) for r in best_rows)
    
    # If baseline is already very good (e.g., close to lower bound), we might not need heavy search.
    # But we want to improve.
    
    # Pre-compute candidate rows for a given M?
    # The number of subsets of size k+1 from {0..M} is C(M+1, k+1).
    # For M=112, k=5, C(113, 6) is approx 113^6 / 720 ~ 2e11 / 720 ~ 3e8. Too many to pre-compute all.
    
    # Alternative: Backtracking that generates rows on the fly but uses pruning.
    # Or: Backtracking that picks a row, then restricts the next row's differences.
    
    # Let's try a different structural approach:
    # 1. Generate a pool of "good" candidate rows (low max value) using randomized greedy.
    # 2. Build a conflict graph where nodes are candidate rows, edges if they share diffs.
    # 3. Find a clique of size n_rows in this graph.
    # 4. Minimize the max value of the clique.
    
    # How to generate candidate rows?
    # We can generate rows with max value <= M_candidate.
    # Let's try to find a solution with M < best_max.
    
    def find_solution_with_max(M_limit, time_budget):
        """
        Try to find n_rows rows with max entry <= M_limit such that diffs are disjoint.
        Uses backtracking with pruning.
        """
        if time.time() > time_budget:
            return None
            
        # To make this efficient, we can pre-compute valid rows up to M_limit?
        # No, too many.
        
        # Instead, we use a recursive backtracking that builds rows one by one.
        # At each step, we have a set of 'used' diffs.
        # We need to choose a row from the remaining possibilities.
        
        # To prune effectively, we can order candidate rows by their max value or some heuristic.
        # But generating candidates on the fly is tricky.
        
        # Let's stick to the "Pool + Graph" idea but with a smaller pool.
        # Generate P candidate rows using randomized greedy constrained to M_limit.
        # If we can't generate enough, return None.
        
        candidates = []
        seen_diffs = set() # Not strictly necessary for generation, but for diversity
        
        # We want diverse candidates.
        # Use different seeds/temperatures.
        num_candidates = 2000
        for i in range(num_candidates):
            if time.time() > time_budget:
                break
            random.seed(i)
            used = set()
            row = [0]
            x = 0
            valid = True
            while len(row) < k + 1:
                x += 1
                if x > M_limit:
                    valid = False
                    break
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    # Add randomness
                    if random.random() < 0.1 and x < M_limit:
                        continue
                    row.append(x)
                    used |= new_diffs
            if valid and max(row) <= M_limit:
                # Check if this row is unique (by its diff set)
                # We can just add it. Duplicates in pool are okay but wasteful.
                candidates.append(row)
                
        if len(candidates) < n_rows:
            return None
            
        # Remove duplicates based on diff sets
        unique_candidates = []
        seen_diff_sets = set()
        for r in candidates:
            ds = frozenset({r[j]-r[i] for i in range(len(r)) for j in range(i+1, len(r))})
            if ds not in seen_diff_sets:
                seen_diff_sets.add(ds)
                unique_candidates.append((r, ds))
                
        if len(unique_candidates) < n_rows:
            return None
            
        # Build conflict graph (adjacency list)
        # Node i is compatible with node j if ds_i & ds_j is empty.
        # Since n_candidates can be large (up to 2000), O(N^2) check is 4M ops, which is fine.
        N = len(unique_candidates)
        adj = [set() for _ in range(N)]
        
        for i in range(N):
            if time.time() > time_budget:
                return None
            ds_i = unique_candidates[i][1]
            for j in range(i+1, N):
                ds_j = unique_candidates[j][1]
                if not (ds_i & ds_j):
                    adj[i].add(j)
                    adj[j].add(i)
                    
        # Find a clique of size n_rows in this graph.
        # Use Bron-Kerbosch or simple backtracking.
        # Since we just need ONE clique of size n_rows, and we want to minimize max value,
        # we can sort candidates by max value and try to pick from the "best" ones first.
        
        # Sort by max value
        unique_candidates.sort(key=lambda x: max(x[0]))
        
        # Re-index adj after sorting? No, adj is built on indices.
        # Let's just run a backtracking search on the graph.
        # To bias towards low max values, we can prioritize nodes with lower max value in the search.
        
        # Simple recursive clique finder
        best_clique = []
        
        def expand_clique(candidates_idx, current_clique):
            nonlocal best_clique
            if time.time() > time_budget:
                return
            if len(current_clique) == n_rows:
                best_clique = list(current_clique)
                return
            if not candidates_idx:
                return
            # Heuristic: pick a node from candidates_idx that has the smallest max value
            # This biases the search towards "smaller" rows.
            # Find index in unique_candidates with min max value
            min_idx = min(candidates_idx, key=lambda i: max(unique_candidates[i][0]))
            
            # Try adding min_idx
            new_candidates = candidates_idx.intersection(adj[min_idx])
            current_clique.append(min_idx)
            expand_clique(new_candidates, current_clique)
            current_clique.pop()
            
            if best_clique:
                return
                
            # Try other nodes
            # To avoid exponential blowup, we can limit branching or use a better heuristic.
            # For now, let's just try the next best few.
            # Actually, standard Bron-Kerbosch with pivot is better.
            # But for small N (2000) and target size 7, simple backtracking with ordering might work.
            
            # Let's try a few other candidates from the top of the sorted list
            # candidates_idx is a set. Let's convert to sorted list by max value.
            sorted_cands = sorted(candidates_idx, key=lambda i: max(unique_candidates[i][0]))
            
            for idx in sorted_cands:
                if idx == min_idx:
                    continue
                new_candidates = candidates_idx.intersection(adj[idx])
                current_clique.append(idx)
                expand_clique(new_candidates, current_clique)
                current_clique.pop()
                if best_clique:
                    return
                # Limit branching to avoid timeout
                if len(sorted_cands) > 10 and sorted_cands.index(idx) > 5:
                    break

        # Initial candidates for search: all nodes
        # But we want to start with the "best" (smallest max) nodes.
        # Let's initialize with the first K nodes in sorted order.
        initial_set = set(range(min(100, N)))
        expand_clique(initial_set, [])
        
        if best_clique:
            return [unique_candidates[i][0] for i in best_clique]
        return None

    # Search for better solutions
    # Start from best_max - 1 down to lower_bound
    current_limit = best_max - 1
    
    # To save time, we can jump.
    # If we can't find a solution with M=100, we might not find one with M=99.
    # But the problem is discrete.
    
    # Let's try a few specific limits.
    # The known best is 112. Our greedy probably got ~120-130.
    # Let's try to improve.
    
    search_limits = []
    # Try to beat the current best
    for m in range(current_limit, lower_bound - 1, -1):
        if time.time() > deadline - 5:
            break
        # Don't try every single integer, it's slow.
        # Try every 1st, but with a small time budget per attempt.
        # Or try a few strategic points.
        
        # If we are far from lower bound, we can be more aggressive.
        # Let's just iterate.
        
        # Time budget per M: 5 seconds?
        # If we have 100s left, and range is 20, that's 5s each.
        
        time_budget = min(deadline - 5, time.time() + 5)
        
        # Optimization: If we failed for M+1, we likely need more candidates for M.
        # But the pool generation is the bottleneck.
        
        # Let's only try this if we have enough time.
        # If best_max is 120, and lower_bound is 105, range is 15.
        
        sol = find_solution_with_max(m, time_budget)
        if sol:
            best_rows = sol
            best_max = m
            # Continue searching for even smaller?
            # Yes, the loop continues to m-1.
            # But we should update time_budget logic?
            # The loop condition handles it.
            
    return best_rows
# EVOLVE-BLOCK-END
