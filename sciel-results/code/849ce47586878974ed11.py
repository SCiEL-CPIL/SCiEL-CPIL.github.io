"""Program for difference triangle set (7,5) using branch-and-bound with bitsets."""

def construct(n_rows, k):
    import itertools
    import sys
    sys.setrecursionlimit(100000)
    
    # We need n_rows rows, each with k+1 entries starting at 0, strictly increasing.
    # All positive differences within each row must be distinct across all rows.
    # Minimize the largest entry.
    
    # Precompute: for a given max_val, generate all valid rows (sequences of length k+1
    # starting at 0, strictly increasing, max <= max_val) and their difference sets.
    
    # Strategy: binary search / incremental on max_val.
    # For each max_val, check if we can select n_rows mutually non-conflicting rows.
    
    # Use bitsets for difference tracking. Differences range from 1 to max_val.
    
    def generate_rows(max_val):
        """Generate all valid rows of length k+1 starting at 0, strictly increasing, max <= max_val.
        Return list of (row_tuple, diff_mask) where diff_mask is a bitset of differences used."""
        rows = []
        # Generate all combinations of k elements from 1..max_val (since first is 0)
        # Actually: row = (0, a1, a2, ..., ak) with 0 < a1 < a2 < ... < ak <= max_val
        # Differences: a_j - a_i for all i < j, where a_0 = 0.
        
        # Number of combinations: C(max_val, k) which can be large.
        # For k=5, max_val=112: C(112,5) ≈ 112*111*110*109*108/120 ≈ 60M - too many.
        # We need a smarter approach.
        
        # Instead, let's use a different strategy:
        # Start with a greedy/known-good solution and try to improve.
        # Or use iterative deepening on max_val with smart pruning.
        
        # Given time constraints (120s), let's use a hybrid:
        # 1. Use the greedy approach to get an initial solution
        # 2. Try to improve via local search / backtracking on smaller instances
        
        return rows
    
    # Given the complexity, let's use a more practical approach:
    # 1. Greedy to get initial solution
    # 2. Try to optimize by reordering / backtracking on the greedy construction
    # 3. Use iterative deepening with pruning
    
    # First, let's get a baseline with greedy
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    # Get initial solution
    rows = greedy_construct(n_rows, k)
    max_entry = max(max(r) for r in rows)
    
    # Now try to improve using backtracking with pruning
    # We'll try to construct rows one by one, but with the ability to backtrack
    # and try different choices for earlier rows.
    
    # For efficiency, let's try a limited backtracking search
    # with the goal of finding a solution with max_entry < current_best
    
    best_rows = [list(r) for r in rows]
    best_max = max_entry
    
    # Try iterative improvement: for each target_max from best_max-1 down to some bound,
    # try to find n_rows non-conflicting rows with all entries <= target_max
    
    # Precompute all possible rows up to a certain max for efficiency
    # For k=5, we need sequences (0, a1, a2, a3, a4, a5) with 0 < a1 < ... < a5 <= M
    
    # Let's use a recursive backtracking approach with bitsets
    
    def solve_with_max(M, n_rows, k):
        """Try to find n_rows mutually non-conflicting rows with all entries <= M.
        Returns list of rows or None if not possible."""
        if M < k:  # minimum possible max is at least k
            return None
        
        # Generate candidate rows: all (k+1)-tuples (0, a1, ..., ak) with 0 < a1 < ... < ak <= M
        # and compute their difference bitsets
        
        # For efficiency, precompute candidates
        candidates = []
        
        # Generate combinations of k elements from 1..M
        # This can be expensive for large M, so we limit
        if M > 150:  # safety limit
            return None
        
        from itertools import combinations
        
        # For each combination of k values from 1..M
        # The row is (0) + sorted(combination)
        # Compute differences
        
        # To reduce candidates, we can use a heuristic: prefer rows with smaller max
        # and rows that use "rare" differences
        
        # Build list of (row, diff_set)
        # diff_set is a frozenset of differences
        
        # For memory, use integer bitmasks if M is small enough
        # Differences range from 1 to M, so we need M bits
        
        # Let's use Python sets for simplicity, but this may be slow
        # For M up to ~112, differences are 1..112, so 112 bits
        
        # Use integer bitmasks
        candidates = []
        for comb in combinations(range(1, M + 1), k):
            row = (0,) + comb
            # Compute differences
            diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
            # Convert to bitmask
            mask = 0
            for d in diffs:
                if d <= M:
                    mask |= (1 << d)
                else:
                    # Shouldn't happen since max diff <= M
                    mask |= (1 << d)
            candidates.append((row, mask))
        
        # Sort candidates by some heuristic: prefer those with smaller max element
        # and those that use fewer "popular" differences
        # For now, just sort by max element
        candidates.sort(key=lambda x: x[0][-1])
        
        # Backtracking search
        n_cand = len(candidates)
        if n_cand < n_rows:
            return None
        
        # Use branch and bound
        # State: index in candidates, current used mask, list of selected rows
        # We want to select n_rows candidates such that no two share a difference
        
        selected = []
        used_mask = 0
        
        # Pruning: if remaining candidates < n_rows - len(selected), fail
        # Also, if we can't fit n_rows - len(selected) more non-conflicting rows, fail
        
        # For efficiency, limit the search
        # Use a node limit
        node_limit = 500000
        node_count = 0
        found = [False]
        
        def backtrack(start_idx, used_mask, selected):
            nonlocal node_count, found
            if found[0]:
                return True
            node_count += 1
            if node_count > node_limit:
                return False
            
            if len(selected) == n_rows:
                found[0] = True
                return True
            
            # Pruning: not enough candidates left
            if start_idx + (n_rows - len(selected)) > n_cand:
                return False
            
            # Try each candidate from start_idx
            for i in range(start_idx, n_cand):
                row, mask = candidates[i]
                # Check conflict
                if mask & used_mask:
                    continue
                # Select this candidate
                selected.append(row)
                if backtrack(i + 1, used_mask | mask, selected):
                    return True
                selected.pop()
            
            return False
        
        if backtrack(0, 0, selected):
            return [list(r) for r in selected]
        else:
            return None
    
    # Try to find a better solution
    # Start from best_max - 1 and go down
    # But this could be very slow, so limit the search
    
    # Given time constraints, let's try a few values below the greedy solution
    # and also try to optimize the greedy solution via local search
    
    # Local search: try swapping elements in rows to reduce max
    # This is complex, so let's try a simpler approach:
    # Re-run greedy with different orderings or perturbations
    
    # For now, return the greedy solution if we can't find better quickly
    # Try a few target_max values
    
    for target_max in range(best_max - 1, max(best_max - 10, k), -1):
        result = solve_with_max(target_max, n_rows, k)
        if result is not None:
            best_rows = result
            best_max = target_max
            break
    
    return best_rows
