import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 second deadline with 5s safety margin
    
    # Generate smooth numbers <= 128
    smooth = []
    for x in range(1, 129):
        y = x
        for p in (2, 3, 5):
            while y % p == 0:
                y //= p
        if y == 1:
            smooth.append(x)
    
    # Precompute divisibility relations
    # divs[i] = list of indices j where smooth[j] divides smooth[i]
    n_smooth = len(smooth)
    divs = [[] for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[i] % smooth[j] == 0:
                divs[i].append(j)
    
    solutions = [None] * 129
    
    # For each target, find minimal antichain representation
    # Use backtracking with pruning
    def find_minimal_antichain(target, max_terms=15):
        """Find antichain of smooth numbers summing to target with minimum terms."""
        # Try increasing number of terms
        for k in range(1, max_terms + 1):
            result = backtrack_antichain(target, k, [], 0, set())
            if result is not None:
                return result
        return None
    
    def backtrack_antichain(target, k, current, sum_so_far, used_indices):
        """Backtracking search for antichain of exactly k terms summing to target."""
        if k == 0:
            return current if sum_so_far == target else None
        if sum_so_far > target:
            return None
        if len(current) + k > 30:  # Max 30 terms total
            return None
            
        # Pruning: if remaining sum is too small to be made with k more terms
        # Minimum possible sum of k more terms (all >= 1)
        if sum_so_far + k > target:
            return None
        
        # Try adding terms in increasing order to avoid duplicates
        for idx in range(n_smooth):
            if idx in used_indices:
                continue
            val = smooth[idx]
            if val > target - sum_so_far:
                break
            
            # Check antichain property: new term must not divide or be divisible by any existing term
            valid = True
            for prev_idx in used_indices:
                if val % smooth[prev_idx] == 0 or smooth[prev_idx] % val == 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Time check
            if time.time() > deadline:
                return None
            
            new_used = used_indices | {idx}
            result = backtrack_antichain(target, k - 1, current + [val], sum_so_far + val, new_used)
            if result is not None:
                return result
        
        return None
    
    # Process each target
    for n in range(1, 129):
        if time.time() > deadline:
            break
        
        sol = find_minimal_antichain(n, max_terms=10)
        if sol is not None:
            solutions[n] = sol
    
    # Fallback for any unsolved targets: try with more terms
    for n in range(1, 129):
        if solutions[n] is None:
            if time.time() > deadline:
                break
            sol = find_minimal_antichain(n, max_terms=30)
            if sol is not None:
                solutions[n] = sol
    
    # Final fallback: if still unsolved, use a greedy approach
    for n in range(1, 129):
        if solutions[n] is None:
            # Simple greedy: pick largest smooth numbers that don't conflict
            remaining = n
            chosen = []
            chosen_vals = set()
            for val in sorted(smooth, reverse=True):
                if val > remaining:
                    continue
                # Check antichain
                valid = True
                for prev in chosen:
                    if val % prev == 0 or prev % val == 0:
                        valid = False
                        break
                if valid:
                    chosen.append(val)
                    chosen_vals.add(val)
                    remaining -= val
                    if remaining == 0:
                        break
            if remaining == 0 and len(chosen) >= 1:
                solutions[n] = chosen
    
    return solutions[1:129]
