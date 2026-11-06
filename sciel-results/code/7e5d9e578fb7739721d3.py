import time

def _smooth_numbers_up_to(n):
    """Return sorted list of all 2^a*3^b*5^c <= n (positive)."""
    out = set()
    p2 = 1
    while p2 <= n:
        p23 = p2
        while p23 <= n:
            p235 = p23
            while p235 <= n:
                out.add(p235)
                p235 *= 5
            p23 *= 3
        p2 *= 2
    return sorted(out)

def _is_antichain(terms):
    """No term divides another (terms are distinct positives)."""
    for i in range(len(terms)):
        a = terms[i]
        for j in range(i + 1, len(terms)):
            b = terms[j]
            if a < b:
                if b % a == 0:
                    return False
            else:
                if a % b == 0:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Generate smooth numbers up to 128
    smooth = _smooth_numbers_up_to(128)
    num_smooth = len(smooth)
    
    # 2. Precompute divisibility constraints between smooth numbers
    # div_mask[i] has bit j set if smooth[i] divides smooth[j] (i != j)
    div_mask = [0] * num_smooth
    for i in range(num_smooth):
        for j in range(num_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                div_mask[i] |= (1 << j)
    
    # 3. Use DP to find minimal antichain representations
    # dp[mask] = (min_terms, sum) but we want to minimize terms for each sum
    # Better approach: iterate over all subsets, check if antichain, record sum and size
    
    # For efficiency, we'll do a BFS-like search:
    # For each target, find the smallest k such that there exists an antichain of size k summing to target
    
    # Precompute all valid antichains (subsets where no element divides another)
    # and their sums and sizes, then pick the minimal size for each sum
    
    # Since num_smooth is small (~100), we can use a smarter approach:
    # Iterate k from 1 to max_k, and for each k, find all antichains of size k
    # and record which sums they achieve
    
    # Use a different approach: for each subset, check if it's an antichain
    # But 2^100 is too large. Instead, use the fact that antichains have structure.
    
    # Alternative: use the divisibility poset structure
    # Group smooth numbers by their "type" (exponent vector)
    # Two numbers are comparable if one's exponent vector dominates the other's
    
    # Let's use a direct approach with pruning:
    # For k from 1 to 30:
    #   Find all antichains of size k with sum <= 128
    #   For each sum, if not yet solved, record it
    
    # To make this efficient, use backtracking with pruning
    
    # Actually, let's precompute using a different strategy:
    # For each smooth number, precompute which other smooth numbers it conflicts with
    # Then use a recursive search that builds antichains
    
    best_rep = [None] * 129  # Index 0 unused, 1..128
    solved_count = 0
    
    # For each k, find antichains of size k
    for k in range(1, 31):
        if time.time() > deadline:
            break
        
        # Prune: if sum of k smallest smooth numbers > 128, stop
        if num_smooth < k:
            break
        min_sum_k = sum(smooth[:k])
        if min_sum_k > 128:
            break
        
        # Recursive search for antichains of size k
        # We'll use a list to collect results
        found = {}  # sum -> list of terms (first found, which is minimal k)
        
        def search(start_idx, current, current_sum, remaining):
            nonlocal solved_count
            if time.time() > deadline:
                return
            if len(current) == k:
                if current_sum <= 128 and current_sum >= 1:
                    if current_sum not in found:
                        found[current_sum] = list(current)
                return
            
            if len(current) + remaining < k:
                return
            
            # Try adding each smooth number from start_idx onwards
            for i in range(start_idx, num_smooth):
                val = smooth[i]
                new_sum = current_sum + val
                if new_sum > 128:
                    break  # Since smooth is sorted, all further will be larger
                
                # Check if val is compatible with all elements in current
                compatible = True
                for existing in current:
                    if existing < val:
                        if val % existing == 0:
                            compatible = False
                            break
                    else:
                        if existing % val == 0:
                            compatible = False
                            break
                
                if compatible:
                    current.append(val)
                    search(i + 1, current, new_sum, remaining - 1)
                    current.pop()
        
        search(0, [], 0, k)
        
        # Update best_rep with found antichains
        for s, terms in found.items():
            if best_rep[s] is None:
                best_rep[s] = terms
                solved_count += 1
        
        if solved_count == 128:
            break
    
    # Fallback for any unsolved targets
    for n in range(1, 129):
        if best_rep[n] is None:
            # Try a more exhaustive search for this specific target
            # This should rarely be needed
            pass
    
    return best_rep[1:129]
