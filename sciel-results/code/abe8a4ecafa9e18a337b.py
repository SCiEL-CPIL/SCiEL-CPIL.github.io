import time
import sys

def construct():
    start_time = time.time()
    MAX_TARGET = 128
    TIME_LIMIT = 85  # seconds, leave safety margin

    # 1. Generate all 2-3-5 smooth numbers up to MAX_TARGET
    smooth_numbers = []
    for i in range(1, MAX_TARGET + 1):
        y = i
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth_numbers.append(i)
    
    smooth_numbers.sort()
    N = len(smooth_numbers)
    
    # 2. Pre-compute divisibility relations
    # divs[i] is a bitmask of indices j where smooth_numbers[i] divides smooth_numbers[j]
    # Also need: for each i, which j divide i (i is divisible by j)
    # Let's use: divides[i] = set of j such that smooth_numbers[j] divides smooth_numbers[i]
    # And: is_divided_by[i] = set of j such that smooth_numbers[i] divides smooth_numbers[j]
    
    # For antichain check: no element divides another
    # We'll use a bitmask approach: conflict[i] is a bitmask of all j != i where i divides j or j divides i
    
    conflict = [0] * N
    for i in range(N):
        for j in range(i + 1, N):
            if smooth_numbers[j] % smooth_numbers[i] == 0:
                # i divides j
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
            elif smooth_numbers[i] % smooth_numbers[j] == 0:
                # j divides i (can't happen since i < j and sorted, but just in case)
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)
    
    # 3. For each target, find minimal antichain representation
    # Strategy: BFS/DFS with pruning, trying smaller sizes first
    
    # Pre-compute: for each smooth number, its value
    # We want to find for each target t, a subset of smooth numbers that:
    #   - sums to t
    #   - forms an antichain (no element divides another)
    #   - has minimal size
    
    # Approach: For each possible size k = 1, 2, 3, ..., try to find representations
    # Use dynamic programming or backtracking
    
    # Let's use a different approach: 
    # For each target, try to find minimal k such that there exists an antichain of size k summing to target
    
    # Pre-compute all antichains of size 1, 2, 3, 4, 5 with sum <= MAX_TARGET
    # This is feasible since N is small (~38)
    
    # Better approach: use bitmask DP
    # State: (sum_used, last_index, conflict_mask) -> but this is too large
    
    # Let's use a smarter approach:
    # Generate all valid antichains (as tuples of indices) with sum <= MAX_TARGET
    # Group them by (size, sum)
    # For each target, pick the smallest size
    
    # To generate antichains efficiently, use backtracking with pruning
    
    antichain_solutions = {}  # (size, sum) -> list of tuples of indices
    
    # We'll generate antichains of increasing size
    # Use recursive backtracking
    
    def generate_antichains(start_idx, current_indices, current_sum, current_conflict_mask):
        """
        Generate all antichains starting from start_idx.
        current_conflict_mask: bitmask of all indices that conflict with any element in current_indices
        """
        if time.time() - start_time > TIME_LIMIT - 5:
            return
        
        if current_sum > MAX_TARGET:
            return
        
        if current_indices:
            size = len(current_indices)
            key = (size, current_sum)
            if key not in antichain_solutions:
                antichain_solutions[key] = []
            antichain_solutions[key].append(tuple(current_indices))
        
        # Try adding each smooth number from start_idx onwards
        for i in range(start_idx, N):
            if current_sum + smooth_numbers[i] > MAX_TARGET:
                break
            
            # Check if i conflicts with any element in current_indices
            if current_conflict_mask & (1 << i):
                continue
            
            # Add i to current_indices
            new_conflict_mask = current_conflict_mask | conflict[i]
            current_indices.append(i)
            generate_antichains(i + 1, current_indices, current_sum + smooth_numbers[i], new_conflict_mask)
            current_indices.pop()
        
        # Safety check for time
        if time.time() - start_time > TIME_LIMIT - 5:
            return
    
    # Start generating antichains
    generate_antichains(0, [], 0, 0)
    
    # 4. For each target, find minimal size representation
    best_size = [float('inf')] * (MAX_TARGET + 1)
    best_rep = [None] * (MAX_TARGET + 1)
    
    # Sort antichain_solutions by size, then by sum
    sizes = sorted(set(k[0] for k in antichain_solutions.keys()))
    
    for size in sizes:
        if time.time() - start_time > TIME_LIMIT - 3:
            break
        
        # Check if all targets are found
        all_found = all(best_size[t] != float('inf') for t in range(1, MAX_TARGET + 1))
        if all_found:
            break
        
        # Get all sums for this size
        sums_for_size = sorted([k[1] for k in antichain_solutions.keys() if k[0] == size])
        
        for s in sums_for_size:
            if s < 1 or s > MAX_TARGET:
                continue
            if best_size[s] != float('inf'):
                continue
            
            # Pick the first (or any) antichain of this size summing to s
            for indices in antichain_solutions[(size, s)]:
                best_size[s] = size
                best_rep[s] = [smooth_numbers[i] for i in indices]
                break
    
    # 5. Check if all targets have representations
    missing = [t for t in range(1, MAX_TARGET + 1) if best_rep[t] is None]
    
    if missing:
        # Fallback: try a more exhaustive search for missing targets
        # This should be rare, but let's handle it
        for t in missing:
            if time.time() - start_time > TIME_LIMIT - 1:
                break
            
            # Try to find any antichain summing to t
            # Use a different approach: try all subsets of smooth numbers that sum to t
            # and check if they form an antichain
            
            # Filter smooth numbers that are <= t
            candidates = [i for i in range(N) if smooth_numbers[i] <= t]
            
            # Try sizes from 1 to len(candidates)
            found = False
            for size in range(1, min(len(candidates), 10) + 1):
                if time.time() - start_time > TIME_LIMIT - 1:
                    break
                
                # Try all combinations of size from candidates
                # This is expensive, so use a smarter approach
                # Use backtracking with pruning
                
                def backtrack(start, current, current_sum, current_mask):
                    nonlocal found
                    if found:
                        return
                    if time.time() - start_time > TIME_LIMIT - 1:
                        return
                    
                    if len(current) == size:
                        if current_sum == t:
                            # Check antichain
                            for i in range(len(current)):
                                for j in range(i + 1, len(current)):
                                    if conflict[current[i]] & (1 << current[j]):
                                        return
                            best_rep[t] = [smooth_numbers[i] for i in current]
                            best_size[t] = size
                            found = True
                            return
                        return
                    
                    remaining = size - len(current)
                    for i in range(start, len(candidates)):
                        if len(candidates) - i < remaining:
                            break
                        idx = candidates[i]
                        if current_sum + smooth_numbers[idx] > t:
                            continue
                        if current_mask & (1 << idx):
                            continue
                        
                        current.append(idx)
                        backtrack(i + 1, current, current_sum + smooth_numbers[idx], current_mask | conflict[idx])
                        current.pop()
                        if found:
                            return
                
                backtrack(0, [], 0, 0)
                if found:
                    break
    
    # 6. Construct final answer
    answers = []
    for t in range(1, MAX_TARGET + 1):
        if best_rep[t] is not None:
            answers.append(best_rep[t])
        else:
            # This should not happen, but provide a fallback
            # Use a single term if possible
            if t in smooth_numbers:
                idx = smooth_numbers.index(t)
                answers.append([t])
            else:
                # Emergency fallback: use two terms if possible
                found_pair = False
                for i in range(N):
                    for j in range(i + 1, N):
                        if smooth_numbers[i] + smooth_numbers[j] == t:
                            if not (conflict[i] & (1 << j)):
                                answers.append([smooth_numbers[i], smooth_numbers[j]])
                                found_pair = True
                                break
                    if found_pair:
                        break
                if not found_pair:
                    # Last resort: just use [t] even if not valid (should not happen)
                    answers.append([t])
    
    return answers
