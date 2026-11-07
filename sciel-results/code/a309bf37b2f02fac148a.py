import math
from itertools import combinations

def construct():
    N = 256
    
    # Small primes: those p where p^2 <= N or p is small enough to form composites
    # Actually, any prime p > 16 cannot form a composite with another prime > 16.
    # And p * q <= 256 implies if p > 16, q < 16.
    # So "Large Primes" are those > 16.
    # "Small Primes" are those <= 16: 2, 3, 5, 7, 11, 13.
    
    small_primes = [2, 3, 5, 7, 11, 13]
    large_primes = [p for p in range(17, N + 1) if all(p % i != 0 for i in range(2, int(math.sqrt(p)) + 1))]
    
    # Generate all valid small cores k composed of small primes
    # k must be <= 256
    cores = []
    def gen_cores(idx, current_val, current_mask):
        if idx == len(small_primes):
            if current_val <= N:
                cores.append((current_val, current_mask))
            return
        p = small_primes[idx]
        # Option 1: Don't use p
        gen_cores(idx + 1, current_val, current_mask)
        # Option 2: Use p^k
        power = 1
        while power * p <= N:
            power *= p
            gen_cores(idx + 1, current_val * power, current_mask | (1 << idx))
            if current_val * power > N:
                break
        # Wait, the loop above is slightly inefficient.
        # Let's just iterate powers.
        
    # Reset and implement correctly
    cores = []
    def gen_cores_rec(idx, current_val):
        if idx == len(small_primes):
            if current_val <= N:
                cores.append(current_val)
            return
        p = small_primes[idx]
        # Try 0, 1, 2... powers of p
        power = 1
        while current_val * power <= N:
            gen_cores_rec(idx + 1, current_val * power)
            power *= p
            if power > N: break
            if current_val * power > N: break
            
    gen_cores_rec(0, 1)
    
    # Remove duplicates and sort
    cores = list(set(cores))
    cores.sort(reverse=True)
    
    # Precompute prime mask for each core to check coprimality
    # Mask bit i corresponds to small_primes[i]
    core_masks = []
    for k in cores:
        mask = 0
        for i, p in enumerate(small_primes):
            if k % p == 0:
                mask |= (1 << i)
        core_masks.append(mask)
        
    # Precompute compatible large primes for each core
    # compatible_large_primes[i] is list of large primes p such that p * cores[i] <= N
    compatible_large_primes = []
    for k in cores:
        max_p = N // k
        compatible_large_primes.append([p for p in large_primes if p <= max_p])
        
    best_sum = 1 # Always include 1
    
    # Branch and Bound on selecting pairwise coprime cores
    # State: index in cores list, current_mask, current_sum
    # We iterate through cores in descending order of value.
    
    def solve(idx, current_mask, current_sum):
        nonlocal best_sum
        
        # Pruning: if current_sum + sum of remaining cores (even with max large prime) < best_sum
        # Upper bound calculation:
        # For each remaining core j >= idx, max value is core[j] * max(compatible_large_primes[j]) if available, else core[j].
        # This is an optimistic bound.
        ub = current_sum
        for j in range(idx, len(cores)):
            if (current_mask & core_masks[j]) == 0: # Coprime
                # Add max possible value for this core
                if compatible_large_primes[j]:
                    ub += cores[j] * compatible_large_primes[j][0] # largest prime
                else:
                    ub += cores[j]
            else:
                # If not coprime, we can't pick it, so 0 contribution?
                # Wait, the bound should be sum of max values of ALL remaining, regardless of mask, 
                # to be a valid upper bound (looser but faster to calc).
                # But if mask conflicts, we can't pick.
                # A tighter bound: for each remaining core, if it conflicts, we gain 0.
                # If it doesn't conflict, we gain max_val.
                # This is still an upper bound because it assumes we can pick all non-conflicting ones.
                pass
        
        # Let's use a simpler, tighter bound:
        # Sum of max values of all remaining cores that are coprime to current_mask.
        # This is still an upper bound.
        
        if ub < best_sum:
            return

        if idx == len(cores):
            # We have selected a set of cores. Now we need to assign large primes.
            # The set of selected cores is not explicitly stored, but we can re-derive or pass it.
            # To avoid passing list, let's just calculate the max sum for the current selection?
            # No, we need to know WHICH cores were selected to assign primes.
            # This approach is flawed because the value of a core depends on the prime assigned.
            
            # Better approach:
            # The B&B should track the selected cores.
            # Or, we can pre-compute the "best value" for a set of cores?
            # Since the number of selected cores is small, we can just store the list of indices.
            
            # Let's restart the design of the B&B state.
            pass

    # Revised B&B
    def solve_bnb(idx, current_mask, selected_cores):
        nonlocal best_sum
        
        # Upper Bound
        ub = 0
        for k in selected_cores:
            # Max value for k is k * max_prime_compatible_with_k
            # But primes are shared. So this is not a valid upper bound for the sum.
            # A valid upper bound:
            # Sum of (k * max_possible_prime_for_k) for selected + 
            # Sum of (k_j * max_possible_prime_for_k_j) for remaining coprime k_j.
            # This ignores the constraint that primes must be distinct.
            # It is a valid upper bound.
            pass
            
        # Simpler: Just calculate the exact max sum for the current `selected_cores` + 0 for remaining?
        # No, we need to consider adding more cores.
        
        # Let's use a different strategy.
        # Enumerate all subsets of pairwise coprime cores.
        # The number of such subsets is small.
        # For each subset, calculate the max sum by assigning primes.
        
        pass

    # Strategy: Enumerate all valid subsets of cores.
    # Two cores are compatible if their masks are disjoint.
    
    valid_subsets = []
    
    def enum_subsets(idx, current_mask, current_subset):
        # Pruning: if current_subset is empty, we can still add.
        # If we reach end, add to valid_subsets.
        # To reduce search, we can just iterate all combinations?
        # Number of cores is ~100?
        # Let's check count of cores.
        pass

    # Let's just use a recursive search that builds the subset and calculates the bound.
    
    def search(idx, current_mask, selected_indices):
        nonlocal best_sum
        
        # Calculate upper bound
        # For selected indices, we don't know the exact sum yet (depends on prime assignment).
        # But we can compute a loose bound: sum of max possible values for selected + remaining.
        # Max possible value for core k is k * max_prime_compatible[k].
        
        ub = 0
        for i in selected_indices:
            if compatible_large_primes[i]:
                ub += cores[i] * compatible_large_primes[i][0]
            else:
                ub += cores[i]
                
        for j in range(idx, len(cores)):
            if (current_mask & core_masks[j]) == 0:
                if compatible_large_primes[j]:
                    ub += cores[j] * compatible_large_primes[j][0]
                else:
                    ub += cores[j]
                    
        if ub < best_sum:
            return

        if idx == len(cores):
            # Evaluate this subset
            # We have selected_indices.
            # We need to assign distinct large primes to maximize sum.
            # This is a bipartite matching.
            # Since size is small, use DP or simple recursion.
            
            m = len(selected_indices)
            if m == 0:
                val = 1
            else:
                # Sort selected cores by value descending?
                # We want to assign largest primes to largest cores.
                # Let's just use a simple recursive assignment.
                
                # Get list of (core_index, core_value, compatible_primes)
                items = []
                for i in selected_indices:
                    items.append((cores[i], compatible_large_primes[i]))
                
                # Sort by core value descending to assign largest primes to largest cores?
                # Actually, to maximize sum of products, sort both descending.
                # So sort items by core value descending.
                items.sort(key=lambda x: x[0], reverse=True)
                
                # We need to choose distinct primes.
                # Since m is small, we can use bitmask DP on primes?
                # No, primes are many.
                # We can just pick the top m primes from the union of compatible primes?
                # No, a prime might not be compatible with the top core.
                
                # Greedy might fail. Use simple recursion with memoization on (idx, used_primes_mask)?
                # Used primes mask is too large.
                
                # Let's just use a simple backtracking for assignment.
                # m <= 6.
                
                def assign(idx, used_primes, current_val):
                    if idx == m:
                        return current_val
                    best = -1
                    c_val, c_primes = items[idx]
                    # Option: don't assign
                    best = max(best, assign(idx + 1, used_primes, current_val + c_val))
                    # Option: assign a prime
                    for p in c_primes:
                        if p not in used_primes:
                            new_used = used_primes | {p}
                            val = assign(idx + 1, new_used, current_val + c_val * p)
                            if val > best:
                                best = val
                    return best

                val = assign(0, set(), 0) + 1 # +1 for the number 1
                
            if val > best_sum:
                best_sum = val
            return

        # Branch: Exclude cores[idx]
        search(idx + 1, current_mask, selected_indices)
        
        # Branch: Include cores[idx] if coprime
        if (current_mask & core_masks[idx]) == 0:
            new_mask = current_mask | core_masks[idx]
            search(idx + 1, new_mask, selected_indices + [idx])

    search(0, 0, [])
    
    return [1] # Placeholder, need to return the list of ints?
    # The task says "Return a list of ints. The raw value is their sum."
    # But the evaluator measures the sum.
    # "Return a list of ints."
    # I need to reconstruct the list?
    # The prompt says "Return a list of ints."
    # But I only calculated the sum.
    # I need to store the best configuration.

    # Let's modify to store best config.
    
    best_config = None
    
    def search2(idx, current_mask, selected_indices):
        nonlocal best_sum, best_config
        
        ub = 0
        for i in selected_indices:
            if compatible_large_primes[i]:
                ub += cores[i] * compatible_large_primes[i][0]
            else:
                ub += cores[i]
                
        for j in range(idx, len(cores)):
            if (current_mask & core_masks[j]) == 0:
                if compatible_large_primes[j]:
                    ub += cores[j] * compatible_large_primes[j][0]
                else:
                    ub += cores[j]
                    
        if ub < best_sum:
            return

        if idx == len(cores):
            m = len(selected_indices)
            items = []
            for i in selected_indices:
                items.append((cores[i], compatible_large_primes[i]))
            items.sort(key=lambda x: x[0], reverse=True)
            
            def assign2(idx, used_primes, current_val, current_assignments):
                if idx == m:
                    return current_val, current_assignments
                best_val = -1
                best_assign = None
                c_val, c_primes = items[idx]
                
                # Option: don't assign
                val, assigns = assign2(idx + 1, used_primes, current_val + c_val, current_assignments + [c_val])
                if val > best_val:
                    best_val = val
                    best_assign = assigns
                
                # Option: assign a prime
                for p in c_primes:
                    if p not in used_primes:
                        new_used = used_primes | {p}
                        val, assigns = assign2(idx + 1, new_used, current_val + c_val * p, current_assignments + [c_val * p])
                        if val > best_val:
                            best_val = val
                            best_assign = assigns
                            
                return best_val, best_assign

            val, assigns = assign2(0, set(), 0, [])
            total_val = val + 1
            
            if total_val > best_sum:
                best_sum = total_val
                best_config = [1] + assigns
            return

        search2(idx + 1, current_mask, selected_indices)
        
        if (current_mask & core_masks[idx]) == 0:
            new_mask = current_mask | core_masks[idx]
            search2(idx + 1, new_mask, selected_indices + [idx])

    search2(0, 0, [])
    
    return best_config if best_config else [1]

# EVOLVE-BLOCK-END
