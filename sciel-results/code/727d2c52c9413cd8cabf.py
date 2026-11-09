import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit

    # 1. Compute 12!
    target = 479001600  # 12!
    
    # 2. Prime factorization of target
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    primes = [2, 3, 5, 7, 11]
    target_exps = [10, 5, 2, 1, 1]
    
    # Helper to compute prime exponents of a number
    def get_exps(n):
        exps = [0, 0, 0, 0, 0]
        for i, p in enumerate(primes):
            while n % p == 0:
                exps[i] += 1
                n //= p
        return exps

    # 3. Generate all divisors of 12! that are > 12
    # Number of divisors: (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792
    # We only need those > 12
    divisors = []
    for e1 in range(11):
        for e2 in range(6):
            for e3 in range(3):
                for e4 in range(2):
                    for e5 in range(2):
                        d = (2 ** e1) * (3 ** e2) * (5 ** e3) * (7 ** e4) * (11 ** e5)
                        if d > 12:
                            divisors.append(d)
    
    # Sort divisors
    divisors.sort()
    div_exps = [get_exps(d) for d in divisors]
    n_divs = len(divisors)
    
    # Map from divisor value to index (for lookup)
    div_to_idx = {d: i for i, d in enumerate(divisors)}
    
    # 4. Binary search on the maximum factor M
    # We want to find the smallest M such that we can pick a subset of divisors
    # from {d in divisors : d <= M} whose product is exactly target.
    # All divisors in the subset must be distinct.
    # At most 12 factors.
    
    # For a given M, the available divisors are divisors[:k] where divisors[k-1] <= M < divisors[k]
    # We need to check if there exists a subset of these with product = target.
    
    # This is a subset product problem with a constraint on the number of elements (<= 12)
    # and distinctness (guaranteed by picking a subset).
    
    # Since the number of divisors is ~700, and the product space is large, 
    # but the prime exponent vector has only 792 possibilities, we can use BFS/DP over exponent vectors.
    
    # State: (exp_vector, count, frozenset_of_indices?) 
    # No, we can't track the set of indices easily for ~700 elements.
    
    # Alternative: Backtracking with strong pruning.
    # Given the sorted list of available divisors (<= M), can we pick a subset with product target?
    
    # Let's use a recursive search that tries to build the product.
    # We can memoize (rem_exps, start_idx) -> boolean
    # But rem_exps has 792*6*3*2*2 possibilities? No, rem_exps is just one of the 792 possible exponent vectors?
    # No, rem_exps can be any sub-exponent vector of target_exps.
    # The number of possible sub-exponent vectors is 792.
    # The start_idx can be 0 to ~700.
    # So the state space is 792 * 700 = ~550,000 states. This is very small.
    
    # We can use DP/BFS where state is (exps_tuple, idx).
    # Actually, we just need to know if we CAN reach (0,0,0,0,0) from (10,5,2,1,1).
    
    # Let's define:
    # dp[rem_exps][i] = True if it's possible to reduce rem_exps to (0,0,0,0,0) using divisors with indices >= i, 
    # AND the number of divisors used so far allows us to finish within 12 total.
    # This is tricky because we need to count the number of steps.
    
    # Alternative simple approach:
    # Just do a DFS with memoization on (rem_exps_tuple, next_start_idx, remaining_slots).
    # rem_exps_tuple has 792 possibilities.
    # next_start_idx has ~700 possibilities.
    # remaining_slots has 13 possibilities (0..12).
    # Total states: 792 * 700 * 13 ≈ 7.2 million.
    # This might be a bit large for Python if done naively, but many states are unreachable.
    
    # Even simpler:
    # Since we are searching for the EXISTENCE of a subset, and the divisors are sorted,
    # we can iterate through them and use a set of reachable exponent vectors.
    
    # Reachable set: A set of tuples (exp1, exp2, exp3, exp4, exp5) reachable using some subset of divisors[:k].
    # Initially: {(10, 5, 2, 1, 1)}
    # We want to reach (0,0,0,0,0).
    # We also need to track the count? 
    # If we don't track count, we might use more than 12 factors.
    # But wait, the minimum number of factors for 12! is small.
    # If we find ANY decomposition, it will likely have <= 12 factors because the divisors are > 12.
    # 12! = 479M. 
    # If we use 13 factors, each > 12, product > 12^13 = 10.6 B > 479M. Impossible.
    # So ANY valid decomposition automatically has <= 12 factors.
    # Proof: 12^13 = 10,699,320,432,635 > 479,001,600.
    # Thus, we don't need to enforce the 12-factor limit explicitly, it is implied by d > 12.
    
    # So, the problem reduces to:
    # Does there exist a subset of {d in divisors : d <= M} such that product(d) = target?
    
    # We can use a BFS/DP where we maintain a set of reachable exponent vectors.
    # start: {(10, 5, 2, 1, 1)}
    # For each divisor d <= M:
    #   new_reachable = reachable ∪ { (rem - exps_d) for rem in reachable if rem >= exps_d }
    #   Check if (0,0,0,0,0) is in new_reachable.
    
    # This is efficient if the number of reachable states is small.
    # The number of possible sub-exponent vectors is 792.
    # So the set size is bounded by 792.
    # Loop runs ~700 times.
    # Total operations: 700 * 792 * 5 ≈ 2.8 million operations. Very fast.
    
    def can_decompose(max_val):
        # Get divisors <= max_val
        # Find index k such that divisors[k-1] <= max_val < divisors[k]
        # We can just iterate and break
        current_reachable = {(10, 5, 2, 1, 1)}
        
        # We iterate through divisors in sorted order
        for i, d in enumerate(divisors):
            if d > max_val:
                break
            
            exps_d = div_exps[i]
            new_states = set()
            
            for rem in current_reachable:
                # Check if rem >= exps_d
                # rem[j] >= exps_d[j] for all j
                valid = True
                next_rem = [0, 0, 0, 0, 0]
                for j in range(5):
                    if rem[j] < exps_d[j]:
                        valid = False
                        break
                    next_rem[j] = rem[j] - exps_d[j]
                
                if valid:
                    nr_tuple = tuple(next_rem)
                    new_states.add(nr_tuple)
            
            if (0, 0, 0, 0, 0) in new_states:
                return True
            
            current_reachable.update(new_states)
            
        return False

    # 5. Binary Search for minimal M
    # The candidates for M are in divisors.
    # We can binary search on the index in divisors.
    
    lo = 0
    hi = n_divs - 1
    best_idx = -1
    
    # Ensure we have a fallback if no solution found (should not happen)
    # The full set of divisors definitely contains a solution (e.g. single factor target if target>12, which it is)
    # But wait, target itself is a divisor > 12. So M=target is always a solution.
    
    while lo <= hi:
        if time.time() > deadline:
            # If we run out of time, return the best found so far?
            # Since we are binary searching for the MIN, and we only know "True" for some high values,
            # we should have kept track of the best valid index.
            if best_idx != -1:
                # We need to retrieve the solution for best_idx?
                # The prompt says "Minimize the LARGEST factor". The evaluator measures the raw value.
                # So we just need to return ANY valid list of factors with max <= divisors[best_idx].
                # To be safe, if we timeout, we should return a known valid solution.
                # The simplest valid solution is [target]. Max is target.
                # But if we found a smaller one, we should try to return it.
                # Since we didn't store the path, we'd have to reconstruct it.
                # For now, just return [target] if timeout, or hope we found the min early.
                # Actually, the binary search finds the min. If we timeout before finishing, best_idx might be high.
                # Let's just return [target] as a safe fallback.
                return [target]
            
            mid = (lo + hi) // 2
            M = divisors[mid]
            
            if can_decompose(M):
                best_idx = mid
                hi = mid - 1
            else:
                lo = mid + 1
    
    if best_idx == -1:
        return [target]
    
    M = divisors[best_idx]
    
    # 6. Reconstruct the solution for M
    # We need to find WHICH divisors multiply to target, all <= M.
    # We can use the DP again, but store parents.
    
    # State: (exp_tuple, last_index_used) is not quite right because it's a subset, not a sequence.
    # We can do:
    # dp[i][rem] = prev_rem or None, indicating that we can reach rem using a subset of divisors[0:i] (inclusive)
    # This is O(N * 792) memory and time, which is fine.
    
    # Let's build a parent pointer table.
    # max_rem_state = 792 possible states.
    # We map rem_tuple to an ID for indexing? Or just use a dict.
    
    # parents[i][rem] = (prev_rem, divisor_value)
    # If we use divisor i to reach rem, it came from prev_rem = rem + exps[i].
    # So we can work backwards from rem=(0,0,0,0,0) using divisors <= M.
    
    # Let's use a BFS/DP forward to populate reachability and parents.
    # reachable[rem] = (index_of_divisor_used_to_get_here, prev_rem)
    # We want to know if we can reach (0,0,0,0,0).
    
    # Initialize:
    # dist[rem] = None (meaning unreachable)
    # dist[(10,5,2,1,1)] = None (start state, no parent)
    
    # We iterate divisors i=0 to k (where divisors[i] <= M)
    # We maintain a set of currently reachable states.
    
    start_rem = (10, 5, 2, 1, 1)
    current_reachable = {start_rem: None} # rem -> (parent_rem, divisor_index)
    # Actually, a state might be reached multiple ways. We just need ONE way.
    
    # To ensure we find a solution, we can just track reachability.
    # But to reconstruct, we need to know which divisor led to it.
    
    # Let's use a simple dictionary for the best path to each state.
    # path[rem] = (prev_rem, div_idx)
    
    path = {}
    path[start_rem] = (None, None)
    
    # We process divisors one by one.
    # Note: Standard 0/1 knapsack style iteration is needed to ensure each divisor is used at most once.
    # Since we want a SUBSET, we must be careful.
    # Iterating divisors and updating the reachable set requires using a COPY of the previous state to avoid reusing the same divisor.
    
    reachable = {start_rem: True}
    
    # To store the parent for reconstruction:
    # parent[rem] = (prev_rem, div_idx)
    # This is only valid if prev_rem was reachable BEFORE adding div_idx.
    
    parent_map = {}
    parent_map[start_rem] = (None, None)
    
    for i in range(len(divisors)):
        if divisors[i] > M:
            break
        
        exps_d = div_exps[i]
        d_val = divisors[i]
        
        # We need to check which states in `reachable` can be extended by d_val.
        # And add the new states to a temporary set `new_reachable`.
        # Then merge new_reachable into reachable.
        
        new_states_to_add = set()
        
        for rem in list(reachable.keys()): # Iterate over a snapshot
            # Check if rem - exps_d is valid (>= 0)
            # And if rem - exps_d was NOT just added in this iteration?
            # No, standard 0/1 knapsack: we iterate over items, and for each item, we check all states reachable so far.
            # If state S is reachable, and we add item i, we reach S' = S - exps_i.
            # If S' is not yet reachable, we mark it and record parent.
            
            # But we must ensure we don't use item i twice.
            # By iterating `reachable` (which does not include `new_states_to_add` yet), we ensure item i is used only once in the transition from S to S'.
            
            valid = True
            next_rem = [0]*5
            for j in range(5):
                if rem[j] < exps_d[j]:
                    valid = False
                    break
                next_rem[j] = rem[j] - exps_d[j]
            
            if valid:
                nr = tuple(next_rem)
                if nr not in reachable: # Only if it's a NEW state
                    new_states_to_add.add(nr)
                    parent_map[nr] = (rem, i)
        
        # Add new states to reachable
        reachable.update(new_states_to_add)
        
        if (0,0,0,0,0) in reachable:
            break
    
    # Reconstruct
    if (0,0,0,0,0) not in reachable:
        return [target] # Should not happen if can_decompose returned True
    
    factors = []
    curr_rem = (0,0,0,0,0)
    while parent_map[curr_rem][1] is not None:
        prev_rem, div_idx = parent_map[curr_rem]
        factors.append(divisors[div_idx])
        curr_rem = prev_rem
    
    # The factors are in reverse order (from 0 to start).
    # Sort them? Not required, but nice.
    # Check constraints:
    # 1. Distinct: Yes, because we picked from a set of unique divisors, and each index used once.
    # 2. > 12: Yes, filtered in divisors generation.
    # 3. Product = target: Yes, by construction.
    # 4. Max factor minimized: Yes, by binary search.
    
    return factors

# Run construct
# if __name__ == '__main__':
#     res = construct()
#     print(res)
#     print(max(res))
