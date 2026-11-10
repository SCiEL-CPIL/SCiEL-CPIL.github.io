import time
from math import factorial
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s limit to be safe
    
    N = factorial(12)
    
    # Prime factorization of 12!
    # 12! = 479001600
    # 2: floor(12/2)+floor(12/4)+floor(12/8) = 6+3+1 = 10
    # 3: floor(12/3)+floor(12/9) = 4+1 = 5
    # 5: floor(12/5) = 2
    # 7: floor(12/7) = 1
    # Exponents: (10, 5, 2, 1)
    
    primes = [2, 3, 5, 7]
    max_exps = [10, 5, 2, 1]
    target_state = tuple(max_exps)
    
    # Generate all divisors of N and their exponent vectors
    divisors_info = [] # List of (value, exp_tuple)
    for e2 in range(max_exps[0] + 1):
        for e3 in range(max_exps[1] + 1):
            for e5 in range(max_exps[2] + 1):
                for e7 in range(max_exps[3] + 1):
                    val = (2 ** e2) * (3 ** e3) * (5 ** e5) * (7 ** e7)
                    if val > 12:
                        divisors_info.append((val, (e2, e3, e5, e7)))
    
    # Sort divisors by value to allow binary search on max factor
    divisors_info.sort(key=lambda x: x[0])
    
    # Number of states in DP
    num_states = (max_exps[0] + 1) * (max_exps[1] + 1) * (max_exps[2] + 1) * (max_exps[3] + 1)
    
    # Map state tuple to index
    def state_to_idx(state):
        e2, e3, e5, e7 = state
        return ((e2 * (max_exps[1] + 1) + e3) * (max_exps[2] + 1) + e5) * (max_exps[3] + 1) + e7
    
    def idx_to_state(idx):
        e7 = idx % (max_exps[3] + 1)
        idx //= (max_exps[3] + 1)
        e5 = idx % (max_exps[2] + 1)
        idx //= (max_exps[2] + 1)
        e3 = idx % (max_exps[1] + 1)
        e2 = idx // (max_exps[1] + 1)
        return (e2, e3, e5, e7)
    
    target_idx = state_to_idx(target_state)
    
    # Feasibility check for a given max factor M
    # Returns True if we can form N with distinct factors > 12 and <= M, using <= 12 factors
    def is_feasible(M):
        if time.time() > deadline:
            return False
            
        # Get candidates: divisors d such that 12 < d <= M
        candidates = []
        for val, exp in divisors_info:
            if val <= M:
                candidates.append((val, exp))
            else:
                break
                
        if not candidates:
            return False
            
        # DP: dp[idx] = min number of factors to reach state idx
        # We use a list of size num_states
        # Initialize with infinity
        INF = 1000
        dp = [INF] * num_states
        dp[0] = 0 # State (0,0,0,0) requires 0 factors
        
        # Process each candidate
        # To allow backtracking later, we might need to store history, 
        # but for binary search we just need feasibility.
        # For the final solution, we will re-run with history.
        
        for val, exp in candidates:
            exp_idx = state_to_idx(exp)
            # Update dp in reverse order of indices? 
            # No, standard 0/1 knapsack requires careful iteration to avoid reusing items.
            # Since state space is small, we can copy dp or iterate carefully.
            # Iterating all states for each candidate is O(NumCandidates * NumStates).
            # NumCandidates <= 396, NumStates = 396. 396*396 = 156k ops.
            # This is fast.
            
            # We need to update dp[new_state] from dp[old_state]
            # To avoid using the same candidate twice, we should use a copy of dp for the current step
            # or iterate in a way that prevents reuse. 
            # Since the "weight" is the exponent vector, and we are adding to it, 
            # simply iterating states 0..N-1 and updating dp[state+exp] might use the same item multiple times 
            # if state+exp is also a valid state that gets processed later in the same loop.
            # To be safe and simple, use a temporary array or copy.
            new_dp = dp[:]
            
            for i in range(num_states):
                if dp[i] == INF:
                    continue
                # Try to add this candidate to state i
                state = idx_to_state(i)
                new_state = (
                    state[0] + exp[0],
                    state[1] + exp[1],
                    state[2] + exp[2],
                    state[3] + exp[3]
                )
                # Check bounds
                if new_state[0] > max_exps[0] or new_state[1] > max_exps[1] or \
                   new_state[2] > max_exps[2] or new_state[3] > max_exps[3]:
                    continue
                
                new_idx = state_to_idx(new_state)
                if dp[i] + 1 < new_dp[new_idx]:
                    new_dp[new_idx] = dp[i] + 1
            
            dp = new_dp
            
        return dp[target_idx] <= 12

    # Binary search for minimum M
    lo = 13
    hi = N
    best_M = N
    feasible = False
    
    # Ensure hi is feasible (N is always feasible with 1 factor)
    # We know N is a divisor > 12.
    
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        if is_feasible(mid):
            best_M = mid
            feasible = True
            hi = mid - 1
        else:
            lo = mid + 1
            
    if not feasible:
        # Fallback to single factor
        return [N]
        
    # Now find the actual factors for best_M
    # Re-run DP storing history for backtracking
    candidates = []
    for val, exp in divisors_info:
        if val <= best_M:
            candidates.append((val, exp))
        else:
            break
            
    # dp[i][state_idx] = min factors using first i candidates
    # To save memory and time, we can store prev_state and prev_candidate_index
    # But simpler: store the full dp table for each step? 
    # 396 steps * 396 states * 4 bytes = ~600KB. Very small.
    
    # Let's use a list of arrays for each step
    # dp_table[step][state_idx]
    dp_table = []
    # Initialize step 0 (no candidates used)
    init_dp = [1000] * num_states
    init_dp[0] = 0
    dp_table.append(init_dp)
    
    for val, exp in candidates:
        prev_dp = dp_table[-1]
        curr_dp = prev_dp[:]
        exp_idx = state_to_idx(exp)
        
        for i in range(num_states):
            if prev_dp[i] == 1000:
                continue
            state = idx_to_state(i)
            new_state = (
                state[0] + exp[0],
                state[1] + exp[1],
                state[2] + exp[2],
                state[3] + exp[3]
            )
            if new_state[0] > max_exps[0] or new_state[1] > max_exps[1] or \
               new_state[2] > max_exps[2] or new_state[3] > max_exps[3]:
                continue
            
            new_idx = state_to_idx(new_state)
            if prev_dp[i] + 1 < curr_dp[new_idx]:
                curr_dp[new_idx] = prev_dp[i] + 1
        
        dp_table.append(curr_dp)
        
    # Backtrack to find factors
    # We are at step = len(candidates), state = target_idx
    factors = []
    current_state_idx = target_idx
    current_step = len(candidates)
    
    # We need to check if the target was reached
    if dp_table[current_step][current_state_idx] > 12:
        # Should not happen if is_feasible returned True
        return [N]
        
    while current_step > 0:
        prev_step = current_step - 1
        prev_dp = dp_table[prev_step]
        curr_dp = dp_table[current_step]
        
        if curr_dp[current_state_idx] == prev_dp[current_state_idx]:
            # We did not use candidate current_step-1
            current_step = prev_step
            continue
        else:
            # We used candidate current_step-1
            val, exp = candidates[current_step - 1]
            factors.append(val)
            
            # Find the previous state
            # curr_dp[current_state_idx] = prev_dp[prev_state_idx] + 1
            # So we need to find prev_state_idx such that prev_dp[prev_state_idx] == curr_dp[current_state_idx] - 1
            # And state(prev_state_idx) + exp == state(current_state_idx)
            
            target_prev_count = curr_dp[current_state_idx] - 1
            current_state = idx_to_state(current_state_idx)
            prev_state = (
                current_state[0] - exp[0],
                current_state[1] - exp[1],
                current_state[2] - exp[2],
                current_state[3] - exp[3]
            )
            prev_state_idx = state_to_idx(prev_state)
            
            current_state_idx = prev_state_idx
            current_step = prev_step
            
    # factors is the list of selected divisors
    # Verify product
    prod = 1
    for f in factors:
        prod *= f
        
    if prod == N and all(f > 12 for f in factors) and len(set(factors)) == len(factors) and len(factors) <= 12:
        return factors
    else:
        # Fallback
        return [N]

if __name__ == "__main__":
    construct()
