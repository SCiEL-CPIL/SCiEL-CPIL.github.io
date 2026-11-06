import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 10s safety margin
    
    N = 256
    # Weights are 1/i. We want to maximize sum.
    # Order vertices by weight descending: 1, 2, 3, ...
    order = list(range(1, N + 1))
    weights = [0.0] * (N + 1)
    for i in range(1, N + 1):
        weights[i] = 1.0 / i
        
    # Precompute suffix sums of weights for the upper bound
    # suffix_sum[i] = sum(weights[order[i]] + ... + weights[order[-1]])
    # Note: order is 1..N.
    suffix_sum = [0.0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[order[i]]
        
    best_score = 0.0
    best_set = set()
    
    # in_S array for O(1) lookup
    in_S = [False] * (N + 1)
    current_sum = 0.0
    
    # To speed up is_valid, we can maintain a list of elements in S
    S_list = []
    
    def is_valid_addition(x):
        # Check if x can be added to current S_list
        # 1. x is middle: a + c = 2x => c = 2x - a
        # 2. x is endpoint: a + x = 2c => c = (a+x)/2
        # 3. a is middle: x + c = 2a => c = 2a - x
        
        for a in S_list:
            # x is middle
            c = 2 * x - a
            if 1 <= c <= N and in_S[c]:
                return False
            
            # x is endpoint (a is middle? No, a is in S, x is new. If a is middle, then x and c are endpoints)
            # a is middle: x + c = 2a => c = 2a - x
            c2 = 2 * a - x
            if 1 <= c2 <= N and in_S[c2]:
                return False
            
            # x is endpoint, a is endpoint? Then midpoint is (x+a)/2.
            # If midpoint is in S, then {a, mid, x} is a 3-AP.
            s = a + x
            if s % 2 == 0:
                mid = s // 2
                if 1 <= mid <= N and in_S[mid]:
                    return False
        return True

    def greedy_construct():
        """Construct a greedy set by adding vertices in order 1..N if valid."""
        nonlocal in_S, S_list, current_sum
        # Reset state
        for i in range(1, N + 1):
            in_S[i] = False
        S_list.clear()
        current_sum = 0.0
        
        for x in range(1, N + 1):
            if is_valid_addition(x):
                in_S[x] = True
                S_list.append(x)
                current_sum += weights[x]
                
        return set(S_list), current_sum

    def steepest_ascent(S_init):
        """Perform steepest ascent local search starting from S_init."""
        nonlocal in_S, S_list, current_sum
        
        # Initialize state from S_init
        for i in range(1, N + 1):
            in_S[i] = False
        S_list.clear()
        current_sum = 0.0
        
        for x in S_init:
            in_S[x] = True
            S_list.append(x)
            current_sum += weights[x]
            
        improved = True
        while improved and time.time() < deadline - 10.0:
            improved = False
            best_gain = 0.0
            best_x_out = -1
            best_x_in = -1
            
            # Try all swaps (x_out, x_in)
            # x_out in S_init, x_in not in S_init
            # We can restrict x_in to be smaller than x_out to ensure gain > 0 usually,
            # but not strictly. 1/x_in > 1/x_out iff x_in < x_out.
            # So we only need to check x_in < x_out for positive gain.
            
            for x_out in S_list:
                if best_gain <= 0:
                    break # Optional optimization: if we found a good move, we could stop early? 
                    # No, steepest ascent requires checking all to find the best.
                    
                for x_in in range(1, x_out):
                    if in_S[x_in]:
                        continue
                    
                    # Check if swapping x_out with x_in is valid
                    # We need to check if x_in can be added to S \ {x_out}
                    # Since x_out is removed, we must ensure no AP is formed with x_in
                    # and any pair in S \ {x_out}.
                    
                    valid = True
                    for a in S_list:
                        if a == x_out:
                            continue
                        # Check if {a, x_in, c} is AP
                        # 1. x_in is middle: c = 2*x_in - a
                        c = 2 * x_in - a
                        if 1 <= c <= N and in_S[c] and c != x_out:
                            valid = False
                            break
                        # 2. a is middle: c = 2*a - x_in
                        c2 = 2 * a - x_in
                        if 1 <= c2 <= N and in_S[c2] and c2 != x_out:
                            valid = False
                            break
                        # 3. a is endpoint, x_in is endpoint: mid = (a+x_in)/2
                        s = a + x_in
                        if s % 2 == 0:
                            mid = s // 2
                            if 1 <= mid <= N and in_S[mid] and mid != x_out:
                                valid = False
                                break
                            
                    if valid:
                        gain = weights[x_in] - weights[x_out]
                        if gain > best_gain:
                            best_gain = gain
                            best_x_out = x_out
                            best_x_in = x_in
            
            if best_gain > 1e-9:
                # Apply the best swap
                in_S[best_x_out] = False
                S_list.remove(best_x_out)
                current_sum -= weights[best_x_out]
                
                in_S[best_x_in] = True
                S_list.append(best_x_in)
                current_sum += weights[best_x_in]
                
                improved = True
            else:
                improved = False
                
        return set(S_list), current_sum

    # 1. Generate initial greedy set and improve it with local search
    greedy_set, greedy_score = greedy_construct()
    improved_set, improved_score = steepest_ascent(greedy_set)
    
    # Update best
    if improved_score > best_score:
        best_score = improved_score
        best_set = improved_set

    # 2. Run B&B with the improved initial bound
    def solve(idx):
        nonlocal best_score, best_set, current_sum
        
        if time.time() > deadline - 5.0:
            return
            
        # Pruning
        if current_sum + suffix_sum[idx] <= best_score:
            return
            
        if idx == N:
            if current_sum > best_score:
                best_score = current_sum
                best_set = set(S_list)
            return
            
        x = order[idx]
        
        # Option 1: Include x
        if is_valid_addition(x):
            # Add x
            in_S[x] = True
            S_list.append(x)
            current_sum += weights[x]
            
            solve(idx + 1)
            
            # Remove x
            current_sum -= weights[x]
            S_list.pop()
            in_S[x] = False
            
        # Option 2: Exclude x
        solve(idx + 1)

    # Reset state for B&B to start from scratch or from greedy?
    # B&B explores the space. It's best to start with empty set and let it find the max.
    # But we have a good lower bound `best_score` now.
    # We should reset in_S and S_list to empty for the B&B root.
    for i in range(1, N + 1):
        in_S[i] = False
    S_list.clear()
    current_sum = 0.0
    
    solve(0)
    
    # 3. Polish the B&B result with local search again
    if best_set:
        final_set, final_score = steepest_ascent(best_set)
        if final_score > best_score:
            best_score = final_score
            best_set = final_set
    else:
        # Fallback if B&B failed completely (unlikely)
        final_set = greedy_set

    return list(final_set)
