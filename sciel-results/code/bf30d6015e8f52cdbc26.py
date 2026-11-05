"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s kill
    
    N = 256
    
    # Precompute reciprocal weights
    recip_float = [0.0] * (N + 1)
    recip_frac = [Fraction(0, 1)] * (N + 1)
    for i in range(1, N + 1):
        recip_float[i] = 1.0 / i
        recip_frac[i] = Fraction(1, i)
    
    def is_valid_addition(S, x):
        """
        Check if x can be added to set S without forming a 3-AP.
        S is a list or set of integers.
        """
        S_set = S if isinstance(S, set) else set(S)
        # Case 1: x is the middle term. a, x, c -> c = 2x - a
        # Need a, c in S.
        # Case 2: x is an end term. a, b, x -> b = (a+x)/2
        # Need a, b in S.
        
        # Efficient check:
        # For each a in S:
        # 1. If 2*x - a in S and 2*x - a != x: Invalid (x is middle)
        # 2. If (a+x) % 2 == 0 and (a+x)//2 in S and (a+x)//2 != a and (a+x)//2 != x: Invalid (x is end)
        
        for a in S_set:
            # Check if x is middle: a, x, c
            c = 2 * x - a
            if c in S_set and c != x:
                return False
            
            # Check if x is end: a, b, x or x, b, a
            # b = (a + x) / 2
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if b in S_set and b != a and b != x:
                    return False
                    
        return True

    def is_valid_set(S):
        """
        Verify the entire set S has no 3-AP.
        """
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a >= b:
                    continue
                # Check a, b, c
                c = 2 * b - a
                if c > b and c in S_set:
                    return False
                # Check a, c, b
                if (a + b) % 2 == 0:
                    c = (a + b) // 2
                    if c > a and c < b and c in S_set:
                        return False
                # Check c, a, b
                c = 2 * a - b
                if c < a and c >= 1 and c in S_set:
                    return False
        return True

    # 1. Greedy Initialization
    # Sort by weight descending (smaller numbers first)
    order = list(range(1, N + 1))
    order.sort(key=lambda x: -recip_float[x])
    
    best_S = set()
    best_weight = 0.0
    
    # Standard Greedy
    S = set()
    for x in order:
        if is_valid_addition(S, x):
            S.add(x)
    
    current_weight = sum(recip_float[x] for x in S)
    if current_weight > best_weight:
        best_weight = current_weight
        best_S = S.copy()
    
    # 2. Local Search
    # We will perform iterations of:
    # - Try adding any valid element
    # - Try swapping any element u in S with any element v not in S
    # - Try 2-for-1 swaps (remove u1, u2, add v)
    # - Try 1-for-2 swaps (remove u, add v1, v2)
    
    current_S = best_S.copy()
    current_weight = best_weight
    
    # Precompute list of all numbers for iteration
    all_nums = list(range(1, N + 1))
    
    max_iterations = 10000
    for _ in range(max_iterations):
        if time.time() > DEADLINE:
            break
            
        improved = False
        
        # A. Try Additions
        # Iterate through candidates not in current_S
        # To speed up, check smaller numbers first as they have higher weight
        for x in all_nums:
            if x in current_S:
                continue
            if is_valid_addition(current_S, x):
                current_S.add(x)
                current_weight += recip_float[x]
                if current_weight > best_weight:
                    best_weight = current_weight
                    best_S = current_S.copy()
                improved = True
                # Restart additions to find more
                break 
        
        if improved:
            continue # Restart the loop to re-evaluate additions
            
        # B. Try Swaps (1-for-1)
        # For each u in S, for each v not in S:
        # Check if S \ {u} U {v} is valid.
        # Optimization: Only try v where recip_float[v] > recip_float[u]
        
        found_swap = False
        elements_in_set = list(current_S)
        
        for u in elements_in_set:
            # Try replacing u with v
            # v should ideally be smaller than u to gain weight
            for v in range(1, u):
                if v in current_S:
                    continue
                
                # Check validity of S' = S \ {u} U {v}
                # Create a temporary set without u
                temp_S = set(current_S)
                temp_S.remove(u)
                
                if is_valid_addition(temp_S, v):
                    new_weight = current_weight - recip_float[u] + recip_float[v]
                    if new_weight > current_weight + 1e-12:
                        current_S = temp_S
                        current_S.add(v)
                        current_weight = new_weight
                        if current_weight > best_weight:
                            best_weight = current_weight
                            best_S = current_S.copy()
                        found_swap = True
                        break # Restart swap search
            if found_swap:
                break
        
        if found_swap:
            continue
            
        # C. Try 2-for-1 Swaps (Remove u1, u2, Add v)
        # This is more expensive. O(N^3) roughly.
        # We can limit this to a few iterations or only if time is left.
        
        if time.time() < DEADLINE - 5:
            elements_in_set = list(current_S)
            found_2_1 = False
            
            for i in range(len(elements_in_set)):
                if time.time() > DEADLINE - 2:
                    break
                u1 = elements_in_set[i]
                for j in range(i + 1, len(elements_in_set)):
                    u2 = elements_in_set[j]
                    
                    temp_S = set(current_S)
                    temp_S.remove(u1)
                    temp_S.remove(u2)
                    weight_loss = recip_float[u1] + recip_float[u2]
                    
                    # Look for best v
                    best_v = -1
                    best_v_w = 0.0
                    for v in all_nums:
                        if v in current_S:
                            continue
                        # Optimization: if w[v] <= weight_loss, no point
                        if recip_float[v] <= weight_loss:
                            continue
                        if is_valid_addition(temp_S, v):
                            if recip_float[v] > best_v_w:
                                best_v_w = recip_float[v]
                                best_v = v
                                
                    if best_v != -1:
                        new_weight = current_weight - weight_loss + best_v_w
                        if new_weight > current_weight + 1e-12:
                            current_S = temp_S
                            current_S.add(best_v)
                            current_weight = new_weight
                            if current_weight > best_weight:
                                best_weight = current_weight
                                best_S = current_S.copy()
                            found_2_1 = True
                            break
                if found_2_1:
                    break
                    
            if found_2_1:
                continue

        # D. Try 1-for-2 Swaps (Remove u, Add v1, v2)
        # Similar logic.
        if time.time() < DEADLINE - 10:
            elements_in_set = list(current_S)
            found_1_2 = False
            
            for u in elements_in_set:
                if time.time() > DEADLINE - 5:
                    break
                temp_S = set(current_S)
                temp_S.remove(u)
                weight_loss = recip_float[u]
                
                # Find two numbers v1, v2 not in current_S
                # such that v1, v2 can be added to temp_S and w[v1]+w[v2] > weight_loss
                # This is O(N^2) per u.
                candidates = [v for v in all_nums if v not in current_S]
                # Sort by weight desc
                candidates.sort(key=lambda x: -recip_float[x])
                
                found_pair = False
                for v1 in candidates:
                    if recip_float[v1] <= weight_loss: # Even with 0, can't win
                        break
                    # Check if v1 can be added to temp_S
                    if not is_valid_addition(temp_S, v1):
                        continue
                    
                    temp_S_with_v1 = set(temp_S)
                    temp_S_with_v1.add(v1)
                    
                    for v2 in candidates:
                        if v2 == v1:
                            continue
                        # Optimization: if w[v1]+w[v2] <= weight_loss, break inner loop?
                        # Since sorted desc, if w[v2] is small, further v2 are smaller.
                        if recip_float[v1] + recip_float[v2] <= weight_loss:
                            break
                        if v2 in current_S:
                            continue
                        if is_valid_addition(temp_S_with_v1, v2):
                            new_weight = current_weight - weight_loss + recip_float[v1] + recip_float[v2]
                            if new_weight > current_weight + 1e-12:
                                current_S = temp_S_with_v1
                                current_S.add(v2)
                                current_weight = new_weight
                                if current_weight > best_weight:
                                    best_weight = current_weight
                                    best_S = current_S.copy()
                                found_pair = True
                                found_1_2 = True
                                break
                    if found_pair:
                        break
                if found_1_2:
                    break
            
            if found_1_2:
                continue

        # If no improvement found in A, B, C, D, we are at a local optimum for these moves.
        # Break to avoid wasting time.
        break

    # Final Result
    # Extract numbers from best_S
    result = sorted(list(best_S))
    
    # Final verification (just in case)
    if not is_valid_set(result):
        # Fallback to simple greedy if something went wrong
        result = []
        S_set = set()
        for x in range(1, N + 1):
            if is_valid_addition(S_set, x):
                result.append(x)
                S_set.add(x)

    if not result:
        result = [1]
        
    return result

# EVOLVE-BLOCK-END
