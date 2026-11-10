import time
from math import gcd
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit

    N = 256
    values = list(range(2, N + 1))
    size = len(values)  # 255 elements: 2 to 256
    
    # Map value to index for quick lookup if needed, but we work with indices
    # indices 0..254 correspond to values 2..256
    
    # Precompute conflict masks
    # conflict_mask[i] is a bitmask where bit j is 1 if values[i] and values[j] are INCOMPATIBLE
    conflict_mask = [0] * size
    for i in range(size):
        vi = values[i]
        for j in range(i + 1, size):
            vj = values[j]
            g = gcd(vi, vj)
            lcm_val = vi * vj // g
            if lcm_val <= N:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)
                
    # Precompute weights as Fractions for exact arithmetic
    weights = [Fraction(1, values[i]) for i in range(size)]
    
    # Helper to check if element i is compatible with set represented by mask
    def is_compatible(i, mask):
        return (conflict_mask[i] & mask) == 0

    def get_score(mask):
        total = Fraction(0, 1)
        for i in range(size):
            if mask & (1 << i):
                total += weights[i]
        return total

    def greedy_construct(order):
        S_mask = 0
        for i in order:
            if is_compatible(i, S_mask):
                S_mask |= (1 << i)
        return S_mask

    def local_search(S_mask, time_limit):
        current_score = get_score(S_mask)
        in_set = [False] * size
        for i in range(size):
            if S_mask & (1 << i):
                in_set[i] = True
        
        # List of indices in set for faster iteration
        set_indices = [i for i in range(size) if in_set[i]]
        
        improved = True
        while improved:
            improved = False
            
            # 1. Try adding one element
            for v in range(size):
                if time.time() > time_limit:
                    return S_mask, current_score
                if in_set[v]:
                    continue
                if is_compatible(v, S_mask):
                    S_mask |= (1 << v)
                    in_set[v] = True
                    set_indices.append(v)
                    current_score += weights[v]
                    improved = True
            
            # 2. Try 1-swap (remove u, add v)
            # To speed up, we can iterate u in set_indices
            # and v in range(size)
            # We want weights[v] > weights[u]
            # Since weights are decreasing with index, v should have smaller index than u
            for u in list(set_indices): # Copy list as we might remove
                if time.time() > time_limit:
                    return S_mask, current_score
                # Only consider v with higher weight (smaller value/index)
                # weights[v] > weights[u] implies values[v] < values[u] implies v < u
                for v in range(u): 
                    if in_set[v]:
                        continue
                    # Check if v is compatible with S \ {u}
                    # S_without_u = S_mask & ~(1 << u)
                    if is_compatible(v, S_mask & ~(1 << u)):
                        gain = weights[v] - weights[u]
                        if gain > 0:
                            S_mask = (S_mask & ~(1 << u)) | (1 << v)
                            in_set[u] = False
                            in_set[v] = True
                            set_indices.remove(u)
                            set_indices.append(v)
                            current_score += gain
                            improved = True
                            break # Restart loops to reflect new state? 
                            # Or just continue? Breaking and restarting is safer for correctness of indices
                            # But might be slow. Let's just mark improved and break inner, 
                            # then the outer while loop will restart.
            
            # 3. Try 1-for-2 swap (remove u, add v, w)
            # This allows replacing a large element with two smaller ones
            # Condition: v, w compatible with S\{u} and with each other.
            # Gain: weights[v] + weights[w] - weights[u] > 0
            for u in list(set_indices):
                if time.time() > time_limit:
                    return S_mask, current_score
                S_without_u = S_mask & ~(1 << u)
                # We need to find v, w not in S, v < w, compatible with S_without_u and each other
                # And weights[v] + weights[w] > weights[u]
                # Since weights are 1/value, smaller values have higher weight.
                # So we look for small v, w.
                
                # To avoid O(n^3) which is too slow, we can limit search or be smarter.
                # n=255. n^3 is ~16M. If we do this a few times, it's okay in Python if optimized?
                # Maybe too slow.
                # Let's restrict v, w to indices < u? Not necessarily, but likely.
                # Actually, if v > u, weight[v] < weight[u]. It's hard to gain.
                # Let's try v, w in range(0, size).
                
                # Optimization: Precompute list of candidates not in set
                # Candidates are those not in S.
                # Let's just iterate.
                
                found_pair = False
                # Sort candidates by weight descending (index ascending)
                # Candidates: not in_set
                # We want max weight pair.
                # Iterate v from 0 to size-1
                for v in range(size):
                    if in_set[v]:
                        continue
                    if not is_compatible(v, S_without_u):
                        continue
                    # Now find w > v such that w compatible with S_without_u and v
                    # And weights[v] + weights[w] > weights[u]
                    # weights[w] > weights[u] - weights[v]
                    # Since weights are decreasing, we can break if weights[v] + weights[v+1] <= weights[u] ?
                    # No, weights[w] decreases as w increases.
                    
                    for w in range(v + 1, size):
                        if time.time() > time_limit:
                            return S_mask, current_score
                        if in_set[w]:
                            continue
                        # Check if sum of weights is better
                        # weights[v] + weights[w] > weights[u]
                        if weights[v] + weights[w] <= weights[u]:
                            break # Since weights[w] decreases, further w will be worse
                        # Check compatibility
                        if not is_compatible(w, S_without_u):
                            continue
                        if not is_compatible(w, 1 << v): # w and v compatible
                            continue
                        
                        # Perform swap
                        S_mask = S_without_u | (1 << v) | (1 << w)
                        in_set[u] = False
                        in_set[v] = True
                        in_set[w] = True
                        set_indices.remove(u)
                        set_indices.append(v)
                        set_indices.append(w)
                        current_score += weights[v] + weights[w] - weights[u]
                        improved = True
                        found_pair = True
                        break
                    if found_pair:
                        break
                if found_pair:
                    break

            # 4. Try 2-for-1 swap (remove u, v, add w)
            # Remove two elements, add one.
            # Gain: weights[w] - weights[u] - weights[v] > 0
            # This is useful if u, v are large (low weight) and w is small (high weight).
            # w must be compatible with S \ {u, v}.
            # Since w is likely small, and u,v large, w might have been blocked by u or v.
            
            # This is O(n^3) again (pairs u,v and candidate w).
            # n=255. Pairs ~32k. Candidates 255. Total ~8M ops per full pass.
            # Might be too slow if many passes.
            # Let's only do this if we have time and previous moves stalled?
            # Or just do 1-for-2 and 1-swap which are more likely to improve.
            # 2-for-1 is structurally similar to 1-for-2 but reversed.
            # Given 1/x is convex, 1-for-2 is usually better (splitting a large denominator into two smaller ones).
            # 2-for-1 combines two large denominators into one smaller one.
            # Example: remove 100, 101. Add 2.
            # 1/2 = 0.5. 1/100 + 1/101 ~ 0.02. Huge gain.
            # But 2 must be compatible with the rest.
            # If 2 was not in the set, it was blocked by something.
            # If it was blocked by 100 or 101, then removing them allows adding 2.
            
            # Let's implement 2-for-1 as well, but carefully.
            
            for i_idx in range(len(set_indices)):
                if time.time() > time_limit:
                    return S_mask, current_score
                u = set_indices[i_idx]
                for j_idx in range(i_idx + 1, len(set_indices)):
                    v = set_indices[j_idx]
                    S_without_uv = S_mask & ~(1 << u) & ~(1 << v)
                    
                    # Find best w not in S, compatible with S_without_uv
                    # Maximize weights[w] - weights[u] - weights[v]
                    # i.e., maximize weights[w]
                    # So pick smallest w (lowest index) not in S and compatible.
                    
                    best_w = -1
                    for w in range(size):
                        if in_set[w]:
                            continue
                        if is_compatible(w, S_without_uv):
                            # Check gain
                            if weights[w] > weights[u] + weights[v]:
                                best_w = w
                                break # Found the best (smallest index/highest weight)
                    
                    if best_w != -1:
                        S_mask = S_without_uv | (1 << best_w)
                        in_set[u] = False
                        in_set[v] = False
                        in_set[best_w] = True
                        set_indices.remove(u)
                        set_indices.remove(v)
                        set_indices.append(best_w)
                        current_score += weights[best_w] - weights[u] - weights[v]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue

        return S_mask, current_score

    # Generate initial solutions
    solutions = []
    
    # 1. Greedy by smallest value (highest weight)
    order1 = list(range(size))
    solutions.append(greedy_construct(order1))
    
    # 2. Random restarts
    random.seed(42)
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        order = list(range(size))
        random.shuffle(order)
        solutions.append(greedy_construct(order))
        
    # Local search on each solution
    best_mask = solutions[0]
    best_score = get_score(best_mask)
    
    for S in solutions:
        if time.time() > deadline:
            break
        S_opt, val = local_search(S, deadline)
        if val > best_score:
            best_score = val
            best_mask = S_opt

    # Convert best_mask back to list of values
    result = []
    for i in range(size):
        if best_mask & (1 << i):
            result.append(values[i])

    return result
