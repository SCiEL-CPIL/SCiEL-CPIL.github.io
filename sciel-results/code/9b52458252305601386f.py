import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 15s margin
    
    N = 512
    # Weights
    weights = [0.0] * N
    for a in range(1, N):
        weights[a] = 1.0 / (N - a)
        
    # Prime factors as bitmasks
    # There are 97 primes below 512. This fits in a Python int (arbitrary precision).
    primes = []
    is_prime = [True] * N
    is_prime[0] = is_prime[1] = False
    for i in range(2, N):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, N, i):
                is_prime[j] = False
                
    prime_index = {p: i for i, p in enumerate(primes)}
    
    masks = [0] * N
    for a in range(1, N):
        m = a
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                masks[a] |= (1 << prime_index[p])
                while m % p == 0:
                    m //= p
        if m > 1:
            masks[a] |= (1 << prime_index[m])
            
    # Pre-sort candidates by weight descending
    # We only care about top candidates for swaps. 
    # Numbers with very low weight are unlikely to be part of an optimal solution 
    # unless they are primes that don't conflict. But primes are generally high up.
    # Let's keep top 200 candidates for search efficiency.
    candidates = sorted(range(1, N), key=lambda x: -weights[x])
    top_candidates = candidates[:200]
    
    def get_score(S):
        return sum(weights[a] for a in S)
        
    def is_coprime_set(S, a):
        # Check if a is coprime to all in S
        for x in S:
            if masks[a] & masks[x]:
                return False
        return True
        
    def greedy_initial(order):
        S = []
        current_mask = 0
        for a in order:
            if masks[a] & current_mask == 0:
                S.append(a)
                current_mask |= masks[a]
        return S
        
    # 1. Generate several initial solutions
    best_S = []
    best_score = 0.0
    
    # Greedy descending
    S1 = greedy_initial(candidates)
    best_S = S1
    best_score = get_score(best_S)
    
    # Randomized Greedy (Simulated Annealing style initial)
    for _ in range(20):
        if time.time() > deadline: break
        # Shuffle top 50, keep rest in order
        top50 = candidates[:50]
        random.shuffle(top50)
        order = top50 + candidates[50:]
        S_r = greedy_initial(order)
        score_r = get_score(S_r)
        if score_r > best_score:
            best_score = score_r
            best_S = S_r
            
    # 2. Local Search
    def local_search(S, score):
        S = list(S)
        score = float(score)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(S)
            S_set = set(S)
            
            # Move 1: Add
            # Try to add any candidate not in S that is coprime to all
            for a in top_candidates:
                if time.time() > deadline: break
                if a in S_set:
                    continue
                if all((masks[a] & masks[x]) == 0 for x in S):
                    S.append(a)
                    S_set.add(a)
                    score += weights[a]
                    improved = True
                    break
            
            if improved or time.time() > deadline:
                continue
                
            # Move 2: Swap 1-for-1
            # For each x in S, try to replace with y in top_candidates \ S
            # such that y is coprime to S \ {x} and weights[y] > weights[x]
            for i in range(n):
                if time.time() > deadline: break
                x = S[i]
                # Check if any y can replace x
                best_y = -1
                best_y_w = weights[x]
                for y in top_candidates:
                    if y in S_set:
                        continue
                    if weights[y] <= best_y_w:
                        continue # Since top_candidates is sorted, we could break, 
                                 # but y might not be coprime, so we can't break immediately 
                                 # unless we are sure no later y is better? 
                                 # Actually, top_candidates is sorted by weight. 
                                 # If weights[y] <= weights[x], and we want weights[y] > weights[x], 
                                 # we only need to check y with weights[y] > weights[x].
                        # Optimization: break if weights[y] <= weights[x] is not possible 
                        # because we iterate in descending order. 
                        # So if we hit a y with weight <= x, all subsequent have lower weight.
                        # So we can break.
                        break
                    
                    # Check coprime with S \ {x}
                    ok = True
                    for j in range(n):
                        if i == j: continue
                        if masks[y] & masks[S[j]]:
                            ok = False
                            break
                    if ok:
                        best_y = y
                        best_y_w = weights[y]
                        break # First valid y in descending weight order is best
                
                if best_y != -1:
                    S[i] = best_y
                    S_set.remove(x)
                    S_set.add(best_y)
                    score += weights[best_y] - weights[x]
                    improved = True
                    break
                    
            if improved or time.time() > deadline:
                continue

            # Move 3: Swap 1-for-2
            # Replace x in S with {y, z}
            # y, z must be coprime to each other and to S \ {x}
            # weights[y] + weights[z] > weights[x]
            for i in range(n):
                if time.time() > deadline: break
                x = S[i]
                
                # Find candidates y, z
                # We need y, z from top_candidates \ S
                # y, z coprime to S \ {x}
                # y, z coprime to each other
                # w_y + w_z > w_x
                
                # To be fast, iterate y in top_candidates
                # If w_y < w_x, we might still find z? No, w_z > 0.
                # So w_y must be at least w_x - w_z.
                # Since w_z is small for large indices, y should be relatively large.
                
                found_pair = False
                for y in top_candidates:
                    if time.time() > deadline: break
                    if y in S_set:
                        continue
                    if weights[y] <= weights[x]:
                        # Even if z has max weight, is it possible?
                        # Max w_z is ~1. If w_y is small, sum might be > w_x?
                        # But top_candidates are sorted. 
                        # If w_y is very small, w_z is even smaller.
                        # Let's just check if w_y + max_possible_w_z > w_x.
                        # max_possible_w_z is roughly weights[top_candidates[0]] which is 1.
                        # So if w_y + 1 <= w_x, break?
                        # No, w_x can be large.
                        # Let's just iterate.
                        pass
                    
                    # Check y coprime to S \ {x}
                    ok_y = True
                    for j in range(n):
                        if i == j: continue
                        if masks[y] & masks[S[j]]:
                            ok_y = False
                            break
                    if not ok_y:
                        continue
                        
                    # Find z
                    for z in top_candidates:
                        if time.time() > deadline: break
                        if z == y or z in S_set:
                            continue
                        if weights[y] + weights[z] <= weights[x]:
                            break # Since z is sorted desc, further z are smaller
                        
                        # Check z coprime to S \ {x}
                        ok_z = True
                        for j in range(n):
                            if i == j: continue
                            if masks[z] & masks[S[j]]:
                                ok_z = False
                            break # Wait, need to check all
                        # Re-implement ok_z check properly
                        ok_z = True
                        for j in range(n):
                            if i == j: continue
                            if masks[z] & masks[S[j]]:
                                ok_z = False
                                break
                        if not ok_z:
                            continue
                        
                        # Check y, z coprime
                        if masks[y] & masks[z]:
                            continue
                        
                        # Valid swap
                        S.pop(i)
                        S.append(y)
                        S.append(z)
                        S_set.remove(x)
                        S_set.add(y)
                        S_set.add(z)
                        score += weights[y] + weights[z] - weights[x]
                        improved = True
                        found_pair = True
                        break
                        
                    if found_pair:
                        break
                        
                if found_pair:
                    break
                    
            if improved or time.time() > deadline:
                continue

            # Move 4: Swap 2-for-1
            # Replace {x, y} in S with z
            # z coprime to S \ {x, y}
            # w_z > w_x + w_y
            for i in range(n):
                if time.time() > deadline: break
                x = S[i]
                for j in range(i+1, n):
                    if time.time() > deadline: break
                    y = S[j]
                    
                    # Find z
                    for z in top_candidates:
                        if time.time() > deadline: break
                        if z in S_set:
                            continue
                        if weights[z] <= weights[x] + weights[y]:
                            break # z sorted desc
                        
                        # Check z coprime to S \ {x, y}
                        ok_z = True
                        for k in range(n):
                            if k == i or k == j: continue
                            if masks[z] & masks[S[k]]:
                                ok_z = False
                                break
                        if ok_z:
                            # Remove x, y, add z
                            S.pop(j) # Remove y (j > i)
                            S.pop(i) # Remove x
                            S.append(z)
                            S_set.remove(x)
                            S_set.remove(y)
                            S_set.add(z)
                            score += weights[z] - weights[x] - weights[y]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        return S, score

    # Run local search
    final_S, final_score = local_search(best_S, best_score)
    
    return final_S
