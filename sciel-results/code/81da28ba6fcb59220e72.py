import time
import math
import random
import bisect
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin

    # 1. Generate a list of small primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(2000000)
    prime_set = set(primes)
    
    # 2. Fast tau_perp using precomputed divisor values via DP
    def tau_perp_fast(prime_factors):
        k = len(prime_factors)
        if k == 0:
            return 0
        num_divs = 1 << k
        
        # Precompute divisor values using DP
        div_vals = [1] * num_divs
        for mask in range(1, num_divs):
            lsb = mask & (-mask)
            bit_idx = lsb.bit_length() - 1
            prev_mask = mask ^ lsb
            div_vals[mask] = div_vals[prev_mask] * prime_factors[bit_idx]
        
        # Create list of (value, mask) and sort by value
        divs = [(div_vals[mask], mask) for mask in range(num_divs)]
        divs.sort()
        
        # Count adjacent coprime pairs using bitmask AND
        score = 0
        for i in range(num_divs - 1):
            if divs[i][1] & divs[i+1][1] == 0:
                score += 1
        return score

    # 3. Score heuristic for quick filtering
    def score_heuristic(prime_factors):
        ps = sorted(prime_factors)
        score = 0.0
        for i in range(len(ps)):
            score += 1.0 / math.log(ps[i] + 1)
        if len(ps) >= 10:
            score += 0.3 * (len(ps) - 10)
        return score

    # 4. Try multiple initializations with different sizes
    best_n = None
    best_score = 0
    best_start = None
    
    for max_k in [13, 14, 15, 16]:
        if time.time() > deadline - 10.0:
            break
        current_primes = list(primes[:max_k])
        current_n = 1
        for p in current_primes:
            current_n *= p
        if current_n > 10**18:
            continue
        s = tau_perp_fast(current_primes)
        if s > best_score:
            best_score = s
            best_n = current_n
            best_start = list(current_primes)
    
    if best_n is None:
        best_n = 2
        best_score = 0
        best_start = [2, 3]

    current_primes = list(best_start)
    current_n = 1
    for p in current_primes:
        current_n *= p
    current_score = tau_perp_fast(current_primes)
    
    # 5. Local search with multiple strategies
    improved = True
    max_iterations = 3000
    iteration = 0
    no_improve_count = 0
    
    while time.time() < deadline - 15.0 and iteration < max_iterations:
        iteration += 1
        current_prime_set = set(current_primes)
        k = len(current_primes)
        found_improvement = False
        
        # Strategy 1: Single prime swap - try all positions, pick best
        if time.time() < deadline - 10.0:
            best_swap = None
            best_swap_score = current_score
            best_swap_idx = -1
            best_swap_p = None
            
            for idx in range(k):
                if time.time() > deadline - 10.0:
                    break
                    
                p_old = current_primes[idx]
                max_p_new = (10**18 * p_old) // current_n
                
                # Generate diverse candidates
                candidates = []
                
                # Smallest available primes
                for p_new in primes:
                    if p_new in current_prime_set:
                        continue
                    if p_new > max_p_new:
                        break
                    candidates.append(p_new)
                    if len(candidates) >= 60:
                        break
                
                # Some medium-sized primes
                if max_p_new > 100000:
                    start_idx = bisect.bisect_left(primes, 100000)
                    for p_new in primes[start_idx:]:
                        if p_new in current_prime_set:
                            continue
                        if p_new > max_p_new:
                            break
                        if p_new not in candidates:
                            candidates.append(p_new)
                        if len(candidates) >= 100:
                            break
                
                # Some random primes
                if max_p_new > 1000000:
                    for _ in range(15):
                        p_rand = random.randint(1000000, min(max_p_new, 3000000))
                        pos = bisect.bisect_left(primes, p_rand)
                        if pos < len(primes) and primes[pos] <= max_p_new and primes[pos] not in current_prime_set:
                            if primes[pos] not in candidates:
                                candidates.append(primes[pos])
                
                for p_new in candidates:
                    if p_new in current_prime_set:
                        continue
                    if p_new > max_p_new:
                        continue
                    
                    new_primes = current_primes.copy()
                    new_primes[idx] = p_new
                    
                    s_new = tau_perp_fast(new_primes)
                    
                    if s_new > best_swap_score:
                        best_swap_score = s_new
                        best_swap_idx = idx
                        best_swap_p = p_new
            
            if best_swap_idx >= 0 and best_swap_p is not None and best_swap_score > current_score:
                p_old = current_primes[best_swap_idx]
                current_primes[best_swap_idx] = best_swap_p
                current_n = (current_n // p_old) * best_swap_p
                current_score = best_swap_score
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_n
                found_improvement = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # Strategy 2: Pair swap
        if not found_improvement and time.time() < deadline - 20.0 and k >= 2:
            best_pair = None
            best_pair_score = current_score
            best_pair_i = -1
            best_pair_j = -1
            best_pair_p = None
            
            pair_indices = list(combinations(range(k), 2))
            random.shuffle(pair_indices)
            max_pairs_to_try = 40
            
            for i, j in pair_indices[:max_pairs_to_try]:
                if time.time() > deadline - 10.0:
                    break
                    
                p_i = current_primes[i]
                p_j = current_primes[j]
                base = current_n // (p_i * p_j)
                
                candidates_i = []
                candidates_j = []
                
                for p in primes:
                    if p in current_prime_set:
                        continue
                    if p * base * p_j <= 10**18:
                        candidates_i.append(p)
                        if len(candidates_i) >= 20:
                            break
                
                for p in primes:
                    if p in current_prime_set:
                        continue
                    if p * base * p_i <= 10**18:
                        candidates_j.append(p)
                        if len(candidates_j) >= 20:
                            break
                
                for p_new_i in candidates_i[:12]:
                    for p_new_j in candidates_j[:12]:
                        if p_new_i == p_new_j:
                            continue
                        if p_new_i in current_prime_set or p_new_j in current_prime_set:
                            continue
                        
                        new_primes = current_primes.copy()
                        new_primes[i] = p_new_i
                        new_primes[j] = p_new_j
                        
                        s_new = tau_perp_fast(new_primes)
                        
                        if s_new > best_pair_score:
                            best_pair_score = s_new
                            best_pair_i = i
                            best_pair_j = j
                            best_pair_p = (p_new_i, p_new_j)
            
            if best_pair_i >= 0 and best_pair_p is not None and best_pair_score > current_score:
                p_i = current_primes[best_pair_i]
                p_j = current_primes[best_pair_j]
                base = current_n // (p_i * p_j)
                current_primes[best_pair_i] = best_pair_p[0]
                current_primes[best_pair_j] = best_pair_p[1]
                current_n = base * best_pair_p[0] * best_pair_p[1]
                current_score = best_pair_score
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_n
                found_improvement = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # Strategy 3: Try adding a prime (if k < 16)
        if not found_improvement and time.time() < deadline - 20.0 and k < 16:
            best_add = None
            best_add_score = current_score
            
            for p in primes:
                if p in current_prime_set:
                    continue
                new_n = current_n * p
                if new_n > 10**18:
                    break
                
                new_primes = current_primes + [p]
                s_new = tau_perp_fast(new_primes)
                
                if s_new > best_add_score:
                    best_add_score = s_new
                    best_add = p
            
            if best_add is not None and best_add_score > current_score:
                current_primes.append(best_add)
                current_n *= best_add
                current_score = best_add_score
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_n
                found_improvement = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # Strategy 4: Try removing a prime (if k > 12)
        if not found_improvement and time.time() < deadline - 20.0 and k > 12:
            best_remove = None
            best_remove_score = current_score
            
            for idx in range(k):
                p_old = current_primes[idx]
                new_primes = [p for i, p in enumerate(current_primes) if i != idx]
                new_n = current_n // p_old
                if new_n >= 2:
                    s_new = tau_perp_fast(new_primes)
                    if s_new > best_remove_score:
                        best_remove_score = s_new
                        best_remove = idx
            
            if best_remove is not None and best_remove_score > current_score:
                p_old = current_primes[best_remove]
                current_n //= p_old
                current_primes.pop(best_remove)
                current_score = best_remove_score
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_n
                found_improvement = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        # If no improvement for several iterations, stop
        if no_improve_count >= 5:
            break

    # 6. Final polish: try removing primes one at a time
    if time.time() < deadline - 5.0:
        for idx in range(len(current_primes)):
            if time.time() > deadline - 3.0:
                break
            p_old = current_primes[idx]
            new_primes = [p for i, p in enumerate(current_primes) if i != idx]
            new_n = current_n // p_old
            if new_n >= 2 and new_n <= 10**18:
                s_new = tau_perp_fast(new_primes)
                if s_new > best_score:
                    best_score = s_new
                    best_n = new_n
    
    return best_n
