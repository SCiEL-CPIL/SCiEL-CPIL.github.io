"""Rewritten program for few prime factors across pairwise sums."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin
    
    # Precompute Smallest Prime Factor (SPF) up to 20000 for fast factorization
    MAX_SUM = 20000
    spf = list(range(MAX_SUM + 1))
    for i in range(2, int(MAX_SUM**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_SUM + 1, i):
                if spf[j] == j:
                    spf[j] = i
                    
    def get_prime_factors(n):
        if n <= 1:
            return set()
        factors = set()
        while n > 1:
            p = spf[n]
            factors.add(p)
            while n % p == 0:
                n //= p
        return factors

    def evaluate_set(nums):
        """Calculate the number of distinct prime factors of all pairwise sums."""
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                all_primes.update(get_prime_factors(nums[i] + nums[j]))
        return len(all_primes)

    def evaluate_set_with_primes(nums):
        """Calculate the set of distinct prime factors of all pairwise sums."""
        all_primes = set()
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                all_primes.update(get_prime_factors(nums[i] + nums[j]))
        return all_primes

    best_result = None
    best_count = float('inf')

    # --- Phase 1: Exact Clique Search for Prime Set {2, 3} ---
    # We look for 12 numbers x_i such that x_i + x_j is only divisible by 2 and 3.
    # This implies x_i + x_j = 2^a * 3^b.
    # If we set x_i = 6*k_i + 3, then x_i + x_j = 6(k_i + k_j) + 6 = 6(k_i + k_j + 1).
    # We need k_i + k_j + 1 to be of the form 2^a * 3^b.
    
    max_k = 10000 // 6
    # Generate all 2^a * 3^b <= 2*max_k + 1
    smooth_sums = set()
    for a in range(15):
        pow2 = 1 << a
        if pow2 > 2 * max_k + 1:
            break
        for b in range(15):
            val = pow2 * (3 ** b)
            if val > 2 * max_k + 1:
                break
            smooth_sums.add(val)
            
    # k_i must be such that x_i <= 10000 => 6*k_i + 3 <= 10000 => k_i <= 1666
    # Also k_i >= 0.
    valid_ks = list(range(0, 1667))
    
    # Build a heuristic clique finder
    # We want a clique of size 12 in the graph where edge (u, v) exists if (u+v+1) in smooth_sums
    
    found_clique = False
    best_clique = None
    
    # Heuristic Clique Search
    # We will try to build a clique starting from various seeds.
    # To speed up, we can precompute for each k, the list of compatible k's?
    # That's 1667 * 1667 checks, which is ~2.7M checks. Fast enough in Python?
    # 2.7M set lookups is maybe 1-2 seconds. Let's do it.
    
    # compatible[k] = set of k' such that (k+k'+1) in smooth_sums
    compatible = [set() for _ in range(1667)]
    for k1 in range(1667):
        for k2 in range(k1 + 1, 1667):
            if (k1 + k2 + 1) in smooth_sums:
                compatible[k1].add(k2)
                compatible[k2].add(k1)
    
    # Now find a clique of size 12.
    # Greedy approach:
    # 1. Pick a node with high degree.
    # 2. Intersect neighbors with previous picks.
    
    # Let's try multiple random restarts of a greedy clique finder.
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break
            
        # Start with a random node
        # Or start with the node with highest degree?
        # Let's just pick random to diversify
        current_clique = []
        candidates = set(range(1667))
        
        # Pick first node
        if not candidates:
            break
        first_node = random.choice(list(candidates))
        current_clique.append(first_node)
        candidates = compatible[first_node]
        
        for _ in range(11):
            if not candidates:
                break
            # Pick a node from candidates with high degree within current candidates
            # Degree in candidates = len(candidates.intersection(compatible[node]))
            # This is expensive if done naively for all.
            # Just pick random from candidates?
            # Or sort by global degree?
            
            # Let's pick the one with max intersection size
            best_node = None
            best_inter_size = -1
            # Sample a subset if candidates is large
            cand_list = list(candidates)
            if len(cand_list) > 50:
                cand_list = random.sample(cand_list, 50)
                
            for node in cand_list:
                inter = candidates.intersection(compatible[node])
                if len(inter) > best_inter_size:
                    best_inter_size = len(inter)
                    best_node = node
                    
            if best_node is None:
                break
                
            current_clique.append(best_node)
            candidates = candidates.intersection(compatible[best_node])
            
        if len(current_clique) == 12:
            found_clique = True
            best_clique = [6 * k + 3 for k in current_clique]
            break

    if found_clique:
        # Verify
        if len(set(best_clique)) == 12 and all(1 <= x <= 10000 for x in best_clique):
            cnt = evaluate_set(best_clique)
            if cnt == 2:
                return best_clique
            elif cnt < best_count:
                best_count = cnt
                best_result = best_clique[:]

    # --- Phase 2: Local Search ---
    # If clique search failed, use local search.
    # Good initializations:
    # 1. Multiples of 6: 6, 12, ..., 72. Sums are multiples of 6.
    # 2. Numbers of form 6k+3: 3, 9, 15, ... Sums are 6(k1+k2+1).
    # 3. Numbers of form 6k+1: 1, 7, 13, ... Sums are 6(k1+k2) + 2 = 2(3(k1+k2)+1).
    #    If 3(k1+k2)+1 is smooth, good.
    # 4. Numbers of form 6k+5: 5, 11, 17, ... Sums are 6(k1+k2) + 10 = 2(3(k1+k2)+5).
    
    inits = []
    
    # Init 1: 6k
    inits.append([6 * i for i in range(1, 13)])
    
    # Init 2: 6k+3
    inits.append([6 * i + 3 for i in range(0, 12)])
    
    # Init 3: 6k+1
    inits.append([6 * i + 1 for i in range(0, 12)])
    
    # Init 4: 6k+5
    inits.append([6 * i + 5 for i in range(0, 12)])
    
    # Init 5: Random
    inits.append(random.sample(range(1, 10001), 12))
    
    # Init 6: Modulus 210 based?
    # If we pick numbers all congruent to r mod 210, sums are 2r mod 210.
    # 2r is divisible by 2, 3, 5, 7 if r is multiple of 105.
    # So r=0, 105.
    # x_i = 210*k_i + 105.
    # Sums = 210(k1+k2) + 210 = 210(k1+k2+1).
    # This guarantees primes 2, 3, 5, 7.
    # We want k1+k2+1 to be 2-3-5-7 smooth.
    # This is a similar clique problem but with more smooth numbers available.
    # Let's just use it as a starting point for local search.
    inits.append([210 * i + 105 for i in range(0, 12)])
    
    for init in inits:
        if time.time() > deadline - 10.0:
            break
            
        curr_set = sorted(set(x for x in init if 1 <= x <= 10000))
        if len(curr_set) < 12:
            # Fill
            existing = set(curr_set)
            for x in range(1, 10001):
                if len(curr_set) >= 12:
                    break
                if x not in existing:
                    curr_set.append(x)
            curr_set = sorted(curr_set[:12])
            
        curr_score = evaluate_set(curr_set)
        
        if curr_score < best_count:
            best_count = curr_score
            best_result = curr_set[:]
            if best_count <= 2:
                return best_result

        # Local Search
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Pick a random index to change
            idx = random.randint(0, 11)
            old_val = curr_set[idx]
            
            # Generate candidates for new value
            candidates = set()
            
            # 1. Local perturbations
            for delta in [-5, -2, -1, 1, 2, 5]:
                candidates.add(old_val + delta)
            
            # 2. Random
            candidates.add(random.randint(1, 10000))
            
            # 3. Structured: Try to keep same residue mod 6
            r = old_val % 6
            candidates.add(random.randint(1, 1666) * 6 + r)
            
            # 4. Structured: Try to make sums smooth
            # For each other element y, we want x+y to be smooth.
            # This is hard to do directly.
            
            # 5. If we are in the "6k+3" family, try other 6k+3
            if r == 3:
                candidates.add(random.randint(0, 1666) * 6 + 3)
            
            # 6. If we are in the "210k+105" family, try other 210k+105
            if old_val % 210 == 105:
                candidates.add(random.randint(0, 47) * 210 + 105)
                
            for new_val in candidates:
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in curr_set:
                    continue
                
                new_set = curr_set[:]
                new_set[idx] = new_val
                new_score = evaluate_set(new_set)
                
                if new_score < curr_score:
                    curr_score = new_score
                    curr_set = new_set
                    improved = True
                    if curr_score < best_count:
                        best_count = curr_score
                        best_result = curr_set[:]
                        if best_count <= 2:
                            return best_result
                    # Restart the while loop to pick a new index
                    break # Break inner for loop to restart while loop

    # Final check
    if best_result is None:
        best_result = list(range(1, 13))
        
    # Verify distinctness and bounds
    best_result = sorted(set(x for x in best_result if 1 <= x <= 10000))
    if len(best_result) < 12:
        existing = set(best_result)
        for i in range(1, 10001):
            if len(best_result) >= 12:
                break
            if i not in existing:
                best_result.append(i)
        best_result = sorted(best_result[:12])
    
    return best_result

# EVOLVE-BLOCK-END
