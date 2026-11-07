"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (90s limit)
    
    N = 256
    
    # 1. Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Calculate largest prime power for each prime
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power
    
    # Base: 1 + sum of all prime powers
    base_sum = 1 + sum(prime_powers.values())
    
    # 3. Generate potential merge groups
    active_primes = [p for p in primes if p <= N // 2]
    
    potential_groups = []
    
    def time_check():
        return time.time() > deadline
    
    n_active = len(active_primes)
    
    # Pairs
    for i in range(n_active):
        if time_check():
            return [1] + [prime_powers[p] for p in primes]
        p1 = active_primes[i]
        for j in range(i + 1, n_active):
            p2 = active_primes[j]
            prod = p1 * p2
            if prod > N:
                break
            cost = prime_powers[p1] + prime_powers[p2]
            gain = prod - cost
            if gain > 0:
                potential_groups.append((frozenset([p1, p2]), gain))
    
    # Triples
    for i in range(n_active):
        if time_check():
            return [1] + [prime_powers[p] for p in primes]
        p1 = active_primes[i]
        for j in range(i + 1, n_active):
            p2 = active_primes[j]
            prod12 = p1 * p2
            if prod12 * 2 > N:
                continue
            for k in range(j + 1, n_active):
                p3 = active_primes[k]
                prod = prod12 * p3
                if prod > N:
                    break
                cost = prime_powers[p1] + prime_powers[p2] + prime_powers[p3]
                gain = prod - cost
                if gain > 0:
                    potential_groups.append((frozenset([p1, p2, p3]), gain))
    
    # Quads (only very small primes)
    small_active = active_primes[:12]
    for combo in combinations(small_active, 4):
        prod = 1
        for p in combo:
            prod *= p
        if prod > N:
            continue
        cost = sum(prime_powers[p] for p in combo)
        gain = prod - cost
        if gain > 0:
            potential_groups.append((frozenset(combo), gain))
    
    # Sort by gain descending
    potential_groups.sort(key=lambda x: x[1], reverse=True)
    
    n_groups = len(potential_groups)
    
    # Build index: prime -> list of group indices containing it
    prime_to_groups = {p: [] for p in active_primes}
    for gi, (gs, g) in enumerate(potential_groups):
        for p in gs:
            prime_to_groups[p].append(gi)
    
    # Greedy initialization: pick groups in order of gain, skip conflicts
    best_groups = []
    used_primes = set()
    for gs, g in potential_groups:
        if time.time() > deadline:
            break
        if gs.isdisjoint(used_primes):
            best_groups.append(gs)
            used_primes |= gs
    
    # Local search: iterative improvement via swaps
    def compute_gain(groups):
        return sum(g for gs, g in potential_groups if gs in [frozenset(x) for x in groups])
    
    best_gain = compute_gain(best_groups)
    
    # Multiple restarts with randomized greedy
    import random
    random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Randomized greedy initialization
        shuffled = list(range(n_groups))
        random.shuffle(shuffled)
        
        # Sort by gain with random perturbation
        perturbed = []
        for gi in shuffled:
            g = potential_groups[gi][1]
            perturbed.append((g * (0.8 + 0.4 * random.random()), gi))
        perturbed.sort(reverse=True)
        
        cur_groups = []
        cur_used = set()
        for _, gi in perturbed:
            gs = potential_groups[gi][0]
            if gs.isdisjoint(cur_used):
                cur_groups.append(gs)
                cur_used |= gs
        
        cur_gain = sum(potential_groups[gi][1] for gi in range(n_groups) if potential_groups[gi][0] in cur_groups)
        
        # Hill climbing: try removing one group and adding up to 2 others
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for ri, grp_to_remove in enumerate(list(cur_groups)):
                if time.time() > deadline:
                    break
                
                new_used = set()
                for g in cur_groups:
                    if g != grp_to_remove:
                        new_used |= g
                
                removed_gain = sum(potential_groups[gi][1] for gi in range(n_groups) if potential_groups[gi][0] == grp_to_remove)
                
                # Try adding one group
                for gi, (gs, g) in enumerate(potential_groups):
                    if gs.isdisjoint(new_used) and gs != grp_to_remove:
                        new_gain = cur_gain - removed_gain + g
                        if new_gain > cur_gain:
                            cur_groups = [x for x in cur_groups if x != grp_to_remove] + [gs]
                            cur_gain = new_gain
                            improved = True
                            break
                
                if improved:
                    break
                
                # Try adding two groups
                for gi1 in range(n_groups):
                    gs1 = potential_groups[gi1][0]
                    g1 = potential_groups[gi1][1]
                    if not gs1.isdisjoint(new_used) or gs1 == grp_to_remove:
                        continue
                    
                    for gi2 in range(gi1 + 1, n_groups):
                        gs2 = potential_groups[gi2][0]
                        g2 = potential_groups[gi2][1]
                        if not gs2.isdisjoint(new_used) or gs2 == grp_to_remove:
                            continue
                        if not gs1.isdisjoint(gs2):
                            continue
                        
                        new_gain = cur_gain - removed_gain + g1 + g2
                        if new_gain > cur_gain:
                            cur_groups = [x for x in cur_groups if x != grp_to_remove] + [gs1, gs2]
                            cur_gain = new_gain
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if cur_gain > best_gain:
            best_gain = cur_gain
            best_groups = list(cur_groups)
    
    # Final polish: try to add any remaining non-conflicting groups
    used_in_merge = set()
    for group in best_groups:
        for p in group:
            used_in_merge.add(p)
    
    for gs, g in potential_groups:
        if time.time() > deadline:
            break
        if gs.isdisjoint(used_in_merge):
            if gs not in best_groups:
                best_groups.append(gs)
                used_in_merge |= gs
    
    # 6. Construct final result
    result = [1]
    used_in_merge = set()
    for group in best_groups:
        prod = 1
        for p in group:
            prod *= p
            used_in_merge.add(p)
        result.append(prod)
        
    for p in primes:
        if p not in used_in_merge:
            result.append(prime_powers[p])
            
    # Ensure types are Python int and verify pairwise coprimality
    result = [int(x) for x in result]
    
    # Verify pairwise coprimality
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                # Fallback to safe solution
                return [1] + [prime_powers[p] for p in primes]
    
    return result

# EVOLVE-BLOCK-END
