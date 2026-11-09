import time
from fractions import Fraction
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def get_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = 0
        return [i for i in range(2, limit + 1) if sieve[i]]

    primes = get_primes(1000)
    
    # Generate candidate prime sets
    # We want sets with sum(1/p) <= 1
    # Strategy: Greedy fill with small primes, but ensure we have a mix.
    
    def check_budget(prime_list):
        s = Fraction(0, 1)
        for p in prime_list:
            s += Fraction(1, p)
        return s <= 1

    candidate_sets = []
    
    # Set 1: Greedy small primes (excluding 2 to save budget for more primes? Or include 2?)
    # Let's try including 2, as it covers half the numbers.
    # 2, 5, 7, 13, 17, 19, 23?
    # 1/2 + 1/5 + 1/7 + 1/13 + 1/17 + 1/19 + 1/23
    # 0.5 + 0.2 + 0.142 + 0.077 + 0.059 + 0.053 + 0.043 = 1.074 (Too high)
    # Remove 23: 1.031 (Too high)
    # Remove 19: 0.978 (Valid) -> {2, 5, 7, 13, 17}
    candidate_sets.append([2, 5, 7, 13, 17])
    
    # Set 2: No 2, more primes
    # 3, 5, 7, 11, 13, 17, 19, 23
    # 0.333+0.2+0.142+0.090+0.077+0.058+0.052+0.043 = 0.995 (Valid)
    candidate_sets.append([3, 5, 7, 11, 13, 17, 19, 23])
    
    # Set 3: No 2, no 3
    # 5, 7, 11, 13, 17, 19, 23, 29, 31
    # 0.2+0.142+0.090+0.077+0.058+0.052+0.043+0.034+0.032 = 0.728
    # Can add 37, 41...
    # Let's add primes until sum > 1
    p_set = [5, 7, 11, 13, 17, 19, 23, 29, 31]
    s = sum(Fraction(1, p) for p in p_set)
    for p in primes:
        if p in p_set: continue
        if s + Fraction(1, p) <= 1:
            p_set.append(p)
            s += Fraction(1, p)
    candidate_sets.append(p_set)
    
    # Set 4: Randomized greedy
    # Shuffle primes, pick if fits
    random.seed(42)
    shuffled = primes[:]
    random.shuffle(shuffled)
    p_set = []
    s = Fraction(0, 1)
    for p in shuffled:
        if s + Fraction(1, p) <= 1:
            p_set.append(p)
            s += Fraction(1, p)
    if p_set:
        candidate_sets.append(p_set)

    # Deduplicate and validate
    valid_sets = []
    seen = set()
    for ps in candidate_sets:
        key = tuple(sorted(ps))
        if key in seen: continue
        seen.add(key)
        if check_budget(ps):
            valid_sets.append(ps)

    best_L_global = 0
    best_classes_global = []
    max_L_check = 5000 # We search for coverage up to this L

    def solve_sa(prime_list, time_limit):
        """
        Simulated Annealing to find residues for prime_list covering 1..L
        """
        if not prime_list:
            return 0, []
            
        L_max = max_L_check
        k = len(prime_list)
        
        # State: residues for each prime
        residues = [random.randint(0, p - 1) for p in prime_list]
        
        # cover_count[y] = number of primes covering y
        cover_count = [0] * (L_max + 1)
        is_gap = [True] * (L_max + 1) # is_gap[0] is unused
        is_gap[0] = False
        
        # Initialize coverage
        for i, p in enumerate(prime_list):
            r = residues[i]
            start = r if r > 0 else p
            for y in range(start, L_max + 1, p):
                cover_count[y] += 1
                if cover_count[y] == 1:
                    is_gap[y] = False
        # Wait, is_gap[y] should be True if NOT covered.
        # So if cover_count[y] > 0, is_gap[y] = False.
        for y in range(1, L_max + 1):
            is_gap[y] = (cover_count[y] == 0)
            
        # Find first gap
        first_gap = 1
        while first_gap <= L_max and not is_gap[first_gap]:
            first_gap += 1
            
        current_score = first_gap - 1
        best_score = current_score
        best_residues = residues[:]
        
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        rng = random.Random(123)
        
        while time.time() < time_limit and T > T_min:
            # Choose move
            # 50% chance: Greedy move targeting first_gap
            # 50% chance: Random move
            
            if first_gap > L_max:
                # All covered
                break
                
            if rng.random() < 0.5 and first_gap <= L_max:
                # Greedy: Pick a random prime and set its residue to cover first_gap
                i = rng.randint(0, k - 1)
                p = prime_list[i]
                new_r = first_gap % p
            else:
                # Random: Pick random prime, random new residue
                i = rng.randint(0, k - 1)
                p = prime_list[i]
                new_r = rng.randint(0, p - 1)
                
            old_r = residues[i]
            if old_r == new_r:
                continue
                
            # Apply move
            residues[i] = new_r
            
            # Update cover_count and is_gap
            # Remove old coverage
            old_start = old_r if old_r > 0 else p
            for y in range(old_start, L_max + 1, p):
                cover_count[y] -= 1
                if cover_count[y] == 0:
                    if not is_gap[y]:
                        is_gap[y] = True
                        if y < first_gap:
                            first_gap = y
            # Add new coverage
            new_start = new_r if new_r > 0 else p
            for y in range(new_start, L_max + 1, p):
                cover_count[y] += 1
                if cover_count[y] == 1:
                    if is_gap[y]:
                        is_gap[y] = False
                        if y == first_gap:
                            # Advance first_gap
                            while first_gap <= L_max and not is_gap[first_gap]:
                                first_gap += 1
            
            new_score = first_gap - 1
            
            # Accept/Reject
            delta = new_score - current_score
            if delta > 0 or rng.random() < math.exp(delta / T):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues[:]
            else:
                # Revert
                residues[i] = old_r
                # Revert cover_count and is_gap
                # Remove new coverage
                for y in range(new_start, L_max + 1, p):
                    cover_count[y] -= 1
                    if cover_count[y] == 0:
                        if not is_gap[y]:
                            is_gap[y] = True
                            if y < first_gap:
                                first_gap = y
                # Add old coverage
                for y in range(old_start, L_max + 1, p):
                    cover_count[y] += 1
                    if cover_count[y] == 1:
                        if is_gap[y]:
                            is_gap[y] = False
                            if y == first_gap:
                                while first_gap <= L_max and not is_gap[first_gap]:
                                    first_gap += 1
                current_score = first_gap - 1 # Should be same as before revert
                
            T *= cooling
            
        # Construct result
        classes = [[prime_list[i], best_residues[i]] for i in range(k)]
        return best_score, classes

    # Run SA for each valid set
    # Allocate time to each set
    num_sets = len(valid_sets)
    time_per_set = (deadline - time.time()) / num_sets if num_sets > 0 else 0
    
    for ps in valid_sets:
        if time.time() > deadline - 5:
            break
        t_limit = time.time() + min(time_per_set, deadline - time.time() - 5)
        L, classes = solve_sa(ps, t_limit)
        
        if L > best_L_global:
            best_L_global = L
            best_classes_global = classes
            
    # Final verification and return
    if best_classes_global:
        # Verify budget
        s = Fraction(0, 1)
        for p, r in best_classes_global:
            s += Fraction(1, p)
        if s <= 1:
            return best_classes_global
        else:
            # Fallback to seed-like if budget exceeded (shouldn't happen)
            return []
            
    # Fallback
    return [[2, 0]]
