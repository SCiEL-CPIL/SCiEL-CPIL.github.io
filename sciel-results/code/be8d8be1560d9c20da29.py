import time
from fractions import Fraction
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    # Generate primes up to 1000
    def get_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = 0
        return [i for i in range(2, limit + 1) if sieve[i]]

    primes = get_primes(1000)
    prime_set = set(primes)

    def recip_sum(prime_list):
        s = Fraction(0, 1)
        for p in prime_list:
            s += Fraction(1, p)
        return s

    # Simulated Annealing approach with incremental coverage tracking
    def simulated_annealing_cover(L, prime_list, time_limit, rng):
        """
        Use simulated annealing to find residue assignments for prime_list
        that cover 1..L. Returns (success, assignment) or (False, None).
        """
        n = L
        k = len(prime_list)
        
        # Initialize with random residues
        residues = [rng.randint(0, p - 1) for p in prime_list]
        
        # Build coverage array
        covered = bytearray(n + 1)
        for i, p in enumerate(prime_list):
            r = residues[i]
            start = r if r > 0 else p
            for y in range(start, n + 1, p):
                covered[y] = 1
        
        # Compute score: how many of 1..n are covered
        def compute_score():
            return sum(covered[i] for i in range(1, n + 1))
        
        current_score = compute_score()
        best_score = current_score
        best_residues = residues[:]
        
        # Temperature schedule
        T = 10.0
        T_min = 0.001
        cooling = 0.995
        
        # Gap-focused move generation: prioritize changing residues of primes
        # that don't cover the smallest uncovered number
        moves_made = 0
        no_improve_count = 0
        max_no_improve = 5000
        
        while time.time() < time_limit and T > T_min:
            # Find smallest uncovered number
            first_uncovered = -1
            for x in range(1, n + 1):
                if not covered[x]:
                    first_uncovered = x
                    break
            
            if first_uncovered == -1:
                # All covered!
                return (True, {prime_list[i]: best_residues[i] for i in range(k)})
            
            # Choose which prime to modify
            # Bias toward primes that don't cover first_uncovered (they're "failing")
            # Or randomly pick
            if rng.random() < 0.7:
                # Pick a prime that doesn't cover first_uncovered
                candidates = [i for i in range(k) if first_uncovered % prime_list[i] != residues[i]]
                if candidates:
                    idx = candidates[rng.randint(0, len(candidates) - 1)]
                else:
                    idx = rng.randint(0, k - 1)
            else:
                idx = rng.randint(0, k - 1)
            
            p = prime_list[idx]
            old_r = residues[idx]
            new_r = rng.randint(0, p - 1)
            
            if new_r == old_r:
                continue
            
            # Compute delta in score
            # Remove old coverage, add new coverage
            delta = 0
            
            # Count positions covered by old residue but not new
            old_start = old_r if old_r > 0 else p
            new_start = new_r if new_r > 0 else p
            
            # Positions that become uncovered
            for y in range(old_start, n + 1, p):
                if covered[y]:
                    # Check if this position is covered by any other prime
                    # For speed, just check if it's in the new residue class
                    if y % p != new_r:
                        # Might become uncovered if no other prime covers it
                        # Quick check: is it covered by another prime?
                        # For now, approximate: assume it becomes uncovered
                        # This is an overestimate, but SA handles it
                        delta -= 1
            
            # Count positions that become covered
            for y in range(new_start, n + 1, p):
                if not covered[y]:
                    # This position becomes covered (unless already covered by another prime)
                    # But we're adding this prime's coverage, so it becomes covered
                    delta += 1
            
            # Accept or reject
            if delta >= 0:
                # Accept
                # Update covered array
                old_start = old_r if old_r > 0 else p
                new_start = new_r if new_r > 0 else p
                
                # Remove old coverage (only if not covered by other primes)
                for y in range(old_start, n + 1, p):
                    if covered[y] and y % p != new_r:
                        # Check if covered by other primes
                        still_covered = False
                        for j in range(k):
                            if j == idx:
                                continue
                            pj = prime_list[j]
                            rj = residues[j]
                            if y % pj == rj:
                                still_covered = True
                                break
                        if not still_covered:
                            covered[y] = 0
                
                # Add new coverage
                for y in range(new_start, n + 1, p):
                    covered[y] = 1
                
                residues[idx] = new_r
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                moves_made += 1
            else:
                # Probabilistic accept
                import math
                prob = math.exp(delta / T)
                if rng.random() < prob:
                    # Accept
                    old_start = old_r if old_r > 0 else p
                    new_start = new_r if new_r > 0 else p
                    
                    for y in range(old_start, n + 1, p):
                        if covered[y] and y % p != new_r:
                            still_covered = False
                            for j in range(k):
                                if j == idx:
                                    continue
                                pj = prime_list[j]
                                rj = residues[j]
                                if y % pj == rj:
                                    still_covered = True
                                    break
                            if not still_covered:
                                covered[y] = 0
                    
                    for y in range(new_start, n + 1, p):
                        covered[y] = 1
                    
                    residues[idx] = new_r
                    current_score += delta
                    moves_made += 1
            
            T *= cooling
        
        # Check if best solution covers all
        if best_score >= n:
            return (True, {prime_list[i]: best_residues[i] for i in range(k)})
        
        return (False, None)

    # Candidate prime sets
    candidate_sets = []
    
    # Family 1: primes from 3 up to some limit
    candidate_sets.append([3,5,7,11,13,17,19,23])
    candidate_sets.append([3,5,7,11,13,17,19,29])
    
    # Family 2: include 2
    candidate_sets.append([2,5,7,13,17])
    candidate_sets.append([2,3,7])
    candidate_sets.append([2,5,7,11])
    candidate_sets.append([2,5,7,13,17])
    
    # Family 3: medium primes
    candidate_sets.append([5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97])
    
    # Family 4: primes >= 13
    primes_ge13 = [p for p in primes if p >= 13]
    s = 0
    selected = []
    for p in primes_ge13:
        if s + 1.0/p <= 1.0:
            selected.append(p)
            s += 1.0/p
        else:
            break
    candidate_sets.append(selected)
    
    # Mixed set
    candidate_sets.append([2,3,5,7,11,13,17,19,23,29,31])
    
    # Deduplicate and validate
    valid_sets = []
    seen = set()
    for ps in candidate_sets:
        if len(ps) == 0:
            continue
        key = tuple(sorted(ps))
        if key in seen:
            continue
        seen.add(key)
        if recip_sum(ps) <= 1:
            valid_sets.append(ps)
    
    best_L_overall = 0
    best_assignment_overall = None
    best_primes_overall = None
    
    rng = random.Random(42)
    
    for prime_set in valid_sets:
        if time.time() > deadline - 10:
            break
        
        # Try increasing L values
        test_Ls = [500, 700, 800, 900, 950, 999, 1000]
        
        for L in test_Ls:
            if time.time() > deadline - 5:
                break
            if L <= best_L_overall:
                continue
            
            time_limit = deadline - 5
            success, assign = simulated_annealing_cover(L, prime_set, time_limit, rng)
            
            if success and assign is not None:
                # Verify
                total = Fraction(0, 1)
                for p in prime_set:
                    total += Fraction(1, p)
                if total > 1:
                    continue
                
                covered_check = [False] * (L + 1)
                for p, r in assign.items():
                    if p not in prime_set:
                        continue
                    for y in range(1, L + 1):
                        if y % p == r:
                            covered_check[y] = True
                
                if all(covered_check[i] for i in range(1, L + 1)):
                    if L > best_L_overall:
                        best_L_overall = L
                        best_assignment_overall = dict(assign)
                        best_primes_overall = prime_set[:]
                        break  # This prime set works for this L, try next set
    
    # If we found a good solution, return it
    if best_assignment_overall is not None and best_primes_overall is not None:
        result = []
        for p in best_primes_overall:
            if p in best_assignment_overall:
                result.append([p, best_assignment_overall[p]])
        
        total = Fraction(0, 1)
        for p in best_primes_overall:
            total += Fraction(1, p)
        if total <= 1:
            return result
    
    # Fallback
    fallback_primes = [p for p in primes if p >= 13]
    s = 0
    selected = []
    for p in fallback_primes:
        if s + 1.0/p <= 0.99:
            selected.append(p)
            s += 1.0/p
        else:
            break
    
    result = []
    for i, p in enumerate(selected):
        result.append([p, (i + 1) % p])
    
    return result
