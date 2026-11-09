import time
import random
from fractions import Fraction
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Generate primes up to 1000
    def gen_primes(limit):
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, isqrt(limit) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = 0
        return [i for i in range(2, limit + 1) if sieve[i]]
    
    primes = gen_primes(1000)
    prime_set = set(primes)
    
    best_L = 0
    best_classes = []
    
    def evaluate_assignment(assignment, prime_list, max_L=10000):
        """Evaluate how far we can cover with given assignment.
        assignment: dict {p: r}
        Returns the largest L such that 1..L is all covered.
        """
        if not assignment:
            return 0
        
        # Use bytearray for coverage
        covered = bytearray(max_L + 1)
        
        for p, r in assignment.items():
            # Mark all k in 1..max_L with k % p == r
            start = r
            if start == 0:
                start = p
            for k in range(start, max_L + 1, p):
                covered[k] = 1
        
        L = 0
        for n in range(1, max_L + 1):
            if covered[n]:
                L = n
            else:
                break
        return L
    
    def greedy_cover(prime_list, max_L=10000, seed=0, heuristic='smallest'):
        """Greedy: cover smallest uncovered, pick prime by heuristic.
        Returns (assignment, L)
        """
        rng = random.Random(seed)
        covered = bytearray(max_L + 1)
        assignment = {}
        used_primes = set()
        
        # Precompute for each prime the list of positions it would cover for each residue
        # This is too memory heavy, so compute on the fly
        
        L = 0
        for n in range(1, max_L + 1):
            if covered[n]:
                L = n
                continue
            
            # Find best prime to cover n
            best_p = None
            best_gain = -1
            best_r = None
            
            candidates = []
            for p in prime_list:
                if p in used_primes:
                    continue
                r = n % p
                # Compute gain: how many currently uncovered positions would be covered
                # by assigning residue r to prime p
                gain = 0
                start = r
                if start == 0:
                    start = p
                for k in range(start, max_L + 1, p):
                    if not covered[k]:
                        gain += 1
                
                candidates.append((p, r, gain))
            
            if not candidates:
                break
            
            if heuristic == 'smallest':
                # Among those with max gain, pick smallest prime
                max_gain = max(c[2] for c in candidates)
                best_candidates = [c for c in candidates if c[2] == max_gain]
                best_p, best_r, _ = min(best_candidates, key=lambda x: x[0])
            elif heuristic == 'largest_gain':
                best_p, best_r, best_gain = max(candidates, key=lambda x: x[2])
            elif heuristic == 'random':
                # Weighted random by gain
                max_gain = max(c[2] for c in candidates)
                if max_gain <= 0:
                    break
                # Pick randomly among top performers
                top = [c for c in candidates if c[2] >= max_gain * 0.5]
                best_p, best_r, _ = rng.choice(top)
            else:
                break
            
            used_primes.add(best_p)
            assignment[best_p] = best_r
            
            # Update covered
            start = best_r
            if start == 0:
                start = best_p
            for k in range(start, max_L + 1, best_p):
                covered[k] = 1
            
            L = n
        
        return assignment, L
    
    def local_search_extend(assignment, prime_list, current_L, max_L=10000, max_iters=100, seed=0):
        """Try to extend coverage beyond current_L by changing residues or adding primes.
        Returns (new_assignment, new_L)
        """
        rng = random.Random(seed)
        assignment = dict(assignment)
        current_L = current_L
        
        # Build coverage
        covered = bytearray(max_L + 1)
        for p, r in assignment.items():
            start = r
            if start == 0:
                start = p
            for k in range(start, max_L + 1, p):
                covered[k] = 1
        
        used_primes = set(assignment.keys())
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Find the first uncovered position beyond current_L
            next_uncovered = current_L + 1
            if next_uncovered > max_L:
                break
            if not covered[next_uncovered]:
                # Try to cover it
                best_p = None
                best_r = None
                best_gain = 0
                
                for p in prime_list:
                    if p in used_primes:
                        # Try changing residue of already-used prime
                        old_r = assignment[p]
                        new_r = next_uncovered % p
                        if new_r == old_r:
                            continue
                        # Compute gain of changing
                        gain = 0
                        start = new_r
                        if start == 0:
                            start = p
                        for k in range(start, max_L + 1, p):
                            if not covered[k]:
                                gain += 1
                        # Subtract loss: positions that were covered by old_r but not new_r
                        loss = 0
                        old_start = old_r
                        if old_start == 0:
                            old_start = p
                        for k in range(old_start, max_L + 1, p):
                            if covered[k] and (k % p != new_r):
                                # Check if k is covered by another prime
                                is_covered_other = False
                                for p2, r2 in assignment.items():
                                    if p2 == p:
                                        continue
                                    if k % p2 == r2:
                                        is_covered_other = True
                                        break
                                if not is_covered_other:
                                    loss += 1
                        net_gain = gain - loss
                        if net_gain > best_gain:
                            best_gain = net_gain
                            best_p = p
                            best_r = new_r
                    else:
                        # Try adding new prime
                        r = next_uncovered % p
                        gain = 0
                        start = r
                        if start == 0:
                            start = p
                        for k in range(start, max_L + 1, p):
                            if not covered[k]:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_p = p
                            best_r = r
                
                if best_p is None or best_gain <= 0:
                    break
                
                # Apply change
                if best_p in assignment:
                    old_r = assignment[best_p]
                    # Remove old coverage
                    old_start = old_r
                    if old_start == 0:
                        old_start = old_r
                    # We'll rebuild coverage after
                    assignment[best_p] = best_r
                else:
                    used_primes.add(best_p)
                    assignment[best_p] = best_r
                
                # Rebuild coverage
                covered = bytearray(max_L + 1)
                for p, r in assignment.items():
                    start = r
                    if start == 0:
                        start = p
                    for k in range(start, max_L + 1, p):
                        covered[k] = 1
                
                # Update L
                new_L = current_L
                for n in range(current_L + 1, max_L + 1):
                    if covered[n]:
                        new_L = n
                    else:
                        break
                current_L = new_L
            else:
                # current_L+1 is covered, extend L
                new_L = current_L
                for n in range(current_L + 1, max_L + 1):
                    if covered[n]:
                        new_L = n
                    else:
                        break
                current_L = new_L
        
        return assignment, current_L
    
    # Generate candidate prime sets based on reciprocal budget
    def gen_prime_sets():
        sets = []
        
        # Set 1: All primes >= 13 (seed approach)
        s1 = [p for p in primes if p >= 13]
        s1_sum = sum(Fraction(1, p) for p in s1)
        if s1_sum <= 1:
            sets.append(s1)
        
        # Set 2: Small primes that fit
        # Try combinations
        small_primes = [p for p in primes if p <= 23]
        
        # Greedy: sort by 1/p descending, add if fits
        sorted_sp = sorted(small_primes, key=lambda x: 1/x, reverse=True)
        s2 = []
        s2_sum = Fraction(0, 1)
        for p in sorted_sp:
            if s2_sum + Fraction(1, p) <= 1:
                s2.append(p)
                s2_sum += Fraction(1, p)
        if s2:
            sets.append(s2)
        
        # Set 3: {3,5,7,11,13,17,19,23} if fits
        s3 = [3, 5, 7, 11, 13, 17, 19, 23]
        s3_sum = sum(Fraction(1, p) for p in s3)
        if s3_sum <= 1:
            sets.append(s3)
        
        # Set 4: Include 2, exclude some larger
        s4 = [2]
        s4_sum = Fraction(1, 2)
        for p in sorted_sp:
            if p == 2:
                continue
            if s4_sum + Fraction(1, p) <= 1:
                s4.append(p)
                s4_sum += Fraction(1, p)
        if s4:
            sets.append(s4)
        
        # Set 5: Medium primes
        s5 = [p for p in primes if 11 <= p <= 47]
        s5_sum = sum(Fraction(1, p) for p in s5)
        if s5_sum <= 1:
            sets.append(s5)
        
        # Set 6: Larger primes
        s6 = [p for p in primes if 17 <= p <= 61]
        s6_sum = sum(Fraction(1, p) for p in s6)
        if s6_sum <= 1:
            sets.append(s6)
        
        # Remove duplicates
        unique_sets = []
        seen = set()
        for s in sets:
            key = tuple(sorted(s))
            if key not in seen:
                seen.add(key)
                unique_sets.append(s)
        
        return unique_sets
    
    candidate_sets = gen_prime_sets()
    
    # Also add some random sets
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        rand_primes = primes[:]
        random.shuffle(rand_primes)
        s = Fraction(0, 1)
        rand_set = []
        for p in rand_primes:
            if s + Fraction(1, p) <= 1:
                rand_set.append(p)
                s += Fraction(1, p)
        if rand_set and len(rand_set) >= 5:
            candidate_sets.append(rand_set)
    
    # Evaluate each set with multiple heuristics
    for prime_list in candidate_sets:
        if time.time() > deadline - 5:
            break
        
        # Check budget
        s = sum(Fraction(1, p) for p in prime_list)
        if s > 1:
            continue
        
        # Try different heuristics
        for heuristic in ['smallest', 'largest_gain', 'random']:
            if time.time() > deadline - 5:
                break
            
            for seed in range(3):
                if time.time() > deadline - 5:
                    break
                
                assignment, L = greedy_cover(prime_list, max_L=10000, seed=seed, heuristic=heuristic)
                
                # Try to extend with local search
                if L > 0 and time.time() < deadline - 10:
                    assignment, L = local_search_extend(assignment, prime_list, L, max_L=10000, max_iters=50, seed=seed)
                
                if L > best_L:
                    best_L = L
                    best_classes = [[p, r] for p, r in assignment.items()]
    
    return best_classes
