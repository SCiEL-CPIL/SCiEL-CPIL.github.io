import time
import bisect
from itertools import combinations

def construct():
    """
    Find a maximum-length strictly increasing sequence of integers in [2,128]
    such that the products of all non-empty consecutive blocks are distinct.
    
    Uses a branch-and-bound approach with efficient incremental validity checking
    via prime factorization signatures.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    # Precompute prime factorizations for numbers 2-128
    def factorize(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return tuple(sorted(factors.items()))
    
    # Precompute signatures
    sigs = {}
    for x in range(2, 129):
        sigs[x] = factorize(x)
    
    # For efficient addition of signatures
    def add_sig(a, b):
        """Add two signature tuples (lists of (prime, exp) pairs)"""
        if not a:
            return b
        if not b:
            return a
        d = {}
        for p, e in a:
            d[p] = e
        for p, e in b:
            d[p] = d.get(p, 0) + e
        return tuple(sorted(d.items()))
    
    def check_validity(seq):
        """
        Check if sequence has all distinct consecutive block products.
        Returns True if valid, False otherwise.
        """
        n = len(seq)
        if n == 0:
            return True
        
        seen = set()
        for i in range(n):
            cur = ()
            for j in range(i, n):
                cur = add_sig(cur, sigs[seq[j]])
                if cur in seen:
                    return False
                seen.add(cur)
        return True
    
    # Greedy initialization: build sequence by trying to add each number
    # in order of increasing value
    greedy_seq = []
    for x in range(2, 129):
        candidate = greedy_seq + [x]
        if check_validity(candidate):
            greedy_seq = candidate
    
    best_seq = greedy_seq[:]
    best_len = len(greedy_seq)
    
    # Branch and bound: try to find a better sequence
    # Strategy: work with the full set {2,...,128} and decide which to include
    # Use a heuristic ordering and backtracking
    
    # Precompute for each number, how "conflict-prone" it is
    # Numbers that are products of many small primes are more likely to cause conflicts
    def conflict_score(x):
        """Higher score = more likely to cause conflicts"""
        s = sigs[x]
        # Count total prime factors with multiplicity
        return sum(e for _, e in s)
    
    # Order candidates: try to include "safe" numbers first
    all_numbers = list(range(2, 129))
    
    # Heuristic ordering: prefer numbers with fewer prime factors first
    # Actually, we want to find a maximum subset, so let's use a different approach
    
    # Simulated annealing with strong moves
    import random
    
    def anneal(initial_seq, num_iters=50000):
        """Simulated annealing to improve the sequence"""
        current = initial_seq[:]
        current_set = set(current)
        T = 50.0
        cooling = 0.9999
        
        for it in range(num_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            # Choose a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Insert a missing number
                missing = [x for x in all_numbers if x not in current_set]
                if not missing:
                    continue
                x = random.choice(missing)
                new_seq = current + [x]
                new_seq.sort()
                if check_validity(new_seq):
                    current = new_seq
                    current_set = set(current)
                else:
                    # Accept with probability exp(-delta/T) where delta = 1 (invalid)
                    if random.random() < 0:  # Never accept invalid
                        pass
                    
            elif move_type < 0.7:
                # Remove a random element
                if len(current) == 0:
                    continue
                idx = random.randint(0, len(current) - 1)
                removed = current[idx]
                new_seq = current[:idx] + current[idx+1:]
                current = new_seq
                current_set = set(current)
                
            elif move_type < 0.85:
                # Swap: remove one, add another
                if len(current) == 0:
                    continue
                missing = [x for x in all_numbers if x not in current_set]
                if not missing:
                    continue
                remove_idx = random.randint(0, len(current) - 1)
                add_x = random.choice(missing)
                new_seq = current[:remove_idx] + current[remove_idx+1:] + [add_x]
                new_seq.sort()
                if check_validity(new_seq):
                    current = new_seq
                    current_set = set(current)
                else:
                    # Reject (don't accept invalid states)
                    pass
                    
            else:
                # 2-out, 2-in
                if len(current) < 2:
                    continue
                missing = [x for x in all_numbers if x not in current_set]
                if len(missing) < 2:
                    continue
                rem1, rem2 = random.sample(range(len(current)), 2)
                add1, add2 = random.sample(missing, 2)
                new_seq = [x for i, x in enumerate(current) if i not in (rem1, rem2)]
                new_seq.extend([add1, add2])
                new_seq.sort()
                if check_validity(new_seq):
                    current = new_seq
                    current_set = set(current)
                else:
                    pass
            
            T *= cooling
        
        return current
    
    # Run annealing from multiple starting points
    for restart in range(50):
        if time.time() - start_time > DEADLINE:
            break
        
        if restart == 0:
            start_seq = greedy_seq[:]
        else:
            # Perturb the best solution found so far
            k = random.randint(2, min(5, len(best_seq)))
            if len(best_seq) > k:
                remove_set = set(random.sample(best_seq, k))
                perturbed = [x for x in best_seq if x not in remove_set]
                missing = [x for x in all_numbers if x not in perturbed]
                if len(missing) >= k:
                    perturbed.extend(random.sample(missing, k))
                    perturbed.sort()
                    if check_validity(perturbed):
                        start_seq = perturbed
                    else:
                        start_seq = best_seq[:]
                else:
                    start_seq = best_seq[:]
            else:
                start_seq = best_seq[:]
        
        # Run annealing
        result = anneal(start_seq)
        
        if len(result) > best_len:
            best_seq = result
            best_len = len(result)
    
    # Final improvement: try all possible insertions
    if time.time() - start_time < DEADLINE - 2:
        improved = True
        while improved and time.time() - start_time < DEADLINE - 1:
            improved = False
            current_set = set(best_seq)
            for x in all_numbers:
                if x in current_set:
                    continue
                candidate = best_seq + [x]
                candidate.sort()
                if check_validity(candidate):
                    best_seq = candidate
                    current_set.add(x)
                    improved = True
    
    # Ensure the result is valid
    if not check_validity(best_seq):
        # Fallback to greedy
        best_seq = greedy_seq
    
    return best_seq

