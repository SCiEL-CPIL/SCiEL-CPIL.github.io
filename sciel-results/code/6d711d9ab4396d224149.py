import time
import random
import math
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 115  # seconds, safety margin for 150s limit (or 120s if strict)
    
    # 1. Generate Primes up to 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    all_primes = get_primes(1000)
    
    # 2. Select Prime Set
    # Strategy: Include small primes (starting from 11) to maximize coverage density for small integers.
    # Small primes cover a larger fraction of the prefix 1..L.
    # We must ensure sum(1/p) <= 1.
    # Start with primes >= 11. If sum > 1, remove the largest primes (which contribute least to density per unit cost).
    
    candidates = [p for p in all_primes if p >= 11]
    # Sort by prime value descending to remove largest first
    candidates_sorted_desc = sorted(candidates, reverse=True)
    
    # Use Fraction for exact budget check
    current_sum = sum(Fraction(1, p) for p in candidates)
    primes_set = list(candidates)
    
    while current_sum > 1:
        if not candidates_sorted_desc:
            break
        p_to_remove = candidates_sorted_desc.pop()
        primes_set.remove(p_to_remove)
        current_sum -= Fraction(1, p_to_remove)
        
    primes_set = sorted(primes_set)
    n_primes = len(primes_set)
    
    # Verify sum exactly
    # assert sum(Fraction(1, p) for p in primes_set) <= 1
    
    L_MAX = 100000
    # We will search in phases with increasing L limits to avoid memory/time blowup initially
    # Phase 1: 5000
    # Phase 2: 20000
    # Phase 3: 100000 (if time permits)
    
    # Since we need to return a single set of classes, and the "score" is the max L covered,
    # we should optimize for the largest L we can cover within the time limit.
    # A good strategy is to start with a moderate L limit (e.g., 5000) to find a good configuration,
    # then try to extend it if possible, or just stick to a high limit from the start if efficient.
    
    # Let's use a dynamic L limit that starts small and grows, or just pick a reasonable one like 10000.
    # The inspiration used 2000. The current program used 20000.
    # Let's use 10000 as a balance. If we cover 10000, we can try to extend to 100000 in a final pass?
    # Actually, extending L requires re-evaluating coverage for the new range.
    # It's safer to pick a fixed L_CHECK_LIMIT that is high enough to get a good score but low enough to be fast.
    # Score 1.0184 implies L ~ 1018. 
    # Score 1.8957 implies L ~ 1895.
    # If we can get L to 5000, score is ~5.
    # Let's aim for L_CHECK_LIMIT = 5000 first.
    
    L_CHECK_LIMIT = 5000
    
    def evaluate_coverage(residues, L_limit):
        cover_count = [0] * (L_limit + 1)
        for i, p in enumerate(primes_set):
            r = residues[i]
            start = r
            if start == 0:
                start = p
            if start <= L_limit:
                for n in range(start, L_limit + 1, p):
                    cover_count[n] += 1
        
        L = 0
        for n in range(1, L_limit + 1):
            if cover_count[n] == 0:
                break
            L = n
        return L, cover_count

    def calculate_L(cover_count, L_limit):
        L = 0
        for n in range(1, L_limit + 1):
            if cover_count[n] == 0:
                break
            L = n
        return L

    def update_counts(cover_count, p, r_old, r_new, L_limit):
        if r_old == r_new:
            return
        # Decrement
        start = r_old
        if start == 0:
            start = p
        if start <= L_limit:
            for n in range(start, L_limit + 1, p):
                cover_count[n] -= 1
        
        # Increment
        start = r_new
        if start == 0:
            start = p
        if start <= L_limit:
            for n in range(start, L_limit + 1, p):
                cover_count[n] += 1

    def init_residues():
        # Heuristic: Cover i+1 with i-th prime
        res = [0] * n_primes
        for i, p in enumerate(primes_set):
            target = i + 1
            res[i] = target % p
        return res

    best_L = 0
    best_classes = None
    best_residues = None
    
    # Main Search Loop
    # We will perform local search to maximize L.
    # To avoid getting stuck in local minima, we can use random restarts or simulated annealing-like acceptance.
    # Given the time constraint, simple hill climbing with random restarts is robust.
    
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() - start_time > DEADLINE:
            break
            
        if restart == 0:
            residues = init_residues()
        else:
            residues = [random.randint(0, p - 1) for p in primes_set]
            
        L, cover_count = evaluate_coverage(residues, L_CHECK_LIMIT)
        
        if L > best_L:
            best_L = L
            best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
            best_residues = residues[:]
            
        stuck_counter = 0
        max_stuck = 2000 # Allow more exploration per restart
        
        while time.time() - start_time < DEADLINE:
            if L >= L_CHECK_LIMIT:
                break
            
            # Find the first hole
            hole = L + 1
            
            # Try to cover this hole by changing one prime's residue
            best_new_L = L
            best_idx = -1
            best_r_new = -1
            best_r_old = -1
            
            # Optimization: Only try primes that are not already covering the hole (which is none, by definition of hole)
            # Actually, we just try all primes.
            # To speed up, we can randomize the order of primes to avoid bias.
            indices = list(range(n_primes))
            random.shuffle(indices)
            
            for idx in indices:
                p = primes_set[idx]
                r_old = residues[idx]
                r_new = hole % p
                
                if r_old == r_new:
                    continue
                
                # Apply change
                update_counts(cover_count, p, r_old, r_new, L_CHECK_LIMIT)
                residues[idx] = r_new
                
                new_L = calculate_L(cover_count, L_CHECK_LIMIT)
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_idx = idx
                    best_r_new = r_new
                    best_r_old = r_old
                    # Revert to test next prime against original state
                    update_counts(cover_count, p, r_new, r_old, L_CHECK_LIMIT)
                    residues[idx] = r_old
                else:
                    # Revert
                    update_counts(cover_count, p, r_new, r_old, L_CHECK_LIMIT)
                    residues[idx] = r_old
            
            if best_idx != -1:
                # Apply the best change found
                p = primes_set[best_idx]
                update_counts(cover_count, p, best_r_old, best_r_new, L_CHECK_LIMIT)
                residues[best_idx] = best_r_new
                L = best_new_L
                stuck_counter = 0
                
                if L > best_L:
                    best_L = L
                    best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
                    best_residues = residues[:]
            else:
                stuck_counter += 1
                if stuck_counter > max_stuck:
                    break
    
    # Final Phase: Try to extend coverage beyond L_CHECK_LIMIT if we have time
    # If best_L is close to L_CHECK_LIMIT, we can increase L_CHECK_LIMIT and continue search.
    # However, re-initializing cover_count for a larger limit is expensive.
    # Given the 115s deadline, and that L_CHECK_LIMIT=5000 is already a significant jump from 1000,
    # let's see if we can afford a second phase with L=20000.
    
    if best_L >= L_CHECK_LIMIT * 0.9 and time.time() - start_time < DEADLINE - 30:
        L_CHECK_LIMIT_2 = 20000
        # Use best_residues as start point
        residues = best_residues[:]
        L, cover_count = evaluate_coverage(residues, L_CHECK_LIMIT_2)
        
        if L > best_L:
            best_L = L
            best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
            best_residues = residues[:]
        
        stuck_counter = 0
        max_stuck = 1000
        
        while time.time() - start_time < DEADLINE - 5:
            if L >= L_CHECK_LIMIT_2:
                break
            
            hole = L + 1
            best_new_L = L
            best_idx = -1
            best_r_new = -1
            best_r_old = -1
            
            indices = list(range(n_primes))
            random.shuffle(indices)
            
            for idx in indices:
                p = primes_set[idx]
                r_old = residues[idx]
                r_new = hole % p
                
                if r_old == r_new:
                    continue
                
                update_counts(cover_count, p, r_old, r_new, L_CHECK_LIMIT_2)
                residues[idx] = r_new
                
                new_L = calculate_L(cover_count, L_CHECK_LIMIT_2)
                
                if new_L > best_new_L:
                    best_new_L = new_L
                    best_idx = idx
                    best_r_new = r_new
                    best_r_old = r_old
                    update_counts(cover_count, p, r_new, r_old, L_CHECK_LIMIT_2)
                    residues[idx] = r_old
                else:
                    update_counts(cover_count, p, r_new, r_old, L_CHECK_LIMIT_2)
                    residues[idx] = r_old
            
            if best_idx != -1:
                p = primes_set[best_idx]
                update_counts(cover_count, p, best_r_old, best_r_new, L_CHECK_LIMIT_2)
                residues[best_idx] = best_r_new
                L = best_new_L
                stuck_counter = 0
                
                if L > best_L:
                    best_L = L
                    best_classes = [[primes_set[i], residues[i]] for i in range(n_primes)]
                    best_residues = residues[:]
            else:
                stuck_counter += 1
                if stuck_counter > max_stuck:
                    break

    if best_classes is None:
        # Fallback
        best_classes = [[p, (i + 1) % p] for i, p in enumerate(primes_set)]

    return best_classes

# EVOLVE-BLOCK-END
