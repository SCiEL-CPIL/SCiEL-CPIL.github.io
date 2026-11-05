"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    N = 256
    
    # Precompute all 3-term AP conflicts for fast bitmask-based validity checks
    # For each x, we store the list of (a, b) pairs such that {x, a, b} forms a 3-term AP.
    # If x is added, we need to ensure that for no such pair are BOTH a and b already in the set.
    # 
    # A 3-term AP is (u, v, w) with u < v < w and u + w = 2v.
    # If we add x, a conflict arises if:
    # 1. x is the middle (v): exists u, w in S such that u + w = 2x.
    # 2. x is the first (u): exists v, w in S such that x + w = 2v => w = 2v - x.
    # 3. x is the last (w): exists u, v in S such that u + x = 2v => u = 2v - x.
    
    # We can unify this: For a fixed x, find all pairs (a, b) with a < b such that 
    # {x, a, b} is a 3-term AP. Then x is valid iff for all such pairs, NOT (a in S AND b in S).
    
    # Let's precompute conflicts[x] = list of (a, b) where a < b and {x, a, b} is a 3-AP.
    conflicts = [[] for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        # x as middle: a + c = 2x, a < x < c
        # Pairs are (a, c) with a < c.
        for a in range(1, x):
            c = 2 * x - a
            if c <= N and c > x:
                conflicts[x].append((a, c))
        
        # x as first: x + c = 2b, x < b < c => c = 2b - x
        # Pairs are (b, c) with b < c.
        for b in range(x + 1, N + 1):
            c = 2 * b - x
            if c <= N and c > b:
                conflicts[x].append((b, c))
        
        # x as last: a + x = 2b, a < b < x => a = 2b - x
        # Pairs are (a, b) with a < b.
        for b in range(1, x):
            a = 2 * b - x
            if a >= 1 and a < b:
                conflicts[x].append((a, b))
    
    def is_valid_addition(bitmask, x):
        """Check if x can be added to set represented by bitmask."""
        if (bitmask >> (x - 1)) & 1:
            return False
        
        for a, b in conflicts[x]:
            if ((bitmask >> (a - 1)) & 1) and ((bitmask >> (b - 1)) & 1):
                return False
        
        return True
    
    def reciprocal_sum_float(bitmask):
        """Compute float reciprocal sum for fast comparison."""
        total = 0.0
        for x in range(1, N + 1):
            if (bitmask >> (x - 1)) & 1:
                total += 1.0 / x
        return total
    
    def reciprocal_sum_frac(bitmask):
        """Compute exact reciprocal sum as Fraction."""
        total = Fraction(0, 1)
        for x in range(1, N + 1):
            if (bitmask >> (x - 1)) & 1:
                total += Fraction(1, x)
        return total
    
    def greedy_initial(order):
        """Greedy: add elements in given order if valid."""
        bitmask = 0
        for x in order:
            if is_valid_addition(bitmask, x):
                bitmask |= (1 << (x - 1))
        return bitmask
    
    def local_search(bitmask, max_iter=500, time_limit=None):
        """
        Local search with additions and swaps using bitmask.
        """
        if time_limit is None:
            time_limit = DEADLINE
        
        current = bitmask
        current_val = reciprocal_sum_float(current)
        best = bitmask
        best_val = current_val
        
        # Precompute reciprocals for speed
        recip = [0.0] + [1.0 / i for i in range(1, N + 1)]
        
        for _ in range(max_iter):
            if (time.time() - start_time) > time_limit:
                break
            
            improved = False
            
            # 1. Try single additions
            # Iterate x from 1 to N (smaller x has higher reciprocal weight)
            # We can optimize by only trying x not in current.
            for x in range(1, N + 1):
                if (current >> (x - 1)) & 1:
                    continue
                if is_valid_addition(current, x):
                    new_bitmask = current | (1 << (x - 1))
                    new_val = current_val + recip[x]
                    if new_val > best_val:
                        best_val = new_val
                        best = new_bitmask
                        current = new_bitmask
                        current_val = new_val
                        improved = True
                        break # Restart search with new state
            
            # 2. Try swaps (remove one, add one)
            if not improved:
                # For each element in current, try removing it and adding a smaller element
                # that was previously blocked or just not added.
                # To be efficient, we only look for swaps that improve the sum:
                # Remove y, add x where 1/x > 1/y => x < y.
                
                # Get list of elements in current
                elems_in_current = [x for x in range(1, N + 1) if (current >> (x - 1)) & 1]
                
                for y in elems_in_current:
                    # Try adding x < y
                    # We can break early if we find the smallest valid x?
                    # Not necessarily, because removing y might allow multiple x, 
                    # but we want the best swap. The best swap for removing y is 
                    # adding the smallest valid x not in current \ {y}.
                    
                    temp_bitmask = current & ~(1 << (y - 1))
                    best_x = None
                    # Search for smallest x < y not in temp_bitmask that is valid
                    for x in range(1, y):
                        if (temp_bitmask >> (x - 1)) & 1:
                            continue
                        if is_valid_addition(temp_bitmask, x):
                            best_x = x
                            break # Smallest x is best for this y
                    
                    if best_x is not None:
                        new_bitmask = temp_bitmask | (1 << (best_x - 1))
                        new_val = current_val - recip[y] + recip[best_x]
                        if new_val > best_val:
                            best_val = new_val
                            best = new_bitmask
                            current = new_bitmask
                            current_val = new_val
                            improved = True
                            break # Restart search with new state
            
            if not improved:
                break
        
        return best
    
    # Phase 1: Generate diverse initial solutions
    # 1. Ascending order
    S1 = greedy_initial(list(range(1, N + 1)))
    # 2. Descending order
    S2 = greedy_initial(list(range(N, 0, -1)))
    # 3. Random order
    order3 = list(range(1, N + 1))
    random.shuffle(order3)
    S3 = greedy_initial(order3)
    # 4. Order by prime status? (Primes are often good for AP-free sets)
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True
    primes = [x for x in range(1, N + 1) if is_prime(x)]
    composites = [x for x in range(1, N + 1) if not is_prime(x)]
    S4 = greedy_initial(primes + composites)
    
    candidates = [S1, S2, S3, S4]
    
    # Pick the best initial candidate
    best_bitmask = max(candidates, key=reciprocal_sum_float)
    best_val = reciprocal_sum_float(best_bitmask)
    
    # Phase 2: Local search on the best initial
    best_bitmask = local_search(best_bitmask, max_iter=200)
    best_val = reciprocal_sum_float(best_bitmask)
    
    # Phase 3: Random restarts with perturbations
    # Use remaining time for multiple restarts
    restarts = 0
    while (time.time() - start_time) < DEADLINE - 5:
        restarts += 1
        
        # Perturb the current best
        # Strategy: Remove a few random elements, then re-fill greedily
        S_perturbed = best_bitmask
        num_remove = random.randint(1, 5)
        elems = [x for x in range(1, N + 1) if (S_perturbed >> (x - 1)) & 1]
        if len(elems) > num_remove:
            to_remove = random.sample(elems, num_remove)
            for r in to_remove:
                S_perturbed &= ~(1 << (r - 1))
        
        # Re-fill with greedy in random order
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Start from perturbed set? Or just greedy from scratch?
        # Let's try a hybrid: Start with perturbed set, then try to add remaining elements greedily.
        # But valid_addition depends on the set.
        # Simpler: Just run greedy from scratch with random order, then local search.
        # Or: Take the perturbed set, and try to add elements from 1..N in random order.
        
        current = S_perturbed
        for x in order:
            if is_valid_addition(current, x):
                current |= (1 << (x - 1))
        
        # Local search on this new candidate
        current = local_search(current, max_iter=100)
        val = reciprocal_sum_float(current)
        
        if val > best_val:
            best_val = val
            best_bitmask = current
    
    # Final polish on the global best
    best_bitmask = local_search(best_bitmask, max_iter=100)
    
    # Convert bitmask to list
    result = [x for x in range(1, N + 1) if (best_bitmask >> (x - 1)) & 1]
    
    return result
# EVOLVE-BLOCK-END
