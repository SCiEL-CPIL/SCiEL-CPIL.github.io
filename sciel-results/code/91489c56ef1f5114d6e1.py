import time
import sys

def construct():
    deadline = time.time() + 85
    
    # Target: find 12 distinct integers <= 10000
    # Minimize number of distinct prime factors of pairwise sums
    
    # Strategy: Try to find a set of 12 numbers such that all pairwise sums 
    # are composed only of primes from a small set P.
    # If we find such a set for P, the answer is |P|.
    
    # We try sets of primes in increasing order of size.
    # Common small prime sets:
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13]
    ]
    
    # Precompute smooth numbers for max sum 20000
    MAX_SUM = 20000
    
    def get_smooth_numbers(primes, limit):
        smooths = [1]
        # Simple generation: iterate and multiply
        # Better: recursive generation
        res = set()
        
        def gen(idx, val):
            if val > limit:
                return
            res.add(val)
            if idx == len(primes):
                return
            p = primes[idx]
            # Multiply by p repeatedly
            v = val
            while v <= limit:
                v *= p
                if v > limit:
                    break
                gen(idx + 1, v)
        
        # Start with 1
        gen(0, 1)
        # Ensure 1 is included? Sums are >= 1+2=3, so 1 is not a sum, but useful for generation.
        # Actually, we just need the set of smooth numbers <= limit.
        return list(res)
    
    # Cache smooth numbers
    smooth_cache = {}
    for ps in prime_sets:
        key = tuple(ps)
        if key not in smooth_cache:
            smooth_cache[key] = get_smooth_numbers(ps, MAX_SUM)
    
    # Helper to check if a number is in the smooth set (using a set for O(1) lookup)
    smooth_set_cache = {}
    for ps in prime_sets:
        key = tuple(ps)
        smooth_set_cache[key] = set(smooth_cache[key])
        
    def find_clique_for_prime_set(primes, n=12, max_val=10000):
        """
        Finds n distinct integers <= max_val such that all pairwise sums are smooth w.r.t primes.
        Returns the list of integers or None.
        """
        smooth_set = smooth_set_cache[tuple(primes)]
        
        # We will iterate over potential x_1
        # To make it efficient, we can restrict x_1 to values that have many smooth-sum partners.
        # But let's just iterate and use a fast clique finder on the induced subgraph.
        
        # Optimization: If we find a solution, return it.
        
        # We can sample x_1 or iterate all.
        # Given 10000 is small, iterating all is feasible if the inner loop is fast.
        # Inner loop: Build candidate pool C(x_1), find clique of size n-1.
        
        # Precompute compatibility for the whole range? No, 10000x10000 is 10^8, too big for Python sets of sets.
        # Instead, for each x_1, compute C(x_1) on the fly.
        
        # To speed up, we can sort x_1 candidates.
        # Let's try iterating x_1 from 1 to 10000.
        
        for x1 in range(1, max_val + 1):
            # Check time
            if time.time() > deadline - 5:
                return None
            
            # Generate candidates for other numbers
            # y must satisfy 1 <= y <= 10000, y != x1, and x1+y in smooth_set
            candidates = []
            for s in smooth_set:
                y = s - x1
                if 1 <= y <= max_val and y != x1:
                    candidates.append(y)
            
            # Remove duplicates
            candidates = list(set(candidates))
            
            if len(candidates) < n - 1:
                continue
            
            # We need to find a clique of size n-1 in the graph of candidates
            # where edge (u,v) exists if u+v in smooth_set
            
            # Sort candidates by "degree" (number of compatible partners in candidates)
            # to help backtracking
            # This requires O(|C|^2) checks. |C| is small (~few hundreds/thousands).
            
            # To avoid O(|C|^2) for every x1, we can be smarter.
            # But let's try this first.
            
            # Build a set of candidates for O(1) lookup
            cand_set = set(candidates)
            
            # Calculate degrees
            degs = []
            for u in candidates:
                deg = 0
                for v in candidates:
                    if u < v and (u + v) in smooth_set:
                        deg += 1
                degs.append(deg)
            
            # Sort candidates by degree descending
            # Create list of (deg, val)
            sorted_cands = sorted(zip(degs, candidates), key=lambda x: -x[0])
            
            # Backtracking search for clique of size n-1
            target_size = n - 1
            found_clique = []
            
            def backtrack(start_idx, current_clique):
                if len(current_clique) == target_size:
                    return current_clique[:]
                
                # Pruning: if not enough candidates left
                if len(current_clique) + (len(sorted_cands) - start_idx) < target_size:
                    return None
                
                for i in range(start_idx, len(sorted_cands)):
                    val = sorted_cands[i][1]
                    
                    # Check if val is compatible with all in current_clique
                    compatible = True
                    for v in current_clique:
                        if (v + val) not in smooth_set:
                            compatible = False
                            break
                    
                    if not compatible:
                        continue
                    
                    current_clique.append(val)
                    result = backtrack(i + 1, current_clique)
                    if result is not None:
                        return result
                    current_clique.pop()
                
                return None
            
            clique = backtrack(0, [])
            if clique is not None:
                # We found a clique of size n-1 in candidates
                # Add x1
                full_set = [x1] + clique
                # Verify distinctness and bounds
                if len(set(full_set)) == n and all(1 <= x <= max_val for x in full_set):
                    return full_set
                    
        return None

    # Try prime sets in order of size
    best_solution = None
    best_prime_count = float('inf')
    
    for ps in prime_sets:
        if time.time() > deadline - 10:
            break
            
        sol = find_clique_for_prime_set(ps)
        if sol is not None:
            best_solution = sol
            best_prime_count = len(ps)
            # Since we are trying in order of size, the first one found is optimal 
            # within this family of prime sets.
            # However, a larger set of primes might yield a solution with fewer distinct primes 
            # if the "distinct primes" count is different? No, the count is exactly |ps| 
            # if all sums use only primes from ps.
            # So smaller |ps| is better.
            break
            
    if best_solution is None:
        # Fallback: Use the seed-like heuristic or a simple set
        # If we couldn't find a smooth clique, we return a heuristic solution.
        # The seed program returned a value. We should try to beat it.
        # If we fail the clique search, we can use a random local search with a tighter budget.
        
        # Let's implement a quick local search as fallback
        import random
        random.seed(42)
        
        def _pf(n):
            if n <= 1: return set()
            s = set()
            while n % 2 == 0: s.add(2); n //= 2
            while n % 3 == 0: s.add(3); n //= 3
            f = 5
            while f * f <= n:
                while n % f == 0: s.add(f); n //= f
                f += 2
            if n > 1: s.add(n)
            return s

        def score(xs):
            pr = set()
            for i in range(len(xs)):
                for j in range(i+1, len(xs)):
                    pr |= _pf(xs[i] + xs[j])
            return len(pr)
            
        # Initialize with a set that might be good
        # e.g. multiples of 60 (2,3,5)
        base = [60 * i for i in range(1, 13)]
        best_x = base[:]
        best_s = score(best_x)
        
        for _ in range(1000):
            if time.time() > deadline - 2:
                break
            xs = best_x[:]
            i = random.randint(0, 11)
            old = xs[i]
            new_x = random.randint(1, 10000)
            if new_x in xs: continue
            xs[i] = new_x
            s = score(xs)
            if s < best_s:
                best_s = s
                best_x = xs[:]
                
        return best_x

    return best_solution

# EVOLVE-BLOCK-END
