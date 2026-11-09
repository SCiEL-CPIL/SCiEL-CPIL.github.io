import random
import time
from fractions import Fraction

def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

def construct():
    t0 = time.time()
    DEADLINE = 145.0
    MAXL = 100000
    primes = _primes_upto(1000)
    
    # Precompute bitmasks for each prime and residue
    # mask[p][r] = bitmask of positions covered by (p, r) in 1..MAXL
    # This is too large: 168 primes * up to 997 residues * 100000 bits
    # Instead, compute on the fly
    
    # Better: for each prime p, precompute the list of positions for each residue
    # positions[p][r] = [x for x in 1..MAXL if x % p == r]
    
    # Even better: use bitsets computed on demand
    
    # Strategy: 
    # 1. Greedy initialization to get a good starting point
    # 2. Local search / branch and bound to improve
    
    # Let me use a different approach: 
    # Process primes in order of decreasing 1/p (i.e., increasing p is cheaper)
    # No wait, we want high coverage, so process small primes first (they cover more)
    
    # Actually, let me use a simulated annealing approach with bitset coverage
    
    # Precompute: for each prime p, the bitmask of positions covered by residue r
    # mask_p_r = sum(1 << x for x in range(1, MAXL+1) if x % p == r)
    # But this is expensive to precompute for all p, r
    
    # Instead, compute the mask for a given (p, r) on the fly:
    # def make_mask(p, r):
    #     m = 0
    #     start = r if r >= 1 else p
    #     for x in range(start, MAXL+1, p):
    #         m |= (1 << x)
    #     return m
    # This is O(MAXL/p) which is fine for large p but slow for p=2
    
    # For p=2: 50000 iterations, for p=3: 33333, etc.
    # Total for all primes: sum(MAXL/p) ≈ MAXL * sum(1/p) ≈ 100000 * 2.2 ≈ 220000
    # But we only need this for the primes we actually use
    
    # Let me use a count array approach but with numpy for speed
    
    import numpy as np
    
    # Count array approach with numpy
    # count[x] = number of classes covering position x
    # L = largest x such that count[1..x] all > 0
    
    # Use numpy for vectorized operations
    
    def make_count(p, r, maxl):
        """Create a count array for one class (p, r)."""
        arr = np.zeros(maxl + 1, dtype=np.int32)
        start = r if r >= 1 else p
        if start <= maxl:
            arr[start::p] += 1
        return arr
    
    # Precompute count arrays for all (p, r) pairs? Too much memory.
    # 168 primes * avg ~500 residues * 100001 * 4 bytes ≈ 3.4 GB. Too much.
    
    # Instead, compute on the fly and cache only what we use.
    
    # Let me use a different strategy: 
    # 1. Start with a greedy solution
    # 2. Use local search: try changing one prime's residue, or swapping primes
    # 3. Use simulated annealing to escape local optima
    
    # For the local search, we need fast evaluation of L.
    # L = first position where count == 0, minus 1.
    # With numpy: L = np.argmax(count[1:] == 0) - 1 (if any zero), else MAXL
    
    # Let me implement this.
    
    best_L = 0
    best_solution = []
    
    # Phase 1: Greedy initialization
    def greedy_init(seed):
        import random
        rng = random.Random(seed)
        count = np.zeros(MAXL + 1, dtype=np.int32)
        used = set()
        budget = Fraction(1, 1)
        solution = []
        
        # Try primes in random order
        order = primes[:]
        rng.shuffle(order)
        
        for p in order:
            if time.time() > DEADLINE:
                break
            if Fraction(1, p) > budget:
                continue
            # Find best residue for this prime
            best_gain = 0
            best_r = None
            # Current L
            # Find first zero in count[1:]
            zeros = np.where(count[1:] == 0)[0]
            if len(zeros) == 0:
                cur_L = MAXL
            else:
                cur_L = zeros[0]  # 0-indexed, so position is zeros[0]+1
            
            # For each residue, compute gain
            # Gain = number of new positions covered in 1..cur_L+K
            K = 500
            for r in range(p):
                start = r if r >= 1 else p
                if start > cur_L + K:
                    continue
                # Count how many positions in 1..cur_L+K are uncovered and would be covered
                positions = range(start, min(cur_L + K + 1, MAXL + 1), p)
                gain = sum(1 for x in positions if count[x] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_r is not None and best_gain > 0:
                count[best_r if best_r >= 1 else p::p] += 1
                used.add(p)
                budget -= Fraction(1, p)
                solution.append((p, best_r))
        
        # Compute L
        zeros = np.where(count[1:] == 0)[0]
        L = zeros[0] if len(zeros) > 0 else MAXL
        return L, solution, count, budget
    
    # Run greedy with multiple seeds
    for seed in range(50):
        if time.time() > DEADLINE:
            break
        L, sol, count, budget = greedy_init(seed)
        if L > best_L:
            best_L = L
            best_solution = sol[:]
            best_count = count.copy()
            best_budget = budget
    
    if not best_solution:
        # Fallback
        return [[2, 1], [3, 2]]
    
    # Phase 2: Local search
    # Try: change residue of one prime, add a new prime, remove a prime, swap primes
    
    def compute_L(count):
        """Compute the covered prefix length."""
        zeros = np.where(count[1:] == 0)[0]
        return zeros[0] if len(zeros) > 0 else MAXL
    
    def try_change_residue(count, solution, budget, p, new_r):
        """Try changing residue of prime p to new_r."""
        # Find current residue
        old_r = None
        for (pp, rr) in solution:
            if pp == p:
                old_r = rr
                break
        if old_r is None or old_r == new_r:
            return False, 0
        
        # Remove old
        old_start = old_r if old_r >= 1 else p
        count[old_start::p] -= 1
        # Add new
        new_start = new_r if new_r >= 1 else p
        count[new_start::p] += 1
        
        new_L = compute_L(count)
        
        # Update solution
        for i, (pp, rr) in enumerate(solution):
            if pp == p:
                solution[i] = (p, new_r)
                break
        
        return True, new_L
    
    # Local search: iterate, try all single changes
    no_improve = 0
    max_iters = 5000
    iters = 0
    
    while iters < max_iters and time.time() < DEADLINE:
        iters += 1
        improved = False
        
        # Try changing residue of each used prime
        for i, (p, r) in enumerate(best_solution):
            if time.time() > DEADLINE:
                break
            # Try a few random residues
            for _ in range(5):
                new_r = random.randrange(p)
                if new_r == r:
                    continue
                # Try the change
                old_r = r
                old_start = old_r if old_r >= 1 else p
                new_start = new_r if new_r >= 1 else p
                
                best_count[old_start::p] -= 1
                best_count[new_start::p] += 1
                new_L = compute_L(best_count)
                
                if new_L > best_L:
                    best_L = new_L
                    best_solution[i] = (p, new_r)
                    improved = True
                    no_improve = 0
                else:
                    # Revert
                    best_count[new_start::p] -= 1
                    best_count[old_start::p] += 1
        
        # Try adding a new prime
        for p in primes:
            if p in [pp for pp, _ in best_solution]:
                continue
            if Fraction(1, p) > best_budget:
                continue
            # Find best residue
            best_gain = 0
            best_r = None
            cur_L = best_L
            K = 500
            for r in range(p):
                start = r if r >= 1 else p
                if start > cur_L + K:
                    continue
                positions = range(start, min(cur_L + K + 1, MAXL + 1), p)
                gain = sum(1 for x in positions if best_count[x] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            if best_r is not None and best_gain > 0:
                start = best_r if best_r >= 1 else p
                best_count[start::p] += 1
                new_L = compute_L(best_count)
                if new_L > best_L:
                    best_L = new_L
                    best_solution.append((p, best_r))
                    best_budget -= Fraction(1, p)
                    improved = True
                    no_improve = 0
                else:
                    best_count[start::p] -= 1
        
        # Try removing a prime (if budget is tight, this might allow adding better primes)
        # Skip for now
        
        if not improved:
            no_improve += 1
            if no_improve > 100:
                break
    
    # Final check
    result = [[p, r] for p, r in best_solution]
    s = Fraction(0, 1)
    for p, _ in result:
        s += Fraction(1, p)
    if s > Fraction(1, 1):
        # Fallback
        result = [[2, 1], [3, 2]]
    
    return result
