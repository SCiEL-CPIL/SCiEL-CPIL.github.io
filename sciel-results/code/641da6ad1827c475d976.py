"""Improved: Conflict graph + greedy + local search with optimized checks."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

DEADLINE = 85.0  # safety margin before 90s kill

def construct():
    start_time = time.time()
    n_max = 192
    
    def elapsed():
        return time.time() - start_time
    
    # Precompute: for each a in 1..192, find all pairs (b,c) with b<c, b,c != a,
    # such that (b+c) % a == 0. Then when checking if we can add x to set S,
    # for each a in S, we need to check if there exist b,c in S\{a,x} with (x+b+c)%a==0
    # i.e., (b+c) % a == (-x) % a
    
    # For each a, precompute a dictionary: residue -> list of pairs (b,c) where (b+c)%a == residue
    # But this is huge. Instead, let's use a different approach.
    
    # Key insight: For a fixed set S and candidate x, the conflict check is:
    # 1. For x: check if x divides sum of any 3 elements in S (all distinct from x)
    # 2. For each a in S: check if a divides x + b + c for any b,c in S\{a,x}
    
    # Let's build a conflict graph approach:
    # Two elements conflict if they cannot both be in a valid set.
    # But the conflict depends on the rest of the set, so it's not pairwise.
    
    # Alternative: Use a smarter greedy + local search with fast incremental checks.
    
    def can_add_fast(S_set, x):
        """Check if adding x to S_set maintains validity. S_set is a set of ints."""
        if x in S_set:
            return True
        
        # Check 1: Does x divide sum of any 3 elements in S_set?
        s_list = sorted(S_set)
        if len(s_list) >= 3:
            for t in combinations(s_list, 3):
                if sum(t) % x == 0:
                    return False
        
        # Check 2: For each a in S_set, does a divide x + b + c for any b,c in S_set\{a,x}?
        # This is equivalent to: for each a in S_set, is there a pair (b,c) in S_set\{a,x}
        # such that (x + b + c) % a == 0?
        # i.e., (b + c) % a == (-x) % a
        
        for a in S_set:
            if a == x:
                continue
            target_res = (-x) % a
            # Find if there exist b, c in S_set \ {a, x} with (b+c) % a == target_res
            # Use a set-based approach: for each b in S_set\{a,x}, check if there's c in S_set\{a,x,b}
            # with (c % a) == (target_res - b % a) % a
            
            # Build a dict of residues for elements in S_set\{a,x}
            others = [v for v in S_set if v != a and v != x]
            if len(others) < 2:
                continue
            
            # For each b in others, we need c in others\{b} such that (b+c) % a == target_res
            # i.e., c % a == (target_res - b % a) % a
            residue_map = {}
            for v in others:
                r = v % a
                if r not in residue_map:
                    residue_map[r] = []
                residue_map[r].append(v)
            
            for b in others:
                need_res = (target_res - b % a) % a
                if need_res in residue_map:
                    # Need a c in residue_map[need_res] that is not b
                    for c in residue_map[need_res]:
                        if c != b:
                            return False
        
        return True
    
    # Phase 1: Multiple greedy constructions with different orderings
    best_set = set()
    best_size = 0
    
    # Try several strategies
    strategies = []
    
    # Strategy 1: Descending order
    strategies.append(list(range(192, 0, -1)))
    
    # Strategy 2: Ascending order (small numbers first - they're more restrictive)
    strategies.append(list(range(1, 193)))
    
    # Strategy 3: Primes first (primes > 96 can't divide sums of 3 numbers <= 192 unless sum >= prime)
    # Actually, a prime p > 96 can only divide a sum if the sum is >= p.
    # Sums of 3 numbers from 1..192 range from 6 to 573.
    # So primes in (96, 573] could potentially divide sums.
    # But primes > 192 can never divide a sum of 3 numbers from 1..192 (max sum = 190+191+192=573)
    # Wait, 573 > 192, so primes between 193 and 573 could divide sums.
    # But we're only selecting numbers from 1..192, so the divisor must be in 1..192.
    # Primes in 1..192: a prime p divides sum s if s is a multiple of p.
    # For p > 96, the only multiple of p in range [6, 573] that's achievable as sum of 3 distinct numbers from 1..192
    # is p itself (if p >= 6) or 2p (if 2p <= 573 and 2p >= 6).
    # For p in (96, 192], 2p is in (192, 384], which is <= 573.
    # So primes in (96, 192] could divide sums equal to p or 2p.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191]
    non_primes = [x for x in range(1, 193) if x not in primes]
    strategies.append(primes + non_primes)  # Primes first
    
    # Strategy 4: Large numbers first, then small
    large = list(range(192, 96, -1))
    small = list(range(96, 0, -1))
    strategies.append(large + small)
    
    # Strategy 5: Random shuffles
    random.seed(42)
    for _ in range(10):
        shuffled = list(range(1, 193))
        random.shuffle(shuffled)
        strategies.append(shuffled)
    
    for order in strategies:
        if elapsed() > DEADLINE:
            break
        
        current_set = set()
        for x in order:
            if elapsed() > DEADLINE:
                break
            if can_add_fast(current_set, x):
                current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    # Phase 2: Local search - try swaps to increase size
    # For each element in best_set, try removing it and adding a different element
    improved = True
    while improved and elapsed() < DEADLINE:
        improved = False
        current_best = set(best_set)
        
        # Try adding elements not in the set
        for x in range(1, 193):
            if elapsed() > DEADLINE:
                break
            if x not in current_best:
                if can_add_fast(current_best, x):
                    current_best.add(x)
                    improved = True
        
        # Try swaps: remove one element, try to add others
        for a in list(current_best):
            if elapsed() > DEADLINE:
                break
            
            # Try removing a and adding a different element
            temp_set = set(current_best)
            temp_set.remove(a)
            
            # Try to add elements that weren't in the set
            added_any = False
            for x in range(1, 193):
                if elapsed() > DEADLINE:
                    break
                if x not in temp_set:
                    if can_add_fast(temp_set, x):
                        temp_set.add(x)
                        added_any = True
                        break  # Only add one per removal attempt
            
            if added_any and len(temp_set) > len(current_best):
                current_best = set(temp_set)
                improved = True
        
        if len(current_best) > best_size:
            best_size = len(current_best)
            best_set = set(current_best)
    
    # Phase 3: More aggressive local search with multiple restarts
    for restart in range(5):
        if elapsed() > DEADLINE:
            break
        
        # Start from a random subset of best_set and try to grow
        if best_size > 10:
            sample_size = min(best_size, max(10, best_size - 5))
            sample = random.sample(sorted(best_set), sample_size)
            current_set = set(sample)
        else:
            current_set = set(best_set)
        
        # Try to add elements
        for x in range(192, 0, -1):
            if elapsed() > DEADLINE:
                break
            if x not in current_set:
                if can_add_fast(current_set, x):
                    current_set.add(x)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
