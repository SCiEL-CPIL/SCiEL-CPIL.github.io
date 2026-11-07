"""Rewritten program: bitset-based backtracking for divisibility-incomparable subset sums."""
import time
from functools import lru_cache

# EVOLVE-BLOCK-START
P = {'n': 256, 'maximum_size': 30}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute: for each value v in 1..256, which other values w have v|w or w|v?
    # We need: no two distinct subset sums s1, s2 such that s1|s2 or s2|s1.
    # Also no two subsets can have the same sum (collision -> 2s divisible by s).
    
    # Strategy: incremental construction with bitset tracking of "forbidden" sums.
    # A sum is "forbidden" if adding it would create a divisibility conflict with any existing sum.
    
    # Max possible subset sum: 30 * 256 = 7680. Let's use max_sum = 7680.
    MAX_SUM = 30 * 256  # 7680
    
    # forbidden[v] = True means we cannot have any subset sum equal to v
    # When we have a set of subset sums S, a new sum s is valid if:
    #   1. s not in S (no collision)
    #   2. For all t in S: s does not divide t and t does not divide s
    
    # Precompute divisibility relation: div_conflict[a][b] = True if a|b or b|a (and a != b)
    # Actually we just need: for a given s, which existing sums t conflict with s?
    # t conflicts with s if s|t or t|s (and s != t)
    
    # For speed, precompute for each s in 1..MAX_SUM:
    #   multiples_of[s] = set of multiples of s up to MAX_SUM (excluding s itself)
    #   divisors_of[s] = set of proper divisors of s
    
    # But iterating over all existing sums each time is O(|S|). With bitsets we can do better.
    
    # Alternative approach: maintain a bitset of "occupied" sums and a bitset of "forbidden" sums.
    # forbidden = union over all occupied sums t of {s : s|t or t|s, s != t}
    # When we add a new sum s, we update forbidden to include all s' where s'|s or s|s'.
    
    # Let's use Python integers as bitsets (arbitrary precision).
    
    def precompute_conflict_masks(max_sum):
        """For each s, compute a bitset of all t != s such that s|t or t|s."""
        masks = [0] * (max_sum + 1)
        for s in range(1, max_sum + 1):
            mask = 0
            # Multiples of s (proper): 2s, 3s, ...
            m = 2 * s
            while m <= max_sum:
                mask |= (1 << m)
                m += s
            # Divisors of s (proper): d where d|s and d != s
            for d in range(1, s):
                if s % d == 0:
                    mask |= (1 << d)
            masks[s] = mask
        return masks
    
    conflict_masks = precompute_conflict_masks(MAX_SUM)
    
    # Greedy + backtracking hybrid:
    # Try to build a set of size k by adding elements one at a time.
    # At each step, the new element x generates new subset sums.
    # We track: current subset sums (as a set), and check validity.
    
    # For efficiency, use a recursive backtracker with memoization on (remaining elements, current sum bitset).
    # But the state space is huge. Instead, use greedy with multiple orderings + local search.
    
    # Better: use a randomized greedy with restarts, each trying to maximize size.
    
    import random
    
    best_set = [31, 32, 34]  # fallback
    
    def is_valid_full(elements):
        """Check if the set of elements is valid (all subset sums distinct and divisibility-incomparable)."""
        if len(elements) > 30:
            return False
        # Compute all nonempty subset sums
        sums = set()
        n = len(elements)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += elements[i]
            if s in sums:
                return False
            sums.add(s)
        # Check divisibility
        sl = sorted(sums)
        for i in range(len(sl)):
            for j in range(i + 1, len(sl)):
                if sl[j] % sl[i] == 0:
                    return False
        return True
    
    def greedy_construct(order, target_size=None):
        """Try to build a valid set by adding elements in the given order."""
        current = []
        current_sums = set()
        
        for x in order:
            if x in current:
                continue
            # Compute new subset sums
            new_sums = set()
            valid = True
            for s in current_sums:
                ns = s + x
                if ns in current_sums or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if not valid:
                continue
            new_sums.add(x)
            
            # Check divisibility conflicts
            all_sums = current_sums | new_sums
            for a in all_sums:
                for b in all_sums:
                    if a == b:
                        continue
                    if b % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            current.append(x)
            current_sums = all_sums
            if target_size and len(current) >= target_size:
                break
        
        return current
    
    # Try multiple random orderings
    candidates = list(range(1, 257))
    
    # First, try some heuristic orderings
    # 1. Random
    # 2. Descending (large numbers first - fewer collisions)
    # 3. Ascending
    # 4. Random with bias toward primes or numbers with few divisors
    
    def count_divisors(n):
        c = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                c += 1
                if i != n // i:
                    c += 1
        return c
    
    # Order by number of divisors (fewer divisors = less likely to cause conflicts)
    div_order = sorted(candidates, key=lambda x: (count_divisors(x), x))
    
    # Try descending
    desc_order = sorted(candidates, reverse=True)
    
    # Try ascending
    asc_order = sorted(candidates)
    
    # Try several random orderings
    rng = random.Random(42)
    
    orderings = [desc_order, asc_order, div_order]
    for _ in range(50):
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    best_size = len(best_set)
    
    for order in orderings:
        if time.time() > deadline:
            break
        result = greedy_construct(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    # Now try backtracking to extend the best set found
    # Use a smarter approach: start from best_set and try to add more elements
    
    def try_extend(base_set, candidates_list):
        """Try to add elements from candidates_list to base_set to make it larger."""
        current = list(base_set)
        current_sums = set()
        n = len(current)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += current[i]
            current_sums.add(s)
        
        # Quick validity check
        sl = sorted(current_sums)
        for i in range(len(sl)):
            for j in range(i + 1, len(sl)):
                if sl[j] % sl[i] == 0:
                    return current
        
        for x in candidates_list:
            if x in current:
                continue
            new_sums = set()
            valid = True
            for s in current_sums:
                ns = s + x
                if ns in current_sums or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if not valid:
                continue
            new_sums.add(x)
            
            all_sums = current_sums | new_sums
            valid = True
            sl2 = sorted(all_sums)
            for i in range(len(sl2)):
                for j in range(i + 1, len(sl2)):
                    if sl2[j] % sl2[i] == 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            current.append(x)
            current_sums = all_sums
        
        return current
    
    # Try to extend best_set with remaining candidates
    if time.time() < deadline:
        remaining = [x for x in range(1, 257) if x not in best_set]
        # Try extending with different orderings of remaining
        extended = try_extend(best_set, remaining)
        if len(extended) > best_size:
            best_size = len(extended)
            best_set = extended
        
        # Try more random extensions
        for _ in range(20):
            if time.time() > deadline:
                break
            shuffled_rem = [x for x in range(1, 257) if x not in best_set]
            rng.shuffle(shuffled_rem)
            extended = try_extend(best_set, shuffled_rem)
            if len(extended) > best_size:
                best_size = len(extended)
                best_set = extended
    
    # Final verification
    if not is_valid_full(best_set):
        # Fallback to known valid set
        best_set = [31, 32, 34]
    
    return list(best_set)
# EVOLVE-BLOCK-END
