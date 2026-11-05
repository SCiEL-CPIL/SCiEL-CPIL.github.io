import time
import random
from fractions import Fraction

def construct():
    """
    Constructs a subset of [1, 256] with no non-constant 3-term arithmetic progression
    that maximizes the sum of reciprocals.
    
    Strategy:
    1. Greedy initialization based on reciprocal weight (smallest x first).
    2. Local search with removal and swap moves, followed by greedy re-filling.
    3. Randomized restarts/perturbations to escape local optima.
    4. Efficient bitset-based conflict checking for speed.
    """
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute reciprocals for speed
    recip = [0] + [1.0 / i for i in range(1, n + 1)]
    
    # Precompute conflict graph using bitsets for fast intersection checks
    # conflicts[i] is a bitmask where bit j is set if i and j form a 3-AP with some k
    # Actually, simpler: for each x, precompute the set of y such that x and y conflict.
    # But since n=256, we can use Python integers as bitsets.
    
    # conflicts[x] will be a bitmask of all y != x such that {x, y, z} is a 3-AP for some z in [1, n].
    # Wait, the condition is that adding x to a set S is invalid if there exists a, b in S such that {x, a, b} is a 3-AP.
    # This is equivalent to: x conflicts with a if there exists b such that {x, a, b} is a 3-AP.
    # And x conflicts with b if there exists a such that {x, a, b} is a 3-AP.
    # So we can precompute for each x, the set of all y that conflict with x.
    
    conflicts = [0] * (n + 1)
    for x in range(1, n + 1):
        mask = 0
        # Case 1: x is an endpoint. a, x, b or b, x, a.
        # 2*x = a + b => b = 2*x - a.
        # For any a in [1, n], if b = 2*x - a is in [1, n] and b != x, then a and b conflict via x?
        # No, a and b don't conflict directly. x conflicts with a if b is in the set.
        # The check "can_add(x, S)" fails if (S & conflicts[x]) is not empty?
        # Let's define conflicts[x] as the set of all y such that if y is in S, then x cannot be added?
        # No, that's not right either. If y is in S, x can be added unless there is ANOTHER element z in S that completes the AP.
        
        # Correct bitset approach:
        # We maintain a bitmask `in_set_mask` of the current set.
        # To check if x can be added:
        # We need to ensure that for all a in S, 2*x - a is NOT in S (and != x).
        # And for all a in S, 2*a - x is NOT in S (and != x).
        # And for all a in S, (a+x)/2 is NOT in S (if integer, and != x, a).
        
        # This is hard to do with a single precomputed mask per x because it depends on pairs in S.
        # However, we can precompute for each x, the set of "forbidden" elements?
        # If we add x, it creates conflicts with existing elements.
        # Actually, the most efficient way with bitsets is:
        # For a given x, the elements that CANNOT be in S if x is to be added are those that form an AP with x and another element in S.
        # This is still pairwise.
        
        # Let's stick to the list-based `can_add` but optimize it by iterating only over set bits.
        # Or, precompute for each x, the list of (a, b) pairs such that {a, x, b} is a 3-AP.
        # If a is in S, then b must NOT be in S.
        
        # Let's try a different bitset trick:
        # Precompute `ap_conflicts[x]` as a list of pairs (a, b) such that {a, x, b} is a 3-AP.
        # Then `can_add(x, S)` is False if there exists a pair (a, b) in `ap_conflicts[x]` such that a in S and b in S.
        
        pass # We will implement a faster can_add using precomputed pairs.

    # Precompute AP pairs for each x
    # pairs[x] = list of (a, b) such that a + b = 2*x (x is middle)
    #           OR a + x = 2*b (x is endpoint, b is middle? No, a, b, x AP => a+x=2b)
    #           OR x + b = 2*a (x is endpoint, a is middle)
    
    # Let's just precompute for each x, the set of elements y such that x and y are part of a 3-AP with some z.
    # This is not enough. We need to know if adding x creates a 3-AP with TWO existing elements.
    
    # Optimized can_add using bitsets:
    # Let S be a bitmask.
    # 1. x is middle: need a, b in S such that a + b = 2x.
    #    This is equivalent to: for each a in S, check if 2x - a is in S.
    #    We can precompute for each x, a list of pairs (a, b) with a+b=2x.
    #    Then check if any (a, b) has both in S.
    
    # 2. x is endpoint (smaller): need a, b in S such that x + b = 2a => b = 2a - x.
    #    For each a in S, check if 2a - x is in S.
    
    # 3. x is endpoint (larger): need a, b in S such that a + x = 2b => a = 2b - x.
    #    For each b in S, check if 2b - x is in S.
    
    # We can precompute for each x, three lists of pairs:
    # mid_pairs[x]: [(a, b) such that a+b=2x, a<b]
    # left_pairs[x]: [(a, b) such that x+b=2a, a!=b] -> b = 2a-x. Pair (a, b).
    # right_pairs[x]: [(a, b) such that a+x=2b, a!=b] -> a = 2b-x. Pair (a, b).
    
    # Actually, left and right are symmetric if we just swap roles.
    # Let's just precompute `conflict_pairs[x]` as a list of (u, v) such that if u and v are both in S, then x cannot be added.
    
    conflict_pairs = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        # Case 1: x is middle. a + b = 2x.
        for a in range(1, x):
            b = 2 * x - a
            if b > x and b <= n:
                conflict_pairs[x].append((a, b))
        
        # Case 2: x is smallest endpoint. x, m, b. x + b = 2m => b = 2m - x.
        # m must be > x. b must be > m.
        for m in range(x + 1, n + 1):
            b = 2 * m - x
            if b > m and b <= n:
                conflict_pairs[x].append((m, b))
        
        # Case 3: x is largest endpoint. a, m, x. a + x = 2m => a = 2m - x.
        # m must be < x. a must be < m.
        for m in range(1, x):
            a = 2 * m - x
            if 1 <= a < m:
                conflict_pairs[x].append((a, m))

    def can_add(x, in_set_mask):
        """
        Check if x can be added to the set represented by in_set_mask (bitmask).
        Returns True if valid.
        """
        for u, v in conflict_pairs[x]:
            if (in_set_mask >> u) & 1 and (in_set_mask >> v) & 1:
                return False
        return True

    def greedy_fill(in_set_mask):
        """Add all valid numbers in increasing order."""
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if not ((in_set_mask >> x) & 1):
                    if can_add(x, in_set_mask):
                        in_set_mask |= (1 << x)
                        changed = True
        return in_set_mask

    def get_set(in_set_mask):
        return [i for i in range(1, n + 1) if (in_set_mask >> i) & 1]

    def get_sum(in_set_mask):
        s = 0.0
        for i in range(1, n + 1):
            if (in_set_mask >> i) & 1:
                s += recip[i]
        return s

    # Initialize with standard greedy
    in_set_mask = 0
    for x in range(1, n + 1):
        if can_add(x, in_set_mask):
            in_set_mask |= (1 << x)
    
    greedy_fill(in_set_mask)
    
    best_in_set_mask = in_set_mask
    best_sum = get_sum(best_in_set_mask)
    
    def local_search(current_mask):
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            current_sum = get_sum(current_mask)
            
            # Try removing one element and re-filling
            for rem in range(1, n + 1):
                if not ((current_mask >> rem) & 1):
                    continue
                if time.time() > deadline - 5.0:
                    break
                
                trial_mask = current_mask & ~(1 << rem)
                greedy_fill(trial_mask)
                new_sum = get_sum(trial_mask)
                
                if new_sum > current_sum + 1e-10:
                    current_mask = trial_mask
                    current_sum = new_sum
                    improved = True
                    break
            
            # Try swapping one element with another
            if not improved:
                for a in range(1, n + 1):
                    if not ((current_mask >> a) & 1):
                        continue
                    if time.time() > deadline - 5.0:
                        break
                    for b in range(a + 1, n + 1):
                        if (current_mask >> b) & 1:
                            continue
                        # Try removing a and adding b
                        trial_mask = current_mask & ~(1 << a)
                        if can_add(b, trial_mask):
                            trial_mask |= (1 << b)
                            greedy_fill(trial_mask)
                            new_sum = get_sum(trial_mask)
                            if new_sum > current_sum + 1e-10:
                                current_mask = trial_mask
                                current_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
        return current_mask

    best_in_set_mask = local_search(best_in_set_mask)
    best_sum = get_sum(best_in_set_mask)
    
    # Randomized Restart / Perturbation
    random.seed(42)
    
    while time.time() < deadline - 10.0:
        # Create a perturbed version of the best set
        perturbed_mask = best_in_set_mask
        
        # Remove 1 to 3 random elements
        num_removals = random.randint(1, 3)
        candidates_to_remove = [i for i in range(1, n + 1) if (perturbed_mask >> i) & 1]
        if not candidates_to_remove:
            break
            
        for _ in range(num_removals):
            if not candidates_to_remove:
                break
            idx = random.randint(0, len(candidates_to_remove) - 1)
            rem_val = candidates_to_remove[idx]
            perturbed_mask &= ~(1 << rem_val)
            candidates_to_remove.pop(idx)
        
        # Re-greedy fill
        greedy_fill(perturbed_mask)
        
        # Local search on the perturbed set
        improved_mask = local_search(perturbed_mask)
        new_sum = get_sum(improved_mask)
        
        if new_sum > best_sum + 1e-10:
            best_sum = new_sum
            best_in_set_mask = improved_mask

    result = get_set(best_in_set_mask)
    
    # Final validation
    assert len(result) > 0
    s_set = set(result)
    for i, a in enumerate(result):
        for b in result[i+1:]:
            if (a + b) % 2 == 0:
                mid = (a + b) // 2
                if mid != a and mid != b and mid in s_set:
                    raise AssertionError("3-AP found")
    
    return result
