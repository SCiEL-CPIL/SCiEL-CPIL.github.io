import time
import random
from itertools import combinations

def construct():
    n = 12
    m = 40
    total_masks = 1 << n  # 4096
    deadline = time.time() + 82  # 8s safety margin
    
    # Precompute subset relations for all masks
    # is_subset[a][b] = 1 if a is subset of b
    # We can compute this efficiently: a is subset of b iff (a & ~b) == 0
    
    # Precompute for each mask, the list of masks it is a subset of
    # and the list of masks that are subsets of it
    # But 4096^2 = 16M entries, might be too much memory/time
    # Instead, let's precompute comparability between pairs more efficiently
    
    # For each mask a, count how many masks b in our family are comparable with a
    # comparable means a⊆b or b⊆a
    
    # Let's use a different approach:
    # For a family F, score = number of pairs (a,b) with a⊂b or b⊂a
    # = sum over all a in F of (number of b in F with b⊃a) [since if a⊂b, that's one pair]
    # But we need to be careful not to double count. Actually each unordered pair is counted once.
    # If a⊂b, the pair {a,b} is counted once.
    # So score = sum over a in F of |{b in F : a⊂b}|
    
    # Precompute for each mask a, the list of all masks b such that a⊂b (strict superset)
    # This can be done by: for each a, all b where (a & ~b) == 0 and a != b
    # i.e., b = a | c where c is any non-empty subset of ~a
    
    # Let's precompute strict_supersets[a] = list of b where a⊂b
    # And strict_subsets[a] = list of b where b⊂a
    
    # Actually, for efficiency in the search, let's precompute a bitset representation
    # For each mask a, create a bitmask over the 4096 possible masks indicating which are strict supersets
    
    # But 4096 bits per mask * 4096 masks = 16M bits = 2MB, that's fine
    # Let's use Python integers as bitsets
    
    strict_supersets = [0] * total_masks
    strict_subsets = [0] * total_masks
    
    for a in range(total_masks):
        # Find all b such that a⊂b
        # b must have all bits of a, and at least one more bit
        complement = total_masks - 1 - a  # bits not in a
        # All supersets of a: a | c for c subset of complement
        # We want strict: c != 0
        # Enumerate all submasks of complement
        c = complement
        while c:
            b = a | c
            strict_supersets[a] |= (1 << b)
            strict_subsets[b] |= (1 << a)
            c = (c - 1) & complement
    
    # Now for a family represented as a bitset F (integer with m bits set),
    # score = sum over a in F of popcount(F & strict_supersets[a])
    # But this counts each pair twice? No: if a⊂b, then b is in strict_supersets[a], 
    # and a is in strict_subsets[b]. When we sum over a, we count pair (a,b) once (when at a).
    # When we sum over b, we'd count it again. But we're only summing over a in F of |F ∩ strict_supersets[a]|
    # This gives: for each a, count b in F where a⊂b. Sum over all a.
    # Each pair {a,b} with a⊂b is counted exactly once (when a is the smaller set).
    # So score = sum_{a in F} popcount(F & strict_supersets[a])
    
    # Wait, but this only counts pairs where the first element is the subset. That's correct for unordered pairs
    # since each comparable unordered pair has exactly one subset and one superset.
    
    # Let's verify: if a⊂b, then b ∈ strict_supersets[a], so when we process a, we count this pair.
    # When we process b, a ∈ strict_subsets[b], but we're looking at strict_supersets[b], not strict_subsets[b].
    # So a is not in strict_supersets[b] (unless a⊃b which is false). So the pair is counted once. Good.
    
    # Now implement local search
    # Start with a random family of 40 masks
    
    def popcount(x):
        return bin(x).count('1')
    
    def family_score(family_bits):
        """Compute score for a family represented as a bitset."""
        score = 0
        x = family_bits
        while x:
            # Get lowest set bit
            lsb = x & (-x)
            a = lsb.bit_length() - 1
            score += popcount(family_bits & strict_supersets[a])
            x ^= lsb
        return score
    
    def random_family():
        """Generate a random family of m distinct masks."""
        fam = random.sample(range(total_masks), m)
        bits = 0
        for a in fam:
            bits |= (1 << a)
        return bits, fam
    
    # Local search: try to add a new mask and remove an existing one
    # Or swap: remove one, add one
    
    best_bits, best_fam = random_family()
    best_score = family_score(best_bits)
    
    # Precompute for each mask a, the contribution to score if a is in the family
    # contrib(a, F) = popcount(F & strict_supersets[a])
    # If we add mask x to F, new score = old score + contrib(x, F) + sum over a in F of [x in strict_supersets[a]]
    # = old score + popcount(F & strict_supersets[x]) + popcount(F & strict_subsets[x])
    # Because: pairs involving x: (x, b) where x⊂b => b in strict_supersets[x], count = popcount(F & strict_supersets[x])
    # and (a, x) where a⊂x => a in strict_subsets[x], count = popcount(F & strict_subsets[x])
    # Note: no double counting since x⊂b and a⊂x are different pairs.
    
    # Similarly, if we remove mask y from F:
    # new score = old score - popcount(F\{y} & strict_supersets[y]) - popcount(F\{y} & strict_subsets[y])
    # = old score - popcount((F & ~bit(y)) & strict_supersets[y]) - popcount((F & ~bit(y)) & strict_subsets[y])
    # But strict_supersets[y] doesn't contain y, so F & strict_supersets[y] = (F & ~bit(y)) & strict_supersets[y]
    # Same for strict_subsets. So:
    # removal_loss(y, F) = popcount(F & strict_supersets[y]) + popcount(F & strict_subsets[y])
    
    # For a swap: remove y, add x (x not in F, y in F):
    # delta = [popcount(F & strict_supersets[x]) + popcount(F & strict_subsets[x])] - [popcount(F & strict_supersets[y]) + popcount(F & strict_subsets[y])]
    # But wait, when we remove y and add x, the interactions between x and other elements change,
    # and also the interactions between y and other elements are removed.
    # However, x and y don't interact with each other in the "before" state (x not in F) or "after" state... 
    # Actually in the after state, x is in F and y is not. So x interacts with F\{y}, and y's interactions are gone.
    # The delta is:
    # gain from adding x to F\{y}: popcount((F\{y}) & strict_supersets[x]) + popcount((F\{y}) & strict_subsets[x])
    # loss from removing y from F: popcount((F\{y}) & strict_supersets[y]) + popcount((F\{y}) & strict_subsets[y])
    # Since strict_supersets[x] and strict_subsets[x] don't contain y (unless x and y are comparable, but even then, 
    # y is not in strict_supersets[x] if y⊂x, y is in strict_subsets[x] if y⊂x... wait)
    
    # Let me be more careful. Let F' = F \ {y} ∪ {x}
    # score(F') - score(F) = [sum_{a in F'} popcount(F' & strict_supersets[a])] - [sum_{a in F} popcount(F & strict_supersets[a])]
    
    # This is complex. Let's just compute it directly for simplicity, or use the formula:
    # The change comes from:
    # 1. Pairs involving x (new) and elements in F\{y}
    # 2. Pairs involving y (removed) and elements in F\{y}
    # 3. The pair (x,y) doesn't exist in either F or F'
    
    # gain_x = popcount((F & ~(1<<y)) & strict_supersets[x]) + popcount((F & ~(1<<y)) & strict_subsets[x])
    # loss_y = popcount((F & ~(1<<y)) & strict_supersets[y]) + popcount((F & ~(1<<y)) & strict_subsets[y])
    # delta = gain_x - loss_y
    
    # Note: if x and y are comparable, say x⊂y, then in F, y is in strict_supersets[x] but x is not in F.
    # In F', x is in F and y is not. The pair (x,y) is not present in F (x not in F) nor in F' (y not in F).
    # So no issue.
    
    # Let's implement local search with swaps
    
    def get_family_list(bits):
        fam = []
        x = bits
        while x:
            lsb = x & (-x)
            a = lsb.bit_length() - 1
            fam.append(a)
            x ^= lsb
        return fam
    
    # Local search loop
    no_improve = 0
    max_no_improve = 50
    iters = 0
    
    while time.time() < deadline:
        iters += 1
        if no_improve >= max_no_improve:
            # Restart with a new random family
            best_bits, best_fam = random_family()
            best_score = family_score(best_bits)
            no_improve = 0
            max_no_improve = 50
            continue
        
        # Try random swaps
        improved = False
        for _ in range(200):
            if time.time() > deadline:
                break
            
            # Pick a random y to remove from best_fam
            y = random.choice(best_fam)
            
            # Pick a random x to add, not in best_fam
            # Sample from all masks not in best_fam
            x = random.randint(0, total_masks - 1)
            if (best_bits >> x) & 1:
                continue
            
            # Compute delta
            F_without_y = best_bits & ~(1 << y)
            gain_x = popcount(F_without_y & strict_supersets[x]) + popcount(F_without_y & strict_subsets[x])
            loss_y = popcount(F_without_y & strict_supersets[y]) + popcount(F_without_y & strict_subsets[y])
            delta = gain_x - loss_y
            
            if delta > 0:
                # Accept the swap
                new_bits = (best_bits & ~(1 << y)) | (1 << x)
                best_bits = new_bits
                best_fam.remove(y)
                best_fam.append(x)
                best_score += delta
                improved = True
                no_improve = 0
        
        if not improved:
            no_improve += 1
    
    # Convert back to list of lists
    result = []
    for a in best_fam:
        result.append([b for b in range(n) if (a >> b) & 1])
    
    # Ensure exactly 40 distinct lists
    assert len(result) == m
    assert len(set(tuple(s) for s in result)) == m
    
    return result
