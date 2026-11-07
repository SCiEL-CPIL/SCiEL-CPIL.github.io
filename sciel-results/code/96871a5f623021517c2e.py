import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 85.0
    
    # Bitmask for 1..128 (index 0 unused, bits 1..128)
    FULL_MASK = (1 << 129) - 2  # bits 1..128 set
    
    def make_mask(s):
        m = 0
        for x in s:
            m |= (1 << x)
        return m
    
    def popcount(x):
        return bin(x).count('1')
    
    def is_valid_mask(m):
        """Check if mask m has no bad triple. O(n^3) but with bit ops."""
        # Get sorted list for iteration
        elems = []
        for i in range(1, 129):
            if m & (1 << i):
                elems.append(i)
        n = len(elems)
        if n < 3:
            return True
        # For each triple (a,b,c) with a<b<c, check if a+b, a+c, b+c are all in m
        for i in range(n):
            a = elems[i]
            for j in range(i+1, n):
                b = elems[j]
                ab = a + b
                if ab > 128 or not (m & (1 << ab)):
                    continue
                for k in range(j+1, n):
                    c = elems[k]
                    ac = a + c
                    bc = b + c
                    if ac > 128 or bc > 128:
                        continue
                    if (m & (1 << ac)) and (m & (1 << bc)):
                        return False
        return True
    
    def can_add_to_mask(m, x):
        """O(n^2) check: can we add x to mask m without creating a bad triple?"""
        if m & (1 << x):
            return False
        # A bad triple involving x: {x, a, b} where a,b in m, a≠b, and a+b, a+x, b+x all in m∪{x}
        # Since x is being added, the pair sums that must be in m∪{x} are:
        #   a+b (must be in m, since x≠a+b unless... well a+b could equal x? No, a,b≥1 so a+b≥2, x could be a+b)
        #   a+x (must be in m or equal x? a+x=x means a=0, impossible. So must be in m)
        #   b+x (must be in m)
        # Wait, the condition is: a+b, a+c, b+c are all in S where S = m ∪ {x}.
        # The triple is {a, b, x} with a,b distinct and in m.
        # Pair sums: a+b, a+x, b+x. All three must be in S = m ∪ {x}.
        # a+b: could be x (if a+b=x) or in m.
        # a+x: could be x (impossible, a≥1) or in m.
        # b+x: could be x (impossible) or in m.
        # So: (a+b in m or a+b==x) AND (a+x in m) AND (b+x in m)
        
        # Get elements of m
        elems = []
        for i in range(1, 129):
            if m & (1 << i):
                elems.append(i)
        n = len(elems)
        
        for i in range(n):
            a = elems[i]
            ax = a + x
            if ax > 128:
                continue
            if not (m & (1 << ax)):
                continue
            for j in range(i+1, n):
                b = elems[j]
                bx = b + x
                if bx > 128:
                    continue
                if not (m & (1 << bx)):
                    continue
                ab = a + b
                if ab <= 128:
                    if (m & (1 << ab)) or ab == x:
                        return False
                # If ab > 128, it can't be in S, so no violation
        return True
    
    def can_remove_from_mask(m, x):
        """Check if removing x keeps validity. Always true if m was valid."""
        return True
    
    def remove_and_check(m, x, new_elems):
        """Remove x from m, add new_elems. Check validity."""
        new_m = m & ~(1 << x)
        for ne in new_elems:
            if new_m & (1 << ne):
                return None  # already present
            new_m |= (1 << ne)
        if is_valid_mask(new_m):
            return new_m
        return None
    
    # Initial seed: 65..128
    seed_set = set(range(65, 129))
    seed_mask = make_mask(seed_set)
    best_mask = seed_mask
    best_size = 64
    
    # Verify seed
    if not is_valid_mask(seed_mask):
        # Should not happen, but fallback
        return list(range(65, 129))
    
    # Phase 1: Greedy single-add
    for x in range(1, 129):
        if time.time() > deadline - 10:
            break
        if best_mask & (1 << x):
            continue
        if can_add_to_mask(best_mask, x):
            best_mask |= (1 << x)
            best_size += 1
    
    # Phase 2: Combinatorial k-out-m-in search
    # Try removing k elements and adding m elements where m > k
    random.seed(12345)
    
    for phase in range(10):
        if time.time() > deadline - 5:
            break
        
        # Determine k and m for this phase
        if phase < 3:
            k = 1
            m = 2
        elif phase < 6:
            k = 2
            m = 3
        elif phase < 8:
            k = 2
            m = 4
        else:
            k = 3
            m = 5
        
        # Get current elements
        cur_elems = []
        for i in range(1, 129):
            if best_mask & (1 << i):
                cur_elems.append(i)
        
        # Get candidates not in current set
        candidates = [x for x in range(1, 129) if not (best_mask & (1 << x))]
        
        if not candidates or len(cur_elems) < k:
            continue
        
        # Limit combinations to keep it tractable
        max_combos = 5000
        attempts = 0
        improved = False
        
        # Randomly sample removal combinations
        if len(cur_elems) <= 20:
            removals = list(combinations(cur_elems, k))
        else:
            # Sample random removals
            removals = []
            seen = set()
            while len(removals) < max_combos and attempts < max_combos * 5:
                attempts += 1
                rem = tuple(sorted(random.sample(cur_elems, k)))
                if rem not in seen:
                    seen.add(rem)
                    removals.append(rem)
        
        # Sample random addition combinations
        if len(candidates) <= 15:
            additions = list(combinations(candidates, m))
        else:
            additions = []
            seen_add = set()
            while len(additions) < max_combos and attempts < max_combos * 5:
                attempts += 1
                add = tuple(sorted(random.sample(candidates, m)))
                if add not in seen_add:
                    seen_add.add(add)
                    additions.append(add)
        
        # Try combinations
        for rem in removals:
            if time.time() > deadline - 5:
                break
            rem_set = set(rem)
            # Remove these elements
            trial_mask = best_mask
            for r in rem:
                trial_mask &= ~(1 << r)
            
            for add in additions:
                if time.time() > deadline - 5:
                    break
                # Check additions don't conflict with removals or existing
                valid_add = True
                for a in add:
                    if a in rem_set:
                        continue
                    if trial_mask & (1 << a):
                        valid_add = False
                        break
                if not valid_add:
                    continue
                
                new_mask = trial_mask
                for a in add:
                    new_mask |= (1 << a)
                
                if popcount(new_mask) <= best_size:
                    continue
                
                if is_valid_mask(new_mask):
                    best_mask = new_mask
                    best_size = popcount(new_mask)
                    improved = True
                    # Update cur_elems and candidates
                    cur_elems = []
                    for i in range(1, 129):
                        if best_mask & (1 << i):
                            cur_elems.append(i)
                    candidates = [x for x in range(1, 129) if not (best_mask & (1 << x))]
                    break
        
        if not improved and phase >= 5:
            # Try a few random restarts with perturbation
            for restart in range(5):
                if time.time() > deadline - 5:
                    break
                # Perturb: remove 3-5 random elements
                perturbed = list(best_mask)
                cur_e = []
                for i in range(1, 129):
                    if best_mask & (1 << i):
                        cur_e.append(i)
                p_mask = best_mask
                for _ in range(random.randint(3, 5)):
                    if cur_e:
                        r = random.choice(cur_e)
                        p_mask &= ~(1 << r)
                        cur_e.remove(r)
                
                # Try to greedily add
                for x in range(1, 129):
                    if time.time() > deadline - 5:
                        break
                    if p_mask & (1 << x):
                        continue
                    if can_add_to_mask(p_mask, x):
                        p_mask |= (1 << x)
                
                if popcount(p_mask) > best_size:
                    best_mask = p_mask
                    best_size = popcount(p_mask)
    
    # Phase 3: Final greedy pass
    for x in range(1, 129):
        if time.time() > deadline - 2:
            break
        if best_mask & (1 << x):
            continue
        if can_add_to_mask(best_mask, x):
            best_mask |= (1 << x)
            best_size += 1
    
    # Final validity check
    if not is_valid_mask(best_mask):
        # Fallback to seed
        return list(range(65, 129))
    
    # Extract result
    result = []
    for i in range(1, 129):
        if best_mask & (1 << i):
            result.append(int(i))
    
    return result
