import time
import random
import sys

N = 256

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin before 60s kill

    def get_sum_mask(sel):
        """Compute bitmask of all a+b sums (a<=b) for a Sidon set."""
        m = 0
        s = sorted(sel)
        for i in range(len(s)):
            a = s[i]
            m |= (1 << (2 * a))
            for j in range(i + 1, len(s)):
                m |= (1 << (a + s[j]))
        return m

    def is_valid_sidon(sel):
        """Check Sidon property: all unordered pair sums (including diagonal) distinct."""
        s = sorted(sel)
        seen = set()
        for i in range(len(s)):
            a = s[i]
            if 2 * a in seen:
                return False
            seen.add(2 * a)
            for j in range(i + 1, len(s)):
                b = s[j]
                ab = a + b
                if ab in seen:
                    return False
                seen.add(ab)
        return True

    def can_add(x, sel_set, sum_mask):
        """Check if x can be added to maintain Sidon property."""
        if x in sel_set:
            return False
        if (sum_mask >> (2 * x)) & 1:
            return False
        # Check conflicts with existing elements
        for a in sel_set:
            if (sum_mask >> (x + a)) & 1:
                return False
        return True

    def add_element(x, sel, sel_set, sum_mask):
        """Add x to the set, updating all structures."""
        sel.append(x)
        sel_set.add(x)
        sum_mask |= (1 << (2 * x))
        for a in sel[:-1]:
            sum_mask |= (1 << (x + a))
        return sel, sel_set, sum_mask

    def is_maximal(sel, sum_mask):
        """Check if no missing element can be added."""
        sel_set = set(sel)
        for y in range(1, N + 1):
            if y in sel_set:
                continue
            if can_add(y, sel_set, sum_mask):
                return False
        return True

    def blocking_count(x, sel_set, sum_mask):
        """Count how many currently-unblocked elements become blocked if x is added."""
        if x in sel_set or not can_add(x, sel_set, sum_mask):
            return 0
        
        new_sums = set()
        new_sums.add(2 * x)
        for a in sel_set:
            new_sums.add(x + a)
        
        blocked = 0
        for y in range(1, N + 1):
            if y in sel_set or y == x:
                continue
            # Was y blocked before?
            was_blocked = False
            if (sum_mask >> (2 * y)) & 1:
                was_blocked = True
            else:
                for a in sel_set:
                    if (sum_mask >> (y + a)) & 1:
                        was_blocked = True
                        break
            if was_blocked:
                continue
            # Check if y becomes blocked
            if 2 * y in new_sums:
                blocked += 1
            else:
                for a in sel_set:
                    if (y + a) in new_sums:
                        blocked += 1
                        break
        return blocked

    def greedy_max_block(seed, time_budget=3.0):
        """Greedy: pick element that blocks the most unblocked elements."""
        rng = random.Random(seed)
        t0 = time.time()
        
        sel = []
        sel_set = set()
        sum_mask = 0
        remaining = list(range(1, N + 1))
        rng.shuffle(remaining)
        
        while remaining and time.time() < t0 + time_budget:
            best_x = -1
            best_score = -1
            
            # Sample candidates
            if len(remaining) > 40:
                candidates = rng.sample(remaining, 40)
            else:
                candidates = remaining
            
            for x in candidates:
                if x in sel_set:
                    continue
                if not can_add(x, sel_set, sum_mask):
                    continue
                
                score = blocking_count(x, sel_set, sum_mask)
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x == -1:
                # Fallback: pick first valid
                for x in remaining:
                    if x not in sel_set and can_add(x, sel_set, sum_mask):
                        best_x = x
                        break
            
            if best_x == -1:
                break
            
            sel, sel_set, sum_mask = add_element(best_x, sel, sel_set, sum_mask)
            remaining.remove(best_x)
        
        return sel, sum_mask

    def local_search_reduce(sel, sum_mask, time_budget=2.0):
        """Local search: try removing elements and replacing with others to reduce size."""
        t0 = time.time()
        current = list(sel)
        current_mask = sum_mask
        
        while time.time() < t0 + time_budget:
            improved = False
            n = len(current)
            
            for i in range(n):
                if time.time() > t0 + time_budget:
                    break
                
                x = current[i]
                rem = [e for e in current if e != x]
                rem_set = set(rem)
                rem_mask = get_sum_mask(rem)
                
                # Check if rem alone is maximal
                if is_maximal(rem, rem_mask):
                    current = rem
                    current_mask = rem_mask
                    improved = True
                    break
                
                # Try adding one element to rem
                for y in range(1, N + 1):
                    if y in rem_set:
                        continue
                    if not can_add(y, rem_set, rem_mask):
                        continue
                    
                    new_sel = rem + [y]
                    new_mask = rem_mask
                    new_mask |= (1 << (2 * y))
                    for a in rem:
                        new_mask |= (1 << (y + a))
                    
                    if len(new_sel) < len(current) and is_maximal(new_sel, new_mask):
                        current = new_sel
                        current_mask = new_mask
                        improved = True
                        break
                
                if improved:
                    break
            
            # Also try 2-for-1 reductions
            if not improved and time.time() < t0 + time_budget:
                for i in range(len(current)):
                    if time.time() > t0 + time_budget:
                        break
                    for j in range(i + 1, len(current)):
                        if time.time() > t0 + time_budget:
                            break
                        
                        x1, x2 = current[i], current[j]
                        rem = [e for e in current if e != x1 and e != x2]
                        rem_set = set(rem)
                        rem_mask = get_sum_mask(rem)
                        
                        for y in range(1, N + 1):
                            if y in rem_set:
                                continue
                            if not can_add(y, rem_set, rem_mask):
                                continue
                            
                            new_sel = rem + [y]
                            new_mask = rem_mask
                            new_mask |= (1 << (2 * y))
                            for a in rem:
                                new_mask |= (1 << (y + a))
                            
                            if len(new_sel) < len(current) and is_maximal(new_sel, new_mask):
                                current = new_sel
                                current_mask = new_mask
                                improved = True
                                break
                        
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return current, current_mask

    # Main search
    best_set = None
    best_size = float('inf')
    
    num_trials = 25
    for i in range(num_trials):
        if time.time() > deadline - 5:
            break
        
        sel, sm = greedy_max_block(i, time_budget=1.2)
        
        if len(sel) >= best_size:
            continue
        
        sel, sm = local_search_reduce(sel, sm, time_budget=1.0)
        
        if len(sel) < best_size:
            best_size = len(sel)
            best_set = list(sel)
    
    # Final reduction on best
    if best_set is not None and time.time() < deadline - 2:
        bm = get_sum_mask(best_set)
        best_set, bm = local_search_reduce(best_set, bm, time_budget=1.5)
    
    if best_set is None:
        best_set, _ = greedy_max_block(0, time_budget=3.0)
    
    return best_set
