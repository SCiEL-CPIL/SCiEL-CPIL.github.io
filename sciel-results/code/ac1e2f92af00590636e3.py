import time
import random

def construct():
    t_start = time.time()
    DEADLINE = t_start + 78.0
    
    N = 4096
    
    def greedy_build(order, deadline):
        vals = []
        existing_sums = set()
        for x in order:
            if time.time() > deadline:
                break
            k = len(vals)
            # Check all new triple sums involving x
            conflict = False
            new_sums = []
            
            # 3x
            s = 3 * x
            if s in existing_sums:
                conflict = True
            else:
                new_sums.append(s)
            
            if not conflict:
                # 2x + v for v in vals
                for v in vals:
                    s = 2 * x + v
                    if s in existing_sums or s in new_sums:
                        conflict = True
                        break
                    new_sums.append(s)
            
            if not conflict:
                # x + v1 + v2 for v1 <= v2 in vals
                for i in range(k):
                    v1 = vals[i]
                    for j in range(i, k):
                        v2 = vals[j]
                        s = x + v1 + v2
                        if s in existing_sums or s in new_sums:
                            conflict = True
                            break
                        new_sums.append(s)
                    if conflict:
                        break
            
            if not conflict:
                vals.append(x)
                existing_sums.update(new_sums)
        
        return vals
    
    def _compute_all_sums(selected):
        """Compute set of all unordered triple sums (with replacement)."""
        n = len(selected)
        seen = set()
        for a in range(n):
            sa = selected[a]
            for b in range(a, n):
                sb = selected[b]
                sa_b = sa + sb
                for c in range(b, n):
                    s = sa_b + selected[c]
                    if s in seen:
                        return None  # collision detected
                    seen.add(s)
        return seen
    
    def _incremental_check_add(selected, x, existing_sums):
        """Check if adding x to selected keeps all triple sums unique.
        Returns (True, updated_sums) if valid, (False, None) otherwise."""
        n = len(selected)
        new_sums = set()
        
        # Check (x,x,x)
        s = 3 * x
        if s in existing_sums or s in new_sums:
            return (False, None)
        new_sums.add(s)
        
        # Check (a,x,x) for each a in selected
        for a in selected:
            s = a + 2 * x
            if s in existing_sums or s in new_sums:
                return (False, None)
            new_sums.add(s)
        
        # Check (a,b,x) for a<=b in selected
        for i in range(n):
            sa = selected[i]
            for j in range(i, n):
                sb = selected[j]
                s = sa + sb + x
                if s in existing_sums or s in new_sums:
                    return (False, None)
                new_sums.add(s)
        
        # Merge: return combined set
        combined = existing_sums | new_sums
        return (True, combined)
    
    def greedy_incremental(start_list, limit, deadline):
        """Greedy construction with incremental sum tracking, trying to add the smallest possible next element."""
        base = sorted(start_list)
        if not base:
            base = [1]
        
        # Compute initial sums
        sums = _compute_all_sums(base)
        if sums is None:
            return base
        
        while len(base) < 80:
            if time.time() > deadline:
                break
            next_min = max(base) + 1
            found = False
            for x in range(next_min, limit + 1):
                ok, new_sums = _incremental_check_add(base, x, sums)
                if ok:
                    base.append(x)
                    sums = new_sums
                    found = True
                    break
            if not found:
                break
        return base
    
    def local_search_swap(base, limit, deadline, max_iter=50):
        """Try to improve by removing one element and trying to add two more."""
        base = sorted(base)
        if not base:
            return base
        
        sums = _compute_all_sums(base)
        if sums is None:
            return base
        
        best = list(base)
        best_size = len(base)
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            for remove_idx in range(len(base)):
                if time.time() > deadline:
                    break
                
                # Remove element at remove_idx
                removed = base[remove_idx]
                candidate = base[:remove_idx] + base[remove_idx+1:]
                
                # Recompute sums without the removed element
                new_sums = _compute_all_sums(candidate)
                if new_sums is None:
                    continue
                
                # Try to add two elements to reach a larger set
                # Try adding the smallest possible element first
                next_min = max(candidate) + 1 if candidate else 1
                added = 0
                temp_base = list(candidate)
                temp_sums = set(new_sums)
                
                for _ in range(2):
                    if time.time() > deadline:
                        break
                    found = False
                    for x in range(next_min, limit + 1):
                        ok, updated_sums = _incremental_check_add(temp_base, x, temp_sums)
                        if ok:
                            temp_base.append(x)
                            temp_sums = updated_sums
                            next_min = x + 1
                            added += 1
                            found = True
                            break
                    if not found:
                        break
                
                if len(temp_base) > best_size:
                    best = temp_base
                    best_size = len(best)
                    improved = True
            
            if not improved:
                break
        
        return best
    
    best = []
    
    # Primary: greedy from 1 upward using order-based approach
    order1 = list(range(1, N + 1))
    result = greedy_build(order1, DEADLINE)
    if len(result) > len(best):
        best = result
    
    # Also try the incremental greedy which tries smallest valid next element
    if time.time() < DEADLINE:
        result2 = greedy_incremental([1], N, DEADLINE)
        if len(result2) > len(best):
            best = result2
    
    if time.time() > DEADLINE:
        return best
    
    # Randomized restarts with different orderings
    rng = random.Random(42)
    for attempt in range(200):
        if time.time() > DEADLINE:
            break
        
        # Different strategies for ordering
        strategy = attempt % 5
        
        if strategy == 0:
            # Small bias: random shuffle of [1..N]
            order = list(range(1, N + 1))
            rng.shuffle(order)
        elif strategy == 1:
            # Start from a random offset
            offset = rng.randint(0, 200)
            order = list(range(1 + offset, N + 1)) + list(range(1, 1 + offset))
        elif strategy == 2:
            # Two-phase: small numbers first (shuffled), then larger
            small = list(range(1, 200))
            large = list(range(200, N + 1))
            rng.shuffle(small)
            rng.shuffle(large)
            order = small + large
        elif strategy == 3:
            # Exponential weighting: pick random starting point, go outward
            center = rng.randint(1, N)
            order = [center]
            for d in range(1, N):
                if center - d >= 1:
                    order.append(center - d)
                if center + d <= N:
                    order.append(center + d)
        else:
            # Random small initial set, then extend with incremental greedy
            k = rng.randint(3, 8)
            initial_set = sorted(rng.sample(range(1, N + 1), k))
            # Quick check if initial set is valid
            if _compute_all_sums(initial_set) is not None:
                result = greedy_incremental(initial_set, N, DEADLINE)
                if len(result) > len(best):
                    best = result
                continue
        
        result = greedy_build(order, DEADLINE)
        if len(result) > len(best):
            best = result
            # If we hit the theoretical max, stop
            if len(best) >= 40:
                break
    
    # Try local search on the best result if time permits
    if time.time() < DEADLINE - 5:
        improved = local_search_swap(best, N, DEADLINE, max_iter=30)
        if len(improved) > len(best):
            best = improved
    
    return best
