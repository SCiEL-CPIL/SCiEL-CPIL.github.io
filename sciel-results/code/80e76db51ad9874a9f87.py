"""Program for cardinality-separated subset sums using bitset DP verification."""
import time
import random

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Maximize set size.
    
    Uses greedy construction with bitset-based cardinality-indexed subset-sum DP.
    """
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 256  # max value
    pool = list(range(1, 257))
    
    def compute_sums_by_card(elements):
        """Compute sums_by_card bitset array for the given elements."""
        n = len(elements)
        if n == 0:
            return [1]  # sums_by_card[0] = 1 (bit 0 set)
        
        sums_by_card = [0] * (n + 1)
        sums_by_card[0] = 1  # bit 0 set
        
        for x in elements:
            for k in range(n, 0, -1):
                sums_by_card[k] |= (sums_by_card[k-1] << x)
        
        return sums_by_card
    
    def is_valid(sums_by_card):
        """Check if sums_by_card has pairwise disjoint bitsets across cardinalities."""
        n = len(sums_by_card) - 1
        for k1 in range(n + 1):
            for k2 in range(k1 + 1, n + 1):
                if sums_by_card[k1] & sums_by_card[k2]:
                    return False
        return True
    
    def check_add(sums_by_card, x):
        """
        Check if adding element x to the current set maintains validity.
        Returns (new_sums_by_card, valid) tuple.
        """
        n = len(sums_by_card) - 1
        # new_sums_by_card[k] = sums_by_card[k] | (sums_by_card[k-1] << x)
        new_sums = [0] * (n + 2)
        new_sums[0] = 1
        
        for k in range(1, n + 2):
            new_sums[k] = sums_by_card[k] if k <= n else 0
            if k - 1 <= n:
                new_sums[k] |= (sums_by_card[k-1] << x)
        
        # Check validity
        for k1 in range(n + 2):
            for k2 in range(k1 + 1, n + 2):
                if new_sums[k1] & new_sums[k2]:
                    return None, False
        
        return new_sums, True
    
    def greedy_grow(base, order_list):
        """Grow base by adding elements from order_list if valid."""
        current = list(base)
        current_set = set(current)
        
        if current:
            sums_by_card = compute_sums_by_card(current)
        else:
            sums_by_card = [1]
        
        for x in order_list:
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            
            new_sums, valid = check_add(sums_by_card, x)
            if valid:
                current.append(x)
                current_set.add(x)
                sums_by_card = new_sums
        
        return current
    
    best_set = []
    best_len = 0
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: increasing order from empty
    strategies.append(([], list(range(1, 257))))
    
    # Strategy 2: increasing order from {1}
    strategies.append(([1], [x for x in range(2, 257)]))
    
    # Strategy 3: descending order from empty
    strategies.append(([], list(range(256, 0, -1))))
    
    # Strategy 4: Try starting with various small bases
    for start in [1, 2, 3, 4, 5, 8, 10, 16, 20, 32, 50, 100, 128, 200, 256]:
        strategies.append(([start], [x for x in range(1, 257) if x != start]))
    
    # Strategy 5: Try pairs
    for a in [1, 2, 3, 5, 8, 10, 16, 20, 32, 50, 100, 128]:
        for b in [a+1, a+2, a+5, a+10, a+20, a+50, min(a*2, 256)]:
            if b > 256 or b <= a:
                continue
            strategies.append(([a, b], [x for x in range(1, 257) if x not in (a, b)]))
    
    # Random strategies
    rng = random.Random(42)
    for _ in range(30):
        base_size = rng.randint(1, 3)
        base = rng.sample(pool, base_size)
        random_order = pool[:]
        rng.shuffle(random_order)
        strategies.append((base, random_order))
    
    # Execute all strategies
    for base, order in strategies:
        if time.time() > deadline:
            break
        result = greedy_grow(base, order)
        if len(result) > best_len:
            best_len = len(result)
            best_set = result
    
    # Final polish: try to add remaining elements in multiple orders
    if best_set:
        current_set = set(best_set)
        remaining = [x for x in pool if x not in current_set]
        
        # Try adding in increasing order
        for x in sorted(remaining):
            if time.time() > deadline:
                break
            candidate = best_set + [x]
            sums = compute_sums_by_card(candidate)
            if is_valid(sums):
                best_set = candidate
                current_set.add(x)
        
        # Try adding in decreasing order
        current_set = set(best_set)
        remaining = [x for x in pool if x not in current_set]
        for x in sorted(remaining, reverse=True):
            if time.time() > deadline:
                break
            candidate = best_set + [x]
            sums = compute_sums_by_card(candidate)
            if is_valid(sums):
                best_set = candidate
                current_set.add(x)
    
    return list(best_set)
