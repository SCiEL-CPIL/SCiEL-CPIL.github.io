"""Rewritten program: maximize subset of [1,128] with no equal-gcd triples."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    deadline = time.time() + 82  # safety margin before 90s kill
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
    
    # Precompute: for each x, the list of (a, b) pairs (a < b, both != x)
    # such that gcd(a,b) == gcd(a,x) == gcd(b,x)
    # Using 0-indexed: element i corresponds to value i+1
    # bad_pairs[x_idx] = list of (a_idx, b_idx) where (a+1, b+1, x+1) is a bad triple
    bad_pairs = [[] for _ in range(N)]
    for x in range(1, N + 1):
        xi = x - 1
        for a in range(1, N + 1):
            if a == x:
                continue
            ai = a - 1
            ga = gcd_table[a][x]
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                bi = b - 1
                gb = gcd_table[b][x]
                gab = gcd_table[a][b]
                if ga == gb == gab:
                    bad_pairs[xi].append((ai, bi))
    
    def can_add(mask, x_idx):
        """Check if element x_idx can be added to the set represented by mask."""
        for (ai, bi) in bad_pairs[x_idx]:
            if (mask >> ai) & 1 and (mask >> bi) & 1:
                return False
        return True
    
    def greedy_random(rng):
        """Greedy with random order."""
        order = list(range(N))
        rng.shuffle(order)
        mask = 0
        count = 0
        for x_idx in order:
            if can_add(mask, x_idx):
                mask |= (1 << x_idx)
                count += 1
        return mask, count
    
    # Try many randomized restarts
    best_mask = 0
    best_count = 0
    rng = random.Random(12345)
    
    start_time = time.time()
    iterations = 0
    while time.time() < deadline - 2:
        mask, count = greedy_random(rng)
        iterations += 1
        if count > best_count:
            best_count = count
            best_mask = mask
    
    # Also try some structured orderings
    structured_orders = [
        list(range(N)),  # ascending
        list(range(N - 1, -1, -1)),  # descending
        sorted(range(N), key=lambda i: i + 1, reverse=True),
    ]
    
    # Sort by number of prime factors (fewer first)
    def prime_factor_count(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    structured_orders.append(sorted(range(N), key=lambda i: (-prime_factor_count(i + 1), i)))
    
    for order in structured_orders:
        if time.time() > deadline - 2:
            break
        mask = 0
        count = 0
        for x_idx in order:
            if can_add(mask, x_idx):
                mask |= (1 << x_idx)
                count += 1
        if count > best_count:
            best_count = count
            best_mask = mask
    
    # Local search on best solution: try to add any remaining element
    # Also try swap-based improvements
    if best_count > 0:
        # Try adding elements not in best_mask
        for x_idx in range(N):
            if time.time() > deadline - 1:
                break
            if (best_mask >> x_idx) & 1:
                continue
            if can_add(best_mask, x_idx):
                best_mask |= (1 << x_idx)
                best_count += 1
        
        # Try swap-based local search
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for a_idx in range(N):
                if time.time() > deadline - 1:
                    break
                if not ((best_mask >> a_idx) & 1):
                    continue
                # Try removing a_idx and adding some b_idx
                mask_without_a = best_mask & ~(1 << a_idx)
                for b_idx in range(N):
                    if (best_mask >> b_idx) & 1:
                        continue
                    if can_add(mask_without_a, b_idx):
                        # This swap keeps same size, but might enable future additions
                        # Only do swap if it leads to net gain - skip for now
                        pass
    
    # Convert mask back to list
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    
    if not result:
        result = [1, 2]
    
    return result
# EVOLVE-BLOCK-END
