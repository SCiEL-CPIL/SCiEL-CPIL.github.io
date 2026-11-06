import sys
import time
from math import gcd

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 512
    
    # 1. Precompute primes and prime factor masks
    primes = []
    is_prime = [True] * 512
    is_prime[0] = is_prime[1] = False
    for i in range(2, 512):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, 512, i):
                is_prime[j] = False
                
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)
    
    def get_mask(n):
        mask = 0
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                if d in prime_to_bit:
                    mask |= (1 << prime_to_bit[d])
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            if temp in prime_to_bit:
                mask |= (1 << prime_to_bit[temp])
        return mask

    # 2. Prepare candidates sorted by weight descending
    items = []
    for a in range(511, 0, -1):
        w = 1.0 / (N - a)
        m = get_mask(a)
        items.append((w, a, m))
        
    items.sort(key=lambda x: x[0], reverse=True)
    
    n_items = len(items)
    sorted_weights = [x[0] for x in items]
    sorted_avalues = [x[1] for x in items]
    sorted_masks = [x[2] for x in items]
    
    # Precompute: for each item index, which primes it uses
    # And build a list of items per prime for tighter bounds
    prime_to_items = [[] for _ in range(num_primes)]
    for i in range(n_items):
        m = sorted_masks[i]
        for p_idx in range(num_primes):
            if m & (1 << p_idx):
                prime_to_items[p_idx].append(i)
    
    # Precompute suffix sums (loose bound)
    suffix_sums = [0.0] * (n_items + 1)
    for i in range(n_items - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + sorted_weights[i]
    
    # Precompute for tighter bound: for each index i and each prime p,
    # the max weight among items j >= i that use prime p
    # This allows a tighter upper bound: for each prime, we can pick at most one item
    # So UB = current_score + sum over all "free" items + for each prime, best remaining item using that prime
    # But this is complex. Instead, use: UB = current_score + max over each prime of best remaining
    # Actually, a simpler tight bound: the sum of weights of all remaining items minus for each prime,
    # the (second_best - best) adjustment... This is getting complex.
    
    # Simpler approach: use the fact that for each prime, at most one item can be selected.
    # So UB = current_score + sum over primes p of (max weight among remaining items using p)
    # But this double-counts items that use multiple primes.
    # Even simpler: just use suffix_sums but also track which primes are "used"
    # and compute a better bound.
    
    # For now, let's use a moderately tight bound:
    # UB = current_score + sum of weights of remaining items that don't conflict with used_mask
    #     + for each prime not in used_mask, the best remaining item using that prime (if not already counted)
    # This is still complex. Let's use a different strategy:
    
    # Better: Use a greedy upper bound from the remaining items.
    # From the remaining items (idx to end), greedily pick the highest-weight items
    # that don't conflict with each other or with used_mask.
    # This gives a valid upper bound (actually a feasible solution upper bound).
    
    def greedy_ub(idx, used_mask):
        """Compute a greedy upper bound from items idx..n_items-1 given used_mask."""
        ub_score = 0.0
        local_mask = used_mask
        # We need to be fast here, so use a simple greedy
        for j in range(idx, n_items):
            if (sorted_masks[j] & local_mask) == 0:
                ub_score += sorted_weights[j]
                local_mask |= sorted_masks[j]
        return ub_score
    
    # Precompute greedy UB for all idx values? Too expensive.
    # Instead, compute it on-the-fly but cache when possible.
    
    best_score = 0.0
    best_set = []
    
    # Greedy initial solution
    greedy_set = []
    greedy_mask = 0
    greedy_score = 0.0
    for i in range(n_items):
        if (sorted_masks[i] & greedy_mask) == 0:
            greedy_set.append(sorted_avalues[i])
            greedy_mask |= sorted_masks[i]
            greedy_score += sorted_weights[i]
            
    if greedy_score > best_score:
        best_score = greedy_score
        best_set = list(greedy_set)
        
    # Branch and Bound with tighter bound
    def bb(idx, current_score, used_mask, current_set):
        nonlocal best_score, best_set
        
        if time.time() > deadline:
            return
        
        # Tight upper bound using greedy on remaining
        # Only compute if the loose bound doesn't already prune
        if current_score + suffix_sums[idx] <= best_score:
            return
        
        # Compute tighter greedy UB (only if loose bound is close)
        if current_score + suffix_sums[idx] - best_score < 0.5:
            ub = current_score + greedy_ub(idx, used_mask)
            if ub <= best_score:
                return
        else:
            ub = current_score + suffix_sums[idx]
            if ub <= best_score:
                return
        
        if idx == n_items:
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
            return
        
        a = sorted_avalues[idx]
        m = sorted_masks[idx]
        w = sorted_weights[idx]
        
        # Option 1: Include a (if compatible)
        if (m & used_mask) == 0:
            current_set.append(a)
            bb(idx + 1, current_score + w, used_mask | m, current_set)
            current_set.pop()
            
            # Check time after include branch
            if time.time() > deadline:
                return
        
        # Option 2: Exclude a
        bb(idx + 1, current_score, used_mask, current_set)
        
    bb(0, 0.0, 0, [])
    
    # Local search: try swaps to improve
    if best_set:
        best_set_set = set(best_set)
        best_mask = 0
        for a in best_set:
            best_mask |= sorted_masks[sorted_avalues.index(a)] if a in sorted_avalues else get_mask(a)
        
        # Recompute mask properly
        best_mask = 0
        for a in best_set:
            best_mask |= get_mask(a)
        
        current_score = sum(1.0 / (N - a) for a in best_set)
        
        # Try single swaps: remove one, add one better
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i, a_remove in enumerate(best_set):
                mask_without = best_mask & ~(get_mask(a_remove))
                # Find best candidate to add
                for j in range(n_items):
                    a_add = sorted_avalues[j]
                    if a_add in best_set_set:
                        continue
                    m_add = sorted_masks[j]
                    if (m_add & mask_without) == 0:
                        new_score = current_score - sorted_weights[sorted_avalues.index(a_remove)] + sorted_weights[j]
                        if new_score > current_score + 1e-12:
                            best_set[i] = a_add
                            best_set_set = set(best_set)
                            best_mask = 0
                            for a in best_set:
                                best_mask |= get_mask(a)
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
    
    return best_set
