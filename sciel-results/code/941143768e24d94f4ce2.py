import time
import random
import sys

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    sys.setrecursionlimit(10000)

    def is_valid(subset):
        reachable = 1
        mask = (1 << 201) - 1  # Keep bits 0..200
        for x in subset:
            new_reachable = (reachable | (reachable << x)) & mask
            if (new_reachable >> 200) & 1:
                return False
            reachable = new_reachable
        return True

    def generate_bad_subsets():
        """
        Generates bitmasks of all subsets of [1, 100] that sum to 200.
        Returns None if the list is too large or takes too long.
        """
        # dp[s] is a list of bitmasks
        dp = [[] for _ in range(201)]
        dp[0].append(0)
        
        # Limit generation time
        gen_start = time.time()
        
        for i in range(1, 101):
            bit = 1 << i
            # Iterate downwards to avoid using i multiple times
            for s in range(200, i - 1, -1):
                if dp[s - i]:
                    # Extend dp[s] with new masks
                    # Optimization: if list gets too big, abort
                    if len(dp[s]) > 5000:
                        return None
                    dp[s].extend(m | bit for m in dp[s - i])
            
            if time.time() - gen_start > 5.0:
                return None
        
        return dp[200]

    def greedy_hitting_set(bad_subsets):
        """
        Finds a hitting set R for the given bad subsets.
        Returns the set of elements in R.
        """
        n_bad = len(bad_subsets)
        if n_bad == 0:
            return set()
        
        # cover_mask[i] is a bitmask of indices of bad subsets containing element i
        # We use Python ints as bitsets.
        cover_mask = [0] * 101
        for idx, mask in enumerate(bad_subsets):
            m = mask
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if 1 <= bit_idx <= 100:
                    cover_mask[bit_idx] |= (1 << idx)
                m ^= lsb
        
        uncovered = (1 << n_bad) - 1
        R = set()
        
        while uncovered:
            best_bit = 0
            max_count = 0
            for i in range(1, 101):
                # Count how many uncovered subsets are covered by i
                count = (uncovered & cover_mask[i]).bit_count()
                if count > max_count:
                    max_count = count
                    best_bit = i
            
            if best_bit == 0:
                # Should not happen if uncovered != 0
                break
                
            R.add(best_bit)
            uncovered &= ~cover_mask[best_bit]
            
        return R

    def local_search(best, deadline):
        """Improve via swap and add moves with incremental validity"""
        best = set(best)
        
        # Precompute reachable bitmask for current set
        def compute_reachable(s):
            reachable = 1
            mask = (1 << 201) - 1
            for x in s:
                reachable = (reachable | (reachable << x)) & mask
            return reachable
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            reachable = compute_reachable(best)
            
            # Try adding elements - check if adding x would create sum 200
            for x in range(1, 101):
                if x in best:
                    continue
                # Check if x + some reachable sum = 200, i.e., (200-x) is reachable
                if (200 - x) >= 0 and (200 - x) <= 199:
                    if (reachable >> (200 - x)) & 1:
                        continue  # Would create sum 200
                # Also check if x itself is 200 (not possible here since x <= 100)
                best.add(x)
                reachable = (reachable | (reachable << x)) & ((1 << 201) - 1)
                improved = True
                break
            
            if time.time() >= deadline:
                break
            
            # Try swapping: remove one, add one
            for remove_x in sorted(best):
                if time.time() >= deadline:
                    break
                # Compute reachable without remove_x
                # This is expensive, so only do it if we think it might help
                remaining = best - {remove_x}
                rem_reachable = compute_reachable(remaining)
                
                for add_x in range(1, 101):
                    if add_x in remaining:
                        continue
                    # Check if add_x would create sum 200 with remaining
                    if (200 - add_x) >= 0 and (200 - add_x) <= 199:
                        if (rem_reachable >> (200 - add_x)) & 1:
                            continue
                    best = remaining | {add_x}
                    improved = True
                    break
                if improved:
                    break
        return best

    # Strategy 1: Try multiple greedy initializations with different orderings
    def greedy(order):
        reachable = 1
        mask = (1 << 201) - 1
        selected = set()
        for x in order:
            new_reachable = (reachable | (reachable << x)) & mask
            if not ((new_reachable >> 200) & 1):
                selected.add(x)
                reachable = new_reachable
        return selected

    candidates = []
    # Deterministic orderings
    candidates.append(greedy(range(100, 0, -1)))  # Descending
    candidates.append(greedy(range(1, 101)))       # Ascending
    candidates.append(greedy(range(50, 0, -1)))    # 50 down to 1
    candidates.append(greedy(range(101, 50, -1)))  # 101 down to 51 (empty, just 100..51)
    
    # Interleaved orders
    candidates.append(greedy([i for i in range(1, 101, 2)] + [i for i in range(2, 101, 2)]))
    candidates.append(greedy([i for i in range(100, 0, -2)] + [i for i in range(99, 0, -2)]))
    
    # Random orderings
    random.seed(42)
    for i in range(20):
        order = list(range(1, 101))
        random.shuffle(order)
        candidates.append(greedy(order))
    
    best = max(candidates, key=len)
    
    # Also try hitting set if it finishes quickly
    if time.time() < DEADLINE - 30.0:
        bad_subsets = generate_bad_subsets()
        if bad_subsets is not None and len(bad_subsets) > 0:
            R = greedy_hitting_set(bad_subsets)
            candidate_S = set(range(1, 101)) - R
            if is_valid(candidate_S) and len(candidate_S) > len(best):
                best = candidate_S
        
    # Local search to improve
    best = local_search(best, DEADLINE - 5.0)
    
    # Try multiple local search restarts if time permits
    if time.time() < DEADLINE - 10.0:
        best_copy = set(best)
        for _ in range(10):
            if time.time() >= DEADLINE - 5.0:
                break
            # Perturb: remove 1-3 random elements, then local search
            if len(best_copy) > 3:
                n_remove = random.randint(1, 3)
                to_remove = random.sample(list(best_copy), n_remove)
                perturbed = best_copy - set(to_remove)
                improved = local_search(perturbed, DEADLINE - 5.0)
                if len(improved) > len(best):
                    best = improved
                    best_copy = set(best)

    # Final attempt: try to add any remaining elements
    for x in range(1, 101):
        if time.time() >= DEADLINE - 1.0:
            break
        if x in best:
            continue
        candidate = best | {x}
        if is_valid(candidate):
            best = candidate
    
    # Final verification
    assert len(best) > 0
    assert all(1 <= x <= 100 for x in best)
    assert len(best) == len(set(best))

    # Verify no subset sums to 200
    reachable = 1
    mask = (1 << 201) - 1
    for x in sorted(best):
        new_reachable = (reachable | (reachable << x)) & mask
        if (new_reachable >> 200) & 1:
            return list(best)
        reachable = new_reachable

    return list(best)
