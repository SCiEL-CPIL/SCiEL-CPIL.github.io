"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _fast_check(candidate):
    """
    Fast check: for each element a in candidate, check if a is the mean of
    any subset of >=2 OTHER elements.
    Uses subset-sum DP on the other elements.
    Returns True if valid (no element is the mean of any subset of others).
    """
    n = len(candidate)
    if n < 2:
        return True
    
    # For each element a, we need to check if there exists a subset S of
    # {candidate \ {a}} with |S| >= 2 such that sum(S) = len(S) * a
    # i.e., sum(S) - len(S)*a = 0
    # Let b_i = c_i - a for each other element c_i
    # We need sum of some subset of b's = 0 with subset size >= 2
    
    # But wait, that's not quite right. sum(S) = len(S) * a means
    # sum(c_i for c_i in S) = len(S) * a
    # sum(c_i - a for c_i in S) = 0
    
    # So we need a subset of the transformed values {c_i - a} that sums to 0
    # and has size >= 2.
    
    for idx_a, a in enumerate(candidate):
        others = [c for i, c in enumerate(candidate) if i != idx_a]
        if len(others) < 2:
            continue
        
        # Transform: b_i = c_i - a
        b_vals = [c - a for c in others]
        
        # We need subset of b_vals with sum=0 and size>=2
        # DP: dp[s] = set of possible subset sizes achieving sum s
        # But sums can be negative. Range of b_vals: [-a, 512-a]
        # Min possible sum: sum of most negative values
        # Max possible sum: sum of most positive values
        
        # For efficiency, track min and max achievable sums
        # Use a dict: sum_value -> set of sizes (or just track min/max size for each sum)
        # Actually, we just need to know if sum 0 is achievable with size >= 2.
        
        # DP approach: dp[k][s] = True if we can pick k elements summing to s
        # This is O(n * n * range) which might be too slow for n=48
        
        # Alternative: use a set of (sum, min_size, max_size) or just track
        # for each sum, the set of achievable sizes.
        
        # Let's use: reachable[s] = set of sizes that achieve sum s
        # Start: reachable[0] = {0}
        # For each b in b_vals:
        #   new_reachable = copy of reachable
        #   for s, sizes in reachable.items():
        #     for sz in sizes:
        #       new_reachable[s+b].add(sz+1)
        
        # But this can blow up. Let's limit: we only care about sum=0 with size>=2.
        # Also, we can prune: if at any point we find sum=0 with size>=2, return False (invalid).
        
        # More efficient: track for each sum, the min and max size.
        # If min_size[0] >= 2 at any point... wait, we need to check after adding each element.
        
        # Actually, let's just track reachable sums with their achievable sizes as a bitset or set.
        # For n=48, the number of distinct sums is bounded by the range.
        # Range of b_vals: each is in [-511, 511] roughly.
        # Sum range: [-48*511, 48*511] ~ [-24528, 24528]
        
        # Let's use a dictionary: sum -> set of sizes
        # But to keep it fast, let's cap the size tracking.
        
        # Simpler approach for fast check: only check small subsets (size 2,3,4)
        # and use a heuristic for larger subsets.
        # For size 2: need b_i + b_j = 0, i.e., c_i + c_j = 2a
        # For size 3: need c_i + c_j + c_k = 3a
        # etc.
        
        # Let's do: for k in range(2, min(len(others)+1, 6)): check combinations
        # This is O(n^5) worst case but with n=48, combinations(47,5) ~ 1.5M, too slow.
        
        # Better: use the DP with sum->size tracking, but limit the range.
        
        # Let's try a different fast check:
        # Check if any pair sums to 2a: use a set
        # Check if any triple sums to 3a: use nested loops with pruning
        # For larger subsets, use a probabilistic/heuristic check
        
        # Fast pair check:
        seen = set()
        for b in b_vals:
            if -b in seen and b != 0:  # need distinct elements, but b can be 0 if c==a
                # Wait, we need two DISTINCT OTHER elements. b_i and b_j from different indices.
                # If b==0, then c_i == a, but a is not in others, so b != 0 always? 
                # No, b_i = c_i - a, and c_i != a since a is removed. So b_i != 0.
                # Actually c_i can equal a? No, we removed a from candidate.
                # But two different other elements could have the same value.
                # We need two distinct indices.
                return False
            seen.add(b)
        
        # If we get here, no pair sums to 0. Check triples?
        # For a fast check, let's also check triples using a hash.
        # For each pair (i,j), check if -(b_i+b_j) is in the remaining.
        # This is O(n^2) per a, which is O(n^3) total. For n=48, 48^3 ~ 110K, fine.
        
        n_others = len(b_vals)
        for i in range(n_others):
            for j in range(i+1, n_others):
                target = -(b_vals[i] + b_vals[j])
                # Check if target exists in b_vals at index > j
                # To avoid checking the same pair multiple times, we could use a set
                # but we need to ensure distinct indices.
                # Simple approach: check if target is in b_vals[j+1:]
                # This is O(n) per pair, making it O(n^3) total per a, O(n^4) overall.
                # For n=48, 48^4 ~ 5.3M, might be too slow.
                
                # Alternative: precompute a frequency map for b_vals
                # But we need distinct indices, so we need to be careful.
                
                # Let's just do the O(n) check for now, it's a fast check.
                for k in range(j+1, n_others):
                    if b_vals[k] == target:
                        return False
        
        # For the fast check, we'll skip size >= 4. The slow check will handle those.
    
    return True

def _slow_check(candidate):
    """
    Full DP check: for each element a, use subset-sum DP to verify no subset
    of >=2 others has mean a.
    """
    n = len(candidate)
    if n < 2:
        return True
    
    for idx_a, a in enumerate(candidate):
        others = [c for i, c in enumerate(candidate) if i != idx_a]
        m = len(others)
        if m < 2:
            continue
        
        b_vals = [c - a for c in others]
        
        # DP: reachable[s] = set of sizes
        # We'll use a dict mapping sum to a set of sizes.
        # To keep it manageable, we can limit the size set.
        
        # Initialize
        reachable = {0: {0}}
        
        for b in b_vals:
            new_reachable = {}
            # Copy existing
            for s, sizes in reachable.items():
                if s not in new_reachable:
                    new_reachable[s] = set()
                new_reachable[s].update(sizes)
            
            # Add b to each existing sum
            for s, sizes in reachable.items():
                ns = s + b
                if ns not in new_reachable:
                    new_reachable[ns] = set()
                for sz in sizes:
                    new_reachable[ns].add(sz + 1)
            
            reachable = new_reachable
            
            # Early exit: if 0 is in reachable with any size >= 2
            if 0 in reachable:
                if any(sz >= 2 for sz in reachable[0]):
                    return False
        
        # After processing all, check again
        if 0 in reachable:
            if any(sz >= 2 for sz in reachable[0]):
                return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    # Start with a greedy construction using fast check
    selected = []
    
    # Try to build a large set greedily
    for x in range(1, 513):
        if time.time() > deadline:
            break
        candidate = selected + [x]
        if _fast_check(candidate):
            # Verify with slow check for safety
            if _slow_check(candidate):
                selected = candidate
            else:
                # Fast check passed but slow check failed; this shouldn't happen
                # if fast check is a subset of slow check conditions.
                # Actually, fast check only checks pairs and triples, so it's a subset.
                # If fast check passes, it doesn't mean slow check passes.
                # We should not add x if slow check fails.
                pass
    
    best = list(selected)
    
    # Now try to improve by adding more elements or swapping
    # Try random restarts with different orderings
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Create a random ordering of candidates
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        selected = []
        for x in candidates:
            if time.time() > deadline:
                break
            if x in selected:
                continue
            candidate = selected + [x]
            if _fast_check(candidate):
                if _slow_check(candidate):
                    selected = candidate
        
        if len(selected) > len(best):
            best = list(selected)
            if len(best) >= 48:
                break
    
    # Try local improvement: try to add elements not in best
    if len(best) < 48:
        best_set = set(best)
        for x in range(1, 513):
            if time.time() > deadline:
                break
            if x in best_set:
                continue
            candidate = best + [x]
            if _fast_check(candidate):
                if _slow_check(candidate):
                    best = candidate
                    best_set.add(x)
    
    return best
# EVOLVE-BLOCK-END
