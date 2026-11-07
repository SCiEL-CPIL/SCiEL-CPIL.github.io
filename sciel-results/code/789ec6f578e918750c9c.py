"""Improved program: cardinality-indexed subset-sum DP with bitset acceleration."""
# EVOLVE-BLOCK-START
import time
import random

DEADLINE_SECONDS = 55.0  # safety margin before 60s kill
MAX_SUM = 60 * 256  # 15360


def _check_valid_bitset(dp_bits, n):
    """Check no cross-cardinality collision using bitwise AND."""
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if dp_bits[i] & dp_bits[j]:
                return False
    return True


def _greedy_build_bitset(candidates, deadline):
    """Greedily build valid set using bitset DP for O(1) collision checks."""
    current = []
    dp_bits = [0] * 61  # dp_bits[k] = bitset of sums achievable with exactly k elements
    dp_bits[0] = 1  # sum 0 with 0 elements
    n = 0
    
    for x in candidates:
        if time.time() > deadline:
            break
        if n >= 60:
            break
        
        # Try adding x: new_dp[k] = dp_bits[k] | (dp_bits[k-1] << x)
        # Check collision: for each k, new_dp[k] must not overlap with new_dp[j] for j != k
        # Efficient check: compute new bits, verify no cross-cardinality AND is nonzero
        
        # Build new bitsets
        new_bits = dp_bits[:]  # copy (these are immutable ints)
        for k in range(n + 1, 0, -1):
            new_bits[k] = dp_bits[k] | (dp_bits[k - 1] << x)
        
        # Check for any cross-cardinality collision
        valid = True
        for i in range(n + 2):
            if new_bits[i] == 0:
                continue
            for j in range(i + 1, n + 2):
                if new_bits[j] == 0:
                    break
                if new_bits[i] & new_bits[j]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            dp_bits = new_bits
            current.append(x)
            n += 1
    
    return current


def _greedy_build_bitset_optimized(candidates, deadline):
    """Optimized: precompute cumulative OR to reduce collision check cost."""
    current = []
    dp_bits = [0] * 61
    dp_bits[0] = 1
    n = 0
    
    for x in candidates:
        if time.time() > deadline:
            break
        if n >= 60:
            break
        
        # Compute new bitsets
        new_bits = dp_bits[:]
        for k in range(n + 1, 0, -1):
            new_bits[k] = dp_bits[k] | (dp_bits[k - 1] << x)
        
        # Fast collision check: build prefix OR and check each new bit against others
        # More efficient: check each pair but skip zeros
        valid = True
        nonzero = [k for k in range(n + 2) if new_bits[k]]
        for idx_i in range(len(nonzero)):
            bi = nonzero[idx_i]
            for idx_j in range(idx_i + 1, len(nonzero)):
                bj = nonzero[idx_j]
                if new_bits[bi] & new_bits[bj]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            dp_bits = new_bits
            current.append(x)
            n += 1
    
    return current


def _greedy_build_bitset_fast(candidates, deadline):
    """Fastest variant: single pass collision check using union tracking."""
    current = []
    dp_bits = [0] * 61
    dp_bits[0] = 1
    n = 0
    
    for x in candidates:
        if time.time() > deadline:
            break
        if n >= 60:
            break
        
        # Compute only the new bits that change
        # new_bits[k] = dp_bits[k] | (dp_bits[k-1] << x)
        # The "new" contribution at level k is (dp_bits[k-1] << x) & ~dp_bits[k]
        # But for collision we need full new_bits[k]
        
        new_bits = dp_bits[:]
        for k in range(n + 1, 0, -1):
            new_bits[k] = dp_bits[k] | (dp_bits[k - 1] << x)
        
        # Collision check: for each pair (i,j) with i<j, check new_bits[i] & new_bits[j]
        # Optimization: track which levels are non-zero
        valid = True
        nz = []
        for k in range(n + 2):
            if new_bits[k]:
                nz.append(k)
        
        for a in range(len(nz)):
            ia = nz[a]
            ba = new_bits[ia]
            for b in range(a + 1, len(nz)):
                ib = nz[b]
                if ba & new_bits[ib]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            dp_bits = new_bits
            current.append(x)
            n += 1
    
    return current


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Descending (largest first for max gap)
    candidates = list(range(256, 0, -1))
    current = _greedy_build_bitset_fast(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Strategy 2: Ascending
    candidates = list(range(1, 257))
    current = _greedy_build_bitset_fast(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Strategy 3: Spaced (every other, every third, etc.)
    for step in [2, 3, 4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        candidates = list(range(step, 257, step))
        current = _greedy_build_bitset_fast(candidates, deadline)
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Strategy 4: Random restarts
    random.seed(42)
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        if attempt % 4 == 0:
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        elif attempt % 4 == 1:
            # Descending with random perturbation
            candidates = list(range(256, 0, -1))
            for _ in range(10):
                i, j = random.randint(0, 255), random.randint(0, 255)
                candidates[i], candidates[j] = candidates[j], candidates[i]
        elif attempt % 4 == 2:
            # Spaced with random start
            step = random.randint(2, 10)
            start = random.randint(1, step)
            candidates = list(range(start, 257, step))
            random.shuffle(candidates)
        else:
            # Mix: large values first, then random small
            large = list(range(129, 257))
            small = list(range(1, 129))
            random.shuffle(small)
            candidates = large + small
        
        current = _greedy_build_bitset_fast(candidates, deadline)
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
        
        # Try to extend best_set with remaining candidates
        if best_size >= 5 and best_size < 60:
            in_set = set(best_set)
            remaining = [x for x in range(1, 257) if x not in in_set]
            random.shuffle(remaining)
            improved = best_set[:]
            # Rebuild dp for improved
            dp_bits = [0] * 61
            dp_bits[0] = 1
            n = 0
            for val in improved:
                for k in range(n + 1, 0, -1):
                    dp_bits[k] = dp_bits[k] | (dp_bits[k - 1] << val)
                n += 1
            
            for x in remaining:
                if time.time() > deadline:
                    break
                if n >= 60:
                    break
                new_bits = dp_bits[:]
                for k in range(n + 1, 0, -1):
                    new_bits[k] = dp_bits[k] | (dp_bits[k - 1] << x)
                
                valid = True
                nz = []
                for k in range(n + 2):
                    if new_bits[k]:
                        nz.append(k)
                for a in range(len(nz)):
                    ia = nz[a]
                    ba = new_bits[ia]
                    for b in range(a + 1, len(nz)):
                        ib = nz[b]
                        if ba & new_bits[ib]:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    dp_bits = new_bits
                    improved.append(x)
                    n += 1
            
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved[:]
    
    return list(best_set)
# EVOLVE-BLOCK-END
