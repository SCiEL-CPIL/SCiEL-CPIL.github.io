import time
import random

START = time.time()
DEADLINE = 55.0

def _validate(vals):
    n = len(vals)
    sums_by_k = [0] * (n + 1)
    sums_by_k[0] = 1
    for v in vals:
        for k in range(n - 1, -1, -1):
            if sums_by_k[k]:
                sums_by_k[k + 1] |= (sums_by_k[k] << v)
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if sums_by_k[i] & sums_by_k[j]:
                return False
    return True

def _try_add(sums_by_k, c):
    """Try to add element c to current set represented by sums_by_k.
    Returns new sums_by_k if valid, None if collision detected."""
    n = len(sums_by_k) - 1
    new_sums = sums_by_k[:] + [0]
    for k in range(n, -1, -1):
        if sums_by_k[k]:
            new_sums[k + 1] |= (sums_by_k[k] << c)
    
    # Only check pairs where at least one index is >= n (new or updated levels)
    # Pairs among indices 0..n-1 are already valid (invariant)
    # New collisions can only involve new_sums[n] or new_sums[n+1]
    # new_sums[n] = sums_by_k[n] | (sums_by_k[n-1] << c)
    # new_sums[n+1] = sums_by_k[n] << c
    # Check new_sums[n] against all existing levels 0..n-1
    for i in range(n):
        if new_sums[i] & new_sums[n]:
            return None
    # Check new_sums[n+1] against all existing levels 0..n
    for i in range(n + 1):
        if new_sums[i] & new_sums[n + 1]:
            return None
    return new_sums

def _greedy_lattice(candidate_order):
    """Greedy construction over a structured candidate lattice.
    For each position, try all remaining candidates in order and pick the first valid one.
    This is O(1) per candidate check via bitwise AND."""
    sums_by_k = [1]
    current_vals = []
    used = set()
    
    for c in candidate_order:
        if time.time() - START > DEADLINE - 2:
            break
        if c in used:
            continue
        new_sums = _try_add(sums_by_k, c)
        if new_sums is not None:
            sums_by_k = new_sums
            current_vals.append(c)
            used.add(c)
    
    return current_vals

def _greedy_best_first(candidate_order, max_lookahead=200):
    """Greedy best-first: at each step, examine up to max_lookahead candidates
    and pick the one that maximizes future extensibility."""
    sums_by_k = [1]
    current_vals = []
    used = set()
    
    remaining = [c for c in candidate_order if c not in used]
    
    while remaining:
        if time.time() - START > DEADLINE - 2:
            break
        
        best_c = None
        best_sums = None
        best_score = float('inf')
        best_idx = -1
        
        look_limit = min(len(remaining), max_lookahead)
        
        for idx in range(look_limit):
            c = remaining[idx]
            new_sums = _try_add(sums_by_k, c)
            if new_sums is not None:
                # Score: prefer candidates that leave more room for future additions
                # Fewer total bits = more compact = potentially more room
                # Also penalize large values that consume high sum space
                total_bits = sum(s.bit_count() for s in new_sums if s)
                max_sum = 0
                for s in new_sums:
                    if s:
                        msb = s.bit_length() - 1
                        if msb > max_sum:
                            max_sum = msb
                # Combined score: compactness + max sum penalty
                score = total_bits + max_sum * 2
                if score < best_score:
                    best_score = score
                    best_c = c
                    best_sums = new_sums
                    best_idx = idx
        
        if best_c is None:
            break
        
        sums_by_k = best_sums
        current_vals.append(best_c)
        used.add(best_c)
        remaining.pop(best_idx)
    
    return current_vals

def _generate_lattices():
    """Generate several structured candidate orderings (lattices)."""
    lattices = []
    
    # 1. Powers of 2 (classic)
    p2 = [2**i for i in range(8) if 2**i <= 256]
    lattices.append(p2)
    
    # 2. Geometric with ratio ~1.5
    geo15 = []
    v = 1
    while v <= 256:
        geo15.append(v)
        v = int(v * 1.5) + 1
    lattices.append(geo15)
    
    # 3. Geometric with ratio ~1.3
    geo13 = []
    v = 1
    while v <= 256:
        geo13.append(v)
        v = int(v * 1.3) + 1
    lattices.append(geo13)
    
    # 4. Descending from 256
    lattices.append(list(range(256, 0, -1)))
    
    # 5. Ascending from 1
    lattices.append(list(range(1, 257)))
    
    # 6. Spaced: multiples of various bases
    for base in [3, 5, 7, 11, 13, 17]:
        spaced = [i * base for i in range(1, 257 // base + 1)]
        lattices.append(spaced)
    
    # 7. Fibonacci-like
    fib = [1, 2]
    while fib[-1] + fib[-2] <= 256:
        fib.append(fib[-1] + fib[-2])
    lattices.append(fib)
    
    # 8. Mixed: large powers of 2 first, then fill in
    mixed = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    for i in range(3, 257):
        if i not in mixed:
            mixed.append(i)
    lattices.append(mixed)
    
    # 9. Quadratic growth
    quad = []
    v = 1
    step = 1
    while v <= 256:
        quad.append(v)
        step += 2
        v += step
    lattices.append(quad)
    
    # 10. Randomized permutations of powers of 2 and nearby values
    rng = random.Random(42)
    core = [2**i for i in range(8) if 2**i <= 256]
    for _ in range(5):
        perm = core[:]
        rng.shuffle(perm)
        # Add some random fillers
        extras = [x for x in range(1, 257) if x not in perm]
        rng.shuffle(extras)
        lattices.append(perm + extras[:200])
    
    return lattices

def _local_search_add(vals, max_iterations=200):
    """Try to add more elements to the set by trying all available values repeatedly."""
    if not vals:
        return vals
    
    current_vals = vals[:]
    n = len(current_vals)
    
    # Build sums_by_k
    sums_by_k = [0] * (n + 1)
    sums_by_k[0] = 1
    for v in current_vals:
        for k in range(n - 1, -1, -1):
            if sums_by_k[k]:
                sums_by_k[k + 1] |= (sums_by_k[k] << v)
    
    used = set(current_vals)
    
    # Repeatedly try to add any available value until no more can be added
    for _ in range(max_iterations):
        if time.time() - START > DEADLINE - 3:
            break
        added = False
        for c in range(1, 257):
            if c in used:
                continue
            new_sums = _try_add(sums_by_k, c)
            if new_sums is not None:
                sums_by_k = new_sums
                current_vals.append(c)
                used.add(c)
                added = True
                break  # Restart scan after each addition
        if not added:
            break
    
    return current_vals

def _swap_improve(vals, max_rounds=3):
    """Try to improve the set by swapping elements to make room for more additions."""
    if not vals:
        return vals
    
    current_vals = vals[:]
    
    for round_num in range(max_rounds):
        if time.time() - START > DEADLINE - 3:
            break
        
        improved = False
        n = len(current_vals)
        
        used = set(current_vals)
        available = [x for x in range(1, 257) if x not in used]
        
        # Try removing one element and re-adding others
        for i in range(n):
            if time.time() - START > DEADLINE - 3:
                break
            old_val = current_vals[i]
            
            # Build sums without element i
            reduced_vals = current_vals[:i] + current_vals[i+1:]
            rn = len(reduced_vals)
            sums_by_k = [0] * (rn + 1)
            sums_by_k[0] = 1
            for v in reduced_vals:
                for k in range(rn - 1, -1, -1):
                    if sums_by_k[k]:
                        sums_by_k[k + 1] |= (sums_by_k[k] << v)
            
            # Try to add new elements (including possibly old_val back)
            trial_used = set(reduced_vals)
            trial_vals = reduced_vals[:]
            for c in available:
                if c in trial_used:
                    continue
                new_sums = _try_add(sums_by_k, c)
                if new_sums is not None:
                    sums_by_k = new_sums
                    trial_vals.append(c)
                    trial_used.add(c)
            
            if len(trial_vals) > len(current_vals):
                current_vals = trial_vals
                improved = True
                break
        
        if not improved:
            break
    
    return current_vals

def construct():
    global START, DEADLINE
    
    best_result = [1, 2, 4, 8, 16, 32, 64, 128]
    best_len = len(best_result)
    
    lattices = _generate_lattices()
    
    # Phase 1: Pure lattice greedy (very fast, O(1) per candidate)
    for lattice in lattices:
        if time.time() - START > DEADLINE - 5:
            break
        res = _greedy_lattice(lattice)
        if len(res) > best_len:
            if _validate(res):
                best_result = res
                best_len = len(res)
    
    # Phase 2: Best-first greedy on lattices (slower but more optimal choices)
    for lattice in lattices[:10]:
        if time.time() - START > DEADLINE - 5:
            break
        for lookahead in [50, 100, 200]:
            if time.time() - START > DEADLINE - 3:
                break
            res = _greedy_best_first(lattice, max_lookahead=lookahead)
            if len(res) > best_len:
                if _validate(res):
                    best_result = res
                    best_len = len(res)
    
    # Phase 3: Local search - try to add more elements to best result
    if time.time() - START < DEADLINE - 5:
        improved = _local_search_add(best_result)
        if _validate(improved) and len(improved) >= best_len:
            best_result = improved
            best_len = len(best_result)
    
    # Phase 4: Swap-improve to find room for more elements
    if time.time() - START < DEADLINE - 5:
        improved = _swap_improve(best_result)
        if _validate(improved) and len(improved) >= best_len:
            best_result = improved
            best_len = len(best_result)
    
    # Phase 4b: Another round of local search after swap
    if time.time() - START < DEADLINE - 4:
        improved = _local_search_add(best_result)
        if _validate(improved) and len(improved) >= best_len:
            best_result = improved
            best_len = len(best_result)
    
    # Phase 5: More lattice variations with randomization
    rng = random.Random(12345)
    for trial in range(20):
        if time.time() - START > DEADLINE - 3:
            break
        # Create a randomized lattice
        base = list(range(1, 257))
        # Weight towards larger values
        weighted = []
        for v in base:
            weight = v // 16 + 1
            weighted.extend([v] * weight)
        rng.shuffle(weighted)
        # Deduplicate while preserving order
        seen = set()
        ordered = []
        for v in weighted:
            if v not in seen:
                ordered.append(v)
                seen.add(v)
        res = _greedy_lattice(ordered)
        if len(res) > best_len:
            if _validate(res):
                best_result = res
                best_len = len(res)
    
    # Phase 6: Final local search
    if time.time() - START < DEADLINE - 2:
        improved = _local_search_add(best_result)
        if _validate(improved) and len(improved) >= best_len:
            best_result = improved
            best_len = len(best_result)
    
    # Phase 7: Final swap + local search
    if time.time() - START < DEADLINE - 3:
        improved = _swap_improve(best_result)
        if _validate(improved) and len(improved) >= best_len:
            best_result = improved
            best_len = len(best_result)
        if time.time() - START < DEADLINE - 2:
            improved = _local_search_add(best_result)
            if _validate(improved) and len(improved) >= best_len:
                best_result = improved
                best_len = len(best_result)
    
    return [int(x) for x in best_result]

if __name__ == "__main__":
    res = construct()
    print(f"Found size: {len(res)}")
    print(res)
    print("Valid:", _validate(res))
