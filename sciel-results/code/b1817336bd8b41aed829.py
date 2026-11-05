"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 25.0
    start = time.time()
    
    def get_sums_bitmask(S):
        """Return a bitmask where bit s is set if sum s appears in S."""
        mask = 0
        n = len(S)
        for i in range(n):
            si = S[i]
            mask |= (1 << (si + si))
            for j in range(i + 1, n):
                mask |= (1 << (si + S[j]))
        return mask
    
    def can_add_bitmask(x, sums_mask):
        """Check if x can be added given sums_mask."""
        if (sums_mask >> (2 * x)) & 1:
            return False
        # Need to check x+a for all a in S, but we don't have S here
        # This is a simplified check; full check needs S
        return True
    
    def is_sidon(S):
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def is_maximal_fast(S):
        """Fast maximality check using precomputed sums."""
        sums = set()
        n = len(S)
        for i in range(n):
            si = S[i]
            sums.add(si + si)
            for j in range(i + 1, n):
                sums.add(si + S[j])
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Check if x can be added
            if 2 * x in sums:
                continue
            ok = True
            for a in S:
                if x + a in sums:
                    ok = False
                    break
            if ok:
                return False
        return True
    
    def blocking_power(x, S, sums):
        """Count how many elements x blocks (i.e., how many y cannot be added if x is in S)."""
        # This is expensive; use as heuristic
        count = 0
        for y in range(1, N + 1):
            if y in S:
                continue
            # Check if y can be added to S without x
            S_without_x = [a for a in S if a != x]
            if not S_without_x:
                continue
            sums_wo = set()
            for i in range(len(S_without_x)):
                si = S_without_x[i]
                sums_wo.add(si + si)
                for j in range(i + 1, len(S_without_x)):
                    sums_wo.add(si + S_without_x[j])
            # Check if y can be added to S_without_x
            if 2 * y in sums_wo:
                continue
            can_add_wo = True
            for a in S_without_x:
                if y + a in sums_wo:
                    can_add_wo = False
                    break
            if can_add_wo:
                # y could be added without x, so x blocks y
                count += 1
        return count
    
    def greedy_construct_scored(order, score_func=None):
        """Greedy construction with optional scoring for tie-breaking."""
        S = []
        sums = set()
        remaining = list(order)
        
        while remaining:
            best_x = None
            best_score = -1
            best_block_count = 0
            
            for idx, x in enumerate(remaining):
                ok = True
                if 2 * x in sums:
                    ok = False
                else:
                    for a in S:
                        if x + a in sums:
                            ok = False
                            break
                if not ok:
                    continue
                
                # Calculate blocking power (how many future elements this blocks)
                # Simplified: count conflicts with remaining elements
                block_count = 0
                for y in remaining[idx + 1:]:
                    if y == x:
                        continue
                    # Check if x and y would conflict
                    if 2 * y in sums or (x + y) in sums:
                        block_count += 1
                    for a in S:
                        if (x + a) in sums and (y + a) in sums:
                            # Both would be blocked by same sum
                            block_count += 0.5
                            break
                
                if score_func:
                    score = score_func(x, S, sums)
                else:
                    score = -block_count  # Prefer elements that block fewer
                
                if score > best_score or (score == best_score and block_count < best_block_count):
                    best_score = score
                    best_x = x
                    best_block_count = block_count
            
            if best_x is None:
                break
            
            S.append(best_x)
            sums.add(2 * best_x)
            for a in S[:-1]:
                sums.add(best_x + a)
            remaining.remove(best_x)
        
        return S
    
    def prune_for_maximality_optimized(S):
        """Remove elements while maintaining maximality, prioritizing low-blocking-power elements."""
        S = list(S)
        changed = True
        while changed:
            changed = False
            if time.time() > deadline:
                break
            
            # Compute blocking power for each element
            block_powers = []
            for i in range(len(S)):
                if time.time() > deadline:
                    break
                # Count how many elements this element blocks
                bp = 0
                for x in range(1, N + 1):
                    if x in S:
                        continue
                    # Check if x can be added without S[i]
                    S_wo = S[:i] + S[i + 1:]
                    if not S_wo:
                        bp += 1
                        continue
                    sums_wo = set()
                    for a_idx in range(len(S_wo)):
                        sa = S_wo[a_idx]
                        sums_wo.add(sa + sa)
                        for b_idx in range(a_idx + 1, len(S_wo)):
                            sums_wo.add(sa + S_wo[b_idx])
                    if 2 * x in sums_wo:
                        continue
                    can_add = True
                    for a in S_wo:
                        if x + a in sums_wo:
                            can_add = False
                            break
                    if can_add:
                        bp += 1
                block_powers.append(bp)
            
            # Try removing element with lowest blocking power
            if block_powers:
                min_bp_idx = block_powers.index(min(block_powers))
                new_S = S[:min_bp_idx] + S[min_bp_idx + 1:]
                if new_S and is_maximal_fast(new_S):
                    S = new_S
                    changed = True
        
        return S
    
    def try_remove_pair(S, i, j):
        """Try removing elements at positions i and j, check if still maximal."""
        if i == j:
            return None
        new_S = S[:min(i,j)] + S[min(i,j)+1:max(i,j)] + S[max(i,j)+1:]
        if len(new_S) == 0:
            return None
        if is_maximal_fast(new_S):
            return new_S
        return None
    
    # Phase 1: Build initial Sidon set using ascending greedy
    S = greedy_construct_scored(range(1, N + 1))
    S = prune_for_maximality_optimized(S)
    best_S = list(S)
    best_size = len(S)
    
    # Phase 2: Try different greedy orderings with scoring
    rng = random.Random(12345)
    
    for trial in range(150):
        if time.time() > deadline:
            break
        
        # Various ordering strategies
        if trial % 5 == 0:
            # Random
            order = list(range(1, N + 1))
            rng.shuffle(order)
        elif trial % 5 == 1:
            # Descending
            order = list(range(N, 0, -1))
        elif trial % 5 == 2:
            # Start from middle
            order = list(range(N // 2, N + 1)) + list(range(1, N // 2))
        elif trial % 5 == 3:
            # Alternating
            order = list(range(1, N + 1))
            for i in range(0, len(order) - 1, 2):
                order[i], order[i + 1] = order[i + 1], order[i]
        else:
            # Random with bias toward larger numbers
            order = list(range(1, N + 1))
            for i in range(len(order) - 1, 0, -1):
                j = rng.randint(0, i)
                order[i], order[j] = order[j], order[i]
        
        S = greedy_construct_scored(order)
        S = prune_for_maximality_optimized(S)
        
        if len(S) < best_size:
            best_size = len(S)
            best_S = list(S)
    
    # Phase 3: Local search - try swapping elements with blocking-power guidance
    S = list(best_S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(S)):
            if time.time() > deadline:
                break
            S_without = S[:i] + S[i + 1:]
            if not S_without:
                continue
            
            sums_without = set()
            for a_idx in range(len(S_without)):
                sa = S_without[a_idx]
                sums_without.add(sa + sa)
                for b_idx in range(a_idx + 1, len(S_without)):
                    sums_without.add(sa + S_without[b_idx])
            S_without_set = set(S_without)
            
            # Try to find a replacement that keeps it maximal
            for x in range(1, N + 1):
                if x in S_without_set:
                    continue
                if 2 * x in sums_without:
                    continue
                can_add = True
                for a in S_without:
                    if x + a in sums_without:
                        can_add = False
                        break
                if not can_add:
                    continue
                new_S = sorted(S_without + [x])
                if is_maximal_fast(new_S):
                    S = new_S
                    improved = True
                    if len(S) < best_size:
                        best_size = len(S)
                        best_S = list(S)
                    break
    
    # Phase 4: Aggressive removal with blocking-power prioritization
    S = list(best_S)
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        if len(S) > 10:
            new_S = list(S)
            # Try removing elements from the end, prioritizing low blocking power
            for i in range(len(new_S) - 1, -1, -1):
                if time.time() > deadline:
                    break
                candidate = new_S[:i] + new_S[i + 1:]
                if is_maximal_fast(candidate):
                    new_S = candidate
            
            if len(new_S) < len(best_S):
                best_S = new_S
                best_size = len(new_S)
    
    # Phase 5: Pair removal - try removing two elements at once
    S = list(best_S)
    for attempt in range(20):
        if time.time() > deadline:
            break
        if len(S) <= 12:
            break
        
        found_removal = False
        for i in range(len(S) - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(S)):
                if time.time() > deadline:
                    break
                result = try_remove_pair(S, i, j)
                if result is not None:
                    S = result
                    found_removal = True
                    if len(S) < best_size:
                        best_size = len(S)
                        best_S = list(S)
                    break
            if found_removal:
                break
    
    # Phase 6: Final aggressive single-element removal pass
    S = list(best_S)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline:
                break
            new_S = S[:i] + S[i + 1:]
            if len(new_S) == 0:
                continue
            if is_maximal_fast(new_S):
                S = new_S
                changed = True
                if len(S) < best_size:
                    best_size = len(S)
                    best_S = list(S)
                break
    
    # Final verification and repair
    S = list(best_S)
    if not is_sidon(S):
        # Fall back to ascending greedy
        S = greedy_construct_scored(range(1, N + 1))
        S = prune_for_maximality_optimized(S)
    
    if not is_maximal_fast(S):
        # Repair: add elements that can be added
        changed = True
        while changed:
            changed = False
            sums = set()
            for a_idx in range(len(S)):
                sa = S[a_idx]
                sums.add(sa + sa)
                for b_idx in range(a_idx + 1, len(S)):
                    sums.add(sa + S[b_idx])
            S_set = set(S)
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                if 2 * x in sums:
                    continue
                can_add = True
                for a in S:
                    if x + a in sums:
                        can_add = False
                        break
                if can_add:
                    S.append(x)
                    S_set.add(x)
                    sums.add(2 * x)
                    for a in S[:-1]:
                        sums.add(x + a)
                    changed = True
                    break
        S = sorted(S)
    
    return S

# EVOLVE-BLOCK-END
