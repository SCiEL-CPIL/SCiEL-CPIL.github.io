import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Target differences 1..30
    N_DIFFS = 30
    # We need at least 9 elements (36 diffs >= 30)
    
    # Bitmask for differences 1..30: bit i-1 represents difference i
    # Full mask = all 30 bits set
    FULL_MASK = (1 << N_DIFFS) - 1
    
    def time_ok():
        return time.time() < deadline
    
    best_solution = None
    best_max_val = float('inf')
    
    def dfs(S, used_mask, depth):
        nonlocal best_solution, best_max_val
        
        if not time_ok():
            return False
        
        # Check if all differences covered
        if used_mask == FULL_MASK:
            max_val = S[-1]
            if max_val < best_max_val:
                best_max_val = max_val
                best_solution = list(S)
            return True
        
        # Pruning: check if remaining capacity is sufficient
        k = len(S)
        remaining_needed = bin(used_mask ^ FULL_MASK).count('1')  # number of uncovered diffs
        # If we add m more elements, max new diffs = m*k + m*(m-1)/2
        # We need m*k + m*(m-1)/2 >= remaining_needed
        # Also m <= 32 - k
        max_m = 32 - k
        max_new_diffs = max_m * k + max_m * (max_m - 1) // 2
        if max_new_diffs < remaining_needed:
            return False
        
        # Generate candidates: x > S[-1] such that x - y is in uncovered diffs for some y in S
        uncovered = used_mask ^ FULL_MASK
        max_S = S[-1]
        
        # For MRV: score each candidate by how many uncovered diffs it covers
        # and prioritize those covering "rare" diffs (diffs that are hard to cover)
        
        # Build candidate set
        candidates = {}
        for i, y in enumerate(S):
            for d in range(1, N_DIFFS + 1):
                if (uncovered >> (d - 1)) & 1:  # d is uncovered
                    x = y + d
                    if x > max_S:
                        # Check if x is valid (no duplicate diffs)
                        valid = True
                        new_diffs_mask = 0
                        for j, y2 in enumerate(S):
                            diff = x - y2
                            if diff <= 0:
                                continue
                            if diff > N_DIFFS:
                                # This diff is "wasted" but must not conflict
                                # We need to track all diffs, not just 1..30
                                # Actually for Sidon property, ALL positive diffs must be distinct
                                # So we need to check against ALL existing diffs
                                pass
                            if diff <= N_DIFFS:
                                bit = 1 << (diff - 1)
                                if (used_mask & bit) or (new_diffs_mask & bit):
                                    valid = False
                                    break
                                new_diffs_mask |= bit
                            else:
                                # diff > 30, just need to check it's not already used
                                # We need to track these too!
                                pass
                        # This is getting complex. Let me simplify by tracking all diffs.
                        if x not in candidates:
                            candidates[x] = 0
                        candidates[x] += 1  # score: number of uncovered 1..30 diffs covered
        
        # Sort candidates by score (descending) - MRV-like heuristic
        sorted_candidates = sorted(candidates.keys(), key=lambda x: (-candidates[x], x))
        
        for x in sorted_candidates:
            if not time_ok():
                return False
            
            # Verify x creates no duplicate differences (full Sidon check)
            new_diffs = [x - y for y in S]
            # All new diffs must be positive (x > all y) and distinct from each other and existing
            if len(set(new_diffs)) != len(new_diffs):
                continue
            
            # Check against existing diffs
            existing_diffs = set()
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    existing_diffs.add(S[j] - S[i])
            
            conflict = False
            for d in new_diffs:
                if d in existing_diffs:
                    conflict = True
                    break
            if conflict:
                continue
            
            # Update mask
            new_mask = used_mask
            for d in new_diffs:
                if 1 <= d <= N_DIFFS:
                    new_mask |= (1 << (d - 1))
            
            new_S = S + [x]
            
            if dfs(new_S, new_mask, depth + 1):
                return True
        
        return False
    
    # Initial call
    dfs([0], 0, 0)
    
    if best_solution is not None:
        return best_solution
    
    # Fallback: greedy construction
    a = [0]
    def all_diffs(s):
        return [s[j] - s[i] for i in range(len(s)) for j in range(i+1, len(s))]
    
    for _ in range(50):  # try to add up to 50 elements
        if not time_ok():
            break
        current_diffs = set(all_diffs(a))
        missing = [d for d in range(1, 31) if d not in current_diffs]
        if not missing:
            break
        
        # Try to find x that covers most missing diffs
        best_x = None
        best_count = 0
        max_a = max(a)
        
        # Try x values
        for x in range(max_a + 1, max_a + 200):
            new_diffs = [x - y for y in a]
            # Check Sidon property
            all_new = current_diffs | set(new_diffs)
            if len(all_new) != len(current_diffs) + len(new_diffs):
                continue
            # Count how many missing diffs are covered
            count = sum(1 for d in new_diffs if d in missing)
            if count > best_count:
                best_count = count
                best_x = x
        
        if best_x is None:
            # Try larger x
            for x in range(max_a + 200, max_a + 2000):
                new_diffs = [x - y for y in a]
                all_new = current_diffs | set(new_diffs)
                if len(all_new) != len(current_diffs) + len(new_diffs):
                    continue
                count = sum(1 for d in new_diffs if d in missing)
                if count > best_count:
                    best_count = count
                    best_x = x
                if best_count >= len(missing):
                    break
        
        if best_x is not None:
            a.append(best_x)
        else:
            # Add a large element to avoid conflicts
            x = 2 * max_a + 100
            while True:
                new_diffs = [x - y for y in a]
                all_new = current_diffs | set(new_diffs)
                if len(all_new) == len(current_diffs) + len(new_diffs):
                    a.append(x)
                    break
                x += 1
    
    # Verify and return
    return a
