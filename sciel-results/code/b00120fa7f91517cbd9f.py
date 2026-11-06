import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 256
    
    # Precompute: for each a, the set of sums in [2, 2n] divisible by a
    # This is used in the fallback or if we don't use the new precompute
    divs = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for mult in range(2, 2 * n // a + 1):
            v = a * mult
            if v <= 2 * n:
                divs[a].add(v)
                
    def precompute_can_add(S):
        """
        Precompute a boolean array can_add[1..n] where can_add[x] is True
        if x can be added to S without violating constraints.
        """
        S_list = sorted(S)
        S_set = set(S_list)
        k = len(S_list)
        can_add = [False] * (n + 1)
        
        # Precompute residues for S for each modulus a in S?
        # Or just check on the fly for each x.
        
        # For each x in 1..n not in S:
        for x in range(1, n + 1):
            if x in S_set:
                continue
                
            valid = True
            
            # Case 1: x is 'a'. Need b, c in S with b > x, c >= b, x | (b+c)
            # We need to check if there exist b, c in S_{>x} such that (b+c) % x == 0
            # Let S_gt_x = [y for y in S_list if y > x]
            # We can optimize by checking residues.
            # For small x, this is fast.
            
            # Optimization: 
            # If x is small, S_gt_x is large.
            # If x is large, S_gt_x is small.
            
            # Let's use a helper to check Case 1
            # Check if there are b, c in S_gt_x with b+c divisible by x
            # We can iterate b in S_gt_x and check if (-b % x) is in residues of S_gt_x
            # But we need to handle b=c case.
            
            residues_count = {}
            for y in S_list:
                if y > x:
                    r = y % x
                    residues_count[r] = residues_count.get(r, 0) + 1
            
            for r, cnt in residues_count.items():
                target = (-r) % x
                if target == r:
                    if cnt >= 2:
                        valid = False
                        break
                else:
                    if target in residues_count and residues_count[target] >= 1:
                        # Need to ensure we don't use the same element twice if cnt was 1?
                        # If r != target, they are distinct residues, so distinct elements.
                        # Wait, if r != target, we pick one element with residue r and one with residue target.
                        # They are distinct elements.
                        valid = False
                        break
            
            if not valid:
                can_add[x] = False
                continue
                
            # Case 2: x is 'b'. Need a, c in S with a < x <= c, a | (x+c)
            # Iterate a in S with a < x
            for a in S_list:
                if a >= x:
                    break # S_list is sorted
                # Need c in S with c >= x and (x+c) % a == 0
                # i.e., c % a == (-x) % a
                target_res = (-x) % a
                found_c = False
                for c in S_list:
                    if c < x:
                        continue
                    if c % a == target_res:
                        found_c = True
                        break
                if found_c:
                    valid = False
                    break
            
            if not valid:
                can_add[x] = False
                continue
                
            # Case 3: x is 'c'. Need a, b in S with a < b < x, a | (b+x)
            # Iterate a in S with a < x
            for a in S_list:
                if a >= x:
                    break
                # Need b in S with a < b < x and (b+x) % a == 0
                # i.e., b % a == (-x) % a
                target_res = (-x) % a
                found_b = False
                for b in S_list:
                    if b <= a:
                        continue
                    if b >= x:
                        break
                    if b % a == target_res:
                        found_b = True
                        break
                if found_b:
                    valid = False
                    break
            
            can_add[x] = valid
            
        return can_add

    def full_validate(S):
        """Full validation of set S."""
        S_sorted = sorted(S)
        k = len(S_sorted)
        for i in range(k):
            a = S_sorted[i]
            for j in range(i + 1, k):
                b = S_sorted[j]
                for l in range(j, k):
                    c = S_sorted[l]
                    if (b + c) % a == 0:
                        return False
        return True

    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    # Strategy 1: Decreasing greedy with precomputed checks? 
    # Precomputing for each step is O(n^3) which is ~2.7M. 
    # 256 steps * 2.7M is too slow.
    # So we stick to O(k^2) check for greedy construction.
    
    def check_decreasing_add(S, x):
        """Check if x can be added when all elements in S are > x.
        Only need to check: for all b,c in S with b<=c, x ∤ (b+c)."""
        for i in range(len(S)):
            b = S[i]
            for j in range(i, len(S)):
                c = S[j]
                if (b + c) % x == 0:
                    return False
        return True

    def full_check_add(S_set, x):
        """Full check for adding x to S_set (x can be anywhere in ordering)."""
        S_list = list(S_set)
        
        # Case 1: x is 'a' (smallest in triple)
        for b in S_list:
            if b <= x:
                continue
            for c in S_list:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is 'b' (middle)
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is 'c' (largest)
        for a in S_list:
            for b in S_list:
                if a >= b:
                    continue
                if b >= x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True

    # Strategy 1: Decreasing greedy
    S = []
    for x in range(256, 0, -1):
        if time.time() > deadline - 5.0:
            break
        if check_decreasing_add(S, x):
            S.append(x)
            
    if len(S) > best_size:
        best_size = len(S)
        best_set = list(S)
        
    # Strategy 2: Start with upper half, add lower
    for trial in range(10):
        if time.time() > deadline - 5.0:
            break
        S = list(range(129, 257))
        S_set = set(S)
        for x in range(128, 0, -1):
            if time.time() > deadline - 5.0:
                break
            if full_check_add(S_set, x):
                S.append(x)
                S_set.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            
    # Strategy 3: Different upper ranges
    for upper_start in [140, 150, 160, 170, 180, 190, 200]:
        if time.time() > deadline - 5.0:
            break
        S = list(range(upper_start, 257))
        S_set = set(S)
        if not full_validate(S):
            S = []
            S_set = set()
            for x in range(256, upper_start - 1, -1):
                if check_decreasing_add(S, x):
                    S.append(x)
                    S_set.add(x)
        for x in range(upper_start - 1, 0, -1):
            if time.time() > deadline - 5.0:
                break
            if full_check_add(S_set, x):
                S.append(x)
                S_set.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            
    # Strategy 4: Randomized orderings
    for trial in range(30):
        if time.time() > deadline - 5.0:
            break
        S = []
        S_set = set()
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 5.0:
                break
            if full_check_add(S_set, x):
                S.append(x)
                S_set.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            
    # Strategy 5: Weighted random
    for trial in range(20):
        if time.time() > deadline - 5.0:
            break
        S = []
        S_set = set()
        remaining = list(range(1, 257))
        remaining_weights = [float(x) ** 2 for x in remaining]
        while remaining:
            if time.time() > deadline - 5.0:
                break
            total_w = sum(remaining_weights)
            r = rng.random() * total_w
            cum = 0
            for i, w in enumerate(remaining_weights):
                cum += w
                if r <= cum:
                    x = remaining[i]
                    if full_check_add(S_set, x):
                        S.append(x)
                        S_set.add(x)
                    remaining.pop(i)
                    remaining_weights.pop(i)
                    break
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            
    # Strategy 6: Local Search using precomputed can_add
    # Use the best set found so far
    S = list(best_set)
    S_set = set(S)
    
    # Repeat local search until no improvement or time out
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        can_add = precompute_can_add(S)
        
        # Try to add any valid x
        added = False
        for x in range(1, n + 1):
            if x not in S_set and can_add[x]:
                S.append(x)
                S_set.add(x)
                improved = True
                added = True
                break # Recompute after each addition
        
        if not added:
            # Try 1-swap: remove y, add x
            # This is expensive. Only do if time permits.
            if time.time() < deadline - 5.0:
                for y in list(S_set):
                    S_trial = S_set - {y}
                    can_add_trial = precompute_can_add(S_trial)
                    for x in range(1, n + 1):
                        if x not in S_trial and can_add_trial[x]:
                            # Swap
                            S_set.remove(y)
                            S_set.add(x)
                            S = list(S_set)
                            improved = True
                            break
                    if improved:
                        break
                        
    if len(S) > best_size:
        best_size = len(S)
        best_set = list(S)
        
    # Final validation
    if not full_validate(best_set):
        S = []
        for x in range(256, 0, -1):
            if check_decreasing_add(S, x):
                S.append(x)
        best_set = S
        
    return best_set
