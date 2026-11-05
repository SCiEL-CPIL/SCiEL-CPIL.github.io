import time
import random

def _valid_mean_free(selected):
    """Full verification: no element is mean of any subset of >=2 others."""
    n = len(selected)
    if n < 3:
        return True
    selected = list(set(selected))
    n = len(selected)
    if n < 3:
        return True
    for i, a in enumerate(selected):
        others = [x for j, x in enumerate(selected) if j != i]
        m = len(others)
        if m < 2:
            continue
        max_sum = sum(others)
        dp = [0] * (m + 1)
        dp[0] = 1
        for x in others:
            for k in range(m, 0, -1):
                if dp[k - 1]:
                    dp[k] |= (dp[k - 1] << x)
        for k in range(2, m + 1):
            target = k * a
            if target > max_sum:
                break
            if dp[k] & (1 << target):
                return False
    return True


def _build_dp(lst):
    n = len(lst)
    dp = [0] * (n + 1)
    dp[0] = 1
    for x in lst:
        for k in range(n, 0, -1):
            if dp[k - 1]:
                dp[k] |= (dp[k - 1] << x)
    return dp


def _is_valid_set(s, s_set):
    """Check if set s is mean-free. Uses pair-sum check + DP for larger subsets."""
    n = len(s)
    if n < 3:
        return True
    # Check 2-element means: a = (b+c)/2  <=>  2a = b+c
    # For each a, check if 2a is a sum of two distinct other elements
    for i, a in enumerate(s):
        target = 2 * a
        # Check all pairs of others
        for j in range(n):
            if j == i:
                continue
            for k in range(j + 1, n):
                if k == i:
                    continue
                if s[j] + s[k] == target:
                    return False
    # Check subsets of size >= 3
    for i, a in enumerate(s):
        others = [x for j, x in enumerate(s) if j != i]
        m = len(others)
        if m < 3:
            continue
        max_sum = sum(others)
        dp = [0] * (m + 1)
        dp[0] = 1
        for x in others:
            for k in range(m, 0, -1):
                if dp[k - 1]:
                    dp[k] |= (dp[k - 1] << x)
        for k in range(3, m + 1):
            target = k * a
            if target > max_sum:
                break
            if dp[k] & (1 << target):
                return False
    return True


def _can_add(s, s_set, x):
    """Check if x can be added to valid set s without violating constraints."""
    n = len(s)
    if n < 2:
        return True
    
    # Check 1: x is not mean of any subset (size>=2) of s
    # 2-element: x = (a+b)/2 => 2x = a+b
    target_2 = 2 * x
    for i in range(n):
        for j in range(i + 1, n):
            if s[i] + s[j] == target_2:
                return False
    # Larger subsets of s: x = sum(T)/|T|, |T|>=3
    if n >= 3:
        max_sum = sum(s)
        dp = [0] * (n + 1)
        dp[0] = 1
        for val in s:
            for k in range(n, 0, -1):
                if dp[k - 1]:
                    dp[k] |= (dp[k - 1] << val)
        for k in range(3, n + 1):
            target = k * x
            if target > max_sum:
                break
            if dp[k] & (1 << target):
                return False
    
    # Check 2: no existing a becomes mean of subset of (s \ {a}) ∪ {x}
    for i, a in enumerate(s):
        # Subsets including x: T ⊆ s\{a}, |T|>=1, sum(T) + x = (|T|+1)*a
        # => sum(T) = (|T|+1)*a - x
        others = [s[j] for j in range(n) if j != i]
        m = len(others)
        if m < 1:
            continue
        max_sum_others = sum(others)
        dp_others = [0] * (m + 1)
        dp_others[0] = 1
        for val in others:
            for k in range(m, 0, -1):
                if dp_others[k - 1]:
                    dp_others[k] |= (dp_others[k - 1] << val)
        for t in range(1, m + 1):
            k = t + 1
            target = k * a - x
            if target < 0:
                continue
            if target > max_sum_others:
                break
            if dp_others[t] & (1 << target):
                return False
        # Subsets not including x: already valid since s was valid
    return True


def _initial_set(deadline):
    """Build initial set using greedy with structured candidates."""
    c23 = set()
    i = 0
    while (1 << i) <= 512:
        val_i = 1 << i
        j = 0
        while val_i * (3 ** j) <= 512:
            c23.add(val_i * (3 ** j))
            j += 1
        i += 1
    c23_sorted = sorted(c23)
    
    s = []
    s_set = set()
    for x in c23_sorted:
        if time.time() > deadline:
            break
        if _can_add(s, s_set, x):
            s.append(x)
            s_set.add(x)
    return s, s_set


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Phase 1: Initial greedy construction
    s, s_set = _initial_set(deadline)
    best_s = s[:]
    best_len = len(s)
    
    # Phase 2: Simulated annealing
    if time.time() < deadline:
        # Prepare candidate pool
        all_candidates = list(range(1, 513))
        c23 = set()
        i = 0
        while (1 << i) <= 512:
            val_i = 1 << i
            j = 0
            while val_i * (3 ** j) <= 512:
                c23.add(val_i * (3 ** j))
                j += 1
            i += 1
        
        # Temperature schedule
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        
        s = best_s[:]
        s_set = set(s)
        current_len = len(s)
        
        sa_iter = 0
        no_improve = 0
        max_no_improve = 500
        
        while time.time() < deadline and T > T_min:
            sa_iter += 1
            if sa_iter % 100 == 0 and time.time() > deadline:
                break
            
            # Choose random move: swap, remove, or add
            move_type = random.random()
            
            new_s = s[:]
            new_s_set = set(s)
            
            if move_type < 0.5 and len(s) > 0:
                # Swap: remove one, try to add another
                idx = random.randint(0, len(s) - 1)
                removed = s[idx]
                new_s.pop(idx)
                new_s_set.discard(removed)
                
                # Try to add a random candidate
                candidates = [x for x in all_candidates if x not in new_s_set]
                if candidates:
                    # Weight towards structured candidates
                    weighted = []
                    for x in candidates:
                        w = 3 if x in c23 else 1
                        weighted.extend([x] * w)
                    new_x = random.choice(weighted) if weighted else None
                    if new_x is not None:
                        if _can_add(new_s, new_s_set, new_x):
                            new_s.append(new_x)
                            new_s_set.add(new_x)
                        else:
                            # Revert
                            new_s = s[:]
                            new_s_set = set(s)
                            T *= cooling
                            continue
            elif move_type < 0.75 and len(s) > 2:
                # Remove one element
                idx = random.randint(0, len(s) - 1)
                removed = s[idx]
                new_s.pop(idx)
                new_s_set.discard(removed)
            elif move_type < 0.95:
                # Add one element
                candidates = [x for x in all_candidates if x not in s_set]
                if candidates:
                    weighted = []
                    for x in candidates:
                        w = 3 if x in c23 else 1
                        weighted.extend([x] * w)
                    new_x = random.choice(weighted) if weighted else None
                    if new_x is not None:
                        if _can_add(s, s_set, new_x):
                            new_s.append(new_x)
                            new_s_set.add(new_x)
                        else:
                            T *= cooling
                            continue
                    else:
                        T *= cooling
                        continue
                else:
                    T *= cooling
                    continue
            else:
                # Do nothing (rare)
                T *= cooling
                continue
            
            new_len = len(new_s)
            delta = new_len - current_len
            
            if delta >= 0:
                # Accept improvement
                s = new_s
                s_set = new_s_set
                current_len = new_len
                no_improve = 0
                if current_len > best_len:
                    best_len = current_len
                    best_s = s[:]
            else:
                # Accept with probability exp(delta/T)
                prob = min(1.0, __import__('math').exp(delta / T))
                if random.random() < prob:
                    s = new_s
                    s_set = new_s_set
                    current_len = new_len
                    no_improve += 1
                else:
                    no_improve += 1
            
            T *= cooling
            
            if no_improve > max_no_improve:
                # Restart with fresh greedy
                s, s_set = _initial_set(deadline)
                current_len = len(s)
                T = 5.0
                no_improve = 0
        
        # Final expansion on best
        if time.time() < deadline:
            best_s_set = set(best_s)
            remaining = [x for x in all_candidates if x not in best_s_set]
            remaining.sort(key=lambda x: (0 if x in c23 else 1, x))
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for x in remaining[:]:
                    if time.time() > deadline:
                        break
                    if x in best_s_set:
                        continue
                    if _can_add(best_s, best_s_set, x):
                        best_s.append(x)
                        best_s_set.add(x)
                        remaining.remove(x)
                        improved = True
    
    # Final verification
    if best_s and _valid_mean_free(best_s):
        return best_s
    else:
        # Fallback: return empty or minimal valid set
        return [1, 2, 4] if _valid_mean_free([1, 2, 4]) else []


if __name__ == "__main__":
    s = construct()
    print(f"Found set of size {len(s)}")
    print(s)
    assert _valid_mean_free(s), "Invalid set found!"
    assert len(set(s)) == len(s), "Duplicates found!"
    assert all(1 <= x <= 512 for x in s), "Out of range!"
