import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    n = 4096
    max_sum = 3 * n  # Maximum possible triple sum

    best_set = []
    best_len = 0

    def is_valid_set(S):
        """Check if set S has all unique unordered triple sums."""
        S = list(S)
        k = len(S)
        used = bytearray(max_sum + 1)
        for i in range(k):
            for j in range(i, k):
                for l in range(j, k):
                    s = S[i] + S[j] + S[l]
                    if used[s]:
                        return False
                    used[s] = 1
        return True

    def build_greedy_random(seed_val):
        """Build a set using randomized greedy approach."""
        random.seed(seed_val)
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S = []
        used = bytearray(max_sum + 1)
        
        for x in candidates:
            if time.time() > deadline:
                break
            
            # Check if x can be added
            # New sums: 3x, 2x+y for y in S, x+y+z for y<=z in S
            valid = True
            
            # Check 3x
            s = 3 * x
            if used[s]:
                valid = False
            
            if valid:
                # Check 2x + y for each y in S
                for y in S:
                    s = 2 * x + y
                    if used[s]:
                        valid = False
                        break
            
            if valid:
                # Check x + y + z for y <= z in S
                for i in range(len(S)):
                    for j in range(i, len(S)):
                        s = x + S[i] + S[j]
                        if used[s]:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Add x and update used
                S.append(x)
                used[3 * x] = 1
                for y in S[:-1]:
                    used[2 * x + y] = 1
                for i in range(len(S) - 1):
                    for j in range(i, len(S) - 1):
                        used[x + S[i] + S[j]] = 1
        
        return S

    def backtracking_mrv(S, used, target_len):
        """Backtracking search with MRV heuristic to find larger sets."""
        S = list(S)
        used = bytearray(used)
        
        def can_add_fast(x):
            """Fast check if x can be added using precomputed used array."""
            if used[3 * x]:
                return False
            for y in S:
                if used[2 * x + y]:
                    return False
            for i in range(len(S)):
                for j in range(i, len(S)):
                    if used[x + S[i] + S[j]]:
                        return False
            return True

        def add_element(x):
            """Add element x to S and update used."""
            S.append(x)
            used[3 * x] = 1
            for y in S[:-1]:
                used[2 * x + y] = 1
            for i in range(len(S) - 1):
                for j in range(i, len(S) - 1):
                    used[x + S[i] + S[j]] = 1

        def remove_element(x):
            """Remove element x from S and update used."""
            S.remove(x)
            used[3 * x] = 0
            for y in S:
                used[2 * x + y] = 0
            for i in range(len(S)):
                for j in range(i, len(S)):
                    used[x + S[i] + S[j]] = 0

        # Try to extend S to target_len using backtracking
        if len(S) >= target_len:
            return S
        
        # Get candidates in increasing order (smallest first tends to work well)
        candidates = []
        for x in range(1, n + 1):
            if x not in S and can_add_fast(x):
                candidates.append(x)
        
        if not candidates:
            return S
        
        # Try each candidate
        for x in candidates:
            if time.time() > deadline:
                break
            
            old_len = len(S)
            add_element(x)
            
            result = backtracking_mrv(S, used, target_len)
            if len(result) > old_len:
                return result
            
            remove_element(x)
        
        return S

    # Strategy 1: Power of 3 base
    power3_set = [3**i for i in range(8) if 3**i <= n]
    if is_valid_set(power3_set):
        best_set = list(power3_set)
        best_len = len(power3_set)

    # Strategy 2: Greedy smallest-first
    S_greedy = []
    used_greedy = bytearray(max_sum + 1)
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        valid = True
        s = 3 * x
        if used_greedy[s]:
            valid = False
        if valid:
            for y in S_greedy:
                s = 2 * x + y
                if used_greedy[s]:
                    valid = False
                    break
        if valid:
            for i in range(len(S_greedy)):
                for j in range(i, len(S_greedy)):
                    s = x + S_greedy[i] + S_greedy[j]
                    if used_greedy[s]:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            S_greedy.append(x)
            used_greedy[3 * x] = 1
            for y in S_greedy[:-1]:
                used_greedy[2 * x + y] = 1
            for i in range(len(S_greedy) - 1):
                for j in range(i, len(S_greedy) - 1):
                    used_greedy[x + S_greedy[i] + S_greedy[j]] = 1

    if len(S_greedy) > best_len:
        best_set = S_greedy
        best_len = len(S_greedy)

    # Strategy 3: Multiple randomized greedy trials
    num_trials = 150
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        S = build_greedy_random(trial)
        if len(S) > best_len:
            best_set = S
            best_len = len(S)
            
            # Try to improve with backtracking MRV
            used = bytearray(max_sum + 1)
            for i in range(len(S)):
                for j in range(i, len(S)):
                    for l in range(j, len(S)):
                        used[S[i] + S[j] + S[l]] = 1
            
            # Try to extend by 1, 2, 3 elements
            for target in [len(S) + 1, len(S) + 2, len(S) + 3]:
                if time.time() > deadline:
                    break
                S_ext = backtracking_mrv(S, used, target)
                if len(S_ext) > best_len:
                    best_set = S_ext
                    best_len = len(S_ext)
                    if best_len >= 80:
                        break
            
            if best_len >= 80:
                break

    # Strategy 4: More aggressive local search from current best
    if best_len < 80 and time.time() < deadline - 15.0:
        used = bytearray(max_sum + 1)
        for i in range(len(best_set)):
            for j in range(i, len(best_set)):
                for l in range(j, len(best_set)):
                    used[best_set[i] + best_set[j] + best_set[l]] = 1
        
        # Try to extend current best
        for target in [best_len + 1, best_len + 2, best_len + 3, best_len + 5]:
            if time.time() > deadline:
                break
            S_ext = backtracking_mrv(best_set, used, target)
            if len(S_ext) > best_len:
                best_set = S_ext
                best_len = len(S_ext)
                if best_len >= 80:
                    break

    # Final validation
    if best_set and not is_valid_set(best_set):
        best_set = [3**i for i in range(8) if 3**i <= n]

    return best_set
