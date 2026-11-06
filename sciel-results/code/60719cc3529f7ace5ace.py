import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = 4096
    max_sum = 3 * n
    
    best_set = []
    best_len = 0
    
    # Initial guess: Power of 3s
    power3_set = [3**i for i in range(8) if 3**i <= n]
    best_set = list(power3_set)
    best_len = len(power3_set)

    def is_valid_set(S):
        """Check if set S has all unique unordered triple sums."""
        if not S:
            return True
        sums_used = bytearray(max_sum + 1)
        S = list(S)
        k = len(S)
        for i in range(k):
            si = S[i]
            for j in range(i, k):
                sj = S[j]
                for l in range(j, k):
                    sl = S[l]
                    s = si + sj + sl
                    if sums_used[s]:
                        return False
                    sums_used[s] = 1
        return True

    def get_new_sums(x, S):
        """
        Generate all new sums involving x when adding to S.
        Returns a list of sums.
        """
        new_sums = []
        # Triple (x, x, x)
        new_sums.append(3 * x)
        # Triples (x, x, y) where y in S
        for y in S:
            new_sums.append(2 * x + y)
        # Triples (x, y, z) where y, z in S, y <= z
        for i in range(len(S)):
            yi = S[i]
            for j in range(i, len(S)):
                zj = S[j]
                new_sums.append(x + yi + zj)
        return new_sums

    def can_add(x, S, sums_used):
        """
        Check if x can be added to S given the current sums_used bytearray.
        Returns True if valid, and the list of new sums if valid.
        """
        new_sums = get_new_sums(x, S)
        
        # Check for collisions with existing sums
        for s in new_sums:
            if sums_used[s]:
                return False, None
        
        # Check for internal collisions within new_sums
        if len(set(new_sums)) != len(new_sums):
            return False, None
            
        return True, new_sums

    def greedy_build(ordering, sums_used):
        """
        Build a set using the given ordering of candidates.
        """
        S = []
        for x in ordering:
            valid, new_sums = can_add(x, S, sums_used)
            if valid:
                S.append(x)
                for s in new_sums:
                    sums_used[s] = 1
        return S

    def local_search_improve(S):
        """
        Try to improve S by adding elements or swapping.
        Returns improved set if found, else S.
        """
        if not S:
            return S
        
        S = sorted(S)
        S_set = set(S)
        sums_used = bytearray(max_sum + 1)
        k = len(S)
        for i in range(k):
            si = S[i]
            for j in range(i, k):
                sj = S[j]
                for l in range(j, k):
                    sl = S[l]
                    sums_used[si + sj + sl] = 1
        
        # Try to add any single element
        for x in range(1, n + 1):
            if x in S_set:
                continue
            valid, new_sums = can_add(x, S, sums_used)
            if valid:
                S.append(x)
                S.sort()
                S_set.add(x)
                for s in new_sums:
                    sums_used[s] = 1
                return S
        
        # Try "Remove 1, Add 2"
        for i in range(len(S)):
            removed_val = S[i]
            temp_S = S[:i] + S[i+1:]
            temp_S_set = S_set - {removed_val}
            
            # Recalculate sums for temp_S
            temp_sums_used = bytearray(max_sum + 1)
            tk = len(temp_S)
            for a in range(tk):
                sa = temp_S[a]
                for b in range(a, tk):
                    sb = temp_S[b]
                    for c in range(b, tk):
                        sc = temp_S[c]
                        temp_sums_used[sa + sb + sc] = 1
            
            # Find valid 'a's
            valid_as = []
            for a in range(1, n + 1):
                if a in temp_S_set:
                    continue
                valid, _ = can_add(a, temp_S, temp_sums_used)
                if valid:
                    valid_as.append(a)
                    if len(valid_as) >= 5:
                        break
            
            for a in valid_as:
                # Update temp_S and sums with a
                set_with_a = temp_S + [a]
                set_with_a_set = temp_S_set | {a}
                
                # Update sums
                _, sums_a = can_add(a, temp_S, temp_sums_used)
                for s in sums_a:
                    temp_sums_used[s] = 1
                
                # Try to add b
                for b in range(a + 1, n + 1):
                    if b in set_with_a_set:
                        continue
                    valid, sums_b = can_add(b, set_with_a, temp_sums_used)
                    if valid:
                        new_S = set_with_a + [b]
                        new_S.sort()
                        return new_S
                    # Limit b checks
                    if b > a + 10:
                        break
                        
        return S

    # Strategy 1: Randomized Greedy with multiple seeds
    num_restarts = 100
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for seed in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        random.seed(seed)
        
        # Different ordering strategies
        strategy = seed % 4
        
        if strategy == 0:
            # Smallest first
            candidates = list(range(1, n + 1))
        elif strategy == 1:
            # Largest first
            candidates = list(range(n, 0, -1))
        elif strategy == 2:
            # Random
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
        else:
            # Biased Random: Smaller numbers more likely
            # Use a simple shuffle
            candidates = list(range(1, n + 1))
            random.shuffle(candidates)
            
        sums_used = bytearray(max_sum + 1)
        S = greedy_build(candidates, sums_used)
        
        if len(S) > best_len:
            best_set = S
            best_len = len(S)
            
            # Try to improve immediately
            if time.time() < deadline - 10:
                improved_S = local_search_improve(best_set)
                if len(improved_S) > best_len:
                    best_set = improved_S
                    best_len = len(best_set)

    # Final validation
    if best_set and not is_valid_set(best_set):
        best_set = [3**i for i in range(8) if 3**i <= n]

    return best_set
