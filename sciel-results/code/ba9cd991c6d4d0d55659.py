import time
import random

def construct():
    deadline = time.time() + 55.0
    n_max = 512
    target_size = 48

    def is_valid_set(S):
        """
        Checks if S is a valid set using full DP reconstruction.
        Returns True if valid, False otherwise.
        """
        n = len(S)
        if n < 3:
            return True
        
        # 1. For each element a in S, check if a is the mean of a subset of S \ {a}
        for i, a in enumerate(S):
            others = [x for x in S if x != a]
            m = len(others)
            if m < 2:
                continue
            
            # DP for others
            # dp[k] is bitset of sums of k elements from others
            dp = [0] * (m + 1)
            dp[0] = 1
            for val in others:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << val)
            
            # Check if a is mean of subset of size k >= 2
            # sum(T) = k * a
            for k in range(2, m + 1):
                target = k * a
                if target > sum(others):
                    break
                if (dp[k] >> target) & 1:
                    return False
        return True

    def find_valid_subset(size_target, limit, order=None):
        """
        Find a valid subset of size_target from [1, limit].
        """
        if order is None:
            order = list(range(1, limit + 1))
        
        S = []
        S_set = set()
        
        # dp_all[k] : bitset of sums of k elements from S
        dp_all = [1]
        
        for x in order:
            if len(S) >= size_target:
                break
            if x in S_set:
                continue
            
            # Check 1: Is x the mean of a subset of S?
            valid = True
            for k in range(2, len(S) + 1):
                target = k * x
                if k < len(dp_all):
                    if (dp_all[k] >> target) & 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Check 2: Does adding x make any existing element a_i the mean of a subset of S U {x} \ {a_i}?
            # We need to check if there is a subset T of S \ {a_i} such that
            # {x} U T has mean a_i.
            # sum(T) + x = (|T| + 1) * a_i
            # sum(T) = (|T| + 1) * a_i - x
            # Let j = |T|. We need to check if bit ((j+1)*a_i - x) is set in dp_leave[i][j].
            
            # For efficiency, we only maintain dp_all. 
            # To check condition 2, we can use the full is_valid_set on S+[x] if S is small.
            # Since size_target is small (e.g., 12), is_valid_set is fast.
            
            S_test = S + [x]
            if is_valid_set(S_test):
                S = S_test
                S_set.add(x)
                
                # Update DP_all
                old_dp_all = dp_all
                new_dp_all = [0] * (len(S) + 1)
                new_dp_all[0] = 1
                for k in range(len(S) - 1, -1, -1):
                    if old_dp_all[k]:
                        new_dp_all[k] |= old_dp_all[k]
                        new_dp_all[k+1] |= (old_dp_all[k] << x)
                dp_all = new_dp_all
        
        return S

    def local_search(S):
        """
        Perform local search swaps to improve the set.
        """
        S = list(S)
        S_set = set(S)
        available = [x for x in range(1, n_max + 1) if x not in S_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng = random.Random()
            rng.shuffle(available)
            
            # Try to add if size < target
            if len(S) < target_size:
                for x in available:
                    if time.time() > deadline:
                        break
                    S_test = S + [x]
                    if is_valid_set(S_test):
                        S = S_test
                        S_set.add(x)
                        available.remove(x)
                        improved = True
                        break
            
            if len(S) >= target_size:
                break
            
            # If stuck, try swaps
            if not improved:
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    old = S[i]
                    S_without_old = [x for x in S if x != old]
                    
                    for x in available:
                        if time.time() > deadline:
                            break
                        S_test = S_without_old + [x]
                        if is_valid_set(S_test):
                            S = S_test
                            S_set.discard(old)
                            S_set.add(x)
                            available.remove(x)
                            available.append(old)
                            improved = True
                            break
                    if improved:
                        break
        
        return S

    best_set = []
    best_size = 0

    # Strategy 1: Block Decomposition
    # Try to find a base set of size K in [1, L] and replicate it.
    # 512 / 4 = 128. Target 48 / 4 = 12.
    # Try K=12, L=128.
    
    base_candidates = []
    # Try a few random orders for the base set to find diverse candidates
    for i in range(10):
        if time.time() > deadline:
            break
        rng = random.Random(i)
        order = list(range(1, 129))
        rng.shuffle(order)
        # Add some deterministic orders
        if i == 0:
            order = list(range(1, 129))
        elif i == 1:
            order = list(range(128, 0, -1))
            
        base = find_valid_subset(12, 128, order)
        if len(base) == 12:
            base_candidates.append(base)
            if len(base_candidates) >= 3:
                break

    if base_candidates:
        # Try combinations of base sets (or just the first one)
        for base in base_candidates:
            if time.time() > deadline:
                break
            S = []
            S.extend(base)
            S.extend([x + 128 for x in base])
            S.extend([x + 256 for x in base])
            S.extend([x + 384 for x in base])
            
            if len(S) == 48 and is_valid_set(S):
                return S
            elif len(S) > best_size:
                best_size = len(S)
                best_set = S

    # Strategy 2: Global Greedy (from current program)
    # If block decomposition didn't yield 48, or if we want to try direct greedy.
    
    def greedy_build_fast(order):
        S = []
        S_set = set()
        dp_all = [1]
        
        for x in order:
            if len(S) >= target_size:
                break
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            
            valid = True
            for k in range(2, len(S) + 1):
                target = k * x
                if k < len(dp_all):
                    if (dp_all[k] >> target) & 1:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Full check for safety, as dp_all check only covers "x is mean of S"
            # It does NOT cover "existing a is mean of S\{a} U {x}"
            # So we must use is_valid_set or the more complex dp_leave check.
            # Given the small size, is_valid_set is acceptable for the final check,
            # but for speed in the loop, we can use a heuristic or just accept the risk
            # and verify at the end? No, we need to maintain validity.
            
            # Let's use the dp_leave logic from the original program for speed
            # But maintaining dp_leave is complex.
            # Alternative: Just use is_valid_set(S + [x]).
            # For N=48, is_valid_set is O(N^2 * MaxSum/WordSize).
            # 48^2 * 24576 / 64 ~ 180,000 ops. Very fast.
            
            if is_valid_set(S + [x]):
                S.append(x)
                S_set.add(x)
                
                old_dp_all = dp_all
                new_dp_all = [0] * (len(S) + 1)
                new_dp_all[0] = 1
                for k in range(len(S) - 1, -1, -1):
                    if old_dp_all[k]:
                        new_dp_all[k] |= old_dp_all[k]
                        new_dp_all[k+1] |= (old_dp_all[k] << x)
                dp_all = new_dp_all
        
        return S

    # Try deterministic orders
    S = greedy_build_fast(range(n_max, 0, -1))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S
    
    S = greedy_build_fast(range(1, n_max + 1))
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # Try random orders
    rng = random.Random(42)
    for i in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, n_max + 1))
        rng.shuffle(order)
        S = greedy_build_fast(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # Strategy 3: Local Search on best candidate
    if best_size < target_size and time.time() < deadline:
        S = local_search(best_set)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return best_set

if __name__ == "__main__":
    s = construct()
    print(len(s))
    print(s[:10])
