import time
import random

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 90s, but task says 60s limit in text?
    # "Each evaluation must finish within 60 s."
    # "The evaluator KILLS the program after 90s wall-clock."
    # Let's aim for 50s to be safe.
    deadline = 50.0
    
    MAX_N = 512
    TARGET_SIZE = 48
    MAX_SUM = TARGET_SIZE * MAX_N  # 24576

    def try_greedy(order):
        """
        Attempts to build a valid set by adding elements in the given order.
        Returns the set found.
        """
        selected = []
        n = 0
        # dp[k] is a bitmask where bit s is set if sum s is achievable with exactly k elements from 'selected'
        # dp[0] = 1 (bit 0 set)
        dp = [0] * (n + 1)
        dp[0] = 1
        
        # dp_excl[a] is a list of bitmasks for sums achievable with k elements from 'selected' excluding 'a'
        dp_excl = {}

        for x in order:
            if time.time() - start_time > deadline:
                break
            if n >= TARGET_SIZE:
                break

            # Check 1: Is x the mean of any subset of 'selected' of size k >= 2?
            valid = True
            for k in range(2, n + 1):
                target_sum = k * x
                if target_sum > MAX_SUM:
                    break
                if (dp[k] >> target_sum) & 1:
                    valid = False
                    break

            if not valid:
                continue

            # Check 2: For each a in selected, is a the mean of a subset of (selected \ {a}) U {x} of size >= 2?
            # A subset of (selected \ {a}) U {x} of size k >= 2 that includes x and sums to k*a
            # is equivalent to a subset of (selected \ {a}) of size k-1 >= 1 summing to k*a - x.
            # Let j = k-1. Then j >= 1 and sum = (j+1)*a - x.
            for a in selected:
                if not valid:
                    break
                # We need to check subsets of size j >= 1 from selected\{a}
                # Max j is n-1
                for j in range(1, n):
                    target_sum = (j + 1) * a - x
                    if target_sum < 0 or target_sum > MAX_SUM:
                        continue
                    if (dp_excl[a][j] >> target_sum) & 1:
                        valid = False
                        break

            if not valid:
                continue

            # x is valid. Add it to selected.
            selected.append(x)
            
            # Update dp
            new_dp = [0] * (n + 2)
            new_dp[0] = 1
            for k in range(1, n + 2):
                val = dp[k] if k <= n else 0
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (dp[k - 1] << x)
                new_dp[k] = val
            dp = new_dp

            # Update dp_excl for existing elements
            new_dp_excl = {}
            for a in selected[:-1]:  # All except x
                old_dp_w = dp_excl[a]
                new_list = [0] * (n + 2)
                new_list[0] = 1
                for k in range(1, n + 2):
                    val = old_dp_w[k] if k <= n - 1 else 0
                    if k - 1 >= 0 and k - 1 <= n - 1:
                        val |= (old_dp_w[k - 1] << x)
                    new_list[k] = val
                new_dp_excl[a] = new_list

            # For new element x, dp_excl[x] is the state of dp BEFORE adding x
            new_dp_excl[x] = dp[:]
            dp_excl = new_dp_excl
            n += 1

        return selected

    best_set = []
    best_len = 0

    # Strategy 1: Ascending
    res = try_greedy(range(1, MAX_N + 1))
    if len(res) > best_len:
        best_set = res
        best_len = len(res)

    # Strategy 2: Descending
    if time.time() - start_time < deadline - 5:
        res = try_greedy(range(MAX_N, 0, -1))
        if len(res) > best_len:
            best_set = res
            best_len = len(res)

    # Strategy 3: Randomized Restarts with Modular-inspired Ordering
    # We use multiple random seeds.
    # Heuristic: Sort by hash or random shuffle.
    attempts = 0
    while time.time() - start_time < deadline - 2:
        attempts += 1
        seed = attempts * 104729 + 7919
        random.seed(seed)
        
        candidates = list(range(1, MAX_N + 1))
        
        # Modular-inspired ordering: 
        # Sort by (x mod p) for some small primes to cluster "similar" numbers?
        # Or just random shuffle which is a valid "modular" distribution if we assume uniform.
        # The knowledge suggests "Modular-inspired ordering heuristic".
        # Let's try sorting by x % 7, then x % 5, etc., to create a specific structure.
        # Or, filter out numbers that are "bad" modulo small primes relative to each other?
        # For now, random shuffle is the safest baseline. 
        # To incorporate the hint, let's try a few different sorting keys.
        
        if attempts % 3 == 0:
            random.shuffle(candidates)
        elif attempts % 3 == 1:
            # Sort by random hash
            candidates.sort(key=lambda x: hash((x, seed)))
        else:
            # Sort by modulo small prime to create clusters
            p = 7 + (attempts % 5)
            candidates.sort(key=lambda x: (x % p, x))

        res = try_greedy(candidates)
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= TARGET_SIZE:
                break

    return best_set
