"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random
import numpy as np

N = 256
MAX_SUM = 2 * N  # 512

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 27.0  # Safety margin before 30s limit
    
    def get_sums_array(S):
        """Compute the sum boolean array for set S using vectorized ops."""
        sums = np.zeros(MAX_SUM + 1, dtype=bool)
        n = len(S)
        if n == 0:
            return sums
        S_arr = np.array(S, dtype=np.int32)
        for i in range(n):
            sums[S_arr[i:] + S_arr[i]] = True
        return sums

    def can_add(x, sums_arr, S):
        """Check if x can be added to S given sums as boolean array."""
        if sums_arr[2 * x]:
            return False
        if len(S) > 0:
            if np.any(sums_arr[x + np.array(S, dtype=np.int32)]):
                return False
        return True

    def is_maximal(S, sums_arr=None):
        """Check if S is inclusion-maximal using vectorized checks."""
        if not S:
            return False
        if sums_arr is None:
            sums_arr = get_sums_array(S)
        S_set = set(S)
        missing = [x for x in range(1, N + 1) if x not in S_set]
        if not missing:
            return True
        missing_arr = np.array(missing, dtype=np.int32)
        # Check 2x for all missing at once
        two_x_blocked = sums_arr[2 * missing_arr]
        # For those not blocked by 2x, check x+a
        candidates = missing_arr[~two_x_blocked]
        if len(candidates) == 0:
            return True
        S_arr = np.array(S, dtype=np.int32)
        # Vectorized check: for each candidate x, check if any x+a is in sums
        # sums_arr[x+a] for all a in S
        # Shape: (len(candidates), len(S))
        pair_sums = candidates[:, np.newaxis] + S_arr[np.newaxis, :]
        if np.any(sums_arr[pair_sums]):
            return False
        return True

    def greedy_construct(order):
        """Greedy construction with given order using vectorized can_add."""
        S = []
        sums = np.zeros(MAX_SUM + 1, dtype=bool)
        for x in order:
            if can_add(x, sums, S):
                S.append(x)
                sums[2 * x] = True
                if S[:-1]:
                    sums[x + np.array(S[:-1], dtype=np.int32)] = True
        return S

    def prune_aggressive(S):
        """Aggressively prune: try removing each element, check maximality."""
        S = list(S)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                return S
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline:
                    return S
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                new_sums = get_sums_array(new_S)
                if is_maximal(new_S, new_sums):
                    S = new_S
                    improved = True
                    break
        return S

    def local_search(S, max_iters=100, rng=None):
        """Local search: try removing elements to reduce size while maintaining maximality."""
        if rng is None:
            rng = random.Random()
        S = list(S)
        best = S
        best_size = len(S)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if len(S) <= 1:
                break
            i = rng.randint(0, len(S) - 1)
            new_S = S[:i] + S[i+1:]
            if new_S:
                new_sums = get_sums_array(new_S)
                if is_maximal(new_S, new_sums):
                    S = new_S
                    if len(S) < best_size:
                        best_size = len(S)
                        best = list(S)
        
        return best

    def local_search_swap(S, max_iters=50, rng=None):
        """Local search with swaps: remove one element, add another to maintain size or reduce."""
        if rng is None:
            rng = random.Random()
        S = list(S)
        best = S
        best_size = len(S)
        all_elements = list(range(1, N + 1))
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            if len(S) <= 1:
                break
            # Try removing a random element
            i = rng.randint(0, len(S) - 1)
            new_S = S[:i] + S[i+1:]
            if new_S:
                new_sums = get_sums_array(new_S)
                if is_maximal(new_S, new_sums):
                    S = new_S
                    if len(S) < best_size:
                        best_size = len(S)
                        best = list(S)
                    continue
            # Try swap: remove one, add another
            if len(S) >= 2:
                i = rng.randint(0, len(S) - 1)
                S_no_i = S[:i] + S[i+1:]
                if not S_no_i:
                    continue
                # Try adding a random element
                x = rng.randint(1, N)
                if x in S_no_i:
                    continue
                S_no_i_arr = np.array(S_no_i, dtype=np.int32)
                if can_add(x, get_sums_array(S_no_i), S_no_i):
                    candidate = S_no_i + [x]
                    cand_sums = get_sums_array(candidate)
                    if is_maximal(candidate, cand_sums):
                        if len(candidate) < len(S):
                            S = candidate
                            if len(S) < best_size:
                                best_size = len(S)
                                best = list(S)
        
        return best

    # Phase 1: Multiple deterministic orders
    orders = [
        list(range(N, 0, -1)),
        list(range(1, N + 1)),
        list(range(N // 2, 0, -1)) + list(range(N // 2 + 1, N + 1)),
        list(range(1, N + 1, 2)) + list(range(2, N + 1, 2)),
        list(range(2, N + 1, 2)) + list(range(1, N + 1, 2)),
        [x for x in range(1, N + 1) if x % 3 != 0] + [x for x in range(1, N + 1) if x % 3 == 0],
        [2**i for i in range(1, 9) if 2**i <= N] + [x for x in range(1, N + 1) if x not in [2**i for i in range(1, 9) if 2**i <= N]],
    ]
    
    best_set = None
    best_size = N + 1
    
    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_construct(order)
        S = prune_aggressive(S)
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
    
    # Phase 2: Randomized restarts with local search - many more iterations now
    rng = random.Random(42)
    all_elements = list(range(1, N + 1))
    
    while time.time() < deadline - 3.0:
        r = rng.random()
        if r < 0.2:
            order = list(all_elements)
            rng.shuffle(order)
        elif r < 0.35:
            order = sorted(all_elements, key=lambda x: -x + rng.gauss(0, 80))
        elif r < 0.5:
            order = sorted(all_elements, key=lambda x: x + rng.gauss(0, 80))
        elif r < 0.65:
            # Favor primes and numbers with few divisors
            def is_prime(n):
                if n < 2:
                    return False
                for i in range(2, int(n**0.5) + 1):
                    if n % i == 0:
                        return False
                return True
            primes = [x for x in range(1, N + 1) if is_prime(x)]
            non_primes = [x for x in range(1, N + 1) if not is_prime(x)]
            order = primes + non_primes
            rng.shuffle(order)
        elif r < 0.8:
            # Fibonacci-like spacing
            fibs = [1, 2]
            while fibs[-1] + fibs[-2] <= N:
                fibs.append(fibs[-1] + fibs[-2])
            order = fibs + [x for x in range(1, N + 1) if x not in fibs]
            rng.shuffle(order)
        else:
            order = list(all_elements)
            rng.shuffle(order)
            if rng.random() < 0.3:
                order.reverse()
        
        S = greedy_construct(order)
        S = prune_aggressive(S)
        
        if len(S) < best_size:
            S_ls = local_search(S, max_iters=30, rng=rng)
            if len(S_ls) < best_size:
                best_size = len(S_ls)
                best_set = S_ls
            # Also try swap-based local search
            S_sw = local_search_swap(S_ls, max_iters=20, rng=rng)
            if len(S_sw) < best_size:
                best_size = len(S_sw)
                best_set = S_sw
        elif len(S) == best_size:
            S_ls = local_search(S, max_iters=15, rng=rng)
            if len(S_ls) < best_size:
                best_size = len(S_ls)
                best_set = S_ls
    
    # Final: aggressive local search on best found
    if best_set is None:
        best_set = greedy_construct(list(range(N, 0, -1)))
        best_set = prune_aggressive(best_set)
    
    if time.time() < deadline - 1.0:
        final_rng = random.Random(123)
        final_ls = local_search(best_set, max_iters=100, rng=final_rng)
        if len(final_ls) < len(best_set):
            best_set = final_ls
        
        if time.time() < deadline - 0.5:
            final_sw = local_search_swap(best_set, max_iters=50, rng=final_rng)
            if len(final_sw) < len(best_set):
                best_set = final_sw
    
    # Final verification
    if not is_maximal(best_set):
        best_set = greedy_construct(list(range(N, 0, -1)))
        best_set = prune_aggressive(best_set)
    
    return best_set

# EVOLVE-BLOCK-END
