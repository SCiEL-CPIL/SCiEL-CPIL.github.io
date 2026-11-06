import math
import time

def construct():
    start = time.time()
    deadline = start + 85  # safety margin under 90s kill

    target_val = math.factorial(12)  # 479001600
    primes = [2, 3, 5, 7, 11]

    def get_vec(n):
        v = [0]*5
        for idx, p in enumerate(primes):
            while n % p == 0:
                v[idx] += 1
                n //= p
        return tuple(v)

    target_vec = get_vec(target_val)

    # Generate all divisors of target_val
    divisors = set()
    i = 1
    while i * i <= target_val:
        if target_val % i == 0:
            divisors.add(i)
            divisors.add(target_val // i)
        i += 1

    # Filter: > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    div_vecs = {d: get_vec(d) for d in valid_divs}

    n_divs = len(valid_divs)

    # ---- Precompute reachability tables ----
    # dp[i][vec] = True iff a nonempty subset of valid_divs[0..i-1] (each used at most once)
    #              has prime-exponent vector exactly `vec`.
    # We build incrementally: dp_i = dp_{i-1} U { vec + d_vec : vec in dp_{i-1}, vec + d_vec <= target_vec }
    #
    # A solution with max factor <= valid_divs[M] exists iff dp[M+1][target_vec] is True.
    # We only need the final layer dp[n_divs] for reconstruction, and the sequence of
    # membership checks dp[i+1][target_vec] for binary search.
    #
    # State space: product of (exp+1) over primes = 12*7*4*3*3 = 12096. Very small.

    # Precompute vec->index and index->vec for fast array access
    dims = [e + 1 for e in target_vec]  # [12,7,4,3,3]
    vec_to_idx = {}
    idx_to_vec = []
    # Mixed-radix encoding
    stride = [1]*5
    for j in range(4, -1, -1):
        stride[j] = stride[j+1] * dims[j] if j+1 < 5 else dims[j]
    # Simpler: stride[4]=1, stride[3]=dims[4], stride[2]=dims[3]*dims[4], etc.
    stride = [1]*5
    for j in range(3, -1, -1):
        stride[j] = stride[j+1] * dims[j+1]
    total_states = dims[0] * dims[1] * dims[2] * dims[3] * dims[4]
    for v0 in range(dims[0]):
        for v1 in range(dims[1]):
            for v2 in range(dims[2]):
                for v3 in range(dims[3]):
                    for v4 in range(dims[4]):
                        vec = (v0, v1, v2, v3, v4)
                        idx = v0*stride[0] + v1*stride[1] + v2*stride[2] + v3*stride[3] + v4*stride[4]
                        vec_to_idx[vec] = idx
                        idx_to_vec.append(vec)

    target_idx = vec_to_idx[target_vec]

    # dp is a list of bytearrays, one per layer. dp[0] = all False (no factors used).
    # dp[i] = reachable using subset of valid_divs[0..i-1].
    # We'll store as list of bytearray for memory efficiency.
    dp_layers = [bytearray(total_states)]  # dp[0], all zeros

    # Also store for reconstruction: for each layer i and each reachable vec,
    #   which divisor (index in valid_divs) was last added to reach it (if via addition).
    #   We store a "parent" array: parent[i][idx] = divisor_index used, or -1 if not reachable via this path.
    #   Actually for reconstruction we need: given dp[i][target_idx] is True, find which divisor d
    #   in valid_divs[0..i-1] was used such that dp[i-1][target_vec - vec(d)] is True.
    #   We can just re-check during reconstruction using the stored dp layers.

    for i in range(n_divs):
        if time.time() > deadline:
            break
        d = valid_divs[i]
        dv = div_vecs[d]
        prev = dp_layers[i]
        curr = bytearray(prev)  # start with "skip divisor i"
        # For each reachable vec in prev, try adding dv
        for vec_idx in range(total_states):
            if not prev[vec_idx]:
                continue
            vec = idx_to_vec[vec_idx]
            new_vec = (vec[0]+dv[0], vec[1]+dv[1], vec[2]+dv[2], vec[3]+dv[3], vec[4]+dv[4])
            if new_vec[0] > target_vec[0] or new_vec[1] > target_vec[1] or \
               new_vec[2] > target_vec[2] or new_vec[3] > target_vec[3] or new_vec[4] > target_vec[4]:
                continue
            new_idx = vec_to_idx[new_vec]
            curr[new_idx] = 1
        dp_layers.append(curr)

    # Check if target is reachable at all
    if not dp_layers[n_divs][target_idx]:
        return [target_val]

    # Binary search on M_idx: smallest M such that dp[M+1][target_idx] is True
    lo = 0
    hi = n_divs - 1
    best_M_idx = n_divs - 1  # fallback

    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        # dp[mid+1] corresponds to using divisors valid_divs[0..mid]
        if dp_layers[mid+1][target_idx]:
            best_M_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1

    # Reconstruct: find a subset of valid_divs[0..best_M_idx] whose vec sums to target_vec
    # Work backwards: at layer i (meaning we've considered divisors 0..i-1), if dp[i][target_idx] is True,
    #   either target was reachable without divisor i-1 (dp[i-1][target_idx] is True),
    #   or it was reached by adding divisor i-1 (dp[i-1][target_vec - vec(valid_divs[i-1])] is True).
    # We want ANY valid subset, so we greedily try to include the largest available divisor first
    # (to keep the set small, though size doesn't matter for the objective).
    # Actually, for reconstruction we just need any valid subset. Let's go forward with DFS using dp as a guide.

    # Simpler reconstruction: DFS with dp-based pruning.
    # State: (pos, cur_vec, count) -> can we complete to target using divisors from pos..best_M_idx?
    # Prune: the remaining vector = target_vec - cur_vec must be in dp[best_M_idx+1 - pos] shifted...
    # Actually, we can use: remaining must be reachable by a subset of valid_divs[pos..best_M_idx].
    # That's equivalent to: remaining is in the "suffix dp" which we haven't precomputed.
    # Instead, let's just do DFS with memoization and the dp table as a necessary condition:
    #   At state (pos, cur_vec), the remaining vec r = target_vec - cur_vec.
    #   We need r to be formable from valid_divs[pos..best_M_idx].
    #   A necessary condition: r must be <= target_vec (component-wise), which is guaranteed.
    #   We can also use: r must be reachable from the full set (dp[n_divs]), but that's always true if r <= target.
    #   For pruning, we can check: is there a subset of valid_divs[pos..best_M_idx] with vec sum = r?
    #   We can precompute suffix reachability tables, but that doubles memory.
    #   Alternatively, just do DFS with memoization; the state space is small (pos * 12096 * 13).

    M = best_M_idx
    candidates = valid_divs[:M+1]
    cvecs = [div_vecs[d] for d in candidates]
    n_cand = len(candidates)

    # Memoize DFS states
    memo = {}
    path = []

    def dfs_recon(pos, cur_vec, count):
        if time.time() > deadline:
            return False
        remaining = (target_vec[0]-cur_vec[0], target_vec[1]-cur_vec[1],
                     target_vec[2]-cur_vec[2], target_vec[3]-cur_vec[3],
                     target_vec[4]-cur_vec[4])
        if remaining == (0,0,0,0,0):
            return count >= 1
        if count == 12 or pos >= n_cand:
            return False

        key = (pos, cur_vec, count)
        if key in memo:
            return memo[key]

        # Prune: remaining must be formable by divisors in candidates[pos..]
        # Quick check: the product of remaining primes^exponents must be <= product of largest (12-count) in candidates[pos..]
        rem_int = 1
        for j, p in enumerate(primes):
            rem_int *= p ** remaining[j]
        max_avail = 12 - count
        if max_avail <= 0:
            memo[key] = False
            return False
        # Product of largest max_avail in candidates[pos..]
        start_idx = max(pos, n_cand - max_avail)
        prod_max = 1
        for x in candidates[start_idx:]:
            prod_max *= x
        if rem_int > prod_max:
            memo[key] = False
            return False
        if rem_int < candidates[pos]:
            memo[key] = False
            return False

        # Try each candidate from pos onward
        for i in range(pos, n_cand):
            v = cvecs[i]
            new_vec = (cur_vec[0]+v[0], cur_vec[1]+v[1], cur_vec[2]+v[2],
                       cur_vec[3]+v[3], cur_vec[4]+v[4])
            if new_vec[0] > target_vec[0] or new_vec[1] > target_vec[1] or \
               new_vec[2] > target_vec[2] or new_vec[3] > target_vec[3] or new_vec[4] > target_vec[4]:
                continue
            path.append(candidates[i])
            if dfs_recon(i+1, new_vec, count+1):
                memo[key] = True
                return True
            path.pop()

        memo[key] = False
        return False

    if not dfs_recon(0, (0,0,0,0,0), 0):
        return [target_val]

    # Verify
    prod = 1
    for d in path:
        prod *= d
    assert prod == target_val, f"Product mismatch: {prod} != {target_val}"
    assert len(set(path)) == len(path), "Not distinct"
    assert all(d > 12 for d in path), "Factor not > 12"
    assert 1 <= len(path) <= 12, "Bad count"

    return path
