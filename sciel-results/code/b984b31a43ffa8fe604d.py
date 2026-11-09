import time
import random

def construct():
    deadline = time.time() + 80.0   # explicit guard: return best before the 90s wall-clock kill
    n_max = 512
    target_size = 48

    # ------------------------------------------------------------------
    # Exact validity oracle (bitset subset-sum, "hyperedge" check).
    # A set S is valid iff no element is the exact mean of any subset of
    # >=2 OTHER elements.
    # ------------------------------------------------------------------
    def is_valid(S):
        S = list(S)
        n = len(S)
        for i, x in enumerate(S):
            others = S[:i] + S[i + 1:]
            m = len(others)
            if m < 2:
                continue
            dp = [0] * (m + 1)
            dp[0] = 1
            for val in others:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k + 1] |= (dp[k] << val)
            max_sum = sum(others)
            max_k = min(m, max_sum // x)
            for k in range(2, max_k + 1):
                if (dp[k] >> (k * x)) & 1:
                    return False
        return True

    # ------------------------------------------------------------------
    # Fast incremental check: is S ∪ {x} valid given S is already valid?
    # Only the constraints that involve x need re-checking.
    # ------------------------------------------------------------------
    def is_valid_add(S, x):
        if x in S:
            return False
        n = len(S)
        # (a) x is not the mean of any subset of S of size >= 2
        if n >= 2:
            max_sum = sum(S)
            max_k = min(n, max_sum // x)
            if max_k >= 2:
                dp = [0] * (n + 1)
                dp[0] = 1
                for val in S:
                    for k in range(n - 1, -1, -1):
                        if dp[k]:
                            dp[k + 1] |= (dp[k] << val)
                for k in range(2, max_k + 1):
                    if (dp[k] >> (k * x)) & 1:
                        return False
        # (b) no a in S becomes the mean of a subset of (S\{a}) ∪ {x}
        #     that includes x  ->  M = {x} ∪ T, T ⊆ S\{a}, |T| >= 1
        for a in S:
            others = [s for s in S if s != a]
            m = len(others)
            if m < 1:
                continue
            dp = [0] * (m + 1)
            dp[0] = 1
            for val in others:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k + 1] |= (dp[k] << val)
            max_sum_o = sum(others)
            for j in range(1, m + 1):
                t = (j + 1) * a - x
                if t < 0:
                    continue
                if t > max_sum_o:
                    break
                if (dp[j] >> t) & 1:
                    return False
        return True

    # ------------------------------------------------------------------
    # Hypergraph vertex ordering: inverse of how many hyperedges a vertex
    # participates in (as a mean or as a constituent). Low-degree vertices
    # are "safer" to pick early -> larger maximal independent set.
    # ------------------------------------------------------------------
    def conflict_degree(x):
        deg = 0
        # x as a mean: pairs a<b, a+b=2x within [1,n_max]
        a_min = max(1, 2 * x - n_max)
        a_max = x - 1
        if a_min <= a_max:
            deg += a_max - a_min + 1
        # x as a constituent of a pair whose mean is in [1,n_max]:
        #   x + b = 2m, m in [1,n_max]  ->  b = 2m - x, b in [1,n_max], b != x
        lo = max(1, x + 1)
        hi = min(2 * n_max - x, n_max)
        if lo <= hi:
            deg += (hi - lo) // 2
        return deg

    def greedy(order, S_set):
        S = []
        S_set = set(S_set) if S_set else set()
        for x in order:
            if len(S) >= target_size:
                break
            if time.time() > deadline:
                break
            if x in S_set:
                continue
            if is_valid_add(S, x):
                S.append(x)
                S_set.add(x)
        return S, S_set

    def local_search(S, S_set, available, rng):
        S = list(S)
        S_set = set(S)
        available = list(available)
        improved = True
        it = 0
        while improved and it < 3000 and time.time() < deadline:
            it += 1
            improved = False
            # --- add ---
            for x in available:
                if len(S) >= target_size:
                    break
                if is_valid_add(S, x):
                    S.append(x); S_set.add(x)
                    available.remove(x)
                    improved = True
                    break
            if len(S) >= target_size:
                break
            if improved:
                continue
            # --- 1-swap ---
            rng.shuffle(S)
            for i in range(len(S)):
                if time.time() > deadline:
                    return S, S_set, available
                old = S[i]
                S_temp = S[:i] + S[i + 1:]
                S_temp_set = set(S_temp)
                for x in available:
                    if x in S_temp_set:
                        continue
                    if is_valid_add(S_temp, x):
                        S = S_temp + [x]
                        S_set = set(S)
                        available.remove(x)
                        available.append(old)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # --- 2-swap (deeper local search) ---
            if len(S) >= 2:
                rng.shuffle(S)
                for i in range(len(S)):
                    if time.time() > deadline:
                        return S, S_set, available
                    for j in range(i + 1, len(S)):
                        S_temp = S[:i] + S[i + 1:j] + S[j + 1:]
                        S_temp_set = set(S_temp)
                        o1, o2 = S[i], S[j]
                        for x in available:
                            if x in S_temp_set:
                                continue
                            if is_valid_add(S_temp, x):
                                for y in available:
                                    if y == x or y in S_temp_set:
                                        continue
                                    if is_valid_add(S_temp + [x], y):
                                        S = S_temp + [x, y]
                                        S_set = set(S)
                                        available.remove(x)
                                        available.remove(y)
                                        available.append(o1)
                                        available.append(o2)
                                        improved = True
                                        break
                                if improved:
                                    break
                        if improved:
                            break
                    if improved:
                        break

        return S, S_set, available

    # ------------------------------------------------------------------
    # Search: multiple orderings (degree-based + weighted random), keep best.
    # ------------------------------------------------------------------
    best_set = []
    best_size = 0
    best_available = None

    def record(S):
        nonlocal best_set, best_size, best_available
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            best_available = [x for x in range(1, n_max + 1) if x not in set(S)]

    rng = random.Random(42)

    # 1) low-degree-first (fewer hyperedges -> safer)
    order = sorted(range(1, n_max + 1), key=conflict_degree)
    S, _ = greedy(order, set())
    record(S)

    # 2) high-degree-first
    order = sorted(range(1, n_max + 1), key=lambda x: -conflict_degree(x))
    S, _ = greedy(order, set())
    record(S)

    # 3) weighted random orders biased toward low conflict degree
    weights = [1.0 / (conflict_degree(x) + 1) for x in range(1, n_max + 1)]
    for _ in range(2000):
        if time.time() > deadline:
            break
        order = rng.choices(range(1, n_max + 1), weights=weights, k=n_max)
        order = list(dict.fromkeys(order))          # dedupe, keep first occurrence
        S, _ = greedy(order, set())
        record(S)

    # 4) local search on the best set found
    if time.time() < deadline:
        S, S_set, available = local_search(best_set, set(best_set), best_available, rng)
        record(S)

    # ------------------------------------------------------------------
    # Final safety: guarantee validity of the returned set.
    # ------------------------------------------------------------------
    result = best_set
    if not is_valid(result):
        result = [x for x in result if is_valid([x for y in result if y <= x] + [])]
        # rebuild valid prefix
        acc = []
        for x in result:
            if is_valid_add(acc, x):
                acc.append(x)
        result = acc

    return result
