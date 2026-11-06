"""Greedy-construction + stochastic local search for equal-gcd triple avoidance."""
import time
import random
from math import gcd

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    N = 128
    UNIVERSE = list(range(1, N + 1))

    # Precompute gcd matrix
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    def is_valid(S):
        """Check if set S has no equal-gcd triple. O(|S|^2) using set of seen gcds per pair."""
        S_set = set(S)
        L = len(S)
        for i in range(L):
            a = S[i]
            for j in range(i + 1, L):
                b = S[j]
                gij = gcd_mat[a][b]
                for k in range(j + 1, L):
                    c = S[k]
                    if gcd_mat[a][c] == gij and gcd_mat[b][c] == gij:
                        return False
        return True

    def can_add(S, x, S_set):
        """Check if x can be added to S without creating a bad triple. O(|S|^2)."""
        for i in range(len(S)):
            a = S[i]
            gax = gcd_mat[a][x]
            for j in range(i + 1, len(S)):
                b = S[j]
                if gcd_mat[a][b] == gax and gcd_mat[b][x] == gax:
                    return False
        return True

    def greedy_construct(seed_val, order="natural"):
        """Greedy construction: add elements in a given order if valid."""
        random.seed(seed_val)
        if order == "natural":
            candidates = list(UNIVERSE)
        elif order == "reverse":
            candidates = list(reversed(UNIVERSE))
        elif order == "random":
            candidates = list(UNIVERSE)
            random.shuffle(candidates)
        elif order == "prime_first":
            # Prioritize primes and numbers with large smallest prime factor
            def score(x):
                if x == 1:
                    return -1
                # Find smallest prime factor
                spf = x
                d = 2
                while d * d <= x:
                    if x % d == 0:
                        spf = d
                        break
                    d += 1
                return -spf  # Larger spf (more prime-like) gets higher priority
            candidates = sorted(UNIVERSE, key=score, reverse=True)
        else:
            candidates = list(UNIVERSE)
            random.shuffle(candidates)

        selected = []
        S_set = set()
        for x in candidates:
            if can_add(selected, x, S_set):
                selected.append(x)
                S_set.add(x)
        return selected

    def local_search(S, seed_val, max_iters=5000):
        """Stochastic local search: try to add or swap elements to improve size."""
        random.seed(seed_val)
        S = sorted(S)
        S_set = set(S)
        remaining = [x for x in UNIVERSE if x not in S_set]
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < DEADLINE - 2:
            improved = False
            iters += 1

            # Try to add an element
            if remaining:
                # Try a random few candidates from remaining
                n_try = min(10, len(remaining))
                for _ in range(n_try):
                    idx = random.randint(0, len(remaining) - 1)
                    x = remaining[idx]
                    if can_add(S, x, S_set):
                        S.append(x)
                        S.sort()
                        S_set.add(x)
                        remaining.pop(idx)
                        improved = True
                        break

            # If no add, try a swap: remove one from S, add one from remaining
            if not improved and remaining and len(S) > 1:
                n_try = min(20, len(S) * len(remaining) // 5)
                for _ in range(n_try):
                    i = random.randint(0, len(S) - 1)
                    j = random.randint(0, len(remaining) - 1)
                    a = S[i]
                    b = remaining[j]
                    # Remove a, add b
                    new_S = S[:i] + S[i + 1:] + [b]
                    new_S.sort()
                    new_S_set = set(new_S)
                    if is_valid(new_S):
                        S = new_S
                        S_set = new_S_set
                        remaining.remove(b)
                        remaining.append(a)
                        improved = True
                        break

        return S

    # Initial best via multiple greedy orderings
    best_solution = greedy_construct(0, "natural")
    best_len = len(best_solution)

    # Try other orderings
    for order in ["reverse", "prime_first", "random"]:
        cand = greedy_construct(42, order)
        if len(cand) > best_len:
            best_len = len(cand)
            best_solution = cand

    # Local search on best
    best_solution = local_search(best_solution[:], 0)
    best_len = len(best_solution)

    # Multiple restarts with different seeds
    for seed_val in range(1, 500):
        if time.time() >= DEADLINE - 5:
            break
        # Random greedy construction
        cand = greedy_construct(seed_val, "random")
        # Local search
        cand = local_search(cand, seed_val, max_iters=3000)
        if len(cand) > best_len:
            best_len = len(cand)
            best_solution = cand

    # Final local search on the best solution with more iterations
    if time.time() < DEADLINE - 3:
        final = local_search(best_solution[:], 999, max_iters=10000)
        if len(final) > best_len:
            best_len = len(final)
            best_solution = final

    # Final validity check
    if not is_valid(best_solution):
        # Fallback: simple greedy
        selected = []
        S_set = set()
        for x in UNIVERSE:
            if can_add(selected, x, S_set):
                selected.append(x)
                S_set.add(x)
        if selected:
            best_solution = selected
            best_len = len(selected)
        else:
            best_solution = [1]
            best_len = 1

    return best_solution


# Wrap to ensure nonempty
_orig_construct = construct
def construct():
    result = _orig_construct()
    if not result:
        return [1]
    return result
