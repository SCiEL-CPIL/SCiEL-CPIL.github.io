# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start = time.time()
    deadline = start + 55.0  # Safety margin before 90s kill

    N = 192
    max_sum = 3 * N  # 576

    # Precompute divisors for all possible sums
    divisors = [[] for _ in range(max_sum + 1)]
    for s in range(3, max_sum + 1):
        for d in range(1, N + 1):
            if s % d == 0:
                divisors[s].append(d)

    def can_add(a, S):
        """Check if adding element a to set S keeps the set valid.
        Returns True if S ∪ {a} satisfies the constraint."""
        S_list = list(S)
        K = len(S_list)
        S_set = set(S)

        # Part 1: Check if a divides sum of any 3 distinct elements in S
        # For each multiple M of a, check if 3 distinct elements in S sum to M
        for M in range(a, max_sum + 1, a):
            if M < 3 or M > 3 * N:
                continue
            # Check if there exist x, y, z in S (distinct) with x+y+z = M
            found = False
            for i in range(K):
                x = S_list[i]
                if x > M:
                    continue
                for j in range(i + 1, K):
                    y = S_list[j]
                    if x + y > M:
                        continue
                    z = M - x - y
                    if z == x or z == y:
                        continue
                    if z in S_set:
                        return False
                        # Early exit: a divides this triple sum

        # Part 2: Check if any b in S divides sum of any triple involving a
        # Triples are (a, u, v) where u, v in S, u != v
        # Sum T = a + u + v
        # Check if any b in S (b != a, b != u, b != v) divides T
        for i in range(K):
            u = S_list[i]
            for j in range(i + 1, K):
                v = S_list[j]
                T = a + u + v
                if T <= max_sum:
                    for b in divisors[T]:
                        if b == a or b == u or b == v:
                            continue
                        if b in S_set:
                            return False
        return True

    def is_valid_set(S):
        """Verify that set S satisfies the constraint."""
        S_list = list(S)
        S_set = set(S)
        K = len(S_list)
        if K < 4:
            return True

        for a in S_list:
            others = [x for x in S_list if x != a]
            # Check if a divides sum of any 3 in others
            for M in range(a, max_sum + 1, a):
                if M < 3 or M > 3 * N:
                    continue
                for i in range(len(others)):
                    x = others[i]
                    if x > M:
                        continue
                    for j in range(i + 1, len(others)):
                        y = others[j]
                        if x + y > M:
                            continue
                        z = M - x - y
                        if z == x or z == y:
                            continue
                        if z in S_set:
                            return False
        return True

    # Strategy: Greedy construction with efficient checking
    # Start with larger numbers (they're less likely to divide small sums)
    S = set()
    for x in range(N, 0, -1):
        if time.time() > deadline:
            break
        if can_add(x, S):
            S.add(x)

    best_S = set(S)
    best_size = len(S)

    # Local search: try to improve by adding or swapping
    rng = random.Random(42)

    while time.time() < deadline - 5.0:
        S_list = list(S)
        U_not_S = [x for x in range(1, N + 1) if x not in S]

        if not U_not_S:
            break

        # Randomly choose action: add or swap
        if random.random() < 0.3:
            # Try to add a new element
            a = rng.choice(U_not_S)
            if can_add(a, S):
                S.add(a)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
        else:
            # Try to swap: remove one element, add another
            if S_list:
                r = rng.choice(S_list)
                a = rng.choice(U_not_S)

                # Check if S \ {r} ∪ {a} is valid
                S_mid = S - {r}
                if can_add(a, S_mid):
                    S.remove(r)
                    S.add(a)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)

    # Final verification
    if not is_valid_set(best_S):
        # Fallback: rebuild with greedy
        S = set()
        for x in range(N, 0, -1):
            if can_add(x, S):
                S.add(x)
        best_S = S

    return list(best_S)
# EVOLVE-BLOCK-END
