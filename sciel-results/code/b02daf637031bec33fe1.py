"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    n = 256

    def can_add(a, S):
        """Check if element a can be added to set S without violating property.
        For all b, c in S with a < b <= c: a must not divide (b+c)."""
        elements = sorted(S)
        m = len(elements)
        for j in range(m):
            b = elements[j]
            if b <= a:
                continue
            for k in range(j, m):
                c = elements[k]
                if (b + c) % a == 0:
                    return False
        return True

    def greedy_add(S, upper_limit):
        """Starting from S, greedily add elements from upper_limit down to 1."""
        S = set(S)
        for a in range(upper_limit, 0, -1):
            if time.time() > deadline - 5:
                break
            if a not in S and can_add(a, S):
                S.add(a)
        return S

    def local_search(S, max_iters=500):
        """Try to improve S by removing one element and adding multiple smaller ones."""
        S = set(S)
        best_size = len(S)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 8:
            improved = False
            iters += 1
            S_list = sorted(S)
            # Try removing each element (prioritize larger ones as they block more)
            for a in reversed(S_list):
                if time.time() > deadline - 8:
                    break
                S_test = S - {a}
                # Try to add as many smaller elements as possible
                added = 0
                for b in range(1, a + 1):
                    if b not in S_test and can_add(b, S_test):
                        S_test.add(b)
                        added += 1
                if len(S_test) > best_size:
                    S = S_test
                    best_size = len(S)
                    improved = True
                    break
        return S

    def final_polish(S):
        """Final greedy pass to add any remaining elements."""
        S = set(S)
        for a in range(1, n + 1):
            if time.time() > deadline - 3:
                break
            if a not in S and can_add(a, S):
                S.add(a)
        return S

    # Strategy 1: Base [171,256], descending greedy
    S1 = set(range(171, n + 1))
    S1 = greedy_add(S1, 170)

    # Strategy 2: Wider base [160,256], descending greedy
    S2 = S1
    if time.time() < deadline - 15:
        S2 = set(range(160, n + 1))
        S2 = greedy_add(S2, 159)

    # Strategy 3: Even wider base [144,256]
    S3 = S1
    if time.time() < deadline - 15:
        S3 = set(range(144, n + 1))
        S3 = greedy_add(S3, 143)

    # Strategy 4: Greedy from below (add smallest first)
    S4 = S1
    if time.time() < deadline - 15:
        S4 = set()
        for a in range(1, n + 1):
            if time.time() > deadline - 10:
                break
            if can_add(a, S4):
                S4.add(a)

    # Strategy 5: Greedy from above (add largest first)
    S5 = S1
    if time.time() < deadline - 15:
        S5 = set()
        for a in range(n, 0, -1):
            if time.time() > deadline - 10:
                break
            if can_add(a, S5):
                S5.add(a)

    # Pick the best candidate so far
    candidates = [S1, S2, S3, S4, S5]
    best_S = max(candidates, key=len)
    best_size = len(best_S)

    # Apply local search to the best candidate
    if time.time() < deadline - 10:
        best_S = local_search(best_S, max_iters=400)
        best_size = len(best_S)

    # Also try local search on other good candidates if time permits
    for S_cand in candidates:
        if time.time() > deadline - 6:
            break
        if len(S_cand) >= best_size - 5:
            S_improved = local_search(set(S_cand), max_iters=150)
            if len(S_improved) > best_size:
                best_S = S_improved
                best_size = len(best_S)

    # Final polish pass
    if time.time() < deadline - 3:
        S_final = final_polish(best_S)
        if len(S_final) > best_size:
            best_S = S_final
            best_size = len(S_final)

    return sorted(best_S)
# EVOLVE-BLOCK-END
