import sys
import time
from itertools import combinations

def _compute_smooth(n_max):
    smooth = []
    for i in range(1, n_max + 1):
        x = i
        while x % 2 == 0: x //= 2
        while x % 3 == 0: x //= 3
        while x % 5 == 0: x //= 5
        if x == 1:
            smooth.append(i)
    return smooth

def _is_antichain(terms):
    n = len(terms)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = terms[i], terms[j]
            if a % b == 0 or b % a == 0:
                return False
    return True

def _solve_all():
    start = time.time()
    DEADLINE = 80.0  # Safety margin before 90s kill
    
    def time_ok():
        return time.time() - start < DEADLINE
    
    smooth = _compute_smooth(128)
    N = len(smooth)
    
    # Precompute divisibility: div[i][j] = True if smooth[i] divides smooth[j]
    div = [[False] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j and smooth[j] % smooth[i] == 0:
                div[i][j] = True
    
    def is_anti_set(indices):
        """Check if indices form an antichain using precomputed div table."""
        n = len(indices)
        for a in range(n):
            for b in range(a + 1, n):
                i, j = indices[a], indices[b]
                if div[i][j] or div[j][i]:
                    return False
        return True
    
    answers = [None] * 128
    
    # Level 1: Single term
    for idx, s in enumerate(smooth):
        if s <= 128:
            answers[s - 1] = [s]
    
    # Level 2: Two terms
    for i in range(N):
        a = smooth[i]
        for j in range(i + 1, N):
            b = smooth[j]
            s = a + b
            if s > 128:
                break
            if answers[s - 1] is None:
                if not div[i][j] and not div[j][i]:
                    answers[s - 1] = [a, b]
    
    # Check if all found
    if all(ans is not None for ans in answers):
        return answers
    
    # Level 3: Three terms
    for i in range(N):
        a = smooth[i]
        for j in range(i + 1, N):
            b = smooth[j]
            if a + b > 128:
                break
            if div[i][j] or div[j][i]:
                continue
            for k in range(j + 1, N):
                c = smooth[k]
                s = a + b + c
                if s > 128:
                    break
                if answers[s - 1] is None:
                    if not div[i][k] and not div[k][i] and not div[j][k] and not div[k][j]:
                        answers[s - 1] = [a, b, c]
    
    if all(ans is not None for ans in answers):
        return answers
    
    # Level 4: Four terms
    for i in range(N):
        a = smooth[i]
        for j in range(i + 1, N):
            b = smooth[j]
            if a + b > 128:
                break
            if div[i][j] or div[j][i]:
                continue
            for k in range(j + 1, N):
                c = smooth[k]
                if a + b + c > 128:
                    break
                if div[i][k] or div[k][i] or div[j][k] or div[k][j]:
                    continue
                for l in range(k + 1, N):
                    d = smooth[l]
                    s = a + b + c + d
                    if s > 128:
                        break
                    if answers[s - 1] is None:
                        if (not div[i][l] and not div[l][i] and
                            not div[j][l] and not div[l][j] and
                            not div[k][l] and not div[l][k]):
                            answers[s - 1] = [a, b, c, d]
    
    if all(ans is not None for ans in answers):
        return answers
    
    # Level 5: Five terms
    for i in range(N):
        a = smooth[i]
        for j in range(i + 1, N):
            b = smooth[j]
            if a + b > 128:
                break
            if div[i][j] or div[j][i]:
                continue
            for k in range(j + 1, N):
                c = smooth[k]
                if a + b + c > 128:
                    break
                if div[i][k] or div[k][i] or div[j][k] or div[k][j]:
                    continue
                for l in range(k + 1, N):
                    d = smooth[l]
                    if a + b + c + d > 128:
                        break
                    if (div[i][l] or div[l][i] or
                        div[j][l] or div[l][j] or
                        div[k][l] or div[l][k]):
                        continue
                    for m in range(l + 1, N):
                        e = smooth[m]
                        s = a + b + c + d + e
                        if s > 128:
                            break
                        if answers[s - 1] is None:
                            if (not div[i][m] and not div[m][i] and
                                not div[j][m] and not div[m][j] and
                                not div[k][m] and not div[m][k] and
                                not div[l][m] and not div[m][l]):
                                answers[s - 1] = [a, b, c, d, e]
    
    if all(ans is not None for ans in answers):
        return answers
    
    # Fallback for any remaining: brute force with increasing k
    for idx in range(128):
        if answers[idx] is None:
            target = idx + 1
            for k in range(6, 31):
                if not time_ok():
                    break
                for comb in combinations(smooth, k):
                    if sum(comb) == target and _is_antichain(comb):
                        answers[idx] = list(comb)
                        break
                if answers[idx] is not None:
                    break
    
    return answers

# Compute at module load time
_PRECOMPUTED = _solve_all()

def construct():
    return _PRECOMPUTED
