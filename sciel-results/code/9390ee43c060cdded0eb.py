import time

N = 256


def _validate(S):
    """Return True iff S satisfies: for all a<b<=c in S, a does not divide b+c."""
    L = sorted(S)
    m = len(L)
    for i in range(m):
        a = L[i]
        for j in range(i + 1, m):
            b = L[j]
            for k in range(j, m):
                c = L[k]
                if (b + c) % a == 0:
                    return False
    return True


def _greedy_descending():
    """Construct a valid set using greedy descending order."""
    n = N
    S = []
    for x in range(n, 0, -1):
        valid = True
        # Check if x divides b+c for any b,c in S
        # Since we add in descending order, all b,c in S are > x.
        # Thus x is the smallest element in any new triple (x, b, c).
        for i in range(len(S)):
            b = S[i]
            for j in range(i, len(S)):
                c = S[j]
                if (b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            S.append(x)
    return S


def _interval_seed():
    """Construct a valid set using the interval [floor(2N/3)+1, N]."""
    n = N
    start = (2 * n) // 3 + 1
    if start < 1:
        start = 1
    return list(range(start, n + 1))


def construct():
    # Generate candidates
    cand1 = _greedy_descending()
    cand2 = _interval_seed()
    
    # Validate and pick the largest valid one
    best = []
    if _validate(set(cand1)):
        best = cand1
    if _validate(set(cand2)) and len(cand2) > len(best):
        best = cand2
        
    # Fallback if both fail (should not happen)
    if not best:
        best = list(range(1, N + 1))
        
    return sorted(best)


# Self-check at import (cheap, deterministic).
if __name__ == "__main__":
    _s = construct()
    assert _validate(set(_s)), "constructed set is invalid"
    print("size", len(_s), "first", _s[:5], "last", _s[-5:])
