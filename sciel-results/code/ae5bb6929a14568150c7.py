import random
import time

def _is_valid(S):
    sum_to_card = {}
    n = len(S)
    for mask in range(1 << n):
        s = 0
        k = 0
        for i in range(n):
            if mask & (1 << i):
                s += S[i]
                k += 1
        if s in sum_to_card:
            if sum_to_card[s] != k:
                return False
        else:
            sum_to_card[s] = k
    return True

def _search(n, limit=256, trials=10000):
    for _ in range(trials):
        S = random.sample(range(1, limit + 1), n)
        if _is_valid(S):
            return S
    return None

def construct():
    # Try to find larger sets
    for n in range(11, 8, -1):
        S = _search(n, trials=5000)
        if S:
            return [int(x) for x in S]
    # Fallback to powers of 2
    result = []
    val = 1
    while val <= 256:
        result.append(val)
        val *= 2
    return [int(x) for x in result]
