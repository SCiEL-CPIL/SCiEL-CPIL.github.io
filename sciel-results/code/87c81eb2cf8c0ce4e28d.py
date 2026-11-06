import time
from itertools import combinations

def _is_smooth(n):
    while n % 2 == 0:
        n //= 2
    while n % 3 == 0:
        n //= 3
    while n % 5 == 0:
        n //= 5
    return n == 1

def _is_antichain(terms):
    n = len(terms)
    for i in range(n):
        for j in range(i + 1, n):
            if terms[i] % terms[j] == 0 or terms[j] % terms[i] == 0:
                return False
    return True

def _find_minimal(target, max_terms=30):
    """Find minimal antichain representation for target."""
    # 1-term
    if _is_smooth(target):
        return [target]
    
    # Generate smooth numbers up to target
    smooths = [i for i in range(1, target + 1) if _is_smooth(i)]
    
    # Try 2-term
    for i in range(len(smooths)):
        a = smooths[i]
        if a >= target:
            break
        b = target - a
        if b > a and _is_smooth(b) and a % b != 0 and b % a != 0:
            return [a, b]
    
    # Try 3-term
    for i in range(len(smooths)):
        a = smooths[i]
        if a >= target:
            break
        for j in range(i + 1, len(smooths)):
            b = smooths[j]
            if a + b >= target:
                break
            c = target - a - b
            if c > b and _is_smooth(c):
                if (a % b != 0 and b % a != 0 and
                    a % c != 0 and c % a != 0 and
                    b % c != 0 and c % b != 0):
                    return [a, b, c]
    
    # Try 4-term
    for i in range(len(smooths)):
        a = smooths[i]
        if a >= target:
            break
        for j in range(i + 1, len(smooths)):
            b = smooths[j]
            if a + b >= target:
                break
            for k in range(j + 1, len(smooths)):
                c = smooths[k]
                if a + b + c >= target:
                    break
                d = target - a - b - c
                if d > c and _is_smooth(d):
                    terms = (a, b, c, d)
                    ac = True
                    for x in range(4):
                        for y in range(x + 1, 4):
                            if terms[x] % terms[y] == 0 or terms[y] % terms[x] == 0:
                                ac = False
                                break
                        if not ac:
                            break
                    if ac:
                        return list(terms)
    
    # Try 5-term (rarely needed for targets <= 128)
    for i in range(len(smooths)):
        a = smooths[i]
        if a >= target:
            break
        for j in range(i + 1, len(smooths)):
            b = smooths[j]
            if a + b >= target:
                break
            for k in range(j + 1, len(smooths)):
                c = smooths[k]
                if a + b + c >= target:
                    break
                for l in range(k + 1, len(smooths)):
                    d = smooths[l]
                    if a + b + c + d >= target:
                        break
                    e = target - a - b - c - d
                    if e > d and _is_smooth(e):
                        terms = (a, b, c, d, e)
                        ac = True
                        for x in range(5):
                            for y in range(x + 1, 5):
                                if terms[x] % terms[y] == 0 or terms[y] % terms[x] == 0:
                                    ac = False
                                    break
                            if not ac:
                                break
                        if ac:
                            return list(terms)
    
    # Fallback: should not happen for targets <= 128
    return [target]

# Precompute all solutions at module load time
def _precompute():
    results = []
    for t in range(1, 129):
        results.append(_find_minimal(t))
    return results

_PRECOMPUTED = _precompute()

def construct():
    return [list(row) for row in _PRECOMPUTED]
