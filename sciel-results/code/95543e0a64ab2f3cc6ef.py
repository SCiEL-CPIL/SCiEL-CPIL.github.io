# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _gen_smooth(max_val):
    """Generate all 2^a * 3^b * 5^c numbers up to max_val, sorted."""
    smooth = set()
    a = 1
    while a <= max_val:
        b = a
        while b <= max_val:
            c = b
            while c <= max_val:
                smooth.add(c)
                c *= 5
            b *= 3
        a *= 2
    return sorted(smooth)

def _divides(a, b):
    """Check if a divides b (a <= b assumed)."""
    return b % a == 0

def _incompatible(a, b):
    """Check if a and b are incompatible (neither divides the other)."""
    if a > b:
        a, b = b, a
    return not _divides(a, b)

def _is_antichain(terms):
    """Check if no term divides another."""
    n = len(terms)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = terms[i], terms[j]
            if a > b:
                a, b = b, a
            if _divides(a, b):
                return False
    return True

def _find_min_terms(target, smooth_list, smooth_set, max_k=6):
    """Find minimum number of smooth antichain terms summing to target."""
    # k=1: target itself is smooth
    if target in smooth_set:
        return [target]
    
    candidates = [x for x in smooth_list if x < target]  # exclude target itself for k>=2
    
    # k=2
    for i in range(len(candidates)):
        a = candidates[i]
        b = target - a
        if b in smooth_set and b != a and _incompatible(a, b):
            return [a, b]
    
    # k=3
    for i in range(len(candidates)):
        a = candidates[i]
        remaining = target - a
        if remaining <= 0:
            continue
        for j in range(len(candidates)):
            b = candidates[j]
            if b >= remaining:
                break
            c = remaining - b
            if c in smooth_set and c != a and c != b:
                if _incompatible(a, b) and _incompatible(a, c) and _incompatible(b, c):
                    return [a, b, c]
    
    # k=4
    for i in range(len(candidates)):
        a = candidates[i]
        rem1 = target - a
        if rem1 <= 0:
            continue
        for j in range(len(candidates)):
            b = candidates[j]
            rem2 = rem1 - b
            if rem2 <= 0:
                break
            for k2 in range(len(candidates)):
                c = candidates[k2]
                rem3 = rem2 - c
                if rem3 <= 0:
                    break
                d = rem3
                if d in smooth_set and d != a and d != b and d != c:
                    terms = [a, b, c, d]
                    if _is_antichain(terms):
                        return terms
    
    # k=5
    for i in range(len(candidates)):
        a = candidates[i]
        rem1 = target - a
        if rem1 <= 0:
            continue
        for j in range(len(candidates)):
            b = candidates[j]
            rem2 = rem1 - b
            if rem2 <= 0:
                break
            for k2 in range(len(candidates)):
                c = candidates[k2]
                rem3 = rem2 - c
                if rem3 <= 0:
                    break
                for l in range(len(candidates)):
                    d = candidates[l]
                    rem4 = rem3 - d
                    if rem4 <= 0:
                        break
                    e = rem4
                    if e in smooth_set and e != a and e != b and e != c and e != d:
                        terms = [a, b, c, d, e]
                        if _is_antichain(terms):
                            return terms
    
    # k=6
    for i in range(len(candidates)):
        a = candidates[i]
        rem1 = target - a
        if rem1 <= 0:
            continue
        for j in range(len(candidates)):
            b = candidates[j]
            rem2 = rem1 - b
            if rem2 <= 0:
                break
            for k2 in range(len(candidates)):
                c = candidates[k2]
                rem3 = rem2 - c
                if rem3 <= 0:
                    break
                for l in range(len(candidates)):
                    d = candidates[l]
                    rem4 = rem3 - d
                    if rem4 <= 0:
                        break
                    for m in range(len(candidates)):
                        e = candidates[m]
                        rem5 = rem4 - e
                        if rem5 <= 0:
                            break
                        f = rem5
                        if f in smooth_set and f != a and f != b and f != c and f != d and f != e:
                            terms = [a, b, c, d, e, f]
                            if _is_antichain(terms):
                                return terms
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    smooth_list = _gen_smooth(128)
    smooth_set = set(smooth_list)
    
    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            # Fallback: just use [n] if smooth, else try simple decomposition
            if n in smooth_set:
                answers.append([n])
            else:
                # Try to find any valid representation quickly
                found = False
                for a in smooth_list:
                    if a >= n:
                        break
                    b = n - a
                    if b in smooth_set and b != a and _incompatible(a, b):
                        answers.append([a, b])
                        found = True
                        break
                if not found:
                    answers.append([1] * n if n <= 30 else [1] * 30)
            continue
        
        result = _find_min_terms(n, smooth_list, smooth_set)
        
        if result is None:
            # Fallback
            if n in smooth_set:
                result = [n]
            else:
                # Try brute force with more terms
                found = False
                for k in range(2, 10):
                    if time.time() > deadline:
                        break
                    for combo in combinations(smooth_list, k):
                        if sum(combo) == n and _is_antichain(combo):
                            result = list(combo)
                            found = True
                            break
                    if found:
                        break
                if result is None:
                    result = [1] * n if n <= 30 else [1] * 30
        
        answers.append(result)
    
    return answers
# EVOLVE-BLOCK-END
