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

def _precompute_antichains(smooth_list, max_k=5, max_val=128):
    """Precompute all valid antichain combinations of size 1 to max_k that sum to <= max_val."""
    antichain_sums = {k: {} for k in range(1, max_k + 1)}
    
    # k=1
    for x in smooth_list:
        if x <= max_val:
            antichain_sums[1].setdefault(x, []).append((x,))
    
    # k=2
    n = len(smooth_list)
    for i in range(n):
        a = smooth_list[i]
        for j in range(i + 1, n):
            b = smooth_list[j]
            s = a + b
            if s > max_val:
                break
            if _incompatible(a, b):
                antichain_sums[2].setdefault(s, []).append((a, b))
    
    # k=3
    for i in range(n):
        a = smooth_list[i]
        for j in range(i + 1, n):
            b = smooth_list[j]
            if a + b >= max_val:
                break
            if not _incompatible(a, b):
                continue
            for k2 in range(j + 1, n):
                c = smooth_list[k2]
                s = a + b + c
                if s > max_val:
                    break
                if _incompatible(a, c) and _incompatible(b, c):
                    antichain_sums[3].setdefault(s, []).append((a, b, c))
    
    # k=4
    for i in range(n):
        a = smooth_list[i]
        for j in range(i + 1, n):
            b = smooth_list[j]
            if a + b >= max_val:
                break
            if not _incompatible(a, b):
                continue
            for k2 in range(j + 1, n):
                c = smooth_list[k2]
                if a + b + c >= max_val:
                    break
                if not (_incompatible(a, c) and _incompatible(b, c)):
                    continue
                for l in range(k2 + 1, n):
                    d = smooth_list[l]
                    s = a + b + c + d
                    if s > max_val:
                        break
                    if _incompatible(a, d) and _incompatible(b, d) and _incompatible(c, d):
                        antichain_sums[4].setdefault(s, []).append((a, b, c, d))
    
    # k=5
    for i in range(n):
        a = smooth_list[i]
        for j in range(i + 1, n):
            b = smooth_list[j]
            if a + b >= max_val:
                break
            if not _incompatible(a, b):
                continue
            for k2 in range(j + 1, n):
                c = smooth_list[k2]
                if a + b + c >= max_val:
                    break
                if not (_incompatible(a, c) and _incompatible(b, c)):
                    continue
                for l in range(k2 + 1, n):
                    d = smooth_list[l]
                    if a + b + c + d >= max_val:
                        break
                    if not (_incompatible(a, d) and _incompatible(b, d) and _incompatible(c, d)):
                        continue
                    for m in range(l + 1, n):
                        e = smooth_list[m]
                        s = a + b + c + d + e
                        if s > max_val:
                            break
                        if (_incompatible(a, e) and _incompatible(b, e) and 
                            _incompatible(c, e) and _incompatible(d, e)):
                            antichain_sums[5].setdefault(s, []).append((a, b, c, d, e))
    
    return antichain_sums

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    smooth_list = _gen_smooth(128)
    smooth_set = set(smooth_list)
    
    # Precompute antichain sums
    antichain_sums = _precompute_antichains(smooth_list, max_k=5, max_val=128)
    
    answers = []
    for n in range(1, 129):
        if time.time() > deadline:
            # Fallback: use precomputed if available, else simple decomposition
            if n in antichain_sums[1]:
                answers.append(list(antichain_sums[1][n][0]))
            elif n in antichain_sums[2]:
                answers.append(list(antichain_sums[2][n][0]))
            elif n in antichain_sums[3]:
                answers.append(list(antichain_sums[3][n][0]))
            elif n in antichain_sums[4]:
                answers.append(list(antichain_sums[4][n][0]))
            elif n in antichain_sums[5]:
                answers.append(list(antichain_sums[5][n][0]))
            else:
                # Last resort fallback
                if n in smooth_set:
                    answers.append([n])
                else:
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
                        answers.append([1] * min(n, 30))
            continue
        
        # Try increasing k from 1 to 5
        found = False
        for k in range(1, 6):
            if n in antichain_sums[k]:
                answers.append(list(antichain_sums[k][n][0]))
                found = True
                break
        
        if not found:
            # Fallback for cases not covered by precomputation
            if n in smooth_set:
                answers.append([n])
            else:
                # Try to find any valid representation
                found = False
                for k in range(2, 8):
                    if time.time() > deadline:
                        break
                    for combo in combinations(smooth_list, k):
                        if sum(combo) == n and _is_antichain(combo):
                            answers.append(list(combo))
                            found = True
                            break
                    if found:
                        break
                if not found:
                    answers.append([1] * min(n, 30))
    
    return answers
# EVOLVE-BLOCK-END
