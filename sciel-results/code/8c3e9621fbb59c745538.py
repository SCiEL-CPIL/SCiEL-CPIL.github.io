"""
Divisibility-antichain representations by smooth numbers.
construct() -> 128 rows corresponding to targets 1..128.
Each row: list of 1..30 distinct positive 2^a*3^b*5^c terms summing to target.
No chosen term divides another. Minimize total summands.
"""
import time

_START = time.time()
DEADLINE = 85  # leave 5s margin before 90s kill

def construct():
    t0 = time.time()
    
    # Generate all smooth numbers up to 128
    smooth_set = set()
    for a in range(0, 8):
        v2 = 1 << a
        for b in range(0, 6):
            v23 = v2 * (3 ** b)
            if v23 > 128:
                break
            for c in range(0, 5):
                v = v23 * (5 ** c)
                if v > 128:
                    break
                if v >= 1:
                    smooth_set.add(v)
    smooth = sorted(smooth_set)
    n_smooth = len(smooth)
    
    # Precompute smooth lookup
    is_sm = [False] * 129
    for s in smooth:
        is_sm[s] = True
    
    # Precompute divisibility relationship
    # div_mask[i] = bitmask of indices j where smooth[j] divides smooth[i] or vice versa
    div_mask = [0] * n_smooth
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth[i], smooth[j]
                if b % a == 0 or a % b == 0:
                    div_mask[i] |= (1 << j)
    
    # Precompute valid antichains for each target sum, for k=1 to 8 terms
    # Use a dictionary: antichain_sums[k][target] = list of tuples
    antichain_sums = {k: {} for k in range(1, 9)}
    
    # k=1
    for s in smooth:
        if s <= 128:
            antichain_sums[1].setdefault(s, []).append((s,))
    
    # k=2
    for i in range(n_smooth):
        s1 = smooth[i]
        for j in range(i + 1, n_smooth):
            s2 = smooth[j]
            total = s1 + s2
            if total > 128:
                break
            if div_mask[i] & (1 << j):
                continue
            antichain_sums[2].setdefault(total, []).append((s1, s2))
    
    # k=3
    for i in range(n_smooth):
        s1 = smooth[i]
        for j in range(i + 1, n_smooth):
            s2 = smooth[j]
            if div_mask[i] & (1 << j):
                continue
            for k in range(j + 1, n_smooth):
                s3 = smooth[k]
                total = s1 + s2 + s3
                if total > 128:
                    break
                if (div_mask[i] & (1 << k)) or (div_mask[j] & (1 << k)):
                    continue
                antichain_sums[3].setdefault(total, []).append((s1, s2, s3))
    
    # k=4
    for i in range(n_smooth):
        s1 = smooth[i]
        for j in range(i + 1, n_smooth):
            s2 = smooth[j]
            if div_mask[i] & (1 << j):
                continue
            for k in range(j + 1, n_smooth):
                s3 = smooth[k]
                if (div_mask[i] & (1 << k)) or (div_mask[j] & (1 << k)):
                    continue
                for l in range(k + 1, n_smooth):
                    s4 = smooth[l]
                    total = s1 + s2 + s3 + s4
                    if total > 128:
                        break
                    if ((div_mask[i] | div_mask[j] | div_mask[k]) & (1 << l)):
                        continue
                    antichain_sums[4].setdefault(total, []).append((s1, s2, s3, s4))
    
    # k=5
    for i in range(n_smooth):
        s1 = smooth[i]
        for j in range(i + 1, n_smooth):
            s2 = smooth[j]
            if div_mask[i] & (1 << j):
                continue
            for k in range(j + 1, n_smooth):
                s3 = smooth[k]
                if (div_mask[i] & (1 << k)) or (div_mask[j] & (1 << k)):
                    continue
                for l in range(k + 1, n_smooth):
                    s4 = smooth[l]
                    if ((div_mask[i] | div_mask[j] | div_mask[k]) & (1 << l)):
                        continue
                    for m in range(l + 1, n_smooth):
                        s5 = smooth[m]
                        total = s1 + s2 + s3 + s4 + s5
                        if total > 128:
                            break
                        if ((div_mask[i] | div_mask[j] | div_mask[k] | div_mask[l]) & (1 << m)):
                            continue
                        antichain_sums[5].setdefault(total, []).append((s1, s2, s3, s4, s5))
    
    # k=6
    for i in range(n_smooth):
        s1 = smooth[i]
        for j in range(i + 1, n_smooth):
            s2 = smooth[j]
            if div_mask[i] & (1 << j):
                continue
            for k in range(j + 1, n_smooth):
                s3 = smooth[k]
                if (div_mask[i] & (1 << k)) or (div_mask[j] & (1 << k)):
                    continue
                for l in range(k + 1, n_smooth):
                    s4 = smooth[l]
                    if ((div_mask[i] | div_mask[j] | div_mask[k]) & (1 << l)):
                        continue
                    for m in range(l + 1, n_smooth):
                        s5 = smooth[m]
                        if ((div_mask[i] | div_mask[j] | div_mask[k] | div_mask[l]) & (1 << m)):
                            continue
                        for n in range(m + 1, n_smooth):
                            s6 = smooth[n]
                            total = s1 + s2 + s3 + s4 + s5 + s6
                            if total > 128:
                                break
                            if ((div_mask[i] | div_mask[j] | div_mask[k] | div_mask[l] | div_mask[m]) & (1 << n)):
                                continue
                            antichain_sums[6].setdefault(total, []).append((s1, s2, s3, s4, s5, s6))
    
    # k=7
    for i in range(n_smooth):
        s1 = smooth[i]
        for j in range(i + 1, n_smooth):
            s2 = smooth[j]
            if div_mask[i] & (1 << j):
                continue
            for k in range(j + 1, n_smooth):
                s3 = smooth[k]
                if (div_mask[i] & (1 << k)) or (div_mask[j] & (1 << k)):
                    continue
                for l in range(k + 1, n_smooth):
                    s4 = smooth[l]
                    if ((div_mask[i] | div_mask[j] | div_mask[k]) & (1 << l)):
                        continue
                    for m in range(l + 1, n_smooth):
                        s5 = smooth[m]
                        if ((div_mask[i] | div_mask[j] | div_mask[k] | div_mask[l]) & (1 << m)):
                            continue
                        for n in range(m + 1, n_smooth):
                            s6 = smooth[n]
                            if ((div_mask[i] | div_mask[j] | div_mask[k] | div_mask[l] | div_mask[m]) & (1 << n)):
                                continue
                            for o in range(n + 1, n_smooth):
                                s7 = smooth[o]
                                total = s1 + s2 + s3 + s4 + s5 + s6 + s7
                                if total > 128:
                                    break
                                if ((div_mask[i] | div_mask[j] | div_mask[k] | div_mask[l] | div_mask[m] | div_mask[n]) & (1 << o)):
                                    continue
                                antichain_sums[7].setdefault(total, []).append((s1, s2, s3, s4, s5, s6, s7))
    
    # k=8
    for i in range(n_smooth):
        s1 = smooth[i]
        for j in range(i + 1, n_smooth):
            s2 = smooth[j]
            if div_mask[i] & (1 << j):
                continue
            for k in range(j + 1, n_smooth):
                s3 = smooth[k]
                if (div_mask[i] & (1 << k)) or (div_mask[j] & (1 << k)):
                    continue
                for l in range(k + 1, n_smooth):
                    s4 = smooth[l]
                    if ((div_mask[i] | div_mask[j] | div_mask[k]) & (1 << l)):
                        continue
                    for m in range(l + 1, n_smooth):
                        s5 = smooth[m]
                        if ((div_mask[i] | div_mask[j] | div_mask[k] | div_mask[l]) & (1 << m)):
                            continue
                        for n in range(m + 1, n_smooth):
                            s6 = smooth[n]
                            if ((div_mask[i] | div_mask[j] | div_mask[k] | div_mask[l] | div_mask[m]) & (1 << n)):
                                continue
                            for o in range(n + 1, n_smooth):
                                s7 = smooth[o]
                                if ((div_mask[i] | div_mask[j] | div_mask[k] | div_mask[l] | div_mask[m] | div_mask[n]) & (1 << o)):
                                    continue
                                for p in range(o + 1, n_smooth):
                                    s8 = smooth[p]
                                    total = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8
                                    if total > 128:
                                        break
                                    if ((div_mask[i] | div_mask[j] | div_mask[k] | div_mask[l] | div_mask[m] | div_mask[n] | div_mask[o]) & (1 << p)):
                                        continue
                                    antichain_sums[8].setdefault(total, []).append((s1, s2, s3, s4, s5, s6, s7, s8))
    
    # Build answer for each target
    answers = []
    for target in range(1, 129):
        if time.time() - t0 > DEADLINE:
            # Fallback: use any valid representation found
            found = False
            for k in range(1, 9):
                if target in antichain_sums[k]:
                    answers.append(list(antichain_sums[k][target][0]))
                    found = True
                    break
            if not found:
                # Last resort
                if is_sm[target]:
                    answers.append([target])
                else:
                    answers.append([1, target - 1])
            continue
        
        # Try increasing number of terms
        found = False
        for k in range(1, 9):
            if target in antichain_sums[k]:
                answers.append(list(antichain_sums[k][target][0]))
                found = True
                break
        
        if not found:
            # Fallback for targets not covered by precomputation
            # Try a greedy approach
            candidates = [s for s in smooth if s <= target]
            found_terms = []
            remaining = target
            for s in sorted(candidates, reverse=True):
                if s <= remaining:
                    # Check antichain property
                    compatible = True
                    for t in found_terms:
                        if s % t == 0 or t % s == 0:
                            compatible = False
                            break
                    if compatible:
                        found_terms.append(s)
                        remaining -= s
                        if remaining == 0:
                            break
            
            if remaining == 0 and found_terms:
                answers.append(found_terms)
            elif is_sm[target]:
                answers.append([target])
            else:
                # Last resort
                answers.append([1, target - 1])
    
    return answers
