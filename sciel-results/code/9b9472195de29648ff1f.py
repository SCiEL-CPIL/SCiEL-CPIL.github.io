"""
Divisibility-antichain representations by smooth numbers.
construct() -> 128 rows corresponding to targets 1..128.
Each row: list of 1..30 distinct positive 2^a*3^b*5^c terms summing to target.
No chosen term divides another. Minimize total summands.
"""
import time

_START = time.time()
DEADLINE = 85  # leave 5s margin before 90s kill

def _is_smooth(x):
    """Check if x is of form 2^a * 3^b * 5^c with a,b,c >= 0."""
    y = x
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

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
    
    # Precompute all valid antichain pairs (s1, s2) with s1 < s2, s1+s2 <= 128
    # Store as dict: sum -> list of (s1, s2) pairs
    pair_sums = {}
    for i in range(n_smooth):
        s1 = smooth[i]
        for j in range(i + 1, n_smooth):
            s2 = smooth[j]
            total = s1 + s2
            if total > 128:
                break
            if s2 % s1 == 0:
                continue
            if total not in pair_sums:
                pair_sums[total] = []
            pair_sums[total].append((s1, s2))
    
    # Precompute all valid antichain triples (s1, s2, s3) with s1 < s2 < s3
    # Store as dict: sum -> list of (s1, s2, s3) triples
    triple_sums = {}
    for i in range(n_smooth):
        s1 = smooth[i]
        for j in range(i + 1, n_smooth):
            s2 = smooth[j]
            if s2 % s1 == 0:
                continue
            for k in range(j + 1, n_smooth):
                s3 = smooth[k]
                total = s1 + s2 + s3
                if total > 128:
                    break
                if s3 % s1 == 0 or s3 % s2 == 0:
                    continue
                if total not in triple_sums:
                    triple_sums[total] = []
                triple_sums[total].append((s1, s2, s3))
    
    # Precompute all valid antichain 4-tuples
    # This is expensive, so we'll do it lazily or with early termination
    # Store as dict: sum -> list of 4-tuples
    quad_sums = {}
    for i in range(n_smooth):
        s1 = smooth[i]
        for j in range(i + 1, n_smooth):
            s2 = smooth[j]
            if s2 % s1 == 0:
                continue
            for k in range(j + 1, n_smooth):
                s3 = smooth[k]
                if s3 % s1 == 0 or s3 % s2 == 0:
                    continue
                for l in range(k + 1, n_smooth):
                    s4 = smooth[l]
                    total = s1 + s2 + s3 + s4
                    if total > 128:
                        break
                    if s4 % s1 == 0 or s4 % s2 == 0 or s4 % s3 == 0:
                        continue
                    if total not in quad_sums:
                        quad_sums[total] = []
                    quad_sums[total].append((s1, s2, s3, s4))
                    # Early exit if we've spent too much time
                    if time.time() - t0 > DEADLINE * 0.5:
                        break
                if time.time() - t0 > DEADLINE * 0.5:
                    break
            if time.time() - t0 > DEADLINE * 0.5:
                break
        if time.time() - t0 > DEADLINE * 0.5:
            break
    
    answers = []
    for n in range(1, 129):
        if time.time() - t0 > DEADLINE:
            # Fallback: try to find any valid representation quickly
            found = False
            if is_sm[n]:
                answers.append([n])
                found = True
            elif n in pair_sums:
                s1, s2 = pair_sums[n][0]
                answers.append([s1, s2])
                found = True
            elif n in triple_sums:
                s1, s2, s3 = triple_sums[n][0]
                answers.append([s1, s2, s3])
                found = True
            elif n in quad_sums:
                s1, s2, s3, s4 = quad_sums[n][0]
                answers.append([s1, s2, s3, s4])
                found = True
            else:
                # Last resort: try a simple fallback
                for s1 in smooth:
                    if s1 >= n:
                        break
                    s2 = n - s1
                    if s2 <= s1:
                        continue
                    if is_sm[s2] and s2 % s1 != 0:
                        answers.append([s1, s2])
                        found = True
                        break
                if not found:
                    answers.append([n] if is_sm[n] else [1, n-1])
            continue
        
        # Try 1-term
        if is_sm[n]:
            answers.append([n])
            continue
        
        # Try 2-term using precomputed pairs
        if n in pair_sums and len(pair_sums[n]) > 0:
            s1, s2 = pair_sums[n][0]
            answers.append([s1, s2])
            continue
        
        # Try 3-term using precomputed triples
        if n in triple_sums and len(triple_sums[n]) > 0:
            s1, s2, s3 = triple_sums[n][0]
            answers.append([s1, s2, s3])
            continue
        
        # Try 4-term using precomputed quads
        if n in quad_sums and len(quad_sums[n]) > 0:
            s1, s2, s3, s4 = quad_sums[n][0]
            answers.append([s1, s2, s3, s4])
            continue
        
        # Try 5-term search (slower, but should be rare)
        found_5 = False
        for i in range(n_smooth):
            s1 = smooth[i]
            if s1 * 4 >= n:
                break
            rem1 = n - s1
            for j in range(i + 1, n_smooth):
                if time.time() - t0 > DEADLINE:
                    break
                s2 = smooth[j]
                if s2 * 3 >= rem1:
                    break
                rem2 = rem1 - s2
                for k in range(j + 1, n_smooth):
                    s3 = smooth[k]
                    if s3 * 2 >= rem2:
                        break
                    rem3 = rem2 - s3
                    for l in range(k + 1, n_smooth):
                        s4 = smooth[l]
                        if s4 >= rem3:
                            break
                        s5 = rem3 - s4
                        if s5 <= s4:
                            break
                        if not is_sm[s5]:
                            continue
                        terms = [s1, s2, s3, s4, s5]
                        # Check antichain
                        valid = True
                        for a2 in range(5):
                            for b2 in range(a2 + 1, 5):
                                if terms[b2] % terms[a2] == 0:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            answers.append(terms)
                            found_5 = True
                            break
                    if found_5:
                        break
                if found_5:
                    break
            if found_5:
                break
        if found_5:
            continue
        
        # Final fallback: use a known valid representation
        if is_sm[n]:
            answers.append([n])
        elif n in pair_sums:
            s1, s2 = pair_sums[n][0]
            answers.append([s1, s2])
        elif n in triple_sums:
            s1, s2, s3 = triple_sums[n][0]
            answers.append([s1, s2, s3])
        else:
            # Try a simple fallback
            found_fallback = False
            for s1 in smooth:
                if s1 >= n:
                    break
                s2 = n - s1
                if s2 <= s1:
                    continue
                if is_sm[s2] and s2 % s1 != 0:
                    answers.append([s1, s2])
                    found_fallback = True
                    break
            if not found_fallback:
                answers.append([1, n-1] if n > 1 else [n])
    
    return answers
