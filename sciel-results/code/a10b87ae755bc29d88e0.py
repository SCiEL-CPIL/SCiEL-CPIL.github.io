"""
Divisibility-antichain representations by smooth numbers.
construct() -> 128 rows corresponding to targets 1..128.
Each row: list of 1..30 distinct positive 2^a*3^b*5^c terms summing to target.
No chosen term divides another. Minimize total summands.
"""
import time

_START = time.time()
DEADLINE = 80  # leave 10s margin before 90s kill

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
    
    # Precompute divisibility: divs[i][j] = True if smooth[i] divides smooth[j]
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j:
                a, b = smooth[i], smooth[j]
                if a < b:
                    divs[i][j] = (b % a == 0)
                else:
                    divs[i][j] = (a % b == 0)
    
    # Precompute pair antichain validity
    pair_ok = [[True] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if divs[i][j] or divs[j][i]:
                pair_ok[i][j] = False
                pair_ok[j][i] = False
    
    # Precompute triple antichain validity
    triple_ok = {}
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if not pair_ok[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                if pair_ok[i][k] and pair_ok[j][k]:
                    triple_ok[(i, j, k)] = True
    
    # Precompute quad antichain validity
    quad_ok = {}
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if not pair_ok[i][j]:
                continue
            for k in range(j + 1, n_smooth):
                if not pair_ok[i][k] or not pair_ok[j][k]:
                    continue
                for l in range(k + 1, n_smooth):
                    if pair_ok[i][l] and pair_ok[j][l] and pair_ok[k][l]:
                        quad_ok[(i, j, k, l)] = True
    
    # For 2-term: for each pair (i,j) that is antichain, record sum
    sum_to_pairs = {}
    for i in range(n_smooth):
        for j in range(i + 1, n_smooth):
            if pair_ok[i][j]:
                s = smooth[i] + smooth[j]
                if s <= 128:
                    if s not in sum_to_pairs:
                        sum_to_pairs[s] = []
                    sum_to_pairs[s].append((i, j))
    
    # For 3-term: for each triple that is antichain, record sum
    sum_to_triples = {}
    for (i, j, k) in triple_ok:
        s = smooth[i] + smooth[j] + smooth[k]
        if s <= 128:
            if s not in sum_to_triples:
                sum_to_triples[s] = []
            sum_to_triples[s].append((i, j, k))
    
    # For 4-term
    sum_to_quads = {}
    for (i, j, k, l) in quad_ok:
        s = smooth[i] + smooth[j] + smooth[k] + smooth[l]
        if s <= 128:
            if s not in sum_to_quads:
                sum_to_quads[s] = []
            sum_to_quads[s].append((i, j, k, l))
    
    # Now for each target, pick minimum terms
    answers = []
    for n in range(1, 129):
        if time.time() - t0 > DEADLINE:
            # Fallback: find any valid representation quickly
            if is_sm[n]:
                answers.append([n])
            else:
                # Try 2-term
                found = False
                for i in range(n_smooth):
                    s1 = smooth[i]
                    if s1 >= n:
                        break
                    s2 = n - s1
                    if s2 <= s1:
                        continue
                    if not is_sm[s2]:
                        continue
                    j = smooth.index(s2) if s2 in smooth_set else -1
                    if j >= 0 and pair_ok[i][j]:
                        answers.append([s1, s2])
                        found = True
                        break
                if not found:
                    # Try 3-term
                    found = False
                    for i in range(n_smooth):
                        s1 = smooth[i]
                        if s1 * 3 >= n:
                            break
                        rem1 = n - s1
                        for j in range(i + 1, n_smooth):
                            s2 = smooth[j]
                            if s2 * 2 >= rem1:
                                break
                            s3 = rem1 - s2
                            if s3 <= s2:
                                break
                            if not is_sm[s3]:
                                continue
                            k = smooth.index(s3) if s3 in smooth_set else -1
                            if k >= 0 and pair_ok[i][j] and pair_ok[i][k] and pair_ok[j][k]:
                                answers.append([s1, s2, s3])
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        answers.append([1, n - 1] if n > 1 else [1])
            continue
        
        # Try 1-term
        if is_sm[n]:
            answers.append([n])
            continue
        
        # Try 2-term
        if n in sum_to_pairs:
            i, j = sum_to_pairs[n][0]
            answers.append([smooth[i], smooth[j]])
            continue
        
        # Try 3-term
        if n in sum_to_triples:
            i, j, k = sum_to_triples[n][0]
            answers.append([smooth[i], smooth[j], smooth[k]])
            continue
        
        # Try 4-term
        if n in sum_to_quads:
            i, j, k, l = sum_to_quads[n][0]
            answers.append([smooth[i], smooth[j], smooth[k], smooth[l]])
            continue
        
        # Fallback: search for 5+ terms
        found = False
        for max_terms in range(5, 11):
            if time.time() - t0 > DEADLINE:
                break
            
            def _search(rem, min_idx, chosen, max_t):
                if max_t == 0:
                    return chosen if rem == 0 else None
                if rem <= 0:
                    return None
                # Pruning: minimum possible sum with remaining terms
                if max_t > 0:
                    min_possible = 0
                    for idx in range(min_idx, min(min_idx + max_t, n_smooth)):
                        min_possible += smooth[idx]
                    if min_possible > rem:
                        return None
                
                for idx in range(min_idx, n_smooth):
                    s = smooth[idx]
                    if s > rem:
                        break
                    # Check antichain with all chosen
                    valid = True
                    for c in chosen:
                        if c < s:
                            if s % c == 0:
                                valid = False
                                break
                        else:
                            if c % s == 0:
                                valid = False
                                break
                    if not valid:
                        continue
                    result = _search(rem - s, idx + 1, chosen + [s], max_t - 1)
                    if result is not None:
                        return result
                return None
            
            result = _search(n, 0, [], max_terms)
            if result is not None:
                answers.append(result)
                found = True
                break
        
        if not found:
            # Last resort: try harder with more terms
            for max_terms in range(11, 21):
                if time.time() - t0 > DEADLINE:
                    break
                
                def _search2(rem, min_idx, chosen, max_t):
                    if max_t == 0:
                        return chosen if rem == 0 else None
                    if rem <= 0:
                        return None
                    for idx in range(min_idx, n_smooth):
                        s = smooth[idx]
                        if s > rem:
                            break
                        valid = True
                        for c in chosen:
                            if c < s:
                                if s % c == 0:
                                    valid = False
                                    break
                            else:
                                if c % s == 0:
                                    valid = False
                                    break
                        if not valid:
                            continue
                        result = _search2(rem - s, idx + 1, chosen + [s], max_t - 1)
                        if result is not None:
                            return result
                    return None
                
                result = _search2(n, 0, [], max_terms)
                if result is not None:
                    answers.append(result)
                    found = True
                    break
            
            if not found:
                # Absolute last resort
                if is_sm[n]:
                    answers.append([n])
                else:
                    # Find any valid pair
                    for i in range(n_smooth):
                        s1 = smooth[i]
                        if s1 >= n:
                            break
                        s2 = n - s1
                        if s2 <= s1:
                            continue
                        if not is_sm[s2]:
                            continue
                        j = smooth.index(s2) if s2 in smooth_set else -1
                        if j >= 0 and pair_ok[i][j]:
                            answers.append([s1, s2])
                            found = True
                            break
                    if not found:
                        answers.append([1, n - 1] if n > 1 else [1])
    
    return answers
