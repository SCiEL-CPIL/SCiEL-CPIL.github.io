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
    smooth = []
    smooth_set = set()
    for a in range(0, 8):
        for b in range(0, 6):
            for c in range(0, 5):
                v = (2 ** a) * (3 ** b) * (5 ** c)
                if v >= 1 and v <= 128:
                    if v not in smooth_set:
                        smooth_set.add(v)
                        smooth.append(v)
    smooth.sort()
    n_smooth = len(smooth)
    
    # Precompute smooth lookup
    is_sm = [False] * 129
    for s in smooth:
        is_sm[s] = True
    
    # Precompute "divides" relation
    # divides[i][j] = True if smooth[i] divides smooth[j]
    divides = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                divides[i][j] = True
    
    # Map value -> index in smooth list
    val_to_idx = {s: i for i, s in enumerate(smooth)}
    
    def check_antichain_by_val(terms):
        """Check antichain for a list of values (assumed smooth and distinct)."""
        L = len(terms)
        for a in range(L):
            for b in range(a + 1, L):
                ta, tb = terms[a], terms[b]
                if ta < tb:
                    if tb % ta == 0:
                        return False
                else:
                    if ta % tb == 0:
                        return False
        return True
    
    answers = [None] * 128
    
    # First pass: find minimal term count for each target
    # We'll try 1, 2, 3, 4 terms in order and stop at first valid
    
    for n in range(1, 129):
        if time.time() - t0 > DEADLINE:
            # Fallback: use a valid representation
            if is_sm[n]:
                answers[n-1] = [n]
            else:
                # Find any valid representation quickly
                found = False
                for s1 in smooth:
                    if s1 >= n:
                        break
                    s2 = n - s1
                    if s2 <= s1 or not is_sm[s2]:
                        continue
                    if s2 % s1 == 0:
                        continue
                    answers[n-1] = [s1, s2]
                    found = True
                    break
                if not found:
                    answers[n-1] = [1, n-1] if not is_sm[1] and not is_sm[n-1] else ([n] if is_sm[n] else [1, n-1])
            continue
        
        best = None
        best_count = 99
        
        # Try 1-term
        if is_sm[n]:
            best = [n]
            best_count = 1
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
                if s2 % s1 == 0:
                    continue
                best = [s1, s2]
                best_count = 2
                found = True
                break
            if found:
                answers[n-1] = best
                continue
            
            # Try 3-term
            found = False
            for i in range(n_smooth):
                s1 = smooth[i]
                if s1 >= n:
                    break
                for j in range(n_smooth):
                    if time.time() - t0 > DEADLINE:
                        break
                    s2 = smooth[j]
                    if s2 <= s1:
                        continue
                    if s1 + s2 >= n:
                        break
                    s3 = n - s1 - s2
                    if s3 <= s2:
                        break
                    if not is_sm[s3]:
                        continue
                    # Check antichain
                    if s2 % s1 == 0 or s3 % s1 == 0 or s3 % s2 == 0:
                        continue
                    best = [s1, s2, s3]
                    best_count = 3
                    found = True
                    break
                if found:
                    break
            if found:
                answers[n-1] = best
                continue
            
            # Try 4-term
            found = False
            for i in range(n_smooth):
                s1 = smooth[i]
                if s1 >= n:
                    break
                for j in range(n_smooth):
                    if time.time() - t0 > DEADLINE:
                        break
                    s2 = smooth[j]
                    if s2 <= s1:
                        continue
                    if s1 + s2 >= n:
                        break
                    for k in range(n_smooth):
                        s3 = smooth[k]
                        if s3 <= s2:
                            continue
                        if s1 + s2 + s3 >= n:
                            break
                        s4 = n - s1 - s2 - s3
                        if s4 <= s3:
                            break
                        if not is_sm[s4]:
                            continue
                        terms4 = [s1, s2, s3, s4]
                        if check_antichain_by_val(terms4):
                            best = terms4
                            best_count = 4
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                answers[n-1] = best
                continue
            
            # Try 5-term (safety net)
            found = False
            for i in range(n_smooth):
                s1 = smooth[i]
                if s1 >= n:
                    break
                for j in range(n_smooth):
                    if time.time() - t0 > DEADLINE:
                        break
                    s2 = smooth[j]
                    if s2 <= s1:
                        continue
                    if s1 + s2 >= n:
                        break
                    for k in range(n_smooth):
                        s3 = smooth[k]
                        if s3 <= s2:
                            continue
                        if s1 + s2 + s3 >= n:
                            break
                        for l in range(n_smooth):
                            if time.time() - t0 > DEADLINE:
                                break
                            s4 = smooth[l]
                            if s4 <= s3:
                                continue
                            if s1 + s2 + s3 + s4 >= n:
                                break
                            rem = n - s1 - s2 - s3 - s4
                            if rem <= s4:
                                break
                            if is_sm[rem]:
                                terms = [s1, s2, s3, s4, rem]
                                if check_antichain_by_val(terms):
                                    best = terms
                                    best_count = 5
                                    found = True
                                    break
                            # Try rem as two terms
                            for m in range(n_smooth):
                                s5 = smooth[m]
                                if s5 <= s4:
                                    continue
                                s6 = rem - s5
                                if s6 <= s5:
                                    break
                                if not is_sm[s6]:
                                    continue
                                terms = [s1, s2, s3, s4, s5, s6]
                                if check_antichain_by_val(terms):
                                    best = terms
                                    best_count = 6
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                answers[n-1] = best
                continue
            
            # Ultimate fallback
            answers[n-1] = [n] if is_sm[n] else [1, n-1]
    
    return answers
