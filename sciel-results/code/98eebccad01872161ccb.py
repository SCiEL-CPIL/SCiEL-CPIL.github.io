import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    max_target = 128
    
    # Generate all 2,3,5-smooth numbers up to max_target
    smooth = []
    smooth_set = set()
    for a in range(0, 8):
        p2 = 2 ** a
        if p2 > max_target:
            break
        for b in range(0, 6):
            p3 = 3 ** b
            if p2 * p3 > max_target:
                break
            for c in range(0, 5):
                p5 = 5 ** c
                v = p2 * p3 * p5
                if v > max_target:
                    break
                if v not in smooth_set:
                    smooth_set.add(v)
                    smooth.append(v)
    smooth.sort()
    n_smooth = len(smooth)
    
    # Map value to index for quick lookup
    is_smooth_arr = [False] * (max_target + 1)
    for v in smooth:
        is_smooth_arr[v] = True
    
    # Precompute divisibility matrix for faster antichain checks
    # divs[i][j] is True if smooth[i] divides smooth[j]
    divs = [[False] * n_smooth for _ in range(n_smooth)]
    for i in range(n_smooth):
        for j in range(n_smooth):
            if i != j and smooth[j] % smooth[i] == 0:
                divs[i][j] = True
    
    def is_antichain_indices(idxs):
        """Check if list of indices forms a divisibility antichain."""
        l = len(idxs)
        for i in range(l):
            for j in range(i + 1, l):
                if divs[idxs[i]][idxs[j]] or divs[idxs[j]][idxs[i]]:
                    return False
        return True
    
    def is_antichain(vals):
        """Check if list of values forms a divisibility antichain."""
        l = len(vals)
        for i in range(l):
            vi = vals[i]
            for j in range(i + 1, l):
                vj = vals[j]
                if vi < vj:
                    if vj % vi == 0:
                        return False
                else:
                    if vi % vj == 0:
                        return False
        return True
    
    answers = [None] * (max_target + 1)
    
    # For each target n (1..128), find minimum number of terms
    for n in range(1, max_target + 1):
        if time.time() > deadline:
            break
        
        # Try count = 1
        if is_smooth_arr[n]:
            answers[n] = [n]
            continue
        
        # Try count = 2
        found = False
        for a in smooth:
            if a * 2 >= n:  # a < b implies 2a < n
                break
            b = n - a
            if not is_smooth_arr[b]:
                continue
            if b % a == 0:
                continue
            answers[n] = [a, b]
            found = True
            break
        if found:
            continue
        
        # Try count = 3
        found = False
        for i in range(n_smooth):
            a = smooth[i]
            if a * 3 >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if a + b >= n:
                    break
                c = n - a - b
                if c <= b:
                    continue
                if c > max_target:
                    continue
                if not is_smooth_arr[c]:
                    continue
                # Check antichain for {a, b, c} where a < b < c
                if b % a == 0:
                    continue
                if c % a == 0:
                    continue
                if c % b == 0:
                    continue
                answers[n] = [a, b, c]
                found = True
                break
            if found:
                break
        if found:
            continue
        
        # Try count = 4
        found = False
        for i in range(n_smooth):
            a = smooth[i]
            if a * 4 >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if a + 3 * b + 3 >= n:
                    break
                for k in range(j + 1, n_smooth):
                    c = smooth[k]
                    if a + b + c >= n:
                        break
                    d = n - a - b - c
                    if d <= c:
                        continue
                    if d > max_target:
                        continue
                    if not is_smooth_arr[d]:
                        continue
                    # Check antichain {a, b, c, d} where a < b < c < d
                    if b % a == 0:
                        continue
                    if c % a == 0:
                        continue
                    if d % a == 0:
                        continue
                    if c % b == 0:
                        continue
                    if d % b == 0:
                        continue
                    if d % c == 0:
                        continue
                    answers[n] = [a, b, c, d]
                    found = True
                    break
                if found:
                    break
            if found:
                break
        if found:
            continue
        
        # Try count = 5
        found = False
        for i in range(n_smooth):
            a = smooth[i]
            if a * 5 >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if a + 4 * b >= n:
                    break
                for k in range(j + 1, n_smooth):
                    c = smooth[k]
                    if a + b + 3 * c >= n:
                        break
                    for l in range(k + 1, n_smooth):
                        d = smooth[l]
                        if a + b + c + d >= n:
                            break
                        e = n - a - b - c - d
                        if e <= d:
                            continue
                        if e > max_target:
                            continue
                        if not is_smooth_arr[e]:
                            continue
                        # Check antichain {a, b, c, d, e} where a < b < c < d < e
                        if b % a == 0:
                            continue
                        if c % a == 0:
                            continue
                        if d % a == 0:
                            continue
                        if e % a == 0:
                            continue
                        if c % b == 0:
                            continue
                        if d % b == 0:
                            continue
                        if e % b == 0:
                            continue
                        if d % c == 0:
                            continue
                        if e % c == 0:
                            continue
                        if e % d == 0:
                            continue
                        answers[n] = [a, b, c, d, e]
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            continue
        
        # Try count = 6
        found = False
        for i in range(n_smooth):
            a = smooth[i]
            if a * 6 >= n:
                break
            for j in range(i + 1, n_smooth):
                b = smooth[j]
                if a + 5 * b >= n:
                    break
                for k in range(j + 1, n_smooth):
                    c = smooth[k]
                    if a + b + 4 * c >= n:
                        break
                    for l in range(k + 1, n_smooth):
                        d = smooth[l]
                        if a + b + c + 3 * d >= n:
                            break
                        for m in range(l + 1, n_smooth):
                            e = smooth[m]
                            if a + b + c + d + e >= n:
                                break
                            f = n - a - b - c - d - e
                            if f <= e:
                                continue
                            if f > max_target:
                                continue
                            if not is_smooth_arr[f]:
                                continue
                            # Check antichain {a, b, c, d, e, f}
                            if b % a == 0:
                                continue
                            if c % a == 0:
                                continue
                            if d % a == 0:
                                continue
                            if e % a == 0:
                                continue
                            if f % a == 0:
                                continue
                            if c % b == 0:
                                continue
                            if d % b == 0:
                                continue
                            if e % b == 0:
                                continue
                            if f % b == 0:
                                continue
                            if d % c == 0:
                                continue
                            if e % c == 0:
                                continue
                            if f % c == 0:
                                continue
                            if e % d == 0:
                                continue
                            if f % d == 0:
                                continue
                            if f % e == 0:
                                continue
                            answers[n] = [a, b, c, d, e, f]
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
            continue
        
        # Fallback: use exhaustive search for higher counts
        candidates = [s for s in smooth if s <= n]
        for cnt in range(7, 31):
            if time.time() > deadline:
                break
            for combo in combinations(candidates, cnt):
                if sum(combo) == n:
                    if is_antichain(list(combo)):
                        answers[n] = list(combo)
                        break
            if answers[n] is not None:
                break
    
    # Verify all answers are found
    for n in range(1, max_target + 1):
        if answers[n] is None:
            # Last resort: use n itself if smooth, otherwise find any valid representation
            candidates = [s for s in smooth if s <= n]
            for cnt in range(1, 31):
                for combo in combinations(candidates, cnt):
                    if sum(combo) == n:
                        if is_antichain(list(combo)):
                            answers[n] = list(combo)
                            break
                if answers[n] is not None:
                    break
    
    return answers[1:]
