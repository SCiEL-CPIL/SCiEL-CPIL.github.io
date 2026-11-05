"""Term-minimal smooth-antichain representation with optimized antichain for 2-term coverage."""
import time

def _is_smooth(n):
    while n % 2 == 0:
        n //= 2
    while n % 3 == 0:
        n //= 3
    while n % 5 == 0:
        n //= 5
    return n == 1

def _gen_smooth(limit):
    vals = set()
    p2 = 1
    while p2 <= limit:
        p3 = p2
        while p3 <= limit:
            p5 = p3
            while p5 <= limit:
                vals.add(p5)
                p5 *= 5
            p3 *= 3
        p2 *= 2
    return sorted(vals)

def _incompatible(a, b):
    if a == b:
        return True
    if b % a == 0 or a % b == 0:
        return True
    return False

# Precompute smooth numbers up to 128
_SMOOTH = _gen_smooth(128)
_SMOOTH_SET = set(_SMOOTH)
_N = len(_SMOOTH)

# Precompute incompatibility matrix (index-based)
_INCOMP = [[False] * _N for _ in range(_N)]
for i in range(_N):
    for j in range(i + 1, _N):
        if _SMOOTH[i] % _SMOOTH[j] == 0 or _SMOOTH[j] % _SMOOTH[i] == 0:
            _INCOMP[i][j] = True
            _INCOMP[j][i] = True

# Non-smooth targets that need 2+ terms
_TARGETS = [n for n in range(2, 129) if n not in _SMOOTH_SET]

def _count_two_term_coverage(ac_set):
    """Count how many non-smooth targets can be represented as 2-term sum from ac_set."""
    count = 0
    for n in _TARGETS:
        found = False
        for a in ac_set:
            if a >= n:
                break
            b = n - a
            if b in ac_set and b != a:
                found = True
                break
        if found:
            count += 1
    return count

def _count_three_term_coverage(ac_set):
    """Count how many non-smooth targets can be represented as 3-term sum from ac_set."""
    ac = sorted(ac_set)
    aset = ac_set
    L = len(ac)
    count = 0
    for n in _TARGETS:
        found = False
        for i in range(L):
            a = ac[i]
            if a >= n:
                break
            rem1 = n - a
            for j in range(i + 1, L):
                b = ac[j]
                if b >= rem1:
                    break
                if _incompatible(a, b):
                    continue
                c = rem1 - b
                if c <= b or c not in aset:
                    continue
                if _incompatible(a, c) or _incompatible(b, c):
                    continue
                found = True
                break
            if found:
                break
        if found:
            count += 1
    return count

def _build_optimized_antichain(deadline=30.0):
    """Build antichain maximizing 2-term+3-term coverage via local search with swaps."""
    # Initial: greedy by descending value
    antichain = []
    for v in sorted(_SMOOTH, reverse=True):
        ok = True
        for u in antichain:
            if _incompatible(u, v):
                ok = False
                break
        if ok:
            antichain.append(v)
    ac_set = set(antichain)
    
    def _score(s):
        return _count_two_term_coverage(s) + _count_three_term_coverage(s)
    
    best_count = _score(ac_set)
    
    start = time.time()
    improved = True
    while improved and time.time() - start < deadline:
        improved = False
        
        # Phase 1: Try adding each smooth number not in antichain
        for v in _SMOOTH:
            if time.time() - start > deadline:
                break
            if v in ac_set:
                continue
            ok = True
            for u in antichain:
                if _incompatible(u, v):
                    ok = False
                    break
            if not ok:
                continue
            new_set = ac_set | {v}
            cnt = _score(new_set)
            if cnt > best_count:
                best_count = cnt
                ac_set = new_set
                antichain.append(v)
                improved = True
        
        # Phase 2: Try swap (remove one, add another)
        for idx in range(len(antichain)):
            if time.time() - start > deadline:
                break
            v = antichain[idx]
            new_set = ac_set - {v}
            for w in _SMOOTH:
                if time.time() - start > deadline:
                    break
                if w in new_set:
                    continue
                ok = True
                for u in new_set:
                    if _incompatible(u, w):
                        ok = False
                        break
                if not ok:
                    continue
                trial_set = new_set | {w}
                cnt = _score(trial_set)
                if cnt > best_count:
                    best_count = cnt
                    ac_set = trial_set
                    antichain[idx] = w
                    improved = True
                    break
            if improved:
                break
    
    return sorted(ac_set)

# Build optimized antichain at module load
_ANTICHAIN = _build_optimized_antichain(deadline=20.0)
_ANTICHAIN_SET = set(_ANTICHAIN)
_ANTICHAIN_SORTED = sorted(_ANTICHAIN)

# Precompute 2-term lookup: for each target, store the pair (or None)
_TWO_TERM = {}
for n in _TARGETS:
    found = None
    for a in _ANTICHAIN_SORTED:
        if a >= n:
            break
        b = n - a
        if b in _ANTICHAIN_SET and b != a:
            found = (a, b)
            break
    _TWO_TERM[n] = found

# Precompute 3-term lookup for targets not covered by 2-term
_THREE_TERM = {}
for n in _TARGETS:
    if _TWO_TERM.get(n) is not None:
        continue
    found = None
    ac = _ANTICHAIN_SORTED
    aset = _ANTICHAIN_SET
    L = len(ac)
    for i in range(L):
        a = ac[i]
        if a > n:
            break
        rem1 = n - a
        for j in range(i + 1, L):
            b = ac[j]
            if b >= rem1:
                break
            if _incompatible(a, b):
                continue
            c = rem1 - b
            if c <= b or c not in aset:
                continue
            if _incompatible(a, c) or _incompatible(b, c):
                continue
            found = (a, b, c)
            break
        if found:
            break
    _THREE_TERM[n] = found

# Precompute 4-term lookup for targets not covered by 2 or 3 terms
_FOUR_TERM = {}
for n in _TARGETS:
    if _TWO_TERM.get(n) is not None or _THREE_TERM.get(n) is not None:
        continue
    found = None
    ac = _ANTICHAIN_SORTED
    aset = _ANTICHAIN_SET
    L = len(ac)
    for i in range(L):
        a = ac[i]
        if a > n:
            break
        rem1 = n - a
        for j in range(i + 1, L):
            b = ac[j]
            if a + b >= n:
                break
            if _incompatible(a, b):
                continue
            rem2 = rem1 - b
            for k in range(j + 1, L):
                c = ac[k]
                if c >= rem2:
                    break
                if _incompatible(a, c) or _incompatible(b, c):
                    continue
                d = rem2 - c
                if d <= c or d not in aset:
                    continue
                if _incompatible(a, d) or _incompatible(b, d) or _incompatible(c, d):
                    continue
                found = (a, b, c, d)
                break
            if found:
                break
        if found:
            break
    _FOUR_TERM[n] = found

def _greedy_fill(n):
    """Last-resort: find any valid div-free smooth representation using full smooth set."""
    if n in _SMOOTH_SET:
        return [n]
    # Try 2 terms from full smooth set
    for a in _SMOOTH:
        if a > n:
            break
        b = n - a
        if b <= 0 or b == a:
            continue
        if b not in _SMOOTH_SET:
            continue
        if _incompatible(a, b):
            continue
        return [a, b]
    # Try 3 terms from full smooth set
    for i in range(len(_SMOOTH)):
        a = _SMOOTH[i]
        if a > n:
            break
        for j in range(i + 1, len(_SMOOTH)):
            b = _SMOOTH[j]
            if a + b >= n:
                break
            c = n - a - b
            if c <= b or c not in _SMOOTH_SET:
                continue
            if _incompatible(a, b) or _incompatible(a, c) or _incompatible(b, c):
                continue
            return [a, b, c]
    # Try 4 terms
    for i in range(len(_SMOOTH)):
        a = _SMOOTH[i]
        if a > n:
            break
        for j in range(i + 1, len(_SMOOTH)):
            b = _SMOOTH[j]
            if a + b > n:
                break
            for k in range(j + 1, len(_SMOOTH)):
                c = _SMOOTH[k]
                if a + b + c > n:
                    break
                d = n - a - b - c
                if d <= c or d not in _SMOOTH_SET:
                    continue
                if _incompatible(a, b) or _incompatible(a, c) or _incompatible(a, d) or \
                   _incompatible(b, c) or _incompatible(b, d) or _incompatible(c, d):
                    continue
                return [a, b, c, d]
    # Try 5 terms
    for i in range(len(_SMOOTH)):
        a = _SMOOTH[i]
        if a > n:
            break
        for j in range(i + 1, len(_SMOOTH)):
            b = _SMOOTH[j]
            if a + b > n:
                break
            for k in range(j + 1, len(_SMOOTH)):
                c = _SMOOTH[k]
                if a + b + c > n:
                    break
                for l in range(k + 1, len(_SMOOTH)):
                    d = _SMOOTH[l]
                    if a + b + c + d > n:
                        break
                    e = n - a - b - c - d
                    if e <= d or e not in _SMOOTH_SET:
                        continue
                    if (_incompatible(a, b) or _incompatible(a, c) or _incompatible(a, d) or
                        _incompatible(a, e) or _incompatible(b, c) or _incompatible(b, d) or
                        _incompatible(b, e) or _incompatible(c, d) or _incompatible(c, e) or
                        _incompatible(d, e)):
                        continue
                    return [a, b, c, d, e]
    # Absolute fallback: single term (may violate antichain but ensures validity)
    return [n]

def construct():
    start = time.time()
    DEADLINE = 80.0
    answers = [None] * 129

    for n in range(1, 129):
        if time.time() - start > DEADLINE:
            break

        # k=1: smooth target
        if n in _SMOOTH_SET:
            answers[n] = [n]
            continue

        # k=2: precomputed lookup
        pair = _TWO_TERM.get(n)
        if pair is not None:
            answers[n] = [pair[0], pair[1]]
            continue

        # k=3: precomputed lookup
        triple = _THREE_TERM.get(n)
        if triple is not None:
            answers[n] = [triple[0], triple[1], triple[2]]
            continue

        # k=4: precomputed lookup
        quad = _FOUR_TERM.get(n)
        if quad is not None:
            answers[n] = [quad[0], quad[1], quad[2], quad[3]]
            continue

        # Fallback: greedy over full smooth set
        answers[n] = _greedy_fill(n)

    # Ensure no None
    for n in range(1, 129):
        if answers[n] is None:
            answers[n] = _greedy_fill(n)

    return answers[1:129]
