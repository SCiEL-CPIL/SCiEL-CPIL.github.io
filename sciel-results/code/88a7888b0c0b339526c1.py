import time
import random
from math import gcd

_DEADLINE = 60.0
_START = None
_CAND = list(range(1, 129))
_PRE = None
_BALL_MASK = None
_GCD_PAIRS = None
_PRIMEP = None
_NUM_DIV = None


def _ensure_pre():
    global _PRE, _BALL_MASK, _PRIMEP, _GCD_PAIRS, _NUM_DIV
    if _PRE is not None:
        return
    n = 128
    PRE = [[1] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            PRE[i][j] = PRE[j][i] = gcd(i, j)
    _PRE = PRE

    # Precompute prime power flags
    flag = [False] * (n + 1)
    for x in range(2, n + 1):
        v = x
        f0 = None
        ok = True
        p = 2
        while p * p <= v:
            if v % p == 0:
                if f0 is None:
                    f0 = p
                else:
                    ok = False
                    break
                while v % p == 0:
                    v //= p
            p += 1
        if ok and (v == 1 or v == f0):
            flag[x] = True
    _PRIMEP = flag

    # Precompute number of divisors
    nd = [0] * (n + 1)
    for x in range(1, n + 1):
        cnt = 0
        for d in range(1, int(x ** 0.5) + 1):
            if x % d == 0:
                cnt += 1
                if d != x // d:
                    cnt += 1
        nd[x] = cnt
    _NUM_DIV = nd

    # Precompute GCD pairs for each g
    GCD_PAIRS = {g: [] for g in range(1, n + 1)}
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = PRE[i][j]
            GCD_PAIRS[g].append((i, j))
    _GCD_PAIRS = GCD_PAIRS

    # Precompute Ball Masks
    BALL_MASK = {}
    for x in range(1, n + 1):
        d = {}
        for y in range(1, n + 1):
            g = PRE[x][y]
            if g not in d:
                d[g] = 0
            d[g] |= (1 << y)
        BALL_MASK[x] = d
    _BALL_MASK = BALL_MASK


def _bad_triples_in(S):
    P = _PRE
    s = list(S)
    n = len(s)
    cnt = 0
    for i in range(n):
        a = s[i]
        for j in range(i + 1, n):
            b = s[j]
            g_ab = P[a][b]
            for k in range(j + 1, n):
                c = s[k]
                if P[a][c] == g_ab and P[b][c] == g_ab:
                    cnt += 1
    return cnt


def _can_add_mask(S_mask, x):
    B = _BALL_MASK[x]
    P = _PRE
    for g, ball_mask in B.items():
        S_in_ball = S_mask & ball_mask
        if S_in_ball == 0 or (S_in_ball & (S_in_ball - 1)) == 0:
            continue
        bits = S_in_ball
        elems = []
        while bits:
            lsb = bits & -bits
            idx = lsb.bit_length() - 1
            elems.append(idx)
            bits ^= lsb
        m = len(elems)
        if m < 2:
            continue
        for i in range(m):
            a = elems[i]
            for j in range(i + 1, m):
                b = elems[j]
                if P[a][b] == g:
                    return False
    return True


def _greedy(order, S_mask=0):
    S = []
    for x in order:
        if _can_add_mask(S_mask, x):
            S.append(x)
            S_mask |= (1 << x)
    return S, S_mask


def _local_add(S, S_mask):
    improved = True
    while improved:
        improved = False
        for x in _CAND:
            if not (S_mask & (1 << x)):
                if _can_add_mask(S_mask, x):
                    S.append(x)
                    S_mask |= (1 << x)
                    improved = True
    return S, S_mask


def _local_search(S, S_mask):
    P = _PRE
    max_rounds = 20
    for _ in range(max_rounds):
        if time.time() - _START > _DEADLINE - 5:
            break
        added = True
        while added:
            added = False
            for x in _CAND:
                if not (S_mask & (1 << x)):
                    if _can_add_mask(S_mask, x):
                        S.append(x)
                        S_mask |= (1 << x)
                        added = True
        if time.time() - _START > _DEADLINE - 5:
            break
        swap_found = False
        S_list = list(S)
        for r in S_list:
            if time.time() - _START > _DEADLINE - 5:
                break
            r_mask = 1 << r
            S_minus_r_mask = S_mask & ~r_mask
            for x in _CAND:
                if x == r or (S_mask & (1 << x)):
                    continue
                x_mask = 1 << x
                if _can_add_mask(S_minus_r_mask, x):
                    new_mask = S_minus_r_mask | x_mask
                    can_add_new = False
                    for y in _CAND:
                        if not (new_mask & (1 << y)):
                            if _can_add_mask(new_mask, y):
                                can_add_new = True
                                break
                    if can_add_new:
                        S = [x for x in S if x != r] + [x]
                        S_mask = new_mask
                        swap_found = True
                        break
            if swap_found:
                break
        if not swap_found:
            break
    return S, S_mask


def _construct():
    global _START
    _ensure_pre()
    _START = time.time()
    PRIMEP = _PRIMEP
    ND = _NUM_DIV

    prime_powers = [x for x in range(1, 129) if PRIMEP[x]]
    others = [x for x in range(1, 129) if not PRIMEP[x]]

    best = []
    best_len = -1
    best_mask = 0

    # Multiple orderings to diversify greedy
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        sorted(range(1, 129), key=lambda x: (0 if PRIMEP[x] else 1, x)),
        prime_powers + others,
        others + prime_powers,
        sorted(range(1, 129), key=lambda x: (-ND[x], x)),  # divisor-count ordering
        sorted(range(1, 129), key=lambda x: (ND[x], x)),
        sorted(range(1, 129), key=lambda x: (x % 3, x)),  # residue-class ordering
    ]

    for order in orders:
        if time.time() - _START > _DEADLINE - 10:
            break
        S, S_mask = _greedy(order)
        S, S_mask = _local_add(S, S_mask)
        if len(S) > best_len:
            best = S
            best_len = len(S)
            best_mask = S_mask

    if time.time() - _START < _DEADLINE - 10:
        S, S_mask = _local_search(best, best_mask)
        if len(S) > best_len:
            best = S
            best_len = len(S)
            best_mask = S_mask

    # Random restarts with different orderings
    rng = random.Random(12345)
    while time.time() - _START < _DEADLINE - 10:
        # Randomly choose a strategy
        strat = rng.random()
        if strat < 0.3:
            order = list(range(1, 129))
            rng.shuffle(order)
        elif strat < 0.5:
            pp = list(prime_powers)
            rest = list(others)
            rng.shuffle(pp)
            rng.shuffle(rest)
            order = pp + rest
        elif strat < 0.7:
            order = sorted(range(1, 129), key=lambda x: (-ND[x], x))
            # Perturb: randomly swap a few
            for _ in range(3):
                i, j = rng.sample(range(128), 2)
                order[i], order[j] = order[j], order[i]
        else:
            order = list(range(1, 129))
            rng.shuffle(order)

        S, S_mask = _greedy(order)
        S, S_mask = _local_add(S, S_mask)
        if len(S) > best_len:
            best = S
            best_len = len(S)
            best_mask = S_mask

        if time.time() - _START < _DEADLINE - 12:
            S, S_mask = _local_search(S, S_mask)
            if len(S) > best_len:
                best = S
                best_len = len(S)
                best_mask = S_mask

    if _bad_triples_in(best) != 0:
        best, _ = _greedy(list(range(1, 129)))
        best, _ = _local_add(best, 0)
    if not best:
        best = [2]
    return best


def construct():
    try:
        return _construct()
    except Exception:
        _ensure_pre()
        S = []
        for x in range(1, 129):
            valid = True
            for a in S:
                for b in S:
                    if a == b:
                        continue
                    if gcd(x, a) == gcd(x, b) == gcd(a, b):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.append(x)
        return S if S else [2]
