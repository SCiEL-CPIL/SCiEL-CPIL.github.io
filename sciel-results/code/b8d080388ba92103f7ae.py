import time

_DEADLINE = 80.0
_T0 = time.time()
_P = list(range(2, 129))          # candidate values, increasing
_N = len(_P)                       # 127
_PRIME = [0] * 129
for _i in range(2, 129):
    _PRIME[_i] = _i
for _i in range(2, 129):
    if _PRIME[_i] == _i:
        for _j in range(_i * _i, 129, _i):
            if _PRIME[_j] == _j:
                _PRIME[_j] = _i
_SMALL = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
_SIDX = {p: i for i, p in enumerate(_SMALL)}
_SLEN = len(_SMALL)


def _sig(x):
    v = [0] * _SLEN
    while x > 1:
        p = _PRIME[x]
        c = 0
        while x % p == 0:
            x //= p
            c += 1
        v[_SIDX[p]] = c
    return v


_SIG = {x: _sig(x) for x in _P}


def _sig_mul(a, b):
    r = [0] * _SLEN
    for i in range(_SLEN):
        r[i] = a[i] + b[i]
    return r


def _sig_eq(a, b):
    for i in range(_SLEN):
        if a[i] != b[i]:
            return False
    return True


def _sig_key(a):
    return tuple(a)


def _sig_le(a, b):
    for i in range(_SLEN):
        if a[i] > b[i]:
            return False
    return True


def _sig_div(a, b):
    for i in range(_SLEN):
        if a[i] < b[i]:
            return False
    return True


def _greedy_forward():
    values, products = [], set()
    for x in range(2, 129):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values


def _greedy_backward():
    values, products = [], set()
    for x in range(128, 1, -1):
        tails, prod = [x], x
        for a in reversed(values):
            prod *= a
            tails.append(prod)
        if not products.intersection(tails):
            values.append(x)
            products.update(tails)
    return values


def _upper_bound(mask, min_len, max_len):
    # count distinct tail-product signatures for each prefix length
    cnt = [0] * (max_len + 1)
    cnt[0] = 1
    for L in range(1, max_len + 1):
        s = set()
        for i in range(_N - L + 1):
            if not (mask >> i) & 1:
                continue
            sig = _SIG[_P[i]]
            for j in range(i + 1, _N):
                if (mask >> j) & 1:
                    sig = _sig_mul(sig, _SIG[_P[j]])
            s.add(_sig_key(sig))
        cnt[L] = len(s)
    for L in range(min_len, max_len + 1):
        if cnt[L] < (max_len - L + 1):
            return False
    return True


def _search():
    best = {"mask": 0, "len": 0}

    def dfs(mask, min_len, max_len, cur_len, cur_max, tail_sigs, prod_set, tails, prefix_sigs):
        if time.time() > _DEADLINE:
            return
        if cur_len + (_N - cur_max - 1) <= best["len"]:
            return
        if cur_len > best["len"]:
            best["mask"] = mask
            best["len"] = cur_len
        if cur_len == max_len:
            return
        if not _upper_bound(mask, min_len, max_len):
            return
        for i in range(cur_max + 1, _N):
            if not (mask >> i) & 1:
                continue
            s = _SIG[_P[i]]
            coll = False
            for t in tails:
                if _sig_eq(_sig_mul(s, t), s):
                    pass
                if _sig_key(_sig_mul(s, t)) in prod_set:
                    coll = True
                    break
            if _sig_key(s) in prod_set:
                coll = True
            if coll:
                continue
            new_tails = [_sig_mul(s, t) for t in tails] + [s]
            new_prod = prod_set | {_sig_key(t) for t in new_tails}
            dfs(mask | (1 << i), min_len, max_len, cur_len + 1, i, new_tails, new_prod, new_tails,
                prefix_sigs)
            if time.time() > _DEADLINE:
                return

    # iterative deepening from a good start
    start = len(_greedy_backward())
    for L in range(start, _N + 1):
        if time.time() > _DEADLINE:
            break
        dfs(0, 1, L, 0, -1, [], set(), [], [])
    return best["mask"]


def _mask_to_list(mask):
    return [_P[i] for i in range(_N) if (mask >> i) & 1]


def construct():
    gf = _greedy_forward()
    gb = _greedy_backward()
    best_mask, best_len = 0, 0
    for lst in (gf, gb):
        m = 0
        for x in lst:
            m |= 1 << _P.index(x)
        if len(lst) > best_len:
            best_mask, best_len = m, len(lst)
    try:
        m = _search()
        if m:
            lst = _mask_to_list(m)
            if len(lst) > best_len:
                best_mask, best_len = m, len(lst)
    except Exception:
        pass
    return _mask_to_list(best_mask) if best_mask else gf
