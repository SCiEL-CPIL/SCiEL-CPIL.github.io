import time
import random
from math import gcd
from itertools import combinations

_DEADLINE = 60.0
_START = None
_CAND = list(range(1, 129))
_PRE = None
_BALL = None
_PAIRS_GCD = None  # PAIRS_GCD[g] = list of (a,b) with gcd(a,b)=g
_GCD1 = None

def _ensure_pre():
    global _PRE, _BALL, _PAIRS_GCD, _GCD1
    if _PRE is not None:
        return
    n = 128
    PRE = [[1] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            PRE[i][j] = PRE[j][i] = gcd(i, j)
    _PRE = PRE

    BALL = {}
    for x in range(1, n + 1):
        d = {}
        for y in range(1, n + 1):
            g = PRE[x][y]
            d.setdefault(g, []).append(y)
        BALL[x] = d
    _BALL = BALL
    _GCD1 = {x: BALL[x].get(1, []) for x in range(1, n + 1)}

    # Precompute pairs with gcd g
    PAIRS_GCD = {}
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g = PRE[a][b]
            if g not in PAIRS_GCD:
                PAIRS_GCD[g] = []
            PAIRS_GCD[g].append((a, b))
    _PAIRS_GCD = PAIRS_GCD

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

def _can_add(Sset, x):
    """
    Check if x can be added to Sset.
    Sset is a set.
    """
    P = _PRE
    # Group elements of Sset by gcd with x
    groups = {}
    for u in Sset:
        g = P[x][u]
        if g not in groups:
            groups[g] = []
        groups[g].append(u)
    
    for g, lst in groups.items():
        if len(lst) < 2:
            continue
        # Check if any pair in lst has gcd g
        # Use precomputed pairs for this g
        pairs = _PAIRS_GCD.get(g, [])
        # Create a set of lst for O(1) lookup
        lst_set = set(lst)
        for a, b in pairs:
            if a in lst_set and b in lst_set:
                return False
    return True

def _greedy(order):
    S = []
    Sset = set()
    for x in order:
        if _can_add(Sset, x):
            S.append(x)
            Sset.add(x)
    return S

def _local_search(S):
    P = _PRE
    S = list(S)
    Sset = set(S)
    max_rounds = 100
    for _ in range(max_rounds):
        if time.time() - _START > _DEADLINE - 5:
            break
        # Try adds
        added = True
        while added:
            added = False
            for x in _CAND:
                if x not in Sset and _can_add(Sset, x):
                    S.append(x)
                    Sset.add(x)
                    added = True
        if time.time() - _START > _DEADLINE - 5:
            break
        # Try swaps
        improved = False
        for r in list(S):
            if time.time() - _START > _DEADLINE - 5:
                break
            Sset.discard(r)
            for x in _CAND:
                if x in Sset or x == r:
                    continue
                if _can_add(Sset, x):
                    # Swap
                    S.remove(r)
                    S.append(x)
                    Sset.add(x)
                    improved = True
                    break
            if improved:
                break
            else:
                Sset.add(r)
        if not improved:
            break
    return S

def _construct():
    global _START
    _ensure_pre()
    _START = time.time()
    
    best = []
    best_len = -1
    
    # Deterministic greedy
    S = _greedy(list(range(1, 129)))
    S = _local_search(S)
    if len(S) > best_len:
        best = S
        best_len = len(S)
        
    # Random restarts
    rng = random.Random(42)
    while time.time() - _START < _DEADLINE - 10:
        order = list(range(1, 129))
        rng.shuffle(order)
        S = _greedy(order)
        S = _local_search(S)
        if len(S) > best_len:
            best = S
            best_len = len(S)
            
    return best

def construct():
    try:
        res = _construct()
    except Exception:
        res = [2]
    if not res:
        res = [2]
    return res
