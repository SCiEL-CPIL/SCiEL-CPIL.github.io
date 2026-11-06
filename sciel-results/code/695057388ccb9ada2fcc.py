"""Equal-gcd-triple-free subsets of [1,128].

Structural change: replace the generic bit-mask simulated annealing with a
constraint-satisfaction / conflict-driven local search.

Key structural ideas:
  * Fast incremental violation counting via per-element "conflict partners" so that
    adding/removing a single element is O(degree) rather than O(|S|^2).
  * Greedy construction with a principled ordering (fewest divisors first, i.e. primes
    and prime powers first) because those elements participate in the fewest
    equal-gcd triples.
  * Conflict-based repair: when the set has violations, repeatedly remove the element
    that participates in the most violations, then greedily re-add.
  * Randomized restarts with different orderings and repair strategies to escape
    local optima.
  * Final polish: try every single-element removal + re-add to see if a larger valid
    set is reachable.
"""
import time
import math
import random
from itertools import combinations

_N = 128
_CAND = list(range(1, _N + 1))
_DEADLINE = 85.0  # wall-clock safety margin under the 90 s kill
_START = None

# Precomputation
_PRE = None       # PRE[i][j] = gcd(i,j) for i,j in 0..N
_BUCKETS = None   # BUCKETS[i] = dict g -> list of j with gcd(i,j)==g
# For fast conflict counting:
# For each element x, and each gcd value g, the list of elements y with gcd(x,y)==g.
# A triple (a,b,c) is a violation iff gcd(a,b)=gcd(a,c)=gcd(b,c)=g for some g.
# For a given set S, the number of violations can be counted by:
#   For each g, consider the graph G_g where vertices are elements in S and
#   edges connect pairs with gcd == g. Each triangle in G_g is a violation.
# But counting triangles is expensive. Instead, for incremental updates:
#   When adding x to S, the new violations are triples (x, a, b) where
#   gcd(x,a) = gcd(x,b) = gcd(a,b) = g for some g.
#   So for each g, let N_g(x) = {y in S : gcd(x,y) == g}.
#   Then new violations from x = sum over g of (number of edges in G_g induced by N_g(x)).
#   An edge in G_g between a,b exists iff gcd(a,b) == g.
#   So new_violations = sum_g C_g(x), where C_g(x) = #{(a,b) in N_g(x)^2 : a<b, gcd(a,b)==g}.

def _ensure_pre():
    global _PRE, _BUCKETS
    if _PRE is not None:
        return
    PRE = [[0] * (_N + 1) for _ in range(_N + 1)]
    for i in range(1, _N + 1):
        for j in range(i, _N + 1):
            PRE[i][j] = PRE[j][i] = math.gcd(i, j)
    _PRE = PRE

    BUCKETS = [None] * (_N + 1)
    for i in range(1, _N + 1):
        d = {}
        row = PRE[i]
        for j in range(1, _N + 1):
            g = row[j]
            d.setdefault(g, []).append(j)
        BUCKETS[i] = d
    _BUCKETS = BUCKETS


def _count_violations(S):
    """Count number of equal-gcd triples in set S (list of ints)."""
    P = _PRE
    Sset = set(S)
    n = len(S)
    cnt = 0
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            g = P[a][b]
            for k in range(j + 1, n):
                c = S[k]
                if P[a][c] == g and P[b][c] == g:
                    cnt += 1
    return cnt


def _can_add(S_set, S_list, x):
    """Check if adding x to S creates no new equal-gcd triple."""
    P = _PRE
    B = _BUCKETS
    Bx = B[x]
    # For each g, collect elements in S with gcd(x, .) == g
    for g, lst in Bx.items():
        inball = [u for u in S_list if P[x][u] == g]
        m = len(inball)
        if m < 2:
            continue
        # Check if any pair in inball has gcd == g
        for i in range(m):
            a = inball[i]
            for j in range(i + 1, m):
                b = inball[j]
                if P[a][b] == g:
                    return False
    return True


def _greedy_construct(order):
    """Greedy: add elements in given order if they don't create violations."""
    S = []
    Sset = set()
    for x in order:
        if _can_add(Sset, S, x):
            S.append(x)
            Sset.add(x)
    return S


def _local_add(S):
    """Repeatedly add any excluded element that fits."""
    Sset = set(S)
    changed = True
    while changed:
        changed = False
        for x in _CAND:
            if x not in Sset and _can_add(Sset, S, x):
                S.append(x)
                Sset.add(x)
                changed = True
    return S


def _conflict_count(S):
    """For each element in S, count how many violations it participates in."""
    P = _PRE
    Sset = set(S)
    n = len(S)
    conf = {x: 0 for x in S}
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            g = P[a][b]
            for k in range(j + 1, n):
                c = S[k]
                if P[a][c] == g and P[b][c] == g:
                    conf[a] += 1
                    conf[b] += 1
                    conf[c] += 1
    return conf


def _repair_remove_conflicts(S):
    """Remove elements with highest conflict count until no violations remain."""
    S = list(S)
    while True:
        n = len(S)
        if n < 3:
            break
        # Count violations
        P = _PRE
        total_viol = 0
        for i in range(n):
            a = S[i]
            for j in range(i + 1, n):
                b = S[j]
                g = P[a][b]
                for k in range(j + 1, n):
                    c = S[k]
                    if P[a][c] == g and P[b][c] == g:
                        total_viol += 1
        if total_viol == 0:
            break
        conf = _conflict_count(S)
        # Remove element with max conflict
        worst = max(S, key=lambda x: conf[x])
        S.remove(worst)
    return S


def _randomized_restarts(num_restarts, seed_base):
    """Try multiple random orderings with greedy + repair + local add."""
    rng = random.Random(seed_base)
    best_S = []
    best_size = 0

    # Precompute some useful orderings
    def num_divisors(n):
        c = 0
        i = 1
        while i * i <= n:
            if n % i == 0:
                c += 1
                if i * i != n:
                    c += 1
            i += 1
        return c

    divs = {x: num_divisors(x) for x in range(1, _N + 1)}

    def is_prime_power(x):
        if x < 2:
            return False
        v = x
        f0 = None
        p = 2
        while p * p <= v:
            if v % p == 0:
                if f0 is None:
                    f0 = p
                else:
                    return False
                while v % p == 0:
                    v //= p
                break
            p += 1
        return True

    prime_powers = [x for x in range(1, _N + 1) if is_prime_power(x)]

    base_orders = [
        sorted(range(1, _N + 1), key=lambda x: divs[x]),  # fewest divisors first
        sorted(range(1, _N + 1), key=lambda x: divs[x], reverse=True),
        prime_powers + [x for x in range(1, _N + 1) if x not in prime_powers],
        list(range(1, _N + 1)),
        list(range(_N, 0, -1)),
    ]

    for restart in range(num_restarts):
        if time.time() > _DEADLINE - 10:
            break
        if restart < len(base_orders):
            order = list(base_orders[restart])
            # Shuffle non-prime-power elements to vary
            pp = set(prime_powers)
            non_pp = [x for x in order if x not in pp]
            rng.shuffle(non_pp)
            pp_list = [x for x in order if x in pp]
            order = pp_list + non_pp
        else:
            order = list(_CAND)
            rng.shuffle(order)

        S = _greedy_construct(order)
        S = _local_add(S)

        # If there are violations (shouldn't happen with greedy, but just in case)
        if len(S) >= 3:
            S = _repair_remove_conflicts(S)
            S = _local_add(S)

        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)

    return best_S, best_size


def _polish(S):
    """Try removing each element and re-adding to see if we can do better."""
    best_S = list(S)
    best_size = len(S)
    Sset = set(S)

    for rem in list(S):
        if time.time() > _DEADLINE - 5:
            break
        reduced = [x for x in S if x != rem]
        reduced_set = set(reduced)
        # Try to add back elements
        changed = True
        while changed:
            changed = False
            for x in _CAND:
                if x not in reduced_set and _can_add(reduced_set, reduced, x):
                    reduced.append(x)
                    reduced_set.add(x)
                    changed = True
        if len(reduced) > best_size:
            best_size = len(reduced)
            best_S = list(reduced)
            Sset = set(reduced)

    return best_S, best_size


def _construct():
    global _START
    _ensure_pre()
    _START = time.time()

    # Phase 1: Randomized restarts with greedy + local add
    best_S, best_size = _randomized_restarts(num_restarts=30, seed_base=42)

    # Phase 2: Polish
    if time.time() < _DEADLINE - 5:
        best_S, best_size = _polish(best_S)

    # Phase 3: More restarts from perturbed versions of best_S
    if time.time() < _DEADLINE - 15:
        rng = random.Random(12345)
        for trial in range(20):
            if time.time() > _DEADLINE - 10:
                break
            # Perturb best_S: remove 2-5 random elements, shuffle, re-greedy
            S_pert = list(best_S)
            n_remove = min(len(S_pert), rng.randint(2, 5))
            to_remove = rng.sample(S_pert, n_remove)
            for x in to_remove:
                S_pert.remove(x)
            # Add some random elements from outside
            outside = [x for x in _CAND if x not in S_pert]
            if outside:
                n_add = min(len(outside), rng.randint(1, 3))
                for x in rng.sample(outside, n_add):
                    S_pert.append(x)
            # Greedy repair
            S_pert = _repair_remove_conflicts(S_pert)
            S_pert = _local_add(S_pert)
            if len(S_pert) > best_size:
                best_size = len(S_pert)
                best_S = list(S_pert)

    # Final polish
    if time.time() < _DEADLINE - 3:
        best_S, best_size = _polish(best_S)

    # Safety: ensure validity
    if len(best_S) >= 3:
        v = _count_violations(best_S)
        if v != 0:
            best_S = _repair_remove_conflicts(best_S)
            best_S = _local_add(best_S)

    if not best_S:
        best_S = [2]

    return best_S


def construct():
    try:
        res = _construct()
    except Exception:
        _ensure_pre()
        res = list(range(1, _N + 1))
        # Remove elements causing violations until valid
        while len(res) >= 3:
            v = _count_violations(res)
            if v == 0:
                break
            conf = _conflict_count(res)
            worst = max(res, key=lambda x: conf[x])
            res.remove(worst)
        if not res:
            res = [2]
    return res
