# EVOLVE-BLOCK-START
import time
import random
from math import gcd

N = 128
VALS = list(range(1, N + 1))


def _ensure_pre():
    global _GCD, _TRIPLE_VIOLATIONS
    if '_GCD' in globals():
        return
    # Precompute gcd table
    _GCD = [[0] * N for _ in range(N)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            _GCD[a - 1][b - 1] = _GCD[b - 1][a - 1] = gcd(a, b)
    
    # Precompute for each pair (i,j), the list of k such that
    # gcd(i,k) == gcd(i,j) == gcd(j,k)
    # _TRIPLE_VIOLATIONS[i][j] = frozenset of k values that form a violating triple with i,j
    _TRIPLE_VIOLATIONS = [[frozenset() for _ in range(N)] for _ in range(N)]
    for a in range(N):
        for b in range(a + 1, N):
            gab = _GCD[a][b]
            violators = []
            for c in range(N):
                if c == a or c == b:
                    continue
                if _GCD[a][c] == gab and _GCD[b][c] == gab:
                    violators.append(c)
            _TRIPLE_VIOLATIONS[a][b] = frozenset(violators)
            _TRIPLE_VIOLATIONS[b][a] = frozenset(violators)


def _popcount(m):
    return bin(m).count('1')


def _violations_count(m):
    """Count the number of violating triples in the set represented by mask m."""
    count = 0
    for i in range(N):
        if not ((m >> i) & 1):
            continue
        for j in range(i + 1, N):
            if not ((m >> j) & 1):
                continue
            # Count how many k in the set violate with pair (i,j)
            violators = _TRIPLE_VIOLATIONS[i][j]
            if violators:
                count += sum(1 for k in violators if (m >> k) & 1)
    return count // 3  # each triple counted 3 times (once per pair)


def _delta_violations_add(m, x):
    """Compute change in violation count when adding element x to set m."""
    if (m >> x) & 1:
        return 0
    delta = 0
    for i in range(N):
        if not ((m >> i) & 1):
            continue
        # For pair (i, x), count violators in current set
        violators = _TRIPLE_VIOLATIONS[i][x]
        if violators:
            delta += sum(1 for k in violators if (m >> k) & 1)
    return delta


def _delta_violations_remove(m, x):
    """Compute change in violation count when removing element x from set m."""
    if not ((m >> x) & 1):
        return 0
    delta = 0
    for i in range(N):
        if not ((m >> i) & 1):
            continue
        violators = _TRIPLE_VIOLATIONS[i][x]
        if violators:
            delta += sum(1 for k in violators if (m >> k) & 1)
    return delta


def _greedy_construct():
    """Greedy construction by ascending value."""
    m = 0
    for x in range(N):
        if _delta_violations_add(m, x) == 0:
            m |= (1 << x)
    return m


def _local_search_polish(m, deadline):
    """Local search: try to add elements, remove+add swaps."""
    v = _violations_count(m)
    s = _popcount(m)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding
        for x in range(N):
            if not ((m >> x) & 1):
                dv = _delta_violations_add(m, x)
                if dv == 0:
                    m |= (1 << x)
                    s += 1
                    v += dv
                    improved = True
        # Try swap: remove one, add another
        for r in range(N):
            if not ((m >> r) & 1):
                continue
            m2 = m & ~(1 << r)
            s2 = s - 1
            v2 = v - _delta_violations_remove(m, r)
            for a in range(N):
                if not ((m2 >> a) & 1):
                    dv = _delta_violations_add(m2, a)
                    if s2 + 1 > s and v2 + dv <= v:
                        m = m2 | (1 << a)
                        s = s2 + 1
                        v = v2 + dv
                        improved = True
                        break
    return m, s


def _structural_initial():
    """Create initial solution using divisor-count ordering (more composite first)."""
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count
    
    # Sort by number of divisors descending (more composite first)
    order = sorted(range(N), key=lambda x: num_divisors(VALS[x]), reverse=True)
    m = 0
    for x in order:
        if _delta_violations_add(m, x) == 0:
            m |= (1 << x)
    return m


def _anneal(m, deadline):
    """Simulated annealing with add/remove/swap moves."""
    s = _popcount(m)
    v = _violations_count(m)
    best_m = m
    best_s = s
    best_v = v
    
    T = 5.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.4:
            # Add a random element
            candidates = [x for x in range(N) if not ((m >> x) & 1)]
            if not candidates:
                T *= cooling
                continue
            x = random.choice(candidates)
            dv = _delta_violations_add(m, x)
            new_m = m | (1 << x)
            new_s = s + 1
            new_v = v + dv
            # Accept if improves or with probability exp(-delta/T)
            if new_v < v or (new_v == v and new_s > s) or random.random() < (2.718281828 ** (-dv / T)):
                m, s, v = new_m, new_s, new_v
        elif move_type < 0.8:
            # Remove a random element
            candidates = [x for x in range(N) if (m >> x) & 1]
            if not candidates:
                T *= cooling
                continue
            x = random.choice(candidates)
            dv = _delta_violations_remove(m, x)
            new_m = m & ~(1 << x)
            new_s = s - 1
            new_v = v - dv
            if new_v < v or random.random() < (2.718281828 ** (dv / T)):
                m, s, v = new_m, new_s, new_v
        else:
            # Swap: remove one, add another
            remove_candidates = [x for x in range(N) if (m >> x) & 1]
            add_candidates = [x for x in range(N) if not ((m >> x) & 1)]
            if not remove_candidates or not add_candidates:
                T *= cooling
                continue
            r = random.choice(remove_candidates)
            a = random.choice(add_candidates)
            m2 = m & ~(1 << r)
            dv_remove = _delta_violations_remove(m, r)
            dv_add = _delta_violations_add(m2, a)
            new_m = m2 | (1 << a)
            new_s = s
            new_v = v - dv_remove + dv_add
            delta = new_v - v
            if delta < 0 or random.random() < (2.718281828 ** (-delta / T)):
                m, v = new_m, new_v
        
        # Track best
        if v == 0 and s > best_s:
            best_m, best_s, best_v = m, s, v
        elif v < best_v:
            best_m, best_s, best_v = m, s, v
        
        T *= cooling
    
    return best_m, best_s


def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin before 90s kill
    _ensure_pre()
    
    # Phase 1: Multiple initial solutions
    best_mask = None
    best_size = 0
    
    # Greedy by ascending
    m1 = _greedy_construct()
    m1, s1 = _local_search_polish(m1, deadline - 20.0)
    if s1 > best_size:
        best_mask, best_size = m1, s1
    
    # Structural initial (divisor-count ordering)
    m2 = _structural_initial()
    m2, s2 = _local_search_polish(m2, deadline - 15.0)
    if s2 > best_size:
        best_mask, best_size = m2, s2
    
    # Phase 2: Simulated annealing from best solution
    if time.time() < deadline - 5.0:
        m3, s3 = _anneal(best_mask, deadline - 5.0)
        if s3 > best_size:
            best_mask, best_size = m3, s3
    
    # Phase 3: Final local polish
    m4, s4 = _local_search_polish(best_mask, deadline)
    if s4 > best_size:
        best_mask, best_size = m4, s4
    
    # Extract result
    result = [VALS[i] for i in range(N) if (best_mask >> i) & 1]
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
