import time
import math
import random
from collections import deque

# EVOLVE-BLOCK-START

def _next_prime(n):
    if n <= 2:
        return 2
    if n % 2 == 0:
        n += 1
    while True:
        p = n
        d = 3
        while d * d <= p:
            if p % d == 0:
                break
            d += 2
        else:
            return p
        n += 2

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = 5
    while d * d <= n:
        if n % d == 0 or n % (d + 2) == 0:
            return False
        d += 6
    return True

def _girth_ok_fast(k, g, S, q):
    """Check girth >= g for a 2-sided Cayley graph on Z_q with generating set S.
    Uses BFS of non-backtracking walks; returns True if no closed non-backtracking
    walk of length < g exists."""
    S_list = list(S)
    nS = len(S_list)
    S_neg = {x: (-x) % q for x in S_list}
    # BFS: state = (last_gen_idx, position), track distance
    # Start: all (i, s_i) at depth 1
    # We need to check if any walk of length 2..g-1 returns to 0 without backtracking.
    # Use visited set of (last_idx, pos) to avoid reprocessing.
    visited = set()
    # depth 1 states
    frontier = []
    for i, s in enumerate(S_list):
        state = (i, s % q)
        if state not in visited:
            visited.add(state)
            frontier.append(state)
    
    for depth in range(2, g):
        new_frontier = []
        for last_idx, pos in frontier:
            neg = S_neg[S_list[last_idx]]
            for j, sj in enumerate(S_list):
                if sj == neg:
                    continue
                new_pos = (pos + sj) % q
                if new_pos == 0:
                    # Closed non-backtracking walk of length `depth`
                    return False
                state = (j, new_pos)
                if state not in visited:
                    visited.add(state)
                    new_frontier.append(state)
        frontier = new_frontier
        if not frontier:
            break
    return True

def _girth_ok_bfs(k, g, S, q):
    """Alternative BFS-based girth check using adjacency."""
    S_list = list(S)
    Sset = set(S_list)
    S_neg = {x: (-x) % q for x in Sset}
    
    # Non-backtracking BFS from vertex 0
    # State: (current, prev), dist
    # A cycle of length L exists iff we can return to a previously visited state
    # Actually simpler: BFS of directed edges (u->v), check if we can reach (v->u)
    # No, that's for girth. Let me use the standard approach.
    
    # Standard: for each start vertex, do non-backtracking BFS.
    # But we only need to check from vertex 0 for Cayley graphs (vertex-transitive).
    
    # Non-backtracking walk: sequence v_0, v_1, ..., v_L where v_i != v_{i-2}
    # and v_i adjacent to v_{i-1}.
    # A closed non-backtracking walk of length L means v_L = v_0 and v_{L-1} != v_1.
    
    # BFS over directed edges (prev, curr), starting from all (0, s) for s in S.
    # If we reach (s, 0) where s != the original start... no.
    
    # Let's use: visited[(prev, curr)] = dist. Start: for each s in S, (0, s) at dist 1.
    # From (prev, curr), go to (curr, next) where next in adj[curr] and next != prev.
    # If next == 0 and dist+1 >= 2, we have a cycle of length dist+1.
    # But we need dist+1 < g to fail.
    
    adj = {}
    for v in range(q):
        adj[v] = [(v + s) % q for s in S_list]
    
    visited = {}
    queue = deque()
    for s in S_list:
        key = (0, s % q)
        if key not in visited:
            visited[key] = 1
            queue.append(key)
    
    while queue:
        (prev, curr), dist = queue.popleft()
        for nxt in adj[curr]:
            if nxt == prev:
                continue
            new_dist = dist + 1
            if new_dist >= g:
                continue  # We only care about cycles shorter than g
            if nxt == 0:
                # Closed non-backtracking walk of length new_dist
                return False
            key = (curr, nxt)
            if key not in visited:
                visited[key] = new_dist
                queue.append(key)
    return True

def _luw_construct(k, g):
    # Fallback: Lazebnik-Ustimenco-Woldar / Laszlo bipartite graph (valid, larger).
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def _try_circulant(q, k, g, rng, time_limit=5.0):
    """Try to find a 2-sided generating set S of size k on Z_q with girth >= g.
    Uses random search + local search."""
    if k % 2 == 0:
        # S = {a1, a2, ..., am, -a1, -a2, ..., -am} where m = k/2
        m = k // 2
    else:
        # S = {a1, a2, ..., am, -a1, -a2, ..., -am, c} where 2m+1 = k
        # c must be its own inverse: 2c = 0 mod q, so c = q/2 (q even) or c=0 (invalid)
        # For odd k and odd q, we can't have a self-inverse element other than 0.
        # So for odd k, we need q even or use a different structure.
        # Actually for odd k, the graph must be on an even number of vertices if we want
        # a 2-sided generating set with a fixed point. Let's handle this differently.
        # For odd k, use S = {a1, -a1, ..., am, -am, c} where c = -c mod q => 2c ≡ 0 mod q.
        # This requires q even. If q is odd, we can't do this.
        # Alternative: for odd k, use a different construction.
        # For now, if k is odd and q is odd, skip.
        if q % 2 != 0:
            return None
        m = (k - 1) // 2
        c = q // 2  # self-inverse element

    best = None
    best_score = float('inf')
    t_start = time.time()
    
    # Generate random initial solutions
    def random_solution():
        if k % 2 == 0:
            while True:
                vals = rng.sample(range(1, q), m)
                # Ensure no val is its own negative (val != -val mod q)
                if all(2 * v % q != 0 for v in vals):
                    return vals
        else:
            while True:
                vals = rng.sample(range(1, q // 2), m)
                # Ensure no val equals c
                if all(v != c and v != 0 for v in vals):
                    return vals
    
    def build_S(vals):
        if k % 2 == 0:
            S = list(vals) + [(-v) % q for v in vals]
        else:
            S = list(vals) + [(-v) % q for v in vals] + [c]
        return S
    
    def score_solution(vals):
        """Return girth (higher is better, inf if no cycle < some limit)."""
        S = build_S(vals)
        Sset = set(S)
        if len(Sset) != k or 0 in Sset:
            return -1
        if Sset != {(-x) % q for x in Sset}:
            return -1
        # Check girth
        if _girth_ok_bfs(k, g, S, q):
            return g  # girth >= g
        # Estimate girth by finding shortest cycle
        # For scoring, just return a lower value
        return 1
    
    # Random search
    n_random = 200
    for _ in range(n_random):
        if time.time() - t_start > time_limit:
            break
        vals = random_solution()
        sc = score_solution(vals)
        if sc == g:
            return build_S(vals)
    
    # Local search on best candidates
    # Keep track of all solutions tried, sort by score
    candidates = []
    for _ in range(100):
        if time.time() - t_start > time_limit:
            break
        vals = random_solution()
        sc = score_solution(vals)
        candidates.append((sc, vals))
    
    candidates.sort(key=lambda x: -x[0])
    
    for sc, vals in candidates[:20]:
        if time.time() - t_start > time_limit:
            break
        cur_vals = list(vals)
        cur_sc = sc
        for _ in range(50):
            if time.time() - t_start > time_limit:
                break
            if cur_sc == g:
                return build_S(cur_vals)
            # Mutate: pick a random index and change it
            idx = rng.randrange(len(cur_vals))
            old = cur_vals[idx]
            while True:
                new_val = rng.randrange(1, q if k % 2 == 0 else q // 2)
                if new_val != old and 2 * new_val % q != 0 and new_val not in cur_vals:
                    break
            cur_vals[idx] = new_val
            new_sc = score_solution(cur_vals)
            if new_sc >= cur_sc:
                cur_sc = new_sc
            else:
                cur_vals[idx] = old
            if cur_sc == g:
                return build_S(cur_vals)
    
    return None

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s (kill at 150s)
    
    if k < 3:
        if k == 2:
            n = max(g, 3)
            return n, [(i, (i + 1) % n) for i in range(n)]
        return 1, []
    
    rng = random.Random(42)
    
    # Moore bound
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    # Try circulant constructions for various q values
    # Start from small q, grow
    # For k=5, g=9: Moore bound = 1 + 5*(1+4+16) = 1+5*21 = 106
    # So we need q >= 106 (for odd k, q even)
    
    best_result = None
    best_n = float('inf')
    
    # Generate candidate q values
    # For odd k, q must be even
    # For even k, q can be odd or even
    q_candidates = []
    start_q = max(k + 1, moore)
    if k % 2 == 1:
        start_q = max(start_q, 2)
        if start_q % 2 != 0:
            start_q += 1
    else:
        if start_q < 3:
            start_q = 3
    
    # Try a range of q values
    max_q = start_q + 200
    q = start_q
    trials = 0
    while q <= max_q and time.time() < deadline - 30:
        if k % 2 == 1:
            if q % 2 == 0:
                # Try this q
                S = _try_circulant(q, k, g, rng, time_limit=3.0)
                if S is not None:
                    n = q
                    S_list = list(S)
                    edges = []
                    Sset = set(S_list)
                    for i in range(n):
                        for s in S_list:
                            j = (i + s) % q
                            if j > i:
                                edges.append((i, j))
                    if n < best_n:
                        best_n = n
                        best_result = (n, edges)
                    break  # Found one, stop
        else:
            if _is_prime(q) or q % 2 == 0:
                S = _try_circulant(q, k, g, rng, time_limit=3.0)
                if S is not None:
                    n = q
                    S_list = list(S)
                    edges = []
                    Sset = set(S_list)
                    for i in range(n):
                        for s in S_list:
                            j = (i + s) % q
                            if j > i:
                                edges.append((i, j))
                    if n < best_n:
                        best_n = n
                        best_result = (n, edges)
                    break
        
        q += 1 if k % 2 == 0 else 2
        trials += 1
        if trials > 100:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: LUW construction
    return _luw_construct(k, g)

# EVOLVE-BLOCK-END
