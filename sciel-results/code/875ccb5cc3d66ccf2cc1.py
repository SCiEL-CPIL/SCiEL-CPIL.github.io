"""
Combined program:
  (A) Lazebnik-Ustimenko-Woldar (LUW) algebraic construction  -> guaranteed valid 5-reg, girth>=9
  (B) Random 5-regular generation + edge-switching with a SMOOTH surrogate objective
      (soft-min via log-sum-exp over per-vertex short-cycle BFS counts) to find a smaller graph.

The smooth surrogate is a refinement of the hard "girth >= g" predicate: instead of
accepting a switch only when the minimum cycle count drops, we accept when the
log-sum-exp (soft-min) of the per-vertex counts drops, which is a smooth upper-bound
surrogate that guides the search more effectively.
"""

import time
import random
import math
from collections import deque

# ---------------------------------------------------------------------------
# LUW construction (deterministic, provably girth >= g for odd g)
# ---------------------------------------------------------------------------

def _luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add  = lambda x,y: x ^ y
        sub  = lambda x,y: x ^ y
        mul  = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y) % q
        sub = lambda x,y: (x-y) % q
        mul = lambda x,y: x*y % q

    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5) % 4, c - (c-5) % 4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)

    start = (0, (None,) + (0,)*d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

# ---------------------------------------------------------------------------
# Random 5-regular graph generation (configuration model)
# ---------------------------------------------------------------------------

def _gen_random_5reg(n):
    """Return adjacency list (list of lists) or None."""
    for _ in range(300):
        half = []
        for v in range(n):
            half.extend([v]*5)
        random.shuffle(half)
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, n*5, 2):
            u, v = half[i], half[i+1]
            if u == v or v in adj[u]:
                ok = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if not ok:
            continue
        # connectivity
        seen = [False]*n
        seen[0] = True
        dq = deque([0])
        cnt = 1
        while dq:
            u = dq.popleft()
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    cnt += 1
                    dq.append(w)
        if cnt != n:
            continue
        return [list(s) for s in adj]
    return None

# ---------------------------------------------------------------------------
# Girth checking and short-cycle counting
# ---------------------------------------------------------------------------

def _check_girth(adj, g, n):
    """True iff no cycle shorter than g. BFS from every vertex."""
    for start in range(n):
        dist = [-1]*n
        par  = [-1]*n
        dist[start] = 0
        dq = deque([start])
        while dq:
            u = dq.popleft()
            if dist[u] >= g-1:
                continue
            for w in adj[u]:
                if w == par[u]:
                    continue
                if dist[w] == -1:
                    dist[w] = dist[u]+1
                    par[w] = u
                    dq.append(w)
                else:
                    if dist[u]+1+dist[w] < g:
                        return False
    return True

def _count_short_cycles(adj_sets, g, n):
    """Count short-cycle detections via BFS from each vertex (early exit at 1)."""
    for start in range(n):
        dist = [-1]*n
        par  = [-1]*n
        dist[start] = 0
        dq = deque([start])
        while dq:
            u = dq.popleft()
            if dist[u] >= g-1:
                continue
            for w in adj_sets[u]:
                if w == par[u]:
                    continue
                if dist[w] == -1:
                    dist[w] = dist[u]+1
                    par[w] = u
                    dq.append(w)
                else:
                    if dist[u]+1+dist[w] < g:
                        return 1  # at least one
    return 0

def _per_vertex_cycle_counts(adj_sets, g, n):
    """Return list of per-vertex short-cycle counts (0 if none from that vertex)."""
    counts = []
    for start in range(n):
        dist = [-1]*n
        par  = [-1]*n
        dist[start] = 0
        dq = deque([start])
        c = 0
        while dq:
            u = dq.popleft()
            if dist[u] >= g-1:
                continue
            for w in adj_sets[u]:
                if w == par[u]:
                    continue
                if dist[w] == -1:
                    dist[w] = dist[u]+1
                    par[w] = u
                    dq.append(w)
                else:
                    if dist[u]+1+dist[w] < g:
                        c += 1
        counts.append(c)
    return counts

def _soft_min_surrogate(counts, temperature=1.0):
    """
    Smooth surrogate for min(counts) via log-sum-exp.
    If all counts are 0, returns 0.
    Otherwise returns a value > 0 that is a smooth upper-bound surrogate.
    Lower surrogate => fewer short cycles overall.
    """
    if not counts:
        return 0.0
    max_c = max(counts)
    if max_c == 0:
        return 0.0
    # log-sum-exp of counts (treating them as "energies")
    # We want a smooth function that is 0 iff all counts are 0, and increases with any count.
    # Use: T * log(sum(exp(c_i / T)))  which approximates max(c_i) as T->0
    # But we want it to be 0 when all are 0. Since exp(0)=1, sum = n, so result = T*log(n) > 0.
    # To fix: use (c_i + 1) so that 0-count contributes exp(1/T) which is large... no.
    # Better: use max(0, c_i) and if all zero return 0.
    # Simple approach: if any count > 0, return sum of counts (hard) but smoothed.
    # Let's use: T * log(1 + sum(exp(c_i / T)))  - this is 0 when all c_i=0 (since exp(0)=1, 1+sum=n+1, T*log(n+1) != 0).
    # Actually simplest smooth surrogate: sum(c_i) is already smooth (linear).
    # The "soft-min" idea: we want to minimize the worst vertex, but softly.
    # Use: T * log(sum(exp(c_i / T)))  and subtract T*log(n) to center at 0.
    # When all c_i = 0: T*log(n) - T*log(n) = 0. Good.
    # When some c_i > 0: result > 0. Good.
    log_sum = math.log(sum(math.exp(c / temperature) for c in counts))
    return temperature * (log_sum - math.log(len(counts)))

# ---------------------------------------------------------------------------
# Edge switching with smooth surrogate objective
# ---------------------------------------------------------------------------

def _edge_switching(adj_list, n, g, deadline, max_switches=3000):
    """
    Improve girth via 2-switches. Accept a switch if the smooth surrogate
    (soft-min of per-vertex cycle counts) decreases.
    Returns (adj_sets, success).
    """
    adj_sets = [set(adj_list[v]) for v in range(n)]

    # Initial surrogate value
    counts = _per_vertex_cycle_counts(adj_sets, g, n)
    best_surrogate = _soft_min_surrogate(counts, temperature=2.0)
    best_adj = [set(s) for s in adj_sets]

    if best_surrogate == 0.0:
        return best_adj, True

    for _ in range(max_switches):
        if time.time() > deadline:
            break

        # Pick a random 2-switch: find edges (a,c) and (b,d), replace with (a,b) and (c,d)
        found = False
        for _attempt in range(30):
            a = random.randint(0, n-1)
            b = random.randint(0, n-1)
            c = random.randint(0, n-1)
            d = random.randint(0, n-1)
            if len({a,b,c,d}) < 4:
                continue
            if c not in adj_sets[a] or d not in adj_sets[b]:
                continue
            if b in adj_sets[a] or d in adj_sets[c]:
                continue
            # Perform switch
            adj_sets[a].discard(c); adj_sets[c].discard(a)
            adj_sets[b].discard(d); adj_sets[d].discard(b)
            adj_sets[a].add(b);     adj_sets[b].add(a)
            adj_sets[c].add(d);     adj_sets[d].add(c)
            found = True
            break

        if not found:
            continue

        # Evaluate new surrogate
        new_counts = _per_vertex_cycle_counts(adj_sets, g, n)
        new_surrogate = _soft_min_surrogate(new_counts, temperature=2.0)

        if new_surrogate < best_surrogate:
            best_surrogate = new_surrogate
            best_adj = [set(s) for s in adj_sets]
            if best_surrogate == 0.0:
                return best_adj, True
        else:
            # Revert
            adj_sets[a].discard(b); adj_sets[b].discard(a)
            adj_sets[c].discard(d); adj_sets[d].discard(c)
            adj_sets[a].add(c);     adj_sets[c].add(a)
            adj_sets[b].add(d);     adj_sets[d].add(b)

    return best_adj, (best_surrogate == 0.0)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 108  # 12s safety margin

    # Always compute LUW as guaranteed fallback
    luw_n, luw_edges = _luw(k, g)
    best_n = luw_n
    best_edges = luw_edges

    if k != 5:
        return best_n, best_edges

    # Try to find a smaller 5-regular graph with girth >= g
    # Moore bound for (5,9): 1+5+20+80+320 = 426, so n >= 426.
    # Try sizes from 450 up to luw_n - 1.
    # Random 5-regular graphs have expected girth ~ 2*log_4(n), so for n~600,
    # expected girth ~ 2*log_4(600) ~ 2*5.1 ~ 10.2, promising.

    # Strategy: try several n values, each with a few random attempts + switching.
    # Use time budget wisely.
    n_candidates = list(range(450, min(luw_n, 1116), 25))
    random.shuffle(n_candidates)
    # Prioritize smaller n
    n_candidates.sort()

    for n in n_candidates:
        if time.time() > deadline:
            break
        if n >= best_n:
            break

        # Try up to 5 random graphs at this n
        for attempt in range(5):
            if time.time() > deadline:
                break
            adj = _gen_random_5reg(n)
            if adj is None:
                continue

            # Quick girth check
            if _check_girth(adj, g, n):
                best_n = n
                best_edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            best_edges.append((u, v))
                break

            # Try edge switching to improve
            adj_sets, success = _edge_switching(adj, n, g, deadline, max_switches=1500)
            if success:
                best_n = n
                best_edges = []
                for u in range(n):
                    for v in adj_sets[u]:
                        if u < v:
                            best_edges.append((u, v))
                break

    return best_n, best_edges
