"""Structural rewrite: branch-and-bound maximum independent set on the
conflict graph (Tomita-style), with bitsets and a greedy coloring bound.

A pair {a,b} is a "conflict" if there exists c in [1,128] such that
lcm(a,b)=lcm(a,c)=lcm(b,c).  A valid set is an independent set in this graph.
We maximize |independent set| via exact branch-and-bound (fast for n=128).
"""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85.0

    def _lcm(a, b):
        return a // gcd(a, b) * b

    n = 128

    # ---- Build conflict graph (pair is conflict iff a common bad third exists)
    adj = [0] * (n + 1)          # bit i (0-based) set if (i+1) conflicts with x
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = _lcm(a, b)
            for c in range(b + 1, n + 1):
                if _lcm(a, c) == ab and _lcm(b, c) == ab:
                    adj[a] |= 1 << (b - 1)
                    adj[b] |= 1 << (a - 1)
                    break

    # ---- Fast greedy coloring bound using bit-level operations
    # Returns (num_colors, color_order) where color_order[i] is the color of vertex i
    def color_bound_fast(cands):
        """Greedy coloring: assign colors to vertices in cands in decreasing order.
        Returns the number of colors used."""
        if cands == 0:
            return 0
        # Process vertices in order of decreasing bit position
        # Use a list to track which color each vertex gets
        color_of = {}  # bit -> color
        next_color = 1
        remaining = cands
        while remaining:
            # Pick the highest bit (largest vertex)
            lb = remaining & (-remaining)
            remaining ^= lb
            v = lb.bit_length()  # 1-based vertex number
            
            # Find which colors are used by already-colored neighbors of v
            used_colors = 0
            for bit2, col2 in color_of.items():
                # Check if v and bit2 are adjacent
                if (adj[v] >> (bit2.bit_length() - 1)) & 1:
                    used_colors |= (1 << col2)
            
            # Assign smallest available color
            c = 1
            while used_colors & (1 << (c - 1)):
                c += 1
            color_of[lb] = c
            if c > next_color:
                next_color = c
        return next_color

    # ---- Even faster: precompute adjacency as list of sets for local search
    adj_set = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        mask = adj[a]
        while mask:
            lb = mask & (-mask)
            adj_set[a].add(lb.bit_length())
            mask ^= lb

    # ---- Initial greedy independent set (lower bound)
    def greedy_seed(order):
        chosen = set()
        chosen_mask = 0
        for x in order:
            if (adj[x] & chosen_mask) == 0:
                chosen.add(x)
                chosen_mask |= (1 << (x - 1))
        return chosen

    best = greedy_seed(range(1, n + 1))
    for order in (range(n, 0, -1),
                  sorted(range(1, n + 1), key=lambda x: bin(adj[x]).count('1'))):
        s = greedy_seed(order)
        if len(s) > len(best):
            best = s
    best_size = len(best)

    # ---- Branch and bound (Tomita-style) with fast coloring
    def expand(cands, size):
        nonlocal best, best_size
        if time.time() > deadline - 3.0:
            return
        cb = color_bound_fast(cands)
        if size + cb <= best_size:
            return
        while cands:
            lb = cands & (-cands)
            x = lb.bit_length()
            cands ^= lb
            if size + 1 + color_bound_fast(cands) > best_size:
                expand(cands & ~adj[x], size + 1)
            if size + color_bound_fast(cands) <= best_size:
                return

    if time.time() < deadline - 5.0:
        expand((1 << n) - 1, 0)

    # ---- Local search: aggressive 1-add and 2-exchange
    def can_add(x, S_mask):
        return (adj[x] & S_mask) == 0

    # Phase 1: Add any vertex that can be added
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        S_mask = 0
        for x in best:
            S_mask |= (1 << (x - 1))
        for x in range(1, n + 1):
            if x not in best and can_add(x, S_mask):
                best.add(x)
                best_size += 1
                S_mask |= (1 << (x - 1))
                improved = True

    # Phase 2: 1-for-2 exchanges
    for _ in range(500):
        if time.time() > deadline - 3.0:
            break
        found = False
        best_list = list(best)
        for i, x in enumerate(best_list):
            trial = best - {x}
            trial_mask = 0
            for y in trial:
                trial_mask |= (1 << (y - 1))
            gain = 0
            for y in range(1, n + 1):
                if y == x or y in trial:
                    continue
                if can_add(y, trial_mask):
                    trial.add(y)
                    trial_mask |= (1 << (y - 1))
                    gain += 1
                    if gain >= 2:
                        break
            if len(trial) > len(best):
                best = trial
                best_size = len(trial)
                found = True
                break
        if not found:
            break

    return list(best)
# EVOLVE-BLOCK-END
