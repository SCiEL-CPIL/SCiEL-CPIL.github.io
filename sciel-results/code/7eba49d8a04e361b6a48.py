"""Structural rewrite: branch-and-bound maximum independent set on the
conflict graph (Tomita-style), with bitsets and a greedy coloring bound.

A pair {a,b} is a "conflict" if there exists c in [1,128] such that
lcm(a,b)=lcm(a,c)=lcm(b,c).  A valid set is an independent set in this graph.
We maximize |independent set| via exact branch-and-bound (fast for n=128).
"""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start = time.time()
    deadline = start + 85.0

    def _lcm(a, b):
        return a // gcd(a, b) * b

    n = 128
    full_mask = (1 << n) - 1

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

    # Precompute non-adjacent mask (compatible vertices) for each vertex
    # compat[x] has bits set for all vertices NOT adjacent to x (and not x itself)
    compat = [0] * (n + 1)
    for x in range(1, n + 1):
        compat[x] = full_mask & ~(adj[x] | (1 << (x - 1)))

    # ---- Fast greedy coloring bound (Tomita-style with bit operations)
    # For maximum independent set in G = maximum clique in complement(G).
    # We greedily color complement(G) induced by cands.
    # Two vertices are adjacent in complement if they are NOT adjacent in G.
    # For vertex v, its neighbors in complement are compat[v].
    def color_bound(cands):
        """Greedy coloring bound: returns upper bound on independent set size in cands."""
        if cands == 0:
            return 0
        remaining = cands
        num_colors = 0
        while remaining:
            num_colors += 1
            # Find a maximal clique in complement[remaining] = maximal independent set in G[remaining]
            color_class = 0
            avail = remaining
            while avail:
                lb = avail & (-avail)
                v = lb.bit_length()
                color_class |= lb
                # In complement, v is adjacent to compat[v]. Keep only those.
                avail = avail & compat[v] & ~color_class
            remaining &= ~color_class
        return num_colors

    # Precompute degree for ordering
    degrees = [bin(adj[x]).count('1') for x in range(n + 1)]
    
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
                  sorted(range(1, n + 1), key=lambda x: degrees[x]),
                  sorted(range(1, n + 1), key=lambda x: -degrees[x]),
                  sorted(range(1, n + 1), key=lambda x: (degrees[x], -x))):
        s = greedy_seed(order)
        if len(s) > len(best):
            best = s
    best_size = len(best)

    # ---- Branch and bound (Tomita-style) with fast coloring
    # Use recursion with bitset candidates. Pick vertex with highest degree in cands first.
    def expand(cands, size):
        nonlocal best, best_size
        if time.time() > deadline - 3.0:
            return
        if cands == 0:
            return
        cb = color_bound(cands)
        if size + cb <= best_size:
            return
        # Find vertex with highest degree within cands for better branching
        # Use a heuristic: pick the vertex that appears in the most conflicts within cands
        best_x = 0
        best_deg = -1
        temp = cands
        while temp:
            lb = temp & (-temp)
            x = lb.bit_length()
            deg = bin(adj[x] & cands).count('1')
            if deg > best_deg:
                best_deg = deg
                best_x = x
            temp ^= lb
        
        if best_x == 0:
            return
        
        x = best_x
        x_bit = 1 << (x - 1)
        cands_without = cands & ~x_bit
        new_cands = cands_without & compat[x]
        
        # Branch: include x
        if size + 1 + color_bound(new_cands) > best_size:
            expand(new_cands, size + 1)
        
        # Branch: exclude x (only if still potentially better)
        if size + color_bound(cands_without) > best_size:
            expand(cands_without, size)

    if time.time() < deadline - 5.0:
        expand(full_mask, 0)
    
    # Additional B&B runs with different vertex orderings if time permits
    if time.time() < deadline - 8.0:
        # Try a different initial ordering by re-running with shuffled priorities
        random.seed(42)
        shuffled_vertices = list(range(1, n + 1))
        random.shuffle(shuffled_vertices)
        
        def expand_shuffled(cands, size):
            nonlocal best, best_size
            if time.time() > deadline - 3.0:
                return
            if cands == 0:
                return
            cb = color_bound(cands)
            if size + cb <= best_size:
                return
            # Pick vertex with highest degree in cands, tie-break by shuffle order
            best_x = 0
            best_key = (-1, 0)
            temp = cands
            while temp:
                lb = temp & (-temp)
                x = lb.bit_length()
                deg = bin(adj[x] & cands).count('1')
                # Use negative degree and shuffle index as key
                shuffle_idx = shuffled_vertices.index(x) if x in shuffled_vertices else 0
                key = (-deg, shuffle_idx)
                if key < best_key:
                    best_key = key
                    best_x = x
                temp ^= lb
            
            if best_x == 0:
                return
            
            x = best_x
            x_bit = 1 << (x - 1)
            cands_without = cands & ~x_bit
            new_cands = cands_without & compat[x]
            
            if size + 1 + color_bound(new_cands) > best_size:
                expand_shuffled(new_cands, size + 1)
            
            if size + color_bound(cands_without) > best_size:
                expand_shuffled(cands_without, size)
        
        expand_shuffled(full_mask, 0)

    # ---- Local search with bit-parallel operations
    # Convert best to a mask for fast operations
    def set_to_mask(S):
        m = 0
        for x in S:
            m |= (1 << (x - 1))
        return m

    def mask_to_set(m):
        s = set()
        while m:
            lb = m & (-m)
            s.add(lb.bit_length())
            m ^= lb
        return s

    # Phase 1: Add any vertex that can be added (using bit operations)
    improved = True
    while improved and time.time() < deadline - 3.0:
        improved = False
        S_mask = set_to_mask(best)
        # Find all vertices not in S that are compatible with all of S
        # A vertex x is addable if adj[x] & S_mask == 0
        # Equivalent: x is in compat of every element in S
        # Faster: iterate over candidates not in S
        candidates = full_mask & ~S_mask
        while candidates:
            lb = candidates & (-candidates)
            x = lb.bit_length()
            if (adj[x] & S_mask) == 0:
                best.add(x)
                best_size += 1
                S_mask |= lb
                candidates &= ~lb
                improved = True
            else:
                candidates ^= lb

    # Phase 2: 1-for-2 exchanges using bit operations
    # For each x in best, remove x, then try to add vertices compatible with remaining
    for iteration in range(5000):
        if time.time() > deadline - 3.0:
            break
        found = False
        S_mask = set_to_mask(best)
        best_list = list(best)
        for x in best_list:
            x_bit = 1 << (x - 1)
            trial_mask = S_mask & ~x_bit
            # Find vertices addable to trial
            # A vertex y is addable if y not in trial, y != x, and adj[y] & trial_mask == 0
            candidates = full_mask & ~trial_mask & ~x_bit
            # We need adj[y] & trial_mask == 0 for y
            # Iterate and collect
            addable = 0
            c = candidates
            while c:
                lb = c & (-c)
                y = lb.bit_length()
                if (adj[y] & trial_mask) == 0:
                    addable |= lb
                c ^= lb
            # Greedily add from addable, trying multiple orderings
            best_trial_mask = trial_mask
            best_added = 0
            # Try natural order
            trial_mask2 = trial_mask
            added = 0
            a = addable
            while a:
                lb = a & (-a)
                y = lb.bit_length()
                if (adj[y] & trial_mask2) == 0:
                    trial_mask2 |= lb
                    added += 1
                a ^= lb
            if added > best_added:
                best_added = added
                best_trial_mask = trial_mask2
            # Try reverse order
            trial_mask2 = trial_mask
            added = 0
            a = addable
            while a:
                # Pick highest bit
                lb = 1 << (a.bit_length() - 1)
                y = lb.bit_length()
                if (adj[y] & trial_mask2) == 0:
                    trial_mask2 |= lb
                    added += 1
                a ^= lb
            if added > best_added:
                best_added = added
                best_trial_mask = trial_mask2
            # Try degree-ordered
            trial_mask2 = trial_mask
            added = 0
            a = addable
            # Sort by degree descending
            verts = []
            temp = a
            while temp:
                lb = temp & (-temp)
                y = lb.bit_length()
                verts.append((bin(adj[y] & trial_mask).count('1'), y, lb))
                temp ^= lb
            verts.sort(key=lambda t: -t[0])
            for _, y, lb in verts:
                if (adj[y] & trial_mask2) == 0:
                    trial_mask2 |= lb
                    added += 1
            if added > best_added:
                best_added = added
                best_trial_mask = trial_mask2
            if best_added >= 2:
                # We can improve: remove x, add the greedily selected vertices
                best = mask_to_set(best_trial_mask)
                best_size = len(best)
                found = True
                break
        if not found:
            break

    # Phase 3: Random restarts with local search
    if time.time() < deadline - 5.0:
        for _ in range(200):
            if time.time() > deadline - 3.0:
                break
            # Random subset of current best
            best_list = list(best)
            random.shuffle(best_list)
            trial = set(best_list[:len(best_list) // 2])
            trial_mask = set_to_mask(trial)
            # Greedily add
            candidates = full_mask & ~trial_mask
            while candidates:
                lb = candidates & (-candidates)
                x = lb.bit_length()
                if (adj[x] & trial_mask) == 0:
                    trial.add(x)
                    trial_mask |= lb
                    candidates &= ~lb
                else:
                    candidates ^= lb
            if len(trial) > best_size:
                best = trial
                best_size = len(best)
            # Also try random full restarts
            if _ % 10 == 0:
                random_order = list(range(1, n + 1))
                random.shuffle(random_order)
                trial = set()
                trial_mask = 0
                for x in random_order:
                    if (adj[x] & trial_mask) == 0:
                        trial.add(x)
                        trial_mask |= (1 << (x - 1))
                if len(trial) > best_size:
                    best = trial
                    best_size = len(best)

    return list(best)
# EVOLVE-BLOCK-END
