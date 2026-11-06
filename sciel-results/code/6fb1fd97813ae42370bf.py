import time
import sys
import random
import math

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin before the 90s limit

    n = 256
    all_mask = (1 << n) - 1

    # Precompute conflict masks
    # conflict[i] is a bitmask of indices j such that (i+1) and (j+1) conflict
    conflict = [0] * n
    for i in range(n):
        a = i + 1
        mask_i = 0
        for j in range(n):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                mask_i |= (1 << j)
        conflict[i] = mask_i

    # Complementary adjacency (H): edge (u,v) in H iff NOT conflict
    adj = [0] * n
    for i in range(n):
        adj[i] = (~conflict[i]) & all_mask
        adj[i] &= ~(1 << i)

    # Degree in H
    degrees = [adj[i].bit_count() for i in range(n)]

    # Order vertices by degree descending
    order = list(range(n))
    order.sort(key=lambda x: degrees[x], reverse=True)
    rank = [0] * n
    for idx, v in enumerate(order):
        rank[v] = idx

    # Fast color bound: greedy coloring in rank order using bitmasks
    # Precompute for each vertex its conflict neighbors as a bitmask (already done)
    # Key optimization: use a fixed-size color array and bitmask for used colors
    
    # Precompute: vertices sorted by rank for quick extraction
    # We'll cache color results for frequently seen P values (LIFO cache)
    _color_cache = {}
    _color_cache_size = 0
    _MAX_COLOR_CACHE = 200000
    
    def color_bound(P):
        nonlocal _color_cache_size
        if P == 0:
            return 0
        
        # Check cache
        if P in _color_cache:
            return _color_cache[P]
        
        # Extract vertices from P in rank order
        verts = []
        p = P
        while p:
            lsb = p & -p
            i = lsb.bit_length() - 1
            verts.append(i)
            p ^= lsb
        verts.sort(key=lambda x: rank[x])
        
        # Greedy coloring with bitmask for used colors
        colors = [-1] * n
        max_color = 0
        for v in verts:
            used_mask = 0
            neigh = conflict[v] & P
            np_ = neigh
            while np_:
                lsb = np_ & -np_
                u = lsb.bit_length() - 1
                cu = colors[u]
                if cu >= 0:
                    used_mask |= (1 << cu)
                np_ ^= lsb
            c = 0
            while (used_mask >> c) & 1:
                c += 1
            colors[v] = c
            if c > max_color:
                max_color = c
        
        # Cache result
        if _color_cache_size < _MAX_COLOR_CACHE:
            _color_cache[P] = max_color
            _color_cache_size += 1
        elif len(_color_cache) < _MAX_COLOR_CACHE:
            _color_cache[P] = max_color
        
        return max_color

    best_size = 0
    best_mask = 0

    # Initial greedy solution - multiple passes with different orderings
    def greedy_solve(seed_order):
        """Greedy independent set using a given vertex ordering."""
        mask = 0
        P = all_mask
        for v in seed_order:
            if not (P & (1 << v)):
                continue
            if (conflict[v] & mask) == 0:
                mask |= (1 << v)
                P &= adj[v]
        return mask

    # Try several greedy orderings
    greedy_orders = [
        order,  # degree descending
        list(reversed(order)),  # degree ascending
        sorted(range(n), key=lambda x: -degrees[x] + random.random()),
        sorted(range(n), key=lambda x: random.random()),
    ]

    for go in greedy_orders:
        gm = greedy_solve(go)
        if gm.bit_count() > best_size:
            best_size = gm.bit_count()
            best_mask = gm

    # Also try greedy with dynamic degree (recompute at each step)
    def greedy_dynamic():
        mask = 0
        P = all_mask
        while P:
            # Pick vertex with highest degree in remaining P
            v = -1
            max_deg = -1
            p = P
            while p:
                lsb = p & -p
                i = lsb.bit_length() - 1
                deg = (adj[i] & P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = i
                p ^= lsb
            if v == -1:
                break
            mask |= (1 << v)
            P &= adj[v]
        return mask

    gm = greedy_dynamic()
    if gm.bit_count() > best_size:
        best_size = gm.bit_count()
        best_mask = gm

    # Branch and bound (Tomita-style) with improved vertex ordering
    # We use a recursive approach with bitsets
    def bbmc(P, current, depth):
        nonlocal best_size, best_mask

        if time.time() - start_time > timeout:
            return

        if P == 0:
            if len(current) > best_size:
                best_size = len(current)
                mask = 0
                for v in current:
                    mask |= (1 << v)
                best_mask = mask
            return

        bound = color_bound(P)
        if len(current) + bound <= best_size:
            return

        # Pick vertex with lowest rank (highest degree) among remaining
        v = -1
        min_rank = n + 1
        p = P
        while p:
            lsb = p & -p
            i = lsb.bit_length() - 1
            if rank[i] < min_rank:
                min_rank = rank[i]
                v = i
            p ^= lsb

        if v == -1:
            return

        v_bit = 1 << v

        # Include v
        new_P = P & adj[v]
        current.append(v)
        bbmc(new_P, current, depth + 1)
        current.pop()

        # Exclude v
        new_P_exc = P & ~v_bit
        bbmc(new_P_exc, current, depth + 1)

    # Run BBMC with a time budget
    bbmc_start = time.time()
    bbmc(all_mask, [], 0)
    bbmc_elapsed = time.time() - bbmc_start

    # Local search: try to add vertices
    if best_mask:
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if not ((best_mask >> i) & 1):
                    if (conflict[i] & best_mask) == 0:
                        best_mask |= (1 << i)
                        best_size += 1
                        improved = True
                        break

    # Simulated annealing refinement
    def mask_to_set(mask):
        s = set()
        p = mask
        while p:
            lsb = p & -p
            i = lsb.bit_length() - 1
            s.add(i)
            p ^= lsb
        return s

    def set_to_mask(s):
        m = 0
        for i in s:
            m |= (1 << i)
        return m

    # Precompute conflict sets for fast lookup
    conflict_sets = [set() for _ in range(n)]
    for i in range(n):
        c = conflict[i]
        p = c
        while p:
            lsb = p & -p
            j = lsb.bit_length() - 1
            conflict_sets[i].add(j)
            p ^= lsb

    # Precompute non-conflict sets (vertices that CAN be added to a set containing v)
    non_conflict_sets = [set() for _ in range(n)]
    for i in range(n):
        non_conflict_sets[i] = set(range(n)) - conflict_sets[i] - {i}

    def anneal(S_set, time_budget):
        """Simulated annealing over independent sets with multi-vertex moves."""
        S = set(S_set)
        T0 = 2.0
        T_end = 0.01
        cooling = 0.9997
        T = T0
        best_S = set(S)
        best_size_anneal = len(S)

        # Precompute list of all vertices for random selection
        all_verts = list(range(n))

        while time.time() - start_time < time_budget:
            if T < T_end:
                break

            if not S:
                break

            # Choose a random move type with weighted probabilities
            move_type = random.random()

            if move_type < 0.25 and len(S) > 0:
                # Remove a random vertex and greedily refill
                v = random.choice(tuple(S))
                S_new = S - {v}
                # Find all vertices that can be added to S_new
                cands = [u for u in all_verts if u not in S_new and not (conflict_sets[u] & S_new)]
                # Greedily add a random maximal independent subset of cands
                rem = set(cands)
                add = set()
                while rem:
                    vc = random.choice(tuple(rem))
                    add.add(vc)
                    rem -= conflict_sets[vc]
                    rem.discard(vc)
                S_cand = S_new | add

            elif move_type < 0.5 and len(S) > 1:
                # Swap: remove one, add one
                v = random.choice(tuple(S))
                S_new = S - {v}
                cands = [u for u in all_verts if u not in S_new and not (conflict_sets[u] & S_new)]
                if cands:
                    u = random.choice(cands)
                    S_cand = S_new | {u}
                else:
                    S_cand = S_new
            elif move_type < 0.7:
                # Try to add a random vertex
                not_in_S = [u for u in all_verts if u not in S]
                if not_in_S:
                    u = random.choice(not_in_S)
                    if not (conflict_sets[u] & S):
                        S_cand = S | {u}
                    else:
                        # Remove one conflicting vertex and add u
                        confs = conflict_sets[u] & S
                        if confs:
                            v = random.choice(tuple(confs))
                            S_new = S - {v}
                            if not (conflict_sets[u] & S_new):
                                S_cand = S_new | {u}
                            else:
                                S_cand = S
                        else:
                            S_cand = S
                else:
                    S_cand = S
            else:
                # 2-swap: remove two, add two
                if len(S) > 2:
                    v1, v2 = random.sample(tuple(S), 2)
                    S_new = S - {v1, v2}
                    cands = [u for u in all_verts if u not in S_new and not (conflict_sets[u] & S_new)]
                    add = set()
                    rem = set(cands)
                    while rem and len(add) < 2:
                        vc = random.choice(tuple(rem))
                        add.add(vc)
                        rem -= conflict_sets[vc]
                        rem.discard(vc)
                    S_cand = S_new | add
                else:
                    S_cand = S

            d = len(S_cand) - len(S)

            if d >= 0 or random.random() < math.exp(d / T):
                S = S_cand
                if len(S) > best_size_anneal:
                    best_size_anneal = len(S)
                    best_S = set(S)

            T *= cooling

        return best_S

    # Run annealing on the best solution
    remaining_time = timeout - (time.time() - start_time)
    if best_mask and remaining_time > 5:
        S_set = mask_to_set(best_mask)
        anneal_budget = min(remaining_time - 3, 40.0)
        if anneal_budget > 2:
            S_annealed = anneal(S_set, anneal_budget)
            if len(S_annealed) > best_size:
                best_size = len(S_annealed)
                best_mask = set_to_mask(S_annealed)

                # Try to add more vertices after annealing
                improved = True
                while improved:
                    improved = False
                    for i in range(n):
                        if not ((best_mask >> i) & 1):
                            if (conflict[i] & best_mask) == 0:
                                best_mask |= (1 << i)
                                best_size += 1
                                improved = True
                                break
            
            # Run a second annealing pass with a different random seed
            remaining_time2 = timeout - (time.time() - start_time)
            if remaining_time2 > 10:
                random.seed()
                S_annealed2 = anneal(mask_to_set(best_mask), min(remaining_time2 - 2, 25.0))
                if len(S_annealed2) > best_size:
                    best_size = len(S_annealed2)
                    best_mask = set_to_mask(S_annealed2)
                    improved = True
                    while improved:
                        improved = False
                        for i in range(n):
                            if not ((best_mask >> i) & 1):
                                if (conflict[i] & best_mask) == 0:
                                    best_mask |= (1 << i)
                                    best_size += 1
                                    improved = True
                                    break

    # Second round of BBMC with tighter bound if time permits
    remaining_time = timeout - (time.time() - start_time)
    if remaining_time > 10:
        # Re-run BBMC with the current best as starting point
        # This can help if the first BBMC didn't explore enough
        def bbmc_tight(P, current):
            nonlocal best_size, best_mask

            if time.time() - start_time > timeout:
                return

            if P == 0:
                if len(current) > best_size:
                    best_size = len(current)
                    mask = 0
                    for v in current:
                        mask |= (1 << v)
                    best_mask = mask
                return

            bound = color_bound(P)
            if len(current) + bound <= best_size:
                return

            # Pick vertex with lowest rank
            v = -1
            min_rank = n + 1
            p = P
            while p:
                lsb = p & -p
                i = lsb.bit_length() - 1
                if rank[i] < min_rank:
                    min_rank = rank[i]
                    v = i
                p ^= lsb

            if v == -1:
                return

            v_bit = 1 << v

            # Include v
            new_P = P & adj[v]
            current.append(v)
            bbmc_tight(new_P, current)
            current.pop()

            # Exclude v
            new_P_exc = P & ~v_bit
            bbmc_tight(new_P_exc, current)

        bbmc_tight(all_mask, [])

    # Final local search
    if best_mask:
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if not ((best_mask >> i) & 1):
                    if (conflict[i] & best_mask) == 0:
                        best_mask |= (1 << i)
                        best_size += 1
                        improved = True
                        break

    # Convert to list
    result = []
    for i in range(n):
        if (best_mask >> i) & 1:
            result.append(i + 1)

    if not result:
        return [1]

    return result

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a = res[i]
            b = res[j]
            if (a * b) % (a + b) == 0:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
