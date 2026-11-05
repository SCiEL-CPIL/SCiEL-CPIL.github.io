"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s to be safe under 60s limit
    N = 512
    n = N

    # Precompute compatibility masks:
    # masks[i] is a bitmask where bit j is 1 if lcm(i+1, j+1) <= N
    masks = [0] * n
    for i in range(n):
        a = i + 1
        for j in range(n):
            b = j + 1
            g = gcd(a, b)
            if a * b // g <= N:
                masks[i] |= (1 << j)

    # Degree of each vertex (number of compatible partners)
    degree = [bin(masks[i]).count('1') for i in range(n)]

    # Precompute for each vertex the list of compatible vertices (sorted by degree desc)
    compat_list = []
    for i in range(n):
        compat = [j for j in range(n) if (masks[i] >> j) & 1]
        compat.sort(key=lambda x: -degree[x])
        compat_list.append(compat)

    best_size = 0
    best_set = set()
    best_mask = 0

    def update_best(s, m):
        nonlocal best_size, best_set, best_mask
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
            best_mask = m

    def greedy_clique(order):
        """Greedy: process vertices in given order, add if compatible with all already added."""
        clique = set()
        mask = 0
        for v in order:
            if (masks[v] & mask) == mask:
                clique.add(v)
                mask |= (1 << v)
        return clique, mask

    def try_greedy_add(s, m):
        """Try to greedily add any compatible vertex. Returns (s, m, added)."""
        while True:
            added = False
            for v in range(n):
                if v not in s:
                    if (masks[v] & m) == m:
                        s.add(v)
                        m |= (1 << v)
                        added = True
                        break
            if not added:
                break
        return s, m, added

    def local_search(s, m):
        """Local search: greedy add, single-swap, double-swap (2-opt)."""
        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Phase 1: greedy add
            s, m, added = try_greedy_add(s, m)
            if added:
                improved = True
                update_best(s, m)

            # Phase 2: single-swap (remove u, add v)
            swapped = False
            s_list = list(s)
            for u in s_list:
                if time.time() > deadline:
                    break
                u_bit = 1 << u
                m_wo_u = m ^ u_bit
                for v in compat_list[u]:
                    if v in s:
                        continue
                    if (masks[v] & m_wo_u) == m_wo_u:
                        # Valid swap: remove u, add v
                        s.remove(u)
                        s.add(v)
                        m = m_wo_u | (1 << v)
                        swapped = True
                        update_best(s, m)
                        break
                if swapped:
                    break
            if swapped:
                improved = True

            # Phase 3: double-swap (remove u1,u2, add v1,v2)
            if time.time() < deadline:
                s_list2 = list(s)
                double_swapped = False
                for u1, u2 in combinations(s_list2, 2):
                    if time.time() > deadline:
                        break
                    u1_bit = 1 << u1
                    u2_bit = 1 << u2
                    m_wo_u1u2 = m ^ u1_bit ^ u2_bit
                    # Find two vertices v1, v2 not in s that are compatible with m_wo_u1u2 and with each other
                    candidates = [v for v in range(n) if v not in s and (masks[v] & m_wo_u1u2) == m_wo_u1u2]
                    if len(candidates) >= 2:
                        # Try pairs from candidates
                        for v1, v2 in combinations(candidates, 2):
                            if (masks[v1] >> v2) & 1:
                                # Valid double swap
                                s.remove(u1)
                                s.remove(u2)
                                s.add(v1)
                                s.add(v2)
                                m = m_wo_u1u2 | (1 << v1) | (1 << v2)
                                double_swapped = True
                                update_best(s, m)
                                break
                    if double_swapped:
                        break
                if double_swapped:
                    improved = True

        return s, m

    # Build initial seeds
    inits = []

    # 1. Natural order 1..22
    s1 = set(range(22))
    m1 = 0
    for i in s1:
        m1 |= (1 << i)
    inits.append((s1, m1))

    # 2. Divisors of 512
    divs512 = [i + 1 for i in range(n) if 512 % (i + 1) == 0]
    s2 = set(d - 1 for d in divs512)
    m2 = 0
    for i in s2:
        m2 |= (1 << i)
    inits.append((s2, m2))

    # 3. Powers of 2
    pows = set()
    p = 1
    while p <= 512:
        pows.add(p - 1)
        p *= 2
    m3 = 0
    for i in pows:
        m3 |= (1 << i)
    inits.append((pows, m3))

    # 4. Divisors of 360 (measured to give size 24)
    divs360 = [i + 1 for i in range(n) if 360 % (i + 1) == 0]
    s4 = set(d - 1 for d in divs360)
    m4 = 0
    for i in s4:
        m4 |= (1 << i)
    inits.append((s4, m4))

    # 5. Hub-based: pick high-degree hub, add all its neighbors greedily
    sorted_by_degree = sorted(range(n), key=lambda x: -degree[x])
    for hub_idx in range(min(10, n)):
        if time.time() > deadline:
            break
        hub = sorted_by_degree[hub_idx]
        s_hub = {hub}
        m_hub = 1 << hub
        # Greedily add compatible neighbors sorted by degree
        for v in compat_list[hub]:
            if (masks[v] & m_hub) == m_hub:
                s_hub.add(v)
                m_hub |= (1 << v)
        inits.append((s_hub, m_hub))

    # 6. Random seeds
    random.seed(42)
    for _ in range(10):
        s5 = set(random.sample(range(n), 20))
        m5 = 0
        for i in s5:
            m5 |= (1 << i)
        inits.append((s5, m5))

    # Run local search on each seed
    for s, m in inits:
        if time.time() > deadline:
            break
        s, m = local_search(s, m)

    # Greedy with multiple orderings
    # Natural order
    natural = list(range(n))
    c, cm = greedy_clique(natural)
    update_best(c, cm)
    if time.time() < deadline:
        c, cm = local_search(c, cm)

    # Degree descending
    sorted_desc = sorted(range(n), key=lambda x: -degree[x])
    c, cm = greedy_clique(sorted_desc)
    update_best(c, cm)
    if time.time() < deadline:
        c, cm = local_search(c, cm)

    # Degree ascending
    sorted_asc = sorted(range(n), key=lambda x: degree[x])
    c, cm = greedy_clique(sorted_asc)
    update_best(c, cm)
    if time.time() < deadline:
        c, cm = local_search(c, cm)

    # Random orderings
    for iteration in range(50):
        if time.time() > deadline:
            break
        random_order = list(range(n))
        random.shuffle(random_order)
        c, cm = greedy_clique(random_order)
        if len(c) > best_size:
            update_best(c, cm)
            c, cm = local_search(c, cm)
        elif len(c) >= best_size - 2 and time.time() < deadline - 5:
            # Close to best, try to improve
            c, cm = local_search(c, cm)

    # Final polish: try to add any remaining compatible elements to best_set
    if time.time() < deadline:
        s = set(best_set)
        m = best_mask
        s, m, _ = try_greedy_add(s, m)
        update_best(s, m)

    result = sorted([i + 1 for i in best_set])
    if not result:
        result = [1]
    return result

# EVOLVE-BLOCK-END
