"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s to be safe under 90s limit
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

    # Degree of each vertex
    degree = [bin(masks[i]).count('1') for i in range(n)]

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

    def local_search(s, m):
        """Local search: greedy add, then single-swap, then double-swap if time permits."""
        improved = True
        while improved and time.time() < deadline:
            improved = False

            # Phase 1: greedy add
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
            if added:
                improved = True
                update_best(s, m)

            # Phase 2: single-swap (remove u, add v)
            swapped = False
            s_list = list(s)
            random.shuffle(s_list)
            for u in s_list:
                if time.time() > deadline:
                    break
                u_bit = 1 << u
                m_wo_u = m ^ u_bit
                # Find best v to add
                best_v = -1
                best_degree = -1
                for v in range(n):
                    if v in s:
                        continue
                    if (masks[v] & m_wo_u) == m_wo_u:
                        if degree[v] > best_degree:
                            best_degree = degree[v]
                            best_v = v
                if best_v >= 0:
                    s.remove(u)
                    s.add(best_v)
                    m = m_wo_u | (1 << best_v)
                    swapped = True
                    update_best(s, m)
                    break
            if swapped:
                improved = True

            # Phase 3: double-swap (remove u1,u2, add v1,v2) - only if time remains
            if time.time() < deadline - 5:
                double_swapped = False
                s_list2 = list(s)
                random.shuffle(s_list2)
                for i in range(min(len(s_list2), 20)):
                    if time.time() > deadline - 5:
                        break
                    u1 = s_list2[i]
                    if i + 1 < len(s_list2):
                        u2 = s_list2[i + 1]
                    else:
                        continue
                    u1_bit = 1 << u1
                    u2_bit = 1 << u2
                    m_wo = m ^ u1_bit ^ u2_bit
                    # Find two vertices to add
                    candidates = []
                    for v in range(n):
                        if v not in s:
                            if (masks[v] & m_wo) == m_wo:
                                candidates.append(v)
                    if len(candidates) >= 2:
                        # Try all pairs of candidates (limited)
                        found = False
                        for a_idx in range(min(len(candidates), 15)):
                            if time.time() > deadline - 5:
                                break
                            v1 = candidates[a_idx]
                            v1_bit = 1 << v1
                            m_with_v1 = m_wo | v1_bit
                            for b_idx in range(a_idx + 1, min(len(candidates), 15)):
                                v2 = candidates[b_idx]
                                v2_bit = 1 << v2
                                if (masks[v2] & m_with_v1) == m_with_v1:
                                    # Valid double swap
                                    s.remove(u1)
                                    s.remove(u2)
                                    s.add(v1)
                                    s.add(v2)
                                    m = m_with_v1 | v2_bit
                                    double_swapped = True
                                    update_best(s, m)
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                if double_swapped:
                    improved = True

        return s, m

    # Build initial seeds using various strategies
    inits = []

    # 1. Degree-sorted greedy (descending - prefer high degree)
    order_desc = sorted(range(n), key=lambda x: -degree[x])
    s1, m1 = greedy_clique(order_desc)
    inits.append((s1, m1))

    # 2. Degree-sorted greedy (ascending - prefer low degree)
    order_asc = sorted(range(n), key=lambda x: degree[x])
    s2, m2 = greedy_clique(order_asc)
    inits.append((s2, m2))

    # 3. Natural order
    s3, m3 = greedy_clique(list(range(n)))
    inits.append((s3, m3))

    # 4. Divisors of 512
    divs512 = [i + 1 for i in range(n) if 512 % (i + 1) == 0]
    s4 = set(d - 1 for d in divs512)
    m4 = 0
    for i in s4:
        m4 |= (1 << i)
    inits.append((s4, m4))

    # 5. Powers of 2
    pows = set()
    p = 1
    while p <= 512:
        pows.add(p - 1)
        p *= 2
    m5 = 0
    for i in pows:
        m5 |= (1 << i)
    inits.append((pows, m5))

    # 6. Random restarts
    random.seed(42)
    for _ in range(30):
        random_order = list(range(n))
        random.shuffle(random_order)
        s_r, m_r = greedy_clique(random_order)
        inits.append((s_r, m_r))

    # Run local search on each initial seed
    for s, m in inits:
        if time.time() > deadline:
            break
        s, m = local_search(s, m)

    # Additional: try to improve best solution further with more random swaps
    if best_set and time.time() < deadline - 10:
        s = set(best_set)
        m = best_mask
        # Run additional local search rounds
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            s, m = local_search(s, m)

    result = sorted([i + 1 for i in best_set])
    if not result:
        result = [1]
    return result

# EVOLVE-BLOCK-END
