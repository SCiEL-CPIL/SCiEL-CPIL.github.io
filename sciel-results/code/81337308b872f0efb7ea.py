import time
import random
import numpy as np

# Q3 edges (canonical, vertices 0..7)
_Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
_Q3_ADJ = [0] * 8
for _a, _b in _Q3_EDGES:
    _Q3_ADJ[_a] |= (1 << _b)
    _Q3_ADJ[_b] |= (1 << _a)

# Precompute Q3 edge orientations with candidate ordering
_Q3_EDGE_ORIENTS = []
for qa, qb in _Q3_EDGES:
    for (gu_q, gv_q) in [(qa, qb), (qb, qa)]:
        unassigned = [i for i in range(8) if i != gu_q and i != gv_q]
        def _cc(x, a, b):
            c = 0
            if _Q3_ADJ[x] & (1 << a): c += 1
            if _Q3_ADJ[x] & (1 << b): c += 1
            return c
        unassigned.sort(key=lambda x: _cc(x, gu_q, gv_q), reverse=True)
        _Q3_EDGE_ORIENTS.append((gu_q, gv_q, tuple(unassigned)))

def construct():
    n = 16
    t_start = time.time()
    deadline = t_start + 115.0
    full_mask = (1 << n) - 1

    def has_q3_after_add(adj, u, v, deadline):
        """Check if adding edge (u,v) to current adj creates a Q3 subgraph."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

        for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS:
            for (gu, gv) in [(u, v), (v, u)]:
                assignment = [None] * 8
                assignment[qa] = gu
                assignment[qb] = gv
                used_mask = (1 << gu) | (1 << gv)

                def bt(idx, used_mask):
                    if time.time() > deadline - 2:
                        return False
                    if idx == 6:
                        for a, b in _Q3_EDGES:
                            ga = assignment[a]
                            gb = assignment[b]
                            if not (adj[ga] & (1 << gb)):
                                return False
                        return True

                    x = unassigned[idx]
                    candidates = full_mask & ~used_mask
                    for y in range(8):
                        if assignment[y] is not None and (_Q3_ADJ[x] & (1 << y)):
                            hy = assignment[y]
                            candidates &= adj[hy]

                    while candidates:
                        bit = candidates & (-candidates)
                        g = bit.bit_length() - 1
                        candidates ^= bit

                        assignment[x] = g
                        new_used = used_mask | (1 << g)
                        if bt(idx + 1, new_used):
                            return True
                        assignment[x] = None
                    return False

                if bt(0, used_mask):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    return True

        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return False

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def edge_list(adj):
        es = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    es.append([u, v])
        return es

    def make_k4444():
        """Create K4,4,4,4: 4 groups of 4 vertices, complete 4-partite."""
        adj = [0] * n
        for g1 in range(4):
            for g2 in range(g1+1, 4):
                for v1 in range(g1*4, g1*4+4):
                    for v2 in range(g2*4, g2*4+4):
                        adj[v1] |= (1 << v2)
                        adj[v2] |= (1 << v1)
        return adj

    def make_k313():
        """Create K3,13: 3 vertices connected to 13 vertices."""
        adj = [0] * n
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        return adj

    def local_search(adj, cnt, deadline, max_iter=50):
        """Local search: try to add missing edges, and swap edges if stuck."""
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        existing = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] & (1 << j))]

        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break

            # Try to add a missing edge
            added = False
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline - 5:
                    break
                if not (adj[u] & (1 << v)):
                    if not has_q3_after_add(adj, u, v, deadline):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        cnt += 1
                        missing.remove((u, v))
                        added = True
                        break
            if added:
                continue

            # If stuck, try an edge swap: remove one existing edge, then try to add two missing
            if existing:
                random.shuffle(existing)
                random.shuffle(missing)
                for eu, ev in existing[:5]:
                    if time.time() > deadline - 5:
                        break
                    # Temporarily remove edge (eu, ev)
                    adj[eu] &= ~(1 << ev)
                    adj[ev] &= ~(1 << eu)
                    cnt -= 1

                    # Try to add up to 2 missing edges
                    gains = 0
                    new_missing = list(missing)
                    for u, v in new_missing:
                        if time.time() > deadline - 5:
                            break
                        if gains >= 2:
                            break
                        if not (adj[u] & (1 << v)):
                            if not has_q3_after_add(adj, u, v, deadline):
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
                                cnt += 1
                                gains += 1
                                if (u, v) in missing:
                                    missing.remove((u, v))

                    if gains < 1:
                        # Revert: put back the removed edge
                        adj[eu] |= (1 << ev)
                        adj[ev] |= (1 << eu)
                        cnt += 1
                    else:
                        # Keep the swap if we gained net edges
                        if (eu, ev) in existing:
                            existing.remove((eu, ev))
                        break

        return cnt

    best_edges = None
    best_count = -1

    # Strategy 1: K4,4,4,4 starting point (48 edges)
    adj = make_k4444()
    cnt = count_edges(adj)
    best_count = cnt
    best_edges = edge_list(adj)

    # Strategy 2: K3,13 starting point (39 edges)
    adj2 = make_k313()
    cnt2 = count_edges(adj2)
    if cnt2 > best_count:
        best_count = cnt2
        best_edges = edge_list(adj2)

    # Multiple random restarts with local search
    for attempt in range(30):
        if time.time() > deadline - 15:
            break

        if attempt % 3 == 0:
            adj = make_k4444()
            # Randomly remove some edges to diversify
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            for u, v in all_edges[:random.randint(0, 8)]:
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
        elif attempt % 3 == 1:
            adj = make_k313()
            # Add some random edges within the large part
            for _ in range(random.randint(0, 12)):
                u, v = random.sample(range(3, 16), 2)
                if not (adj[u] & (1 << v)):
                    if not has_q3_after_add(adj, u, v, deadline):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        else:
            # Random dense graph
            adj = [0] * n
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < 0.65:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            # Remove any Q3s by deleting edges
            changed = True
            while changed:
                changed = False
                for (qa, qb, unassigned) in _Q3_EDGE_ORIENTS[:4]:
                    for gu in range(n):
                        for gv in range(gu+1, n):
                            if not (adj[gu] & (1 << gv)):
                                continue
                            assignment = [None] * 8
                            assignment[qa] = gu
                            assignment[qb] = gv
                            used_mask = (1 << gu) | (1 << gv)

                            found = False
                            def bt_check(idx, um):
                                nonlocal found
                                if found:
                                    return
                                if idx == 6:
                                    for a, b in _Q3_EDGES:
                                        ga = assignment[a]
                                        gb = assignment[b]
                                        if not (adj[ga] & (1 << gb)):
                                            return
                                    found = True
                                    return
                                x = unassigned[idx]
                                candidates = full_mask & ~um
                                for y in range(8):
                                    if assignment[y] is not None and (_Q3_ADJ[x] & (1 << y)):
                                        hy = assignment[y]
                                        candidates &= adj[hy]
                                while candidates:
                                    bit = candidates & (-candidates)
                                    g = bit.bit_length() - 1
                                    candidates ^= bit
                                    assignment[x] = g
                                    bt_check(idx + 1, um | (1 << g))
                                    assignment[x] = None
                                    if found:
                                        return

                            bt_check(0, used_mask)
                            if found:
                                # Remove edge (gu, gv) to break this Q3
                                adj[gu] &= ~(1 << gv)
                                adj[gv] &= ~(1 << gu)
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break

        cnt = count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            best_edges = edge_list(adj)

        # Local search
        cnt = local_search(adj, cnt, deadline, max_iter=30)
        if cnt > best_count:
            best_count = cnt
            best_edges = edge_list(adj)

    # Final polish
    if time.time() < deadline - 10:
        adj_final = [0] * n
        for u, v in best_edges:
            adj_final[u] |= (1 << v)
            adj_final[v] |= (1 << u)
        cnt_final = count_edges(adj_final)
        cnt_final = local_search(adj_final, cnt_final, deadline, max_iter=80)
        if cnt_final > best_count:
            best_count = cnt_final
            best_edges = edge_list(adj_final)

    return best_edges
