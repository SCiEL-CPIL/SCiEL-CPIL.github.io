"""Local search for C6-free 4-colorings of K_n with bit-parallel C6 counting."""
import time
import random

_DEADLINE = 145.0

def _c6_count(adj, n):
    """Count C6s in one color class (adj: list of int bitsets)."""
    cnt = 0
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                continue
            # Remove (u,v), check if path of length 5 exists from v to u
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            lvl = 1 << v
            vis = lvl
            hit = False
            for s in range(5):
                nl = 0
                t = lvl
                while t:
                    lsb = t & (-t)
                    i = lsb.bit_length() - 1
                    nl |= adj[i]
                    t ^= lsb
                nl &= ~vis
                if not nl:
                    break
                lvl = nl
                vis |= nl
                if (lvl >> u) & 1 and s + 1 == 5:
                    cnt += 1
                    hit = True
                    break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return cnt


def construct():
    start = time.time()
    deadline = start + _DEADLINE

    for n in [18, 17, 16, 15, 14, 13, 12, 11, 10, 9]:
        if time.time() > deadline:
            break
        m = n * (n - 1) // 2
        # Precompute edge index -> (u, v)
        edge_uv = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_uv.append((i, j))

        for attempt in range(40):
            if time.time() > deadline:
                break
            rng = random.Random(attempt * 31 + n * 7)
            colors = [rng.randint(0, 3) for _ in range(m)]

            # Build adj
            adj = [[0] * n for _ in range(4)]
            for idx in range(m):
                u, v = edge_uv[idx]
                c = colors[idx]
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)

            cost = sum(_c6_count(adj[c], n) for c in range(4))
            if cost == 0:
                return {'n': n, 'colors': colors}

            # Local search
            T = 4.0
            cool = 0.997
            minT = 0.005
            best = cost
            best_col = colors[:]
            no_improve = 0

            while time.time() < deadline and T > minT:
                pos = rng.randint(0, m - 1)
                u, v = edge_uv[pos]
                old_c = colors[pos]
                new_c = rng.randint(0, 3)
                if new_c == old_c:
                    continue

                # Remove old edge
                adj[old_c][u] &= ~(1 << v)
                adj[old_c][v] &= ~(1 << u)
                # Add new edge
                adj[new_c][u] |= (1 << v)
                adj[new_c][v] |= (1 << u)
                colors[pos] = new_c

                new_cost = sum(_c6_count(adj[c], n) for c in range(4))

                if new_cost < best:
                    best = new_cost
                    best_col = colors[:]
                    no_improve = 0
                    if best == 0:
                        return {'n': n, 'colors': best_col}
                else:
                    no_improve += 1

                # Accept or reject
                if new_cost < cost or rng.random() < pow(2.71828, (cost - new_cost) / T):
                    cost = new_cost
                else:
                    # Undo
                    colors[pos] = old_c
                    adj[new_c][u] &= ~(1 << v)
                    adj[new_c][v] &= ~(1 << u)
                    adj[old_c][u] |= (1 << v)
                    adj[old_c][v] |= (1 << u)

                T *= cool

                if no_improve > 5000:
                    break

    # Fallback
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
