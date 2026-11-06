import random
import time
import sys
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n = 16
    rng = random.Random(12345)

    # Precompute 8-subsets and their bipartitions once
    subsets8 = list(itertools.combinations(range(n), 8))
    sub_idx = {s: i for i, s in enumerate(subsets8)}
    sub_masks = []
    for s in subsets8:
        m = 0
        for v in s:
            m |= (1 << v)
        sub_masks.append(m)

    # For each 8-subset, precompute (x_mask, y_mask) for all 70 bipartitions
    part_lists = []
    for s in subsets8:
        parts = []
        full = sub_masks[sub_idx[s]]
        for x in itertools.combinations(range(8), 4):
            xm = 0
            for i in x:
                xm |= (1 << s[i])
            ym = full ^ xm
            parts.append((xm, ym))
        part_lists.append(parts)

    def has_q3(adj):
        for idx in range(len(subsets8)):
            sm = sub_masks[idx]
            s = subsets8[idx]
            e = 0
            for v in s:
                e += (adj[v] & sm).bit_count()
            if e < 12:
                continue
            for xm, ym in part_lists[idx]:
                ok = True
                x = xm
                while x:
                    lsb = x & (-x)
                    v = lsb.bit_length() - 1
                    if (adj[v] & ym).bit_count() < 3:
                        ok = False
                        break
                    x ^= lsb
                if not ok:
                    continue
                y = ym
                while y:
                    lsb = y & (-y)
                    v = lsb.bit_length() - 1
                    if (adj[v] & xm).bit_count() < 3:
                        ok = False
                        break
                    y ^= lsb
                if ok:
                    return True
        return False

    # Build Shrikhande graph = complement of 4x4 rook graph
    adj = [0] * n
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if u // 4 != v // 4 and u % 4 != v % 4:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))

    if has_q3(adj):
        # Fallback to K3,13
        adj = [0] * n
        edges = set()
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                edges.add((a, b))
        for u, v in [(0,1),(1,2),(0,2)]:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.add((u, v))
    else:
        print(f"Shrikhande seed: {len(edges)} edges, Q3-free", file=sys.stderr)

    best_edges = set(edges)
    best_count = len(edges)

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    def try_add(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        ok = not has_q3(adj)
        if ok:
            edges.add((u, v))
        else:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        return ok

    # Greedy add
    while time.time() < deadline - 20:
        progress = False
        missing = [e for e in all_edges if e not in edges]
        rng.shuffle(missing)
        for (u, v) in missing:
            if try_add(u, v):
                progress = True
        if not progress:
            break

    if len(edges) > best_count:
        best_count = len(edges)
        best_edges = set(edges)

    # 1-1 swaps
    for it in range(50000):
        if time.time() > deadline - 15:
            break
        present_list = list(edges)
        if not present_list:
            break
        ru, rv = rng.choice(present_list)
        adj[ru] &= ~(1 << rv)
        adj[rv] &= ~(1 << ru)
        edges.discard((ru, rv))

        cand = None
        for _ in range(50):
            cu, cv = rng.choice(all_edges)
            if (cu, cv) not in edges and (cu, cv) != (ru, rv):
                cand = (cu, cv)
                break
        if cand is None:
            adj[ru] |= (1 << rv)
            adj[rv] |= (1 << ru)
            edges.add((ru, rv))
            continue
        cu, cv = cand
        adj[cu] |= (1 << cv)
        adj[cv] |= (1 << cu)
        if not has_q3(adj):
            edges.add((cu, cv))
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = set(edges)
        else:
            adj[cu] &= ~(1 << cv)
            adj[cv] &= ~(1 << cu)
            adj[ru] |= (1 << rv)
            adj[rv] |= (1 << ru)
            edges.add((ru, rv))

    # Final greedy polish
    missing2 = [e for e in all_edges if e not in edges]
    for (u, v) in missing2:
        if time.time() > deadline - 5:
            break
        try_add(u, v)
    if len(edges) > best_count:
        best_count = len(edges)
        best_edges = set(edges)

    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")
