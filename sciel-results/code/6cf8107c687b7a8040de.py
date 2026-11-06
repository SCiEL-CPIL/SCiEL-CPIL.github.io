import random
import time

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 118  # safety margin vs 150s kill

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    best_edges = []
    best_count = 0

    def greedy_fill():
        """Start empty, add edges in random order whenever valid."""
        adj = [0] * n
        common = [[0] * n for _ in range(n)]
        count = 0

        def valid(u, v):
            if adj[u] & (1 << v):
                return True
            if common[u][v] >= 3:
                return False
            m = adj[v] & ~(1 << u)
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                if common[u][w] >= 2:
                    return False
                m ^= lsb
            m = adj[u] & ~(1 << v)
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                if common[v][w] >= 2:
                    return False
                m ^= lsb
            return True

        def add(u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            m = adj[v] & ~(1 << u)
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                common[u][w] += 1
                common[w][u] += 1
                m ^= lsb
            m = adj[u] & ~(1 << v)
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                common[v][w] += 1
                common[w][v] += 1
                m ^= lsb

        order = all_pairs[:]
        random.shuffle(order)
        for u, v in order:
            if valid(u, v):
                add(u, v)
                count += 1
        return adj, common, count

    def refill(adj, common, count):
        """Add any currently-valid missing edge, to a fixed point."""
        changed = True
        while changed:
            if time.time() > deadline:
                return count
            changed = False
            order = all_pairs[:]
            random.shuffle(order)
            for u, v in order:
                if adj[u] & (1 << v):
                    continue
                if common[u][v] >= 3:
                    continue
                ok = True
                m = adj[v] & ~(1 << u)
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    if common[u][w] >= 2:
                        ok = False
                        break
                    m ^= lsb
                if ok:
                    m = adj[u] & ~(1 << v)
                    while m:
                        lsb = m & -m
                        w = lsb.bit_length() - 1
                        if common[v][w] >= 2:
                            ok = False
                            break
                        m ^= lsb
                if ok:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    m = adj[v] & ~(1 << u)
                    while m:
                        lsb = m & -m
                        w = lsb.bit_length() - 1
                        common[u][w] += 1
                        common[w][u] += 1
                        m ^= lsb
                    m = adj[u] & ~(1 << v)
                    while m:
                        lsb = m & -m
                        w = lsb.bit_length() - 1
                        common[v][w] += 1
                        common[w][v] += 1
                        m ^= lsb
                    count += 1
                    changed = True
        return count

    def extract(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    # --- Simulated annealing with remove-and-refill moves ---
    adj, common, count = greedy_fill()
    count = refill(adj, common, count)
    if count > best_count:
        best_count = count
        best_edges = extract(adj)

    # Track per-vertex degree for a diversity-weighted removal proposal
    def degrees(adj):
        d = [0] * n
        for i in range(n):
            d[i] = bin(adj[i]).count('1')
        return d

    temp_start = 8.0
    temp_end = 0.5
    it = 0
    total_it = 0
    # estimate number of SA iterations we can afford
    while time.time() < deadline - 8:
        it += 1
        if it > 400000:
            break
    total_it = max(it, 100)
    it = 0

    while time.time() < deadline - 8:
        it += 1
        T = temp_start * (temp_end / temp_start) ** (it / total_it)

        # Propose: remove k random edges (k biased by current density)
        present = [(u, v) for (u, v) in all_pairs if adj[u] & (1 << v)]
        if not present:
            break
        k = random.randint(1, 4)
        random.shuffle(present)
        removed = present[:k]

        backup_adj = adj[:]
        backup_common = [row[:] for row in common]
        backup_count = count

        for u, v in removed:
            # remove edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            m = adj[v] & ~(1 << u)
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                common[u][w] -= 1
                common[w][u] -= 1
                m ^= lsb
            m = adj[u] & ~(1 << v)
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                common[v][w] -= 1
                common[w][v] -= 1
                m ^= lsb
            count -= 1

        # Refill greedily
        count = refill(adj, common, count)

        delta = count - backup_count
        if delta >= 0:
            accept = True
        else:
            # accept worse with probability exp(delta / T)
            accept = (random.random() < math.exp(delta / T)) if T > 1e-6 else False
        if accept:
            if count > best_count:
                best_count = count
                best_edges = extract(adj)
        else:
            # restore
            adj = backup_adj
            common = backup_common
            count = backup_count

        # Occasionally re-seed from the best to avoid long drift
        if it % 400 == 0 and count < best_count - 6:
            adj, common, count = greedy_fill()
            count = refill(adj, common, count)
            if count > best_count:
                best_count = count
                best_edges = extract(adj)

    # Final polish: try to add anything valid
    count = refill(adj, common, count)
    if count > best_count:
        best_count = count
        best_edges = extract(adj)

    return best_edges

import math
