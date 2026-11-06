import time
import random
import numpy as np

def has_k33_fast(adj_bitmasks):
    """
    Check if graph is K3,3-free (no triple has >= 3 common neighbors).
    Returns True if valid.
    """
    n = len(adj_bitmasks)
    for i in range(n):
        ai = adj_bitmasks[i]
        for j in range(i + 1, n):
            common_ij = ai & adj_bitmasks[j]
            if common_ij.bit_count() < 3:
                continue
            for k in range(j + 1, n):
                common_ijk = common_ij & adj_bitmasks[k]
                if common_ijk.bit_count() >= 3:
                    return False
    return True

def is_add_valid(adj_bitmasks, u, v):
    """
    Check if adding edge (u,v) keeps the graph K3,3-free.
    Key optimization: when adding (u,v), N(u) gains v and N(v) gains u.
    A triple {u,i,j} can only become invalid if i and j are BOTH in N(v)\{u}
    (i.e., they are new common neighbors of u after adding the edge).
    Similarly for {v,i,j} with i,j in N(u)\{v}.
    Only check pairs within these small sets.
    """
    # New common neighbors of u after adding (u,v): N(v) \ {u}
    new_common_u = adj_bitmasks[v] & ~(1 << u)
    # New common neighbors of v after adding (u,v): N(u) \ {v}
    new_common_v = adj_bitmasks[u] & ~(1 << v)

    # Check triples {u, i, j} where i,j are in new_common_u
    nc_u_list = []
    m = new_common_u
    while m:
        lsb = m & -m
        nc_u_list.append(lsb.bit_length() - 1)
        m -= lsb

    new_au = adj_bitmasks[u] | (1 << v)
    for ii in range(len(nc_u_list)):
        i = nc_u_list[ii]
        ai = adj_bitmasks[i]
        for jj in range(ii + 1, len(nc_u_list)):
            j = nc_u_list[jj]
            common = new_au & ai & adj_bitmasks[j]
            if common.bit_count() >= 3:
                return False

    # Check triples {v, i, j} where i,j are in new_common_v
    nc_v_list = []
    m = new_common_v
    while m:
        lsb = m & -m
        nc_v_list.append(lsb.bit_length() - 1)
        m -= lsb

    new_av = adj_bitmasks[v] | (1 << u)
    for ii in range(len(nc_v_list)):
        i = nc_v_list[ii]
        ai = adj_bitmasks[i]
        for jj in range(ii + 1, len(nc_v_list)):
            j = nc_v_list[jj]
            common = new_av & ai & adj_bitmasks[j]
            if common.bit_count() >= 3:
                return False

    return True

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 150s

    best_edges = 0
    best_adj_bitmasks = None

    def bitmasks_to_edges(bitmasks):
        edges = []
        for i in range(n):
            mask = bitmasks[i]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m -= lsb
        return edges

    def count_edges(bitmasks):
        return sum(b.bit_count() for b in bitmasks) // 2

    def get_non_edges(bitmasks):
        non_edges = []
        for i in range(n):
            mi = bitmasks[i]
            for j in range(i + 1, n):
                if not (mi & (1 << j)):
                    non_edges.append((i, j))
        return non_edges

    def get_all_edges(bitmasks):
        all_edges = []
        for i in range(n):
            mask = bitmasks[i]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    all_edges.append((i, j))
                m -= lsb
        return all_edges

    def greedy_add_exhaustive(bitmasks):
        """Add any valid non-edge repeatedly until none remain (exhaustive)."""
        changed = True
        while changed:
            changed = False
            non_edges = get_non_edges(bitmasks)
            random.shuffle(non_edges)
            for u, v in non_edges:
                if is_add_valid(bitmasks, u, v):
                    bitmasks[u] |= (1 << v)
                    bitmasks[v] |= (1 << u)
                    changed = True
        return bitmasks

    def local_search(initial_bitmasks):
        """Local search: greedy add + 1-swap + 2-swap moves to escape local optima."""
        bitmasks = initial_bitmasks[:]
        current_edges = count_edges(bitmasks)
        best_local_edges = current_edges
        best_local_bitmasks = bitmasks[:]

        # First, greedily add valid edges
        bitmasks = greedy_add_exhaustive(bitmasks)
        current_edges = count_edges(bitmasks)
        if current_edges > best_local_edges:
            best_local_edges = current_edges
            best_local_bitmasks = bitmasks[:]

        no_improve = 0
        while time.time() < deadline:
            moved = False

            # Try to add edges: exhaustive pass
            non_edges = get_non_edges(bitmasks)
            if non_edges:
                random.shuffle(non_edges)
                for u, v in non_edges:
                    if is_add_valid(bitmasks, u, v):
                        bitmasks[u] |= (1 << v)
                        bitmasks[v] |= (1 << u)
                        current_edges += 1
                        moved = True
                        if current_edges > best_local_edges:
                            best_local_edges = current_edges
                            best_local_bitmasks = bitmasks[:]

            if not moved:
                # Try 1-edge swap: remove (a,b), add (c,d)
                all_edges = get_all_edges(bitmasks)
                non_edges = get_non_edges(bitmasks)
                if all_edges and non_edges:
                    for _ in range(60):
                        if time.time() > deadline:
                            break
                        a, b = random.choice(all_edges)
                        c, d = random.choice(non_edges)
                        if (a == c and b == d) or (a == d and b == c):
                            continue
                        bm2 = bitmasks[:]
                        bm2[a] &= ~(1 << b)
                        bm2[b] &= ~(1 << a)
                        if is_add_valid(bm2, c, d):
                            bm2[c] |= (1 << d)
                            bm2[d] |= (1 << c)
                            if has_k33_fast(bm2):
                                bitmasks = bm2
                                current_edges = count_edges(bitmasks)
                                moved = True
                                if current_edges > best_local_edges:
                                    best_local_edges = current_edges
                                    best_local_bitmasks = bitmasks[:]
                                break

            if not moved:
                # Try 2-edge swap: remove (a,b), add (c,d) and (e,f)
                all_edges = get_all_edges(bitmasks)
                non_edges = get_non_edges(bitmasks)
                if len(all_edges) >= 1 and len(non_edges) >= 2:
                    for _ in range(40):
                        if time.time() > deadline:
                            break
                        a, b = random.choice(all_edges)
                        c, d = random.choice(non_edges)
                        e, f = random.choice(non_edges)
                        if (c, d) == (e, f) or (c, d) == (f, e):
                            continue
                        bm2 = bitmasks[:]
                        bm2[a] &= ~(1 << b)
                        bm2[b] &= ~(1 << a)
                        ok = True
                        if not is_add_valid(bm2, c, d):
                            ok = False
                        else:
                            bm2[c] |= (1 << d)
                            bm2[d] |= (1 << c)
                            if not is_add_valid(bm2, e, f):
                                ok = False
                            else:
                                bm2[e] |= (1 << f)
                                bm2[f] |= (1 << e)
                        if ok and has_k33_fast(bm2):
                            bitmasks = bm2
                            current_edges = count_edges(bitmasks)
                            moved = True
                            if current_edges > best_local_edges:
                                best_local_edges = current_edges
                                best_local_bitmasks = bitmasks[:]
                            break

            if not moved:
                no_improve += 1
                if no_improve > 120:
                    break
            else:
                no_improve = 0

        return best_local_bitmasks, best_local_edges

    # Multiple restarts with varied initializations
    num_restarts = 80
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        if r == 0:
            # Start empty, greedy add
            init_bitmasks = [0] * n
        elif r == 1:
            # Start with a random moderate graph, repair, then greedy
            init_bitmasks = [0] * n
            p = 0.3
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        init_bitmasks[i] |= (1 << j)
                        init_bitmasks[j] |= (1 << i)
            attempts = 0
            while not has_k33_fast(init_bitmasks) and attempts < 3000:
                all_edges = get_all_edges(init_bitmasks)
                if not all_edges:
                    break
                u, v = random.choice(all_edges)
                init_bitmasks[u] &= ~(1 << v)
                init_bitmasks[v] &= ~(1 << u)
                attempts += 1
        else:
            # Random sparse start with varying density
            init_bitmasks = [0] * n
            p = 0.15 + 0.02 * (r % 15)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        init_bitmasks[i] |= (1 << j)
                        init_bitmasks[j] |= (1 << i)
            attempts = 0
            while not has_k33_fast(init_bitmasks) and attempts < 3000:
                all_edges = get_all_edges(init_bitmasks)
                if not all_edges:
                    break
                u, v = random.choice(all_edges)
                init_bitmasks[u] &= ~(1 << v)
                init_bitmasks[v] &= ~(1 << u)
                attempts += 1

        if not has_k33_fast(init_bitmasks):
            continue

        local_best_bitmasks, local_best_edges = local_search(init_bitmasks)

        if local_best_edges > best_edges:
            best_edges = local_best_edges
            best_adj_bitmasks = local_best_bitmasks[:]

    # Final: one more greedy pass on the best found
    if best_adj_bitmasks is not None:
        improved = greedy_add_exhaustive(best_adj_bitmasks[:])
        improved_edges = count_edges(improved)
        if improved_edges > best_edges:
            best_edges = improved_edges
            best_adj_bitmasks = improved

    if best_adj_bitmasks is None:
        return []

    return bitmasks_to_edges(best_adj_bitmasks)

# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Found {len(edges)} edges in {t1-t0:.2f}s")
    n = 20
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                common = adj[i] & adj[j] & adj[k]
                if np.sum(common) >= 3:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid K3,3-free: {valid}")
