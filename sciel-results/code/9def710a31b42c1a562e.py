import time
import random
import math

def has_k33_fast(adj_bitmasks):
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
    """Check if adding edge (u,v) keeps graph K3,3-free."""
    n = len(adj_bitmasks)
    new_u = adj_bitmasks[u] | (1 << v)
    new_v = adj_bitmasks[v] | (1 << u)
    
    # Check triples {u, i, j} for all i < j
    for i in range(n):
        if i == u:
            continue
        ai = adj_bitmasks[i]
        for j in range(i + 1, n):
            if j == u:
                continue
            if (new_u & ai & adj_bitmasks[j]).bit_count() >= 3:
                return False
    
    # Check triples {v, i, j} for all i < j
    for i in range(n):
        if i == v:
            continue
        ai = adj_bitmasks[i]
        for j in range(i + 1, n):
            if j == v:
                continue
            if (new_v & ai & adj_bitmasks[j]).bit_count() >= 3:
                return False
    
    return True

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin under 150s

    best_edges = 0
    best_adj_bitmasks = None

    def count_edges(bitmasks):
        return sum(b.bit_count() for b in bitmasks) // 2

    def bitmasks_to_edges(bitmasks):
        edges = []
        for i in range(n):
            m = bitmasks[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m -= lsb
        return edges

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
            m = bitmasks[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    all_edges.append((i, j))
                m -= lsb
        return all_edges

    def greedy_add_exhaustive(bitmasks):
        """Add any valid non-edge repeatedly until none remain."""
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
        """Local search with greedy add and swap moves."""
        bitmasks = initial_bitmasks[:]
        current_edges = count_edges(bitmasks)
        best_local_edges = current_edges
        best_local_bitmasks = bitmasks[:]

        # Greedy add
        bitmasks = greedy_add_exhaustive(bitmasks)
        current_edges = count_edges(bitmasks)
        if current_edges > best_local_edges:
            best_local_edges = current_edges
            best_local_bitmasks = bitmasks[:]

        no_improve = 0
        while time.time() < deadline and no_improve < 150:
            moved = False

            # Try adding edges
            non_edges = get_non_edges(bitmasks)
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
                # Try swap moves
                all_edges = get_all_edges(bitmasks)
                non_edges = get_non_edges(bitmasks)
                if all_edges and non_edges:
                    for _ in range(80):
                        if time.time() > deadline:
                            break
                        a, b = random.choice(all_edges)
                        c, d = random.choice(non_edges)
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
                # Try 2-for-1 swap
                all_edges = get_all_edges(bitmasks)
                non_edges = get_non_edges(bitmasks)
                if len(all_edges) >= 1 and len(non_edges) >= 2:
                    for _ in range(50):
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
                        if is_add_valid(bm2, c, d):
                            bm2[c] |= (1 << d)
                            bm2[d] |= (1 << c)
                            if is_add_valid(bm2, e, f):
                                bm2[e] |= (1 << f)
                                bm2[f] |= (1 << e)
                                if has_k33_fast(bm2):
                                    bitmasks = bm2
                                    current_edges = count_edges(bitmasks)
                                    moved = True
                                    if current_edges > best_local_edges:
                                        best_local_edges = current_edges
                                        best_local_bitmasks = bitmasks[:]
                                    break

            if not moved:
                no_improve += 1
            else:
                no_improve = 0

        return best_local_bitmasks, best_local_edges

    # Strategy: Start from various random dense graphs and repair+improve
    for r in range(80):
        if time.time() > deadline:
            break

        if r == 0:
            # Empty start
            init = [0] * n
        else:
            # Random graph with varying density
            p = 0.20 + 0.12 * (r % 6)
            init = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        init[i] |= (1 << j)
                        init[j] |= (1 << i)
            # Repair
            attempts = 0
            while not has_k33_fast(init) and attempts < 3000:
                all_e = get_all_edges(init)
                if not all_e:
                    break
                u, v = random.choice(all_e)
                init[u] &= ~(1 << v)
                init[v] &= ~(1 << u)
                attempts += 1

        if not has_k33_fast(init):
            continue

        local_best_bm, local_best_e = local_search(init)
        if local_best_e > best_edges:
            best_edges = local_best_e
            best_adj_bitmasks = local_best_bm[:]

    # Final greedy pass
    if best_adj_bitmasks is not None:
        improved = greedy_add_exhaustive(best_adj_bitmasks[:])
        improved_e = count_edges(improved)
        if improved_e > best_edges:
            best_edges = improved_e
            best_adj_bitmasks = improved

    if best_adj_bitmasks is None:
        return []

    return bitmasks_to_edges(best_adj_bitmasks)

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Found {len(edges)} edges in {t1-t0:.2f}s")
    # Verify
    n = 20
    adj = [0]*n
    for u,v in edges:
        adj[u] |= (1<<v)
        adj[v] |= (1<<u)
    valid = True
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                if (adj[i]&adj[j]&adj[k]).bit_count() >= 3:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    print(f"Valid K3,3-free: {valid}")
