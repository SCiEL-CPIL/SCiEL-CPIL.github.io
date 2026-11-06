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
    Only triples containing u or v can become invalid.
    Key optimization: adding edge (u,v) only increases common-neighbor counts
    for triples that already had >=2 common neighbors with the new vertex.
    """
    n = len(adj_bitmasks)
    au = adj_bitmasks[u]
    av = adj_bitmasks[v]

    # For triples {u, i, j}: new common neighbors = (au | {v}) & N(i) & N(j)
    # = (au & N(i) & N(j)) | ({v} & N(i) & N(j))
    # The {v} part is 1 iff v is neighbor of both i and j.
    # So we only need to check if (au & N(i) & N(j)).bit_count() >= 3
    # OR ((au & N(i) & N(j)).bit_count() == 2 AND v is neighbor of both i and j)
    # Simpler: just check (new_au & N(i) & N(j)).bit_count() >= 3
    # But we can pre-filter: only pairs (i,j) where au & N(i) & N(j) has >= 2 bits
    # (since adding v can add at most 1 more bit)
    
    # Triples {u, i, j}: check if (au | (1<<v)) & N(i) & N(j) >= 3
    # This equals (au & N(i) & N(j)) + [v in N(i) and v in N(j)]
    # So we need (au & N(i) & N(j)).bit_count() >= 3 OR
    # ((au & N(i) & N(j)).bit_count() == 2 AND v in N(i) AND v in N(j))
    
    # Fast path: only check pairs (i,j) where au & N(i) & N(j) >= 2
    # because if it's < 2, even adding v (which contributes at most 1) can't reach 3
    
    # Precompute: for each i != u, v_u_i = au & adj_bitmasks[i] (common neighbors of u and i, excluding v since not yet added)
    # Then for pair (i,j): common_u_ij = v_u_i & adj_bitmasks[j]
    # Need common_u_ij.bit_count() >= 3 OR (common_u_ij.bit_count() == 2 AND (1<<v) & adj_bitmasks[i] AND (1<<v) & adj_bitmasks[j])
    
    # Even faster: just iterate and use the direct check but skip early
    for i in range(n):
        if i == u:
            continue
        ai = adj_bitmasks[i]
        common_ui = au & ai  # common neighbors of u and i (before adding edge)
        if common_ui.bit_count() < 2:
            # Even if v is common neighbor of i and j, max is 2+1=3... wait
            # If common_ui has 0 or 1 bits, then common_ui & N(j) has <=1 bit
            # Adding v: at most 1+1 = 2 < 3. So skip.
            # Actually: (au & ai & aj) <= common_ui.bit_count() <= 1
            # Plus possibly v if v in ai and v in aj: total <= 1+1 = 2 < 3
            continue
        for j in range(i + 1, n):
            if j == u:
                continue
            common_u_ij = common_ui & adj_bitmasks[j]
            bc = common_u_ij.bit_count()
            if bc >= 3:
                return False
            if bc == 2:
                # Check if v is neighbor of both i and j
                if (ai & (1 << v)) and (adj_bitmasks[j] & (1 << v)):
                    return False

    # Triples {v, i, j}: symmetric
    for i in range(n):
        if i == v:
            continue
        ai = adj_bitmasks[i]
        common_vi = av & ai
        if common_vi.bit_count() < 2:
            continue
        for j in range(i + 1, n):
            if j == v:
                continue
            common_v_ij = common_vi & adj_bitmasks[j]
            bc = common_v_ij.bit_count()
            if bc >= 3:
                return False
            if bc == 2:
                if (ai & (1 << u)) and (adj_bitmasks[j] & (1 << u)):
                    return False

    return True

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin under 120s

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

    def greedy_add(bitmasks):
        """Add any valid non-edge repeatedly until none remain. Exhaustive per pass."""
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
        """Local search: greedy add + swap moves to escape local optima."""
        bitmasks = initial_bitmasks[:]
        current_edges = count_edges(bitmasks)
        best_local_edges = current_edges
        best_local_bitmasks = bitmasks[:]

        # First, greedily add valid edges
        bitmasks = greedy_add(bitmasks)
        current_edges = count_edges(bitmasks)
        if current_edges > best_local_edges:
            best_local_edges = current_edges
            best_local_bitmasks = bitmasks[:]

        no_improve = 0
        while time.time() < deadline:
            moved = False

            # Try to add edges (exhaustive greedy)
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
                # Try swap: remove (a,b), add (c,d)
                all_edges = get_all_edges(bitmasks)
                if all_edges and non_edges:
                    for _ in range(40):
                        if time.time() > deadline:
                            break
                        a, b = random.choice(all_edges)
                        c, d = random.choice(non_edges)
                        # Skip if same edge
                        if (a == c and b == d) or (a == d and b == c):
                            continue
                        # Remove (a,b)
                        bm2 = bitmasks[:]
                        bm2[a] &= ~(1 << b)
                        bm2[b] &= ~(1 << a)
                        # Try add (c,d)
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
                no_improve += 1
                if no_improve > 120:
                    break
            else:
                no_improve = 0

        return best_local_bitmasks, best_local_edges

    # Multiple restarts with varied initializations
    num_restarts = 50
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
            # Repair until valid
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
            # Repair until valid
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
        improved = greedy_add(best_adj_bitmasks[:])
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
    # Verify K3,3-free
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
