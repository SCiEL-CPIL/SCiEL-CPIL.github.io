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
    """Check if adding edge (u,v) keeps graph K3,3-free.
    Only triples containing u or v can become newly invalid.
    """
    n = len(adj_bitmasks)
    adj_u = adj_bitmasks[u]
    adj_v = adj_bitmasks[v]
    
    # For triples {u,i,j}: check pairs (i,j) where v ∈ N(i)∩N(j)
    nv_list = []
    mv = adj_v
    while mv:
        lsb = mv & -mv
        i = lsb.bit_length() - 1
        if i != u and i != v:
            nv_list.append(i)
        mv -= lsb
    
    for idx_i in range(len(nv_list)):
        i = nv_list[idx_i]
        ai = adj_bitmasks[i]
        for idx_j in range(idx_i + 1, len(nv_list)):
            j = nv_list[idx_j]
            if (adj_u & ai & adj_bitmasks[j]).bit_count() >= 2:
                return False
    
    # For triples {v,i,j}: check pairs (i,j) where u ∈ N(i)∩N(j)
    nu_list = []
    mu = adj_u
    while mu:
        lsb = mu & -mu
        i = lsb.bit_length() - 1
        if i != u and i != v:
            nu_list.append(i)
        mu -= lsb
    
    for idx_i in range(len(nu_list)):
        i = nu_list[idx_i]
        ai = adj_bitmasks[i]
        for idx_j in range(idx_i + 1, len(nu_list)):
            j = nu_list[idx_j]
            if (adj_v & ai & adj_bitmasks[j]).bit_count() >= 2:
                return False
    
    return True

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s

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
        """Add valid non-edges, prioritizing those with fewer common neighbors."""
        changed = True
        while changed:
            changed = False
            non_edges = get_non_edges(bitmasks)
            # Score each non-edge: prefer edges with fewer common neighbors (less likely to create K3,3)
            def edge_score(u, v):
                common = (bitmasks[u] & bitmasks[v]).bit_count()
                return common
            non_edges.sort(key=lambda uv: edge_score(uv[0], uv[1]))
            for u, v in non_edges:
                if is_add_valid(bitmasks, u, v):
                    bitmasks[u] |= (1 << v)
                    bitmasks[v] |= (1 << u)
                    changed = True
        return bitmasks

    def projective_plane_seed():
        """Create a seed from projective plane of order 4 (21 points, remove 1)."""
        # Projective plane of order 4 has 21 points and 21 lines, each line has 5 points
        # Each point is on 5 lines, any two points on exactly 1 line
        # We'll use a known construction for PG(2,4)
        # Points: (x,y) for x,y in GF(4), plus line at infinity with 5 points
        # GF(4) = {0, 1, a, a^2} where a^2 = a + 1
        
        # Use a simpler approach: create a 5-regular graph on 20 vertices that is K3,3-free
        # by using the incidence structure of a projective plane minus one point
        
        # Actually, let's use a known K3,3-free dense graph: the complement of a cycle C20
        # gives 190 - 20 = 170 edges, but that's not K3,3-free.
        
        # Let's try: take vertices 0..19, connect i to j if |i-j| mod 20 is in {1,2,3,4,5,6,7,8,9}
        # That's the complete graph minus a 9-regular graph, giving 190 - 90 = 100 edges
        # But we need to check K3,3-freeness.
        
        # Simpler: use a random 8-regular graph and repair
        bm = [0] * n
        for i in range(n):
            candidates = [j for j in range(n) if j != i]
            random.shuffle(candidates)
            for j in candidates[:8]:
                if not (bm[i] & (1 << j)):
                    bm[i] |= (1 << j)
                    bm[j] |= (1 << i)
        # Symmetrize
        for i in range(n):
            for j in range(i+1, n):
                if (bm[i] & (1 << j)) != (bm[j] & (1 << i)):
                    if bm[i] & (1 << j):
                        bm[j] |= (1 << i)
                    else:
                        bm[i] &= ~(1 << j)
        # Repair if needed
        attempts = 0
        while not has_k33_fast(bm) and attempts < 500:
            all_e = get_all_edges(bm)
            if not all_e:
                break
            u, v = random.choice(all_e)
            bm[u] &= ~(1 << v)
            bm[v] &= ~(1 << u)
            attempts += 1
        return bm

    def local_search(initial_bitmasks, time_limit=10.0):
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
        max_no_improve = 100
        t0 = time.time()
        while time.time() < t0 + time_limit and time.time() < deadline and no_improve < max_no_improve:
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
                # Try swap moves: remove one edge, add one different edge
                all_edges = get_all_edges(bitmasks)
                non_edges = get_non_edges(bitmasks)
                if all_edges and non_edges:
                    for _ in range(50):
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
                # Try 2-for-1 swap: remove one edge, add two different edges
                all_edges = get_all_edges(bitmasks)
                non_edges = get_non_edges(bitmasks)
                if len(all_edges) >= 1 and len(non_edges) >= 2:
                    for _ in range(60):
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

    # Strategy 1: Start from various random dense graphs and repair+improve
    for r in range(120):
        if time.time() > deadline:
            break

        if r == 0:
            # Empty start
            init = [0] * n
        elif r == 1:
            # Projective plane seed
            init = projective_plane_seed()
            if not has_k33_fast(init):
                continue
        elif r < 21:
            # Random graph with varying density
            p = 0.15 + 0.05 * ((r - 1) % 8)
            init = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        init[i] |= (1 << j)
                        init[j] |= (1 << i)
            # Repair
            attempts = 0
            while not has_k33_fast(init) and attempts < 2000:
                all_e = get_all_edges(init)
                if not all_e:
                    break
                u, v = random.choice(all_e)
                init[u] &= ~(1 << v)
                init[v] &= ~(1 << u)
                attempts += 1
        elif r < 41:
            # Start from a 5-regular-ish random graph
            init = [0] * n
            for _ in range(50):
                bm = [0] * n
                for i in range(n):
                    candidates = [j for j in range(n) if j != i]
                    random.shuffle(candidates)
                    for j in candidates[:5]:
                        if not (bm[i] & (1 << j)):
                            bm[i] |= (1 << j)
                            bm[j] |= (1 << i)
                for i in range(n):
                    for j in range(i+1, n):
                        if (bm[i] & (1 << j)) != (bm[j] & (1 << i)):
                            if bm[i] & (1 << j):
                                bm[j] |= (1 << i)
                            else:
                                bm[i] &= ~(1 << j)
                if has_k33_fast(bm):
                    init = bm
                    break
            if not has_k33_fast(init):
                continue
        else:
            # Higher density random graphs
            p = 0.3 + 0.05 * ((r - 41) % 6)
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

        local_best_bm, local_best_e = local_search(init, time_limit=8.0)
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

        # Final local search polish
        if time.time() < deadline - 5:
            final_bm, final_e = local_search(best_adj_bitmasks[:], time_limit=30.0)
            if final_e > best_edges:
                best_edges = final_e
                best_adj_bitmasks = final_bm
            # One more polish pass if time remains
            if time.time() < deadline - 10:
                final_bm2, final_e2 = local_search(best_adj_bitmasks[:], time_limit=20.0)
                if final_e2 > best_edges:
                    best_edges = final_e2
                    best_adj_bitmasks = final_bm2

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
