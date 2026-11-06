import time
import random

def has_k33_fast(adj_bitmasks):
    n = len(adj_bitmasks)
    for i in range(n):
        ai = adj_bitmasks[i]
        for j in range(i + 1, n):
            common_ij = ai & adj_bitmasks[j]
            if common_ij.bit_count() < 3:
                continue
            for k in range(j + 1, n):
                if (common_ij & adj_bitmasks[k]).bit_count() >= 3:
                    return False
    return True

def is_add_valid(adj_bitmasks, u, v):
    """Check if adding edge (u,v) keeps graph K3,3-free.
    After adding (u,v): for triple {u,i,j}, new_common = (adj[u]|(1<<v)) & adj[i] & adj[j].
    Since old graph was valid, adj[u]&adj[i]&adj[j] has bit_count < 3.
    The only way new_common reaches 3 is if v is a common neighbor of i,j AND
    adj[u]&adj[i]&adj[j] already had bit_count >= 2.
    So only check pairs (i,j) in N(v) (for u-triples) and N(u) (for v-triples).
    """
    n = len(adj_bitmasks)
    adj_u = adj_bitmasks[u]
    adj_v = adj_bitmasks[v]

    # Triples {u,i,j}: need (adj_u & adj[i] & adj[j]).bit_count() <= 1
    # for all pairs (i,j) where v in N(i)∩N(j), i.e., i,j in N(v)
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

    # Triples {v,i,j}: need (adj_v & adj[i] & adj[j]).bit_count() <= 1
    # for all pairs (i,j) where u in N(i)∩N(j), i.e., i,j in N(u)
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

def _gf4_table():
    # GF(4) = F2[a]/(a^2+a+1). Elements: 0,1,a,a+1.
    # Represent as 2-bit: 0=0, 1=1, 2=a, 3=a+1.
    # a^2 = a+1 = 3, a^3 = 1, a^0=1.
    add_tbl = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
    mul_tbl = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
    return add_tbl, mul_tbl

def _projective_plane_seed():
    """Build projective plane of order 4: 21 points (1D subspaces of GF(4)^3),
    two points adjacent iff they lie on a common 2D subspace (line).
    Each point is on 5 lines, each line has 5 points. Truncate to 20 vertices."""
    add, mul = _gf4_table()
    def add3(a, b):
        return [add[a[0]][b[0]], add[a[1]][b[1]], add[a[2]][b[2]]]
    def sub3(a, b):
        return add3(a, b)  # char 2
    def mulc(c, a):
        return [mul[c][a[0]], mul[c][a[1]], mul[c][a[2]]]
    def norm(a):
        # representative: first nonzero coordinate = 1
        for k in range(3):
            if a[k] != 0:
                if a[k] == 1:
                    return [0,0,0][:k] + [1] + [0]*(2-k)
                inv = {2:3, 3:2}[a[k]]
                return [0]*k + [1] + [mul[inv][a[t]] for t in range(k+1,3)]
        return None
    # Enumerate 1D subspaces
    pts = []
    seen = set()
    for x in range(4):
        for y in range(4):
            for z in range(4):
                if x == 0 and y == 0 and z == 0:
                    continue
                r = norm([x,y,z])
                if r is None:
                    continue
                key = tuple(r)
                if key not in seen:
                    seen.add(key)
                    pts.append(r)
    # pts should have 21 entries
    # Two points are adjacent iff they span a 2D subspace (i.e., are linearly dependent? No:
    # two distinct 1D subspaces always span a 2D subspace in 3D). Wait — in a projective plane,
    # two points always determine a line. So ALL pairs would be adjacent -> complete graph. That's wrong.
    # Correction: In the projective plane, points are 1D subspaces and lines are 2D subspaces.
    # Two points lie on a common line iff they are linearly dependent... no. Two distinct 1D
    # subspaces in 3D always span a 2D subspace, so they always share a line. That means the
    # incidence graph of points-as-vertices with "same line" is complete. 
    # The K3,3-free graph I want is the POINT-LINE incidence? No, I want a graph ON the points.
    # Actually the right construction: the graph where two points are adjacent if they are
    # NOT on the same line is also complete. Hmm.
    # Let me reconsider: the standard K3,3-free dense graph from a projective plane is the
    # "points adjacent if they lie on a common line" — but that's complete.
    # The correct one: take the projective plane's points; define adjacency by a fixed
    # spread/oval. Simpler: use the known fact that the complement of the line graph... 
    # Given time, fall back to a 5-regular random repair seed if this doesn't yield a
    # proper K3,3-free graph. Return None to signal fallback.
    return None

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin under 150s

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

    def random_regular_seed(target_deg):
        """Generate a roughly target_deg-regular random graph that is K3,3-free."""
        for _ in range(300):
            bm = [0] * n
            for i in range(n):
                cands = [j for j in range(n) if j != i]
                random.shuffle(cands)
                for j in cands[:target_deg]:
                    if not (bm[i] & (1 << j)):
                        bm[i] |= (1 << j)
                        bm[j] |= (1 << i)
            # symmetrize
            for i in range(n):
                for j in range(i+1, n):
                    if (bm[i] & (1 << j)) != (bm[j] & (1 << i)):
                        if bm[i] & (1 << j):
                            bm[j] |= (1 << i)
                        else:
                            bm[i] &= ~(1 << j)
            if has_k33_fast(bm):
                return bm
        return [0] * n

    def local_search(initial_bitmasks):
        bitmasks = initial_bitmasks[:]
        current_edges = count_edges(bitmasks)
        best_local_edges = current_edges
        best_local_bitmasks = bitmasks[:]

        bitmasks = greedy_add_exhaustive(bitmasks)
        current_edges = count_edges(bitmasks)
        if current_edges > best_local_edges:
            best_local_edges = current_edges
            best_local_bitmasks = bitmasks[:]

        no_improve = 0
        while time.time() < deadline and no_improve < 150:
            moved = False
            # Add edges
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
                # 1-for-1 swap
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
                # 2-for-1 swap
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

    # --- Main search ---
    seeds = []
    seeds.append([0] * n)  # empty
    # random regular seeds of varying degree
    for deg in (4, 5, 6, 7):
        seeds.append(random_regular_seed(deg))
    # random dense seeds
    for p in (0.3, 0.4, 0.5):
        init = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    init[i] |= (1 << j)
                    init[j] |= (1 << i)
        # repair
        attempts = 0
        while not has_k33_fast(init) and attempts < 3000:
            all_e = get_all_edges(init)
            if not all_e:
                break
            u, v = random.choice(all_e)
            init[u] &= ~(1 << v)
            init[v] &= ~(1 << u)
            attempts += 1
        if has_k33_fast(init):
            seeds.append(init)

    for init in seeds:
        if time.time() > deadline:
            break
        if not has_k33_fast(init):
            continue
        local_best_bm, local_best_e = local_search(init)
        if local_best_e > best_edges:
            best_edges = local_best_e
            best_adj_bitmasks = local_best_bm[:]

    # Extra restarts from the best found state with random perturbations
    if best_adj_bitmasks is not None:
        for _ in range(30):
            if time.time() > deadline:
                break
            pert = best_adj_bitmasks[:]
            # randomly remove a few edges to escape local optimum
            all_e = get_all_edges(pert)
            for _ in range(random.randint(2, 6)):
                if not all_e:
                    break
                u, v = random.choice(all_e)
                pert[u] &= ~(1 << v)
                pert[v] &= ~(1 << u)
                all_e = get_all_edges(pert)
            if has_k33_fast(pert):
                local_best_bm, local_best_e = local_search(pert)
                if local_best_e > best_edges:
                    best_edges = local_best_e
                    best_adj_bitmasks = local_best_bm[:]

    # Final polish: greedy add with multiple random orderings
    if best_adj_bitmasks is not None:
        for _ in range(10):
            if time.time() > deadline:
                break
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
