import time
import numpy as np

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n = graph['n']
    m = n // 2
    edges = graph['edges']
    E = len(edges)

    # Identify edge indices by type: bottom cycle, top cycle, matching
    bottom_idx = []
    top_idx = []
    match_idx = []
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            bottom_idx.append(i)
        elif u >= m and v >= m:
            top_idx.append(i)
        else:
            match_idx.append(i)

    # Map each bottom/top cycle edge to its position 0..m-1 (edge between k and k+1 mod m)
    bottom_pos = {}
    top_pos = {}
    for i in bottom_idx:
        u, v = edges[i]
        if u < v:
            bottom_pos[i] = u
        else:
            bottom_pos[i] = v - 1
    for i in top_idx:
        u, v = edges[i]
        u0 = u - m
        v0 = v - m
        if u0 < v0:
            top_pos[i] = u0
        else:
            top_pos[i] = v0 - 1

    # Matching: vertex i (bottom) to vertex i+m (top)
    match_pos = {}
    for i in match_idx:
        u, v = edges[i]
        if u < m:
            match_pos[i] = u
        else:
            match_pos[i] = v - m

    # Build the fixed unsigned adjacency A (for the even block eigenvalues)
    A = np.zeros((n, n), dtype=np.float64)
    for (u, v) in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0

    # Even block eigenvalues are fixed (gauge invariant)
    eig_A = np.linalg.eigvalsh(A)

    # Precompute cycle-adjacency eigenvalue coefficients: 2*cos(2*pi*k/m)
    k = np.arange(m)
    cos_vals = 2.0 * np.cos(2.0 * np.pi * k / m)

    # Build the fixed part of the odd block: diag(d) + B_bottom + B_top
    # with all cycle signs = +1 and matching = 0 (matching added separately)
    # Degree vector d: each vertex has 2 (cycle) + 1 (matching) = 3
    # Odd block fixed part (no matching):
    #   W0 = diag(3) + B_bottom(+1) + B_top(+1)
    def build_W0():
        W0 = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(W0, 3.0)
        for i in bottom_idx:
            u, v = edges[i]
            W0[u, v] += 1.0
            W0[v, u] += 1.0
        for i in top_idx:
            u, v = edges[i]
            W0[u, v] += 1.0
            W0[v, u] += 1.0
        return W0

    W0 = build_W0()

    # Matching contribution: for sign w_j on matching edge (j, j+m),
    # it adds w_j to W[j, j+m] and W[j+m, j].
    # We'll build the matching matrix M with M[j, j+m] = 1 for each j.
    M = np.zeros((n, n), dtype=np.float64)
    for i in match_idx:
        j = match_pos[i]
        M[j, j + m] = 1.0
        M[j + m, j] = 1.0

    def spectral_radius_from_w(w):
        """Given matching signs w (length m), compute spectral radius.
        The odd block is W0 + sum_j w_j * (M_ej) where M_ej has 1 at (j,j+m),(j+m,j).
        We build W = W0 + M_w where M_w[j,j+m]=w[j]."""
        W = W0.copy()
        for j in range(m):
            W[j, j + m] = w[j]
            W[j + m, j] = w[j]
        # Spectral norm of W
        try:
            s = np.linalg.norm(W, 2)
        except np.linalg.LinAlgError:
            return np.inf
        # Eigenvalues of lift: lambda_i ± mu_i where mu_i are eigenvalues of W
        # Spectral radius = max_i max(|lambda_i + mu_i|, |lambda_i - mu_i|)
        # = max_i sqrt(lambda_i^2 + mu_i^2)  (since both signs)
        # Actually: max(|l+m|,|l-m|) = sqrt(l^2+m^2) only if we take the right... 
        # |l+m|^2 and |l-m|^2: max = max(l^2+m^2+2lm, l^2+m^2-2lm) = l^2+m^2+2|lm|
        # So spectral radius = max_i sqrt(lambda_i^2 + mu_i^2 + 2*|lambda_i*mu_i|)
        eig_W = np.linalg.eigvalsh(W)
        # Pair them: sort both
        val = 0.0
        for l, mu in zip(eig_A, eig_W):
            r = np.sqrt(l * l + mu * mu + 2.0 * abs(l * mu))
            if r > val:
                val = r
        return val

    def spectral_radius_full(signs):
        """Compute spectral radius from full sign vector via the block structure."""
        # Bottom cycle product
        bprod = 1
        for i in bottom_idx:
            bprod *= signs[i]
        tprod = 1
        for i in top_idx:
            tprod *= signs[i]
        # Build W = W0_scaled + matching
        # W0 was built with +1 cycles. We need to scale B_bottom by bprod? 
        # No: the cycle-adjacency matrix with signs s_k has eigenvalues 2*cos(2*pi*k/m)*prod(s).
        # But W0 = diag(3) + B_bottom(+1) + B_top(+1). We can't just scale B_bottom by bprod
        # because B_bottom(+1) scaled by bprod gives the signed cycle matrix only if all signs
        # equal bprod... which they don't. So we must build W directly from signs.
        W = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(W, 3.0)
        for i in bottom_idx:
            u, v = edges[i]
            s = float(signs[i])
            W[u, v] += s
            W[v, u] += s
        for i in top_idx:
            u, v = edges[i]
            s = float(signs[i])
            W[u, v] += s
            W[v, u] += s
        for i in match_idx:
            u, v = edges[i]
            s = float(signs[i])
            W[u, v] += s
            W[v, u] += s
        try:
            eig_W = np.linalg.eigvalsh(W)
        except np.linalg.LinAlgError:
            return np.inf
        val = 0.0
        for l, mu in zip(eig_A, eig_W):
            r = np.sqrt(l * l + mu * mu + 2.0 * abs(l * mu))
            if r > val:
                val = r
        return val

    def is_connected(signs):
        """Lift connected iff signed graph is not balanced (no x with x_u*x_v=s_uv)."""
        x = np.zeros(n, dtype=np.int8)
        visited = np.zeros(n, dtype=bool)
        x[0] = 1
        visited[0] = True
        stack = [0]
        # adjacency list of edge indices per vertex
        adj = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            adj[u].append((v, i))
            adj[v].append((u, i))
        while stack:
            u = stack.pop()
            for v, i in adj[u]:
                if not visited[v]:
                    x[v] = x[u] * signs[i]
                    visited[v] = True
                    stack.append(v)
        for i, (u, v) in enumerate(edges):
            if x[u] * x[v] != signs[i]:
                return True
        return False

    def validate_and_radius(signs):
        """Use check_lift for authoritative validation and radius."""
        try:
            from autoresearch_suite.science_tcs_expansion import check_lift
            radius, details = check_lift(graph, signs)
            return True, float(radius)
        except Exception:
            return False, np.inf

    # --- Phase 1: Exhaustive search over matching signs w (2^m), with cycles set to +1 ---
    # This covers a large family. We'll also vary cycle products.
    best_signs = None
    best_radius = np.inf

    def make_signs_from(w, bprod=1, tprod=1, b_pattern=None, t_pattern=None):
        """Construct a full sign vector.
        b_pattern/t_pattern: optional arrays of length m giving per-edge cycle signs.
        If None, set all cycle edges to +1 (product +1).
        w: array of length m, matching signs.
        """
        signs = np.ones(E, dtype=np.int8)
        if b_pattern is not None:
            for pos, i in bottom_pos.items():
                signs[i] = b_pattern[pos]
        if t_pattern is not None:
            for pos, i in top_pos.items():
                signs[i] = t_pattern[pos]
        for j in range(m):
            # find matching edge index for position j
            for i in match_idx:
                if match_pos[i] == j:
                    signs[i] = w[j]
                    break
        return signs

    # Exhaustive over w (matching signs), cycles = +1
    best_w = None
    best_w_radius = np.inf
    for mask in range(1 << m):
        if time.time() > deadline - 5.0:
            break
        w = np.array([1 if (mask >> j) & 1 else -1 for j in range(m)], dtype=np.int8)
        r = spectral_radius_from_w(w)
        if r < best_w_radius:
            best_w_radius = r
            best_w = w.copy()

    if best_w is not None:
        signs = make_signs_from(best_w)
        if is_connected(signs):
            valid, radius = validate_and_radius(signs)
            if valid:
                best_radius = radius
                best_signs = signs.copy()

    # --- Phase 2: Local search over the full sign space using the fast spectral radius ---
    # Start from best found (or a good random connected assignment).
    if best_signs is None:
        # Find any connected assignment
        rng = np.random.default_rng(12345)
        for _ in range(2000):
            signs = rng.choice([-1, 1], size=E).astype(np.int8)
            if is_connected(signs):
                best_signs = signs
                best_radius = spectral_radius_full(signs)
                break
    if best_signs is None:
        # Fallback: force connected by making bottom cycle product -1
        signs = np.ones(E, dtype=np.int8)
        signs[bottom_idx[0]] = -1
        best_signs = signs
        best_radius = spectral_radius_full(signs)

    current_signs = best_signs.copy()
    current_radius = best_radius

    # Local search: flip single edges, then pairs
    rng = np.random.default_rng(999)
    no_improve = 0
    max_no_improve = 3000
    iteration = 0

    # Precompute adjacency for connectivity check speed
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    def fast_is_connected(signs):
        x = np.zeros(n, dtype=np.int8)
        visited = np.zeros(n, dtype=bool)
        x[0] = 1
        visited[0] = True
        stack = [0]
        while stack:
            u = stack.pop()
            for v, i in adj[u]:
                if not visited[v]:
                    x[v] = x[u] * signs[i]
                    visited[v] = True
                    stack.append(v)
        for i, (u, v) in enumerate(edges):
            if x[u] * x[v] != signs[i]:
                return True
        return False

    while time.time() < deadline - 8.0 and no_improve < max_no_improve:
        iteration += 1
        # Flip a random edge
        e = int(rng.integers(0, E))
        trial = current_signs.copy()
        trial[e] = -trial[e]
        if not fast_is_connected(trial):
            no_improve += 1
            continue
        r = spectral_radius_full(trial)
        if r < current_radius - 1e-10:
            current_signs = trial
            current_radius = r
            no_improve = 0
            if r < best_radius:
                best_radius = r
                best_signs = current_signs.copy()
        else:
            no_improve += 1

        # Occasionally try a pair flip
        if iteration % 4 == 0:
            e2 = int(rng.integers(0, E))
            if e2 != e:
                trial2 = current_signs.copy()
                trial2[e] = -trial2[e]
                trial2[e2] = -trial2[e2]
                if fast_is_connected(trial2):
                    r2 = spectral_radius_full(trial2)
                    if r2 < current_radius - 1e-10:
                        current_signs = trial2
                        current_radius = r2
                        no_improve = 0
                        if r2 < best_radius:
                            best_radius = r2
                            best_signs = current_signs.copy()

    # --- Phase 3: Polish with more random restarts if time allows ---
    if time.time() < deadline - 15.0:
        for restart in range(30):
            if time.time() > deadline - 8.0:
                break
            rs = rng.choice([-1, 1], size=E).astype(np.int8)
            if not fast_is_connected(rs):
                continue
            lr = spectral_radius_full(rs)
            if lr < best_radius:
                # local search from here
                cs = rs.copy()
                cr = lr
                ni = 0
                for _ in range(2000):
                    if time.time() > deadline - 6.0:
                        break
                    if ni > 500:
                        break
                    e = int(rng.integers(0, E))
                    tr = cs.copy()
                    tr[e] = -tr[e]
                    if not fast_is_connected(tr):
                        ni += 1
                        continue
                    r = spectral_radius_full(tr)
                    if r < cr - 1e-10:
                        cs = tr
                        cr = r
                        ni = 0
                        if r < best_radius:
                            best_radius = r
                            best_signs = cs.copy()
                    else:
                        ni += 1

    # Final authoritative validation
    if best_signs is not None:
        valid, fr = validate_and_radius(best_signs)
        if valid:
            return best_signs
        # If validation failed, try to find a valid one among recent candidates
        # (shouldn't happen since we checked connectivity, but be safe)

    # Last resort: return best_signs (hopefully valid)
    return best_signs
