import random
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix, csr_matrix

def popcount(x):
    return bin(x).count('1')

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    def is_valid_addition(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        common_uv = adj[u] & adj[v]
        if popcount(common_uv) >= 3:
            return False
        neighbors_v = adj[v] & ~(1 << u)
        while neighbors_v:
            lsb = neighbors_v & -neighbors_v
            w = (lsb.bit_length() - 1)
            if popcount(adj[u] & adj[w]) >= 2:
                return False
            neighbors_v ^= lsb
        neighbors_u = adj[u] & ~(1 << v)
        while neighbors_u:
            lsb = neighbors_u & -neighbors_u
            w = (lsb.bit_length() - 1)
            if popcount(adj[v] & adj[w]) >= 2:
                return False
            neighbors_u ^= lsb
        return True

    def get_missing_edges(adj):
        missing = []
        for i in range(n):
            mask = (~adj[i]) & ((1 << n) - 1)
            mask &= ~((1 << (i + 1)) - 1)
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                missing.append((i, j))
                mask ^= lsb
        return missing

    def try_add_all(adj):
        improved = True
        while improved:
            if time.time() > deadline - 5:
                return False
            improved = False
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline - 5:
                    return False
                if is_valid_addition(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
        return True

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def extract_edges(adj):
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

    def verify_edges(edge_list):
        adj_check = [0] * n
        for u, v in edge_list:
            adj_check[u] |= (1 << v)
            adj_check[v] |= (1 << u)
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj_check[u] & adj_check[v]) > 2:
                    return False
        return True

    best_edges = []
    best_count = 0
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # Phase 1: Heuristic Search
    heuristic_deadline = start_time + 40

    for restart in range(300):
        if time.time() > heuristic_deadline:
            break
        adj = [0] * n
        edges = all_pairs[:]
        random.shuffle(edges)
        for u, v in edges:
            if time.time() > heuristic_deadline:
                break
            if is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        try_add_all(adj)

        # Swap-based local search
        improved = True
        while improved:
            if time.time() > heuristic_deadline:
                break
            improved = False
            current_count = edge_count(adj)
            current_edges = extract_edges(adj)
            random.shuffle(current_edges)
            for rem_u, rem_v in current_edges:
                if time.time() > heuristic_deadline:
                    break
                adj[rem_u] &= ~(1 << rem_v)
                adj[rem_v] &= ~(1 << rem_u)
                added = []
                missing = get_missing_edges(adj)
                random.shuffle(missing)
                for u, v in missing:
                    if time.time() > heuristic_deadline:
                        break
                    if is_valid_addition(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        added.append((u, v))
                new_count = edge_count(adj)
                if new_count > current_count:
                    improved = True
                else:
                    for u, v in added:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    adj[rem_u] |= (1 << rem_v)
                    adj[rem_v] |= (1 << rem_u)
        try_add_all(adj)
        count = edge_count(adj)
        if count > best_count:
            best_count = count
            best_edges = extract_edges(adj)

    # Phase 2: MILP with corrected formulation
    if time.time() < deadline - 10:
        try:
            edge_idx = {}
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    edge_idx[(i, j)] = idx
                    idx += 1
            num_edges = idx

            # z variables: for each pair (i,j) with i<j, and each k != i,j
            # Store as list for easy iteration
            z_vars = []
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(n):
                        if k != i and k != j:
                            z_vars.append((i, j, k))
            z_count = len(z_vars)
            total_vars = num_edges + z_count

            c = np.zeros(total_vars)
            c[:num_edges] = -1.0

            # Build constraint matrix
            # For each z_var (i, j, k):
            #   C1: z - x_ik <= 0
            #   C2: z - x_jk <= 0
            #   C3: x_ik + x_jk - z <= 1
            # For each pair (i, j):
            #   C4: sum_k z_ijk <= 2
            num_pairs = n * (n - 1) // 2
            total_constraints = 3 * z_count + num_pairs

            A = lil_matrix((total_constraints, total_vars), dtype=int)
            b_ub = np.zeros(total_constraints)
            b_lb = np.full(total_constraints, -np.inf)

            # Fill z-constraints
            for z_i, (i, j, k) in enumerate(z_vars):
                row_base = 3 * z_i
                zi = num_edges + z_i
                ik = (i, k) if i < k else (k, i)
                jk = (j, k) if j < k else (k, j)
                xi = edge_idx[ik]
                xj = edge_idx[jk]

                # C1: z - x_ik <= 0
                A[row_base, zi] = 1
                A[row_base, xi] = -1
                b_ub[row_base] = 0

                # C2: z - x_jk <= 0
                A[row_base + 1, zi] = 1
                A[row_base + 1, xj] = -1
                b_ub[row_base + 1] = 0

                # C3: x_ik + x_jk - z <= 1
                A[row_base + 2, xi] = 1
                A[row_base + 2, xj] = 1
                A[row_base + 2, zi] = -1
                b_ub[row_base + 2] = 1

            # Fill pair constraints: sum_k z_ijk <= 2
            # Group z_vars by (i,j)
            pair_z_indices = {}
            for z_i, (i, j, k) in enumerate(z_vars):
                pair_z_indices.setdefault((i, j), []).append(z_i)

            pair_iter = 0
            for i in range(n):
                for j in range(i + 1, n):
                    row = 3 * z_count + pair_iter
                    for z_i in pair_z_indices.get((i, j), []):
                        A[row, num_edges + z_i] = 1
                    b_ub[row] = 2
                    pair_iter += 1

            A = csr_matrix(A)
            bounds = Bounds(lb=0, ub=1)
            cons = LinearConstraint(A, b_lb, b_ub)

            milp_time_limit = max(5, int(deadline - time.time() - 2))
            res = milp(c, constraints=cons, integrality=np.ones(total_vars),
                       bounds=bounds, options={"time_limit": milp_time_limit})

            if res.status in (0, 1):
                x_sol = res.x[:num_edges]
                new_edges = []
                for (u, v), idx_e in edge_idx.items():
                    if x_sol[idx_e] > 0.5:
                        new_edges.append([u, v])
                new_count = len(new_edges)
                if new_count > best_count and verify_edges(new_edges):
                    best_count = new_count
                    best_edges = new_edges
        except Exception:
            pass

    return best_edges
