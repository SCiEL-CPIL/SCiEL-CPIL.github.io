import time
import numpy as np
from itertools import combinations, permutations

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    def time_left():
        return deadline - time.time()

    def build_contrib_matrix(U, V, W, rank):
        """Build the full contribution matrix where contribs[i,j,k,r] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
        n_entries = n * m * p
        contribs = np.zeros((n_entries, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contribs[idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    idx += 1
        return contribs

    def verify(U, V, W, rank):
        """Verify decomposition using numpy for speed."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                u_row = U_arr[i*m+j, :]
                for k in range(p):
                    v_row = V_arr[j*p+k, :]
                    w_row = W_arr[k*n+i, :]
                    s = np.dot(u_row, v_row * w_row)
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_fast(U_arr, V_arr, W_arr, rank):
        """Verify using the contribution matrix approach."""
        contribs = build_contrib_matrix(U_arr, V_arr, W_arr, rank)
        # For each (i,j,k), the sum over r should equal 1
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = contribs[idx, :].sum()
                    if abs(s - 1.0) > 1e-9:
                        return False
                    idx += 1
        return True

    def build_strassen_decomp(j_pairs, k_pairs):
        """Build Strassen-based decomposition."""
        num_blocks = len(j_pairs) * len(k_pairs)
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]

        rank = 7 * num_blocks + 2 * len(remaining_j) * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))

        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1 = (A0j0+A1j1)*(Bj0k0+Bj1k1)
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P2 = (A1j0+A1j1)*Bj0k0
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                # P3 = A0j0*(Bj0k1-Bj1k1)
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P4 = A1j1*(Bj1k0-Bj0k0)
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                # P5 = (A0j0+A0j1)*Bj1k1
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                # P6 = (A1j0-A0j0)*(Bj0k0+Bj0k1)
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                # P7 = (A0j1-A1j1)*(Bj1k0+Bj1k1)
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1

        # Remaining j: standard products
        for j in remaining_j:
            for k in range(p):
                U[0*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+0, col] += 1
                col += 1
                U[1*m+j, col] += 1
                V[j*p+k, col] += 1
                W[k*n+1, col] += 1
                col += 1

        return U, V, W, rank

    def try_eliminate_terms(U, V, W, rank):
        """Try to eliminate redundant terms using linear algebra with SVD-based rank check."""
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        cur_rank = rank

        max_iterations = 100
        for iteration in range(max_iterations):
            if time_left() < 15:
                break

            U_arr = np.array(U_list, dtype=float)
            V_arr = np.array(V_list, dtype=float)
            W_arr = np.array(W_list, dtype=float)

            # Get contribution vectors
            contribs = build_contrib_matrix(U_arr, V_arr, W_arr, cur_rank)

            # Find the rank of the contribution matrix
            try:
                svd_rank = np.linalg.matrix_rank(contribs, tol=1e-10)
            except np.linalg.LinAlgError:
                svd_rank = cur_rank

            if svd_rank >= cur_rank:
                # No more linearly dependent terms
                break

            # Find which terms are linearly dependent
            # Use SVD to find the null space
            try:
                U_svd, S, Vt = np.linalg.svd(contribs, full_matrices=True)
            except np.linalg.LinAlgError:
                break

            # The null space is spanned by the last (cur_rank - svd_rank) rows of Vt
            null_start = svd_rank
            if null_start >= cur_rank:
                break

            # Find a term that is in the null space (linearly dependent on others)
            removed = False
            for term_idx in range(cur_rank - 1, -1, -1):
                if time_left() < 15:
                    break

                # Check if this term can be removed
                other_indices = [i for i in range(cur_rank) if i != term_idx]
                other_contribs = contribs[:, other_indices]

                try:
                    # Check if term's contribution is in the span of others
                    sol, residuals, _, _ = np.linalg.lstsq(other_contribs, contribs[:, term_idx], rcond=1e-10)
                    if len(residuals) > 0 and residuals[0] < 1e-9:
                        # Term is linearly dependent, try to remove it
                        U_test = np.array(U_list, dtype=float)
                        V_test = np.array(V_list, dtype=float)
                        W_test = np.array(W_list, dtype=float)
                        U_test[:, term_idx] = 0
                        V_test[:, term_idx] = 0
                        W_test[:, term_idx] = 0

                        if verify_fast(U_test, V_test, W_test, cur_rank):
                            # Remove the term
                            U_list = [row[:term_idx] + row[term_idx+1:] for row in U_list]
                            V_list = [row[:term_idx] + row[term_idx+1:] for row in V_list]
                            W_list = [row[:term_idx] + row[term_idx+1:] for row in W_list]
                            cur_rank -= 1
                            removed = True
                            break
                except np.linalg.LinAlgError:
                    continue

            if not removed:
                break

        return U_list, V_list, W_list, cur_rank

    def generate_partitions(indices, pair_size=2):
        """Generate all ways to partition indices into pairs and singles."""
        if not indices:
            yield [], []
            return
        if len(indices) < pair_size:
            yield [], list(indices)
            return

        first = indices[0]
        rest = indices[1:]

        # Option 1: first is a single
        for pairs, singles in generate_partitions(rest, pair_size):
            yield pairs, [first] + singles

        # Option 2: first is paired with another
        for i in range(len(rest)):
            partner = rest[i]
            remaining = rest[:i] + rest[i+1:]
            for pairs, singles in generate_partitions(remaining, pair_size):
                yield [(first, partner)] + pairs, singles

    j_indices = list(range(m))  # 0..4
    k_indices = list(range(p))  # 0..5

    # Generate all j-partitions
    j_partitions = set()
    for pairs, singles in generate_partitions(j_indices):
        j_partitions.add((tuple(sorted(pairs)), tuple(sorted(singles))))

    # Generate all k-partitions
    k_partitions = set()
    for pairs, singles in generate_partitions(k_indices):
        k_partitions.add((tuple(sorted(pairs)), tuple(sorted(singles))))

    # Build configurations
    configurations = []
    for j_pairs_tuple, j_singles in j_partitions:
        for k_pairs_tuple, k_singles in k_partitions:
            if not j_pairs_tuple or not k_pairs_tuple:
                continue
            num_j_blocks = len(j_pairs_tuple)
            num_k_blocks = len(k_pairs_tuple)
            num_blocks = num_j_blocks * num_k_blocks
            rank = 7 * num_blocks + 2 * len(j_singles) * p
            if rank <= 54:
                configurations.append((list(j_pairs_tuple), list(k_pairs_tuple), rank))

    configurations.sort(key=lambda x: x[2])

    best_result = None
    best_rank = float('inf')

    for j_pairs, k_pairs, est_rank in configurations:
        if time_left() < 20:
            break
        if est_rank >= best_rank:
            continue

        U, V, W, rank = build_strassen_decomp(j_pairs, k_pairs)

        if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
            continue

        # Try to eliminate terms
        U_list, V_list, W_list, new_rank = try_eliminate_terms(U, V, W, rank)

        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}

    if best_result is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best_result
