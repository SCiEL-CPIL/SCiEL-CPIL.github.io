"""Combined program: Strassen block construction + exhaustive term-elimination search."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    def verify(U, V, W, rank):
        """Verify decomposition: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen_decomp(j_pairs, k_pairs):
        """
        Build a Strassen-based decomposition for 2x5x6 matrix multiplication.
        j_pairs: list of (j0, j1) pairs to apply Strassen in the j-dimension
        k_pairs: list of (k0, k1) pairs to apply Strassen in the k-dimension
        Remaining j indices use standard 2*p terms.
        """
        used_j = set()
        for j0, j1 in j_pairs:
            used_j.add(j0)
            used_j.add(j1)
        remaining_j = [j for j in range(m) if j not in used_j]

        num_blocks = len(j_pairs) * len(k_pairs)
        rank = 7 * num_blocks + 2 * len(remaining_j) * p

        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))

        col = 0
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # 7 Strassen terms for the 2x2x2 sub-block
                # P1 = (A[0,j0]+A[1,j1])*(B[j0,k0]+B[j1,k1])
                U[0*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P2 = (A[1,j0]+A[1,j1])*B[j0,k0]
                U[1*m+j0, col] += 1; U[1*m+j1, col] += 1
                V[j0*p+k0, col] += 1
                W[k0*n+1, col] += 1; W[k1*n+1, col] -= 1
                col += 1
                # P3 = A[0,j0]*(B[j0,k1]-B[j1,k1])
                U[0*m+j0, col] += 1
                V[j0*p+k1, col] += 1; V[j1*p+k1, col] -= 1
                W[k1*n+0, col] += 1; W[k1*n+1, col] += 1
                col += 1
                # P4 = A[1,j1]*(B[j1,k0]-B[j0,k0])
                U[1*m+j1, col] += 1
                V[j1*p+k0, col] += 1; V[j0*p+k0, col] -= 1
                W[k0*n+0, col] += 1; W[k0*n+1, col] += 1
                col += 1
                # P5 = (A[0,j0]+A[0,j1])*B[j1,k1]
                U[0*m+j0, col] += 1; U[0*m+j1, col] += 1
                V[j1*p+k1, col] += 1
                W[k0*n+0, col] -= 1; W[k1*n+0, col] += 1
                col += 1
                # P6 = (A[1,j0]-A[0,j0])*(B[j0,k0]+B[j0,k1])
                U[1*m+j0, col] += 1; U[0*m+j0, col] -= 1
                V[j0*p+k0, col] += 1; V[j0*p+k1, col] += 1
                W[k1*n+1, col] += 1
                col += 1
                # P7 = (A[0,j1]-A[1,j1])*(B[j1,k0]+B[j1,k1])
                U[0*m+j1, col] += 1; U[1*m+j1, col] -= 1
                V[j1*p+k0, col] += 1; V[j1*p+k1, col] += 1
                W[k0*n+0, col] += 1
                col += 1

        # Remaining j: standard products (2*p terms each)
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

    def eliminate_terms(U, V, W, rank):
        """
        Iteratively remove redundant terms.
        A term r is redundant if zeroing it out still gives a valid decomposition.
        We try removal in multiple orders to find the minimal rank.
        """
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        current_rank = rank

        # Try multiple removal orders: descending, ascending, random
        orders = [list(range(current_rank - 1, -1, -1)), list(range(current_rank))]
        # Add a few random orders
        np.random.seed(123)
        for _ in range(3):
            order = list(range(current_rank))
            np.random.shuffle(order)
            orders.append(order)

        best_rank = current_rank
        best_U, best_V, best_W = U_list[:], V_list[:], W_list[:]

        for order in orders:
            if time_left() < 10:
                break
            U_cur = [row[:] for row in U_list]
            V_cur = [row[:] for row in V_list]
            W_cur = [row[:] for row in W_list]
            cur_rank = current_rank

            for term_idx in order:
                if term_idx >= cur_rank:
                    continue
                if time_left() < 5:
                    break

                # Try zeroing this term
                for i in range(n*m):
                    U_cur[i][term_idx] = 0
                for j in range(m*p):
                    V_cur[j][term_idx] = 0
                for k in range(p*n):
                    W_cur[k][term_idx] = 0

                if verify(U_cur, V_cur, W_cur, cur_rank):
                    # Remove the column
                    U_cur = [[row[j] for j in range(cur_rank) if j != term_idx] for row in U_cur]
                    V_cur = [[row[j] for j in range(cur_rank) if j != term_idx] for row in V_cur]
                    W_cur = [[row[j] for j in range(cur_rank) if j != term_idx] for row in W_cur]
                    cur_rank -= 1

            if cur_rank < best_rank:
                best_rank = cur_rank
                best_U = [row[:] for row in U_cur]
                best_V = [row[:] for row in V_cur]
                best_W = [row[:] for row in W_cur]

        return best_U, best_V, best_W, best_rank

    def eliminate_terms_exhaustive(U, V, W, rank):
        """
        More aggressive elimination: repeatedly scan and remove any removable term.
        Continue until no more can be removed.
        """
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        cur_rank = rank

        changed = True
        while changed and time_left() > 10:
            changed = False
            for term_idx in range(cur_rank - 1, -1, -1):
                if time_left() < 5:
                    break
                # Try zeroing this term
                for i in range(n*m):
                    U_list[i][term_idx] = 0
                for j in range(m*p):
                    V_list[j][term_idx] = 0
                for k in range(p*n):
                    W_list[k][term_idx] = 0

                if verify(U_list, V_list, W_list, cur_rank):
                    U_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in W_list]
                    cur_rank -= 1
                    changed = True
                    break
                else:
                    # Restore
                    for i in range(n*m):
                        U_list[i][term_idx] = 0  # already 0, but keep consistent
                    # Actually we need to restore original values
                    # Let's rebuild from scratch approach - this is getting complex
                    # Better: save/restore
                    pass

        return U_list, V_list, W_list, cur_rank

    def eliminate_terms_safe(U, V, W, rank):
        """
        Safe iterative elimination with proper save/restore.
        """
        U_list = [row[:] for row in U.tolist()]
        V_list = [row[:] for row in V.tolist()]
        W_list = [row[:] for row in W.tolist()]
        cur_rank = rank

        changed = True
        while changed and time_left() > 10:
            changed = False
            for term_idx in range(cur_rank - 1, -1, -1):
                if time_left() < 5:
                    break
                # Save
                U_save = [U_list[i][term_idx] for i in range(n*m)]
                V_save = [V_list[j][term_idx] for j in range(m*p)]
                W_save = [W_list[k][term_idx] for k in range(p*n)]

                # Zero out
                for i in range(n*m):
                    U_list[i][term_idx] = 0
                for j in range(m*p):
                    V_list[j][term_idx] = 0
                for k in range(p*n):
                    W_list[k][term_idx] = 0

                if verify(U_list, V_list, W_list, cur_rank):
                    # Remove column
                    U_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in W_list]
                    cur_rank -= 1
                    changed = True
                    break
                else:
                    # Restore
                    for i in range(n*m):
                        U_list[i][term_idx] = U_save[i]
                    for j in range(m*p):
                        V_list[j][term_idx] = V_save[j]
                    for k in range(p*n):
                        W_list[k][term_idx] = W_save[k]

        return U_list, V_list, W_list, cur_rank

    # Generate all valid configurations of j_pairs and k_pairs
    # j_pairs: partition of subset of {0,1,2,3,4} into pairs
    # k_pairs: partition of subset of {0,1,2,3,4,5} into pairs

    def generate_pairings(indices):
        """Generate all ways to pair up a subset of indices (leaving some unpaired)."""
        results = []
        n_idx = len(indices)
        # Try all subset sizes that are even
        for size in range(0, n_idx + 1, 2):
            for subset in combinations(indices, size):
                # Generate all perfect matchings of subset
                def gen_matchings(idx_list):
                    if len(idx_list) == 0:
                        yield []
                        return
                    first = idx_list[0]
                    rest = idx_list[1:]
                    for i, second in enumerate(rest):
                        remaining = rest[:i] + rest[i+1:]
                        for rest_matchings in gen_matchings(remaining):
                            yield [(first, second)] + rest_matchings

                for matching in gen_matchings(list(subset)):
                    results.append(matching)
        return results

    j_indices = list(range(5))  # 0,1,2,3,4
    k_indices = list(range(6))  # 0,1,2,3,4,5

    j_pairings = generate_pairings(j_indices)
    k_pairings = generate_pairings(k_indices)

    # Filter: only keep pairings that use at least one pair (to get Strassen benefit)
    # But also try empty pairings as baseline
    best_result = None
    best_rank = float('inf')

    # Sort configurations by expected rank (lower first) to find good solutions early
    configs = []
    for jp in j_pairings:
        for kp in k_pairings:
            num_blocks = len(jp) * len(kp)
            used_j = set()
            for j0, j1 in jp:
                used_j.add(j0)
                used_j.add(j1)
            remaining_j = len(j_indices) - len(used_j)
            est_rank = 7 * num_blocks + 2 * remaining_j * p
            configs.append((est_rank, jp, kp))

    configs.sort(key=lambda x: x[0])

    for est_rank, jp, kp in configs:
        if time_left() < 15:
            break
        if est_rank >= best_rank:
            # This config can't beat our best (optimistic estimate)
            # But after elimination it might, so only skip if clearly worse
            if est_rank > best_rank + 10:
                continue

        U, V, W, rank = build_strassen_decomp(jp, kp)

        if not verify(U.tolist(), V.tolist(), W.tolist(), rank):
            continue

        # Try elimination
        U_list, V_list, W_list, new_rank = eliminate_terms_safe(U, V, W, rank)

        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook if nothing worked
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
# EVOLVE-BLOCK-END
