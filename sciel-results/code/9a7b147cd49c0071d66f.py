import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 130

    def time_left():
        return deadline - time.time()

    def verify_full(U, V, W, rank):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Precompute U, V, W as 3D arrays for easier indexing?
        # U: (n, m, rank)
        # V: (m, p, rank)
        # W: (p, n, rank)
        
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        # The condition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
        # We can compute this for all i,j,k using broadcasting or einsum
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # Using einsum:
        # U[i,j,r] V[j,k,r] -> intermediate[i,j,k,r] ? No, V depends on j,k.
        # Let's just loop over i, j, k but use vectorized dot products.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i*m+j, :], V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def build_strassen(j_pairs, k_pairs):
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

    def try_eliminate(U, V, W, rank):
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        cur_rank = rank

        changed = True
        while changed and time_left() > 5:
            changed = False
            for term_idx in range(cur_rank - 1, -1, -1):
                if time_left() < 5:
                    break
                U_new = [row[:] for row in U_list]
                V_new = [row[:] for row in V_list]
                W_new = [row[:] for row in W_list]
                for i in range(n*m):
                    U_new[i][term_idx] = 0
                for j in range(m*p):
                    V_new[j][term_idx] = 0
                for k in range(p*n):
                    W_new[k][term_idx] = 0

                if verify_full(U_new, V_new, W_new, cur_rank):
                    U_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in U_list]
                    V_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in V_list]
                    W_list = [[row[j] for j in range(cur_rank) if j != term_idx] for row in W_list]
                    cur_rank -= 1
                    changed = True
                    break

        return U_list, V_list, W_list, cur_rank

    def try_linear_reduction(U, V, W, rank):
        """Try to find a linear combination of terms that can be removed."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)

        U_list = U_arr.tolist()
        V_list = V_arr.tolist()
        W_list = W_arr.tolist()
        cur_rank = rank

        # Try removing each term by expressing it as a combination of others
        for term_idx in range(cur_rank - 1, -1, -1):
            if time_left() < 5:
                break

            # Check if term term_idx is a linear combination of the others
            # by checking if the column is in the span of the other columns
            # across the combined tensor
            # For each (i,j,k), the contribution is U[i,j,r]*V[j,k,r]*W[k,i,r]
            # If we remove term r, we need the remaining terms to still give 1 for each (i,j,k)
            # This is equivalent to checking if the removed term's contribution is zero
            # in the tensor, which means it's redundant.
            # Actually, the term is redundant if removing it still gives valid decomposition.
            # This is the same as try_eliminate, but let's try a different approach:
            # Check if the term can be expressed as a linear combination of other terms
            # by solving for coefficients.

            # For each (i,j,k), the term's contribution is:
            # U[i*m+j, term_idx] * V[j*p+k, term_idx] * W[k*n+i, term_idx]
            # If this is zero for all (i,j,k), the term is redundant.
            # Otherwise, we need to check if we can adjust other terms to compensate.
            # This is more complex, so we'll just use the elimination approach.
            pass

        return U_list, V_list, W_list, cur_rank

    # Try different Strassen configurations
    configurations = [
        ([(0,1),(2,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,2),(1,3),(4,5)]),
        ([(0,1),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,3),(4,5)]),
        ([(0,2),(1,3)], [(0,1),(2,3),(4,5)]),
        ([(0,1),(2,3)], [(0,3),(1,4),(2,5)]),
        ([(0,1),(2,3)], [(0,4),(1,5),(2,3)]),
        ([(0,1),(3,4)], [(0,1),(2,4),(3,5)]),
        ([(1,2),(3,4)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,1),(3,4),(2,5)]),
        ([(0,1),(2,3)], [(0,2),(1,4),(3,5)]),
        ([(0,1),(3,4)], [(0,2),(1,3),(4,5)]),
        ([(1,2),(3,4)], [(0,1),(2,4),(3,5)]),
        ([(0,1),(2,3)], [(0,1),(2,5),(3,4)]),
        ([(0,1),(2,3)], [(0,1),(3,5),(2,4)]),
        ([(0,1),(2,3)], [(0,2),(3,4),(1,5)]),
        ([(0,1),(2,3)], [(0,3),(1,2),(4,5)]),
        ([(0,1),(2,3)], [(0,4),(1,2),(3,5)]),
        ([(0,1),(2,3)], [(0,5),(1,2),(3,4)]),
    ]

    best_result = None
    best_rank = float('inf')

    for j_pairs, k_pairs in configurations:
        if time_left() < 10:
            break
        U, V, W, rank = build_strassen(j_pairs, k_pairs)

        if not verify_full(U.tolist(), V.tolist(), W.tolist(), rank):
            continue

        U_list, V_list, W_list, new_rank = try_eliminate(U, V, W, rank)

        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            
            if best_rank <= 50: # Early exit if we find a very good rank
                break

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
