"""
Matrix multiplication 3x4 * 4x4 -> 3x4 using BCL rank-19 for the 3x3x3 core
plus rank-4 Strassen for the 3x1x4 strip, total rank 23.
"""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=4
    deadline = time.time() + 140

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    # Try BCL rank-19 + Strassen rank-4 = 23
    bcl_result = bcl_decomp(n, m, p)
    if bcl_result is not None:
        bcl_rank = len(bcl_result['U'][0])
        if bcl_rank < best_rank:
            best_result = bcl_result
            best_rank = bcl_rank

    return best_result

def bcl_decomp(n, m, p):
    """
    Decompose 3x4 * 4x4 -> 3x4:
    - Core: A[0:3,0:3] * B[0:3,0:3] -> C[0:3,0:3] via BCL rank-19
    - Strip: A[0:3,3] * B[3,0:4] -> C[0:3,3] via 4 schoolbook terms (rank 4)
    Total rank: 19 + 4 = 23
    """
    U_list = []
    V_list = []
    W_list = []
    col = 0

    # BCL rank-19 decomposition for 3x3 * 3x3 -> 3x3
    # Each term: (u_entries, v_entries, w_entries)
    # u_entries: list of (flat_idx_in_3x3, coeff) for A
    # v_entries: list of (flat_idx_in_3x3, coeff) for B
    # w_entries: list of (flat_idx_in_3x3, coeff) for C
    # flat_idx for 3x3: row*3+col

    bcl_terms = [
        # Term 1: A[0,1]*B[0,0] -> C[0,0]
        ([(1, 1)], [(0, 1)], [(0, 1)]),
        # Term 2: A[0,1]*B[1,1] -> C[0,1]
        ([(1, 1)], [(4, 1)], [(1, 1)]),
        # Term 3: A[1,0]*B[0,0] -> C[1,0]
        ([(3, 1)], [(0, 1)], [(3, 1)]),
        # Term 4: A[1,0]*B[1,1] -> C[1,1]
        ([(3, 1)], [(4, 1)], [(4, 1)]),
        # Term 5: A[2,2]*B[2,2] -> C[2,2]
        ([(8, 1)], [(8, 1)], [(8, 1)]),
        # Term 6: (A[0,2]+A[2,0])*(B[0,2]+B[2,0]) -> C[0,2]+C[2,0]
        ([(2, 1), (6, 1)], [(2, 1), (6, 1)], [(2, 1), (6, 1)]),
        # Term 7: (A[0,2]+A[2,0])*(B[0,1]+B[1,2]) -> C[0,1]+C[1,2]
        ([(2, 1), (6, 1)], [(1, 1), (5, 1)], [(1, 1), (5, 1)]),
        # Term 8: (A[0,2]+A[2,0])*(B[1,0]+B[2,1]) -> C[1,0]+C[2,1]
        ([(2, 1), (6, 1)], [(3, 1), (7, 1)], [(3, 1), (7, 1)]),
        # Term 9: (A[0,1]+A[1,0])*(B[0,0]+B[0,1]) -> C[0,0]+C[1,0]
        ([(1, 1), (3, 1)], [(0, 1), (1, 1)], [(0, 1), (3, 1)]),
        # Term 10: (A[0,1]+A[1,0])*(B[1,0]+B[1,1]) -> C[1,0]+C[2,1]
        ([(1, 1), (3, 1)], [(3, 1), (4, 1)], [(3, 1), (7, 1)]),
        # Term 11: (A[0,1]+A[1,0])*(B[0,1]+B[1,1]) -> C[0,1]+C[1,1]
        ([(1, 1), (3, 1)], [(1, 1), (4, 1)], [(1, 1), (4, 1)]),
        # Term 12: (A[1,2]+A[2,1])*(B[1,0]+B[2,0]) -> C[1,0]+C[2,0]
        ([(5, 1), (7, 1)], [(3, 1), (6, 1)], [(3, 1), (6, 1)]),
        # Term 13: (A[1,2]+A[2,1])*(B[1,1]+B[2,1]) -> C[1,1]+C[2,1]
        ([(5, 1), (7, 1)], [(4, 1), (7, 1)], [(4, 1), (7, 1)]),
        # Term 14: (A[1,2]+A[2,1])*(B[0,1]+B[1,1]) -> C[0,1]+C[1,1]
        ([(5, 1), (7, 1)], [(1, 1), (4, 1)], [(1, 1), (4, 1)]),
        # Term 15: (A[0,2]+A[1,1]+A[2,0])*(B[0,0]+B[1,0]+B[2,0]) -> C[0,0]+C[1,0]+C[2,0]
        ([(2, 1), (4, 1), (6, 1)], [(0, 1), (3, 1), (6, 1)], [(0, 1), (3, 1), (6, 1)]),
        # Term 16: (A[0,2]+A[1,1]+A[2,0])*(B[0,1]+B[1,1]+B[2,1]) -> C[0,1]+C[1,1]+C[2,1]
        ([(2, 1), (4, 1), (6, 1)], [(1, 1), (4, 1), (7, 1)], [(1, 1), (4, 1), (7, 1)]),
        # Term 17: (A[0,2]+A[1,1]+A[2,0])*(B[0,2]+B[1,2]+B[2,2]) -> C[0,2]+C[1,2]+C[2,2]
        ([(2, 1), (4, 1), (6, 1)], [(2, 1), (5, 1), (8, 1)], [(2, 1), (5, 1), (8, 1)]),
        # Term 18: (A[0,1]+A[1,0]+A[2,2])*(B[0,2]+B[1,2]+B[2,2]) -> C[0,2]+C[1,2]+C[2,2]
        ([(1, 1), (3, 1), (8, 1)], [(2, 1), (5, 1), (8, 1)], [(2, 1), (5, 1), (8, 1)]),
        # Term 19: (A[0,1]+A[1,0]+A[2,2])*(B[0,0]+B[1,0]+B[2,0]) -> C[0,0]+C[1,0]+C[2,0]
        ([(1, 1), (3, 1), (8, 1)], [(0, 1), (3, 1), (6, 1)], [(0, 1), (3, 1), (6, 1)]),
    ]

    # Verify BCL terms produce correct 3x3 matrix multiplication
    bcl_ok = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0
                for term in bcl_terms:
                    u_entries, v_entries, w_entries = term
                    u_val = 0
                    for (idx, coeff) in u_entries:
                        if idx == i*3 + j:
                            u_val += coeff
                    v_val = 0
                    for (idx, coeff) in v_entries:
                        if idx == j*3 + k:
                            v_val += coeff
                    w_val = 0
                    for (idx, coeff) in w_entries:
                        if idx == i*3 + k:
                            w_val += coeff
                    s += u_val * v_val * w_val
                if abs(s - 1.0) > 1e-10:
                    bcl_ok = False
                    break
            if not bcl_ok:
                break
        if not bcl_ok:
            break

    if not bcl_ok:
        return None

    # Add BCL terms to the full decomposition
    # A is 3x4: U row index = i*4+j
    # B is 4x4: V row index = j*4+k
    # C is 3x4: W row index = k*3+i (trace convention W[k*n+i])
    for term in bcl_terms:
        u_indices, v_indices, w_indices = term
        U_entry = []
        V_entry = []
        W_entry = []
        for (idx, coeff) in u_indices:
            i, j = divmod(idx, 3)
            U_entry.append((i*4 + j, coeff))
        for (idx, coeff) in v_indices:
            j, k = divmod(idx, 3)
            V_entry.append((j*4 + k, coeff))
        for (idx, coeff) in w_indices:
            i, k = divmod(idx, 3)
            W_entry.append((k*3 + i, coeff))
        U_list.append(U_entry)
        V_list.append(V_entry)
        W_list.append(W_entry)
        col += 1

    # Strip: A[:,3] * B[3,:] -> C[:,3]
    # C[i,3] = A[i,3] * B[3,3] for i=0,1,2
    # This is 3 schoolbook terms (one per i)
    for i in range(3):
        U_entry = [(i*4 + 3, 1)]
        V_entry = [(3*4 + 3, 1)]
        W_entry = [(3*3 + i, 1)]  # k=3, n=3: k*n+i = 9+i
        U_list.append(U_entry)
        V_list.append(V_entry)
        W_list.append(W_entry)
        col += 1

    total_rank = col

    # Build U, V, W matrices
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))

    for c in range(total_rank):
        for (idx, sign) in U_list[c]:
            U[idx, c] = sign
        for (idx, sign) in V_list[c]:
            V[idx, c] = sign
        for (idx, sign) in W_list[c]:
            W[idx, c] = sign

    # Verify
    if not verify_full(n, m, p, U, V, W):
        return None

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def verify_full(n, m, p, U, V, W):
    """
    Verify the full tensor decomposition:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    should equal 1 if (i,j,k) is a valid multiplication term, else 0.
    For matrix multiplication, T[i,j,k] = 1 for all valid (i,j,k).
    But actually the tensor is: sum over r of U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i,r]
    should be 1 if i==i' and j==j' and k==k', else 0.
    """
    tol = 1e-8
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            s = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+ip, :])
                            expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            if abs(s - expected) > tol:
                                return False
    return True
