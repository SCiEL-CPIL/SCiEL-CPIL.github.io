import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    nm, mp, pn = n * m, m * p, p * n  # 4, 6, 6

    # Target tensor: T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # i in {0,1}, j in {0,1}, k in {0,1,2}
    # a in {0,1,2,3}, b in {0,1,2,3,4,5}, c in {0,1,2,3,4,5}
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    def compute_tensor(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def error_metric(U, V, W):
        T_prime = compute_tensor(U, V, W)
        return np.sum((T_prime - T_target) ** 2)

    def is_valid(U, V, W, tol=1e-8):
        return error_metric(U, V, W) < tol

    # Schoolbook decomposition (rank 12)
    rank_school = n * m * p
    U_sb = np.zeros((nm, rank_school))
    V_sb = np.zeros((mp, rank_school))
    W_sb = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1

    # Strassen + Schoolbook hybrid (rank 11)
    # We use Strassen for the 2x2x2 part (k=0,1) and schoolbook for the 2x2x1 part (k=2)
    # Strassen decomposition for 2x2x2:
    # M1 = (A11) * (B11 + B22)
    # M2 = (A11 + A22) * B22
    # M3 = (A22) * (B21 - B11)
    # M4 = (A11) * (B12 - B22)
    # M5 = (A11 + A21) * B11
    # M6 = (A22 - A12) * (B12 + B22)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M2 - M5 + M7
    # C12 = M4 + M5
    # C21 = M3 + M6
    # C22 = M1 - M2 + M3 + M6

    # Our indices:
    # A is 2x2 (i=0,1; j=0,1). A[i,j] corresponds to U[i*m+j, :]
    # B is 2x3 (j=0,1; k=0,1,2). B[j,k] corresponds to V[j*p+k, :]
    # C is 2x3 (i=0,1; k=0,1,2). C[i,k] corresponds to W[k*n+i, :]

    # Let's define the Strassen terms for the k=0,1 part.
    # B11 = B[0,0] -> V[0, :] (j=0, k=0)
    # B12 = B[0,1] -> V[1, :] (j=0, k=1)
    # B21 = B[1,0] -> V[2, :] (j=1, k=0)
    # B22 = B[1,1] -> V[3, :] (j=1, k=1)

    # A11 = A[0,0] -> U[0, :]
    # A12 = A[0,1] -> U[1, :]
    # A21 = A[1,0] -> U[2, :]
    # A22 = A[1,1] -> U[3, :]

    # We will construct U, V, W for the 7 Strassen terms.
    # Note: The standard Strassen formulas express C as linear combinations of M's.
    # Here, the tensor decomposition is U * V * W.
    # U[r] * V[r] * W[r] = (linear combo of A) * (linear combo of B) * (linear combo of C indices?)
    # Actually, the tensor is T = sum_r U_r V_r W_r^T (in a sense).
    # Let's just define the 7 rank-1 terms for the 2x2x2 block.

    U_strassen = np.zeros((nm, 7))
    V_strassen = np.zeros((mp, 7))
    W_strassen = np.zeros((pn, 7))

    # Term 1: M1 = A11 * (B11 + B22)
    # U1 = A11 = U[0]
    # V1 = B11 + B22 = V[0] + V[3]
    # W1 = C11 - C22? No, W is the output coefficient.
    # The term contributes to C11 and C22.
    # C11 gets +M1. C22 gets +M1.
    # So W1 has 1 at C11 (k=0, i=0 -> c=0) and 1 at C22 (k=1, i=1 -> c=3).
    U_strassen[0, 0] = 1  # A11
    V_strassen[0, 0] = 1  # B11
    V_strassen[3, 0] = 1  # B22
    W_strassen[0, 0] = 1  # C11
    W_strassen[3, 0] = 1  # C22

    # Term 2: M2 = (A11 + A22) * B22
    # U2 = A11 + A22 = U[0] + U[3]
    # V2 = B22 = V[3]
    # W2 = C11 - C22
    # C11 gets +M2. C22 gets -M2.
    U_strassen[0, 1] = 1
    U_strassen[3, 1] = 1
    V_strassen[3, 1] = 1
    W_strassen[0, 1] = 1
    W_strassen[3, 1] = -1

    # Term 3: M3 = A22 * (B21 - B11)
    # U3 = A22 = U[3]
    # V3 = B21 - B11 = V[2] - V[0]
    # W3 = C21 + C22
    # C21 gets +M3. C22 gets +M3.
    U_strassen[3, 2] = 1
    V_strassen[2, 2] = 1
    V_strassen[0, 2] = -1
    W_strassen[2, 2] = 1  # C21 (k=0, i=1 -> c=2)
    W_strassen[3, 2] = 1  # C22

    # Term 4: M4 = A11 * (B12 - B22)
    # U4 = A11 = U[0]
    # V4 = B12 - B22 = V[1] - V[3]
    # W4 = C12
    # C12 gets +M4.
    U_strassen[0, 3] = 1
    V_strassen[1, 3] = 1
    V_strassen[3, 3] = -1
    W_strassen[1, 3] = 1  # C12 (k=1, i=0 -> c=1)

    # Term 5: M5 = (A11 + A21) * B11
    # U5 = A11 + A21 = U[0] + U[2]
    # V5 = B11 = V[0]
    # W5 = -C11 + C12
    # C11 gets -M5. C12 gets +M5.
    U_strassen[0, 4] = 1
    U_strassen[2, 4] = 1
    V_strassen[0, 4] = 1
    W_strassen[0, 4] = -1
    W_strassen[1, 4] = 1

    # Term 6: M6 = (A22 - A12) * (B12 + B22)
    # U6 = A22 - A12 = U[3] - U[1]
    # V6 = B12 + B22 = V[1] + V[3]
    # W6 = C21 + C22
    # C21 gets +M6. C22 gets +M6.
    U_strassen[3, 5] = 1
    U_strassen[1, 5] = -1
    V_strassen[1, 5] = 1
    V_strassen[3, 5] = 1
    W_strassen[2, 5] = 1
    W_strassen[3, 5] = 1

    # Term 7: M7 = (A12 - A22) * (B21 + B22)
    # U7 = A12 - A22 = U[1] - U[3]
    # V7 = B21 + B22 = V[2] + V[3]
    # W7 = C11
    # C11 gets +M7.
    U_strassen[1, 6] = 1
    U_strassen[3, 6] = -1
    V_strassen[2, 6] = 1
    V_strassen[3, 6] = 1
    W_strassen[0, 6] = 1

    # Now add the schoolbook terms for k=2.
    # We need to compute C[0,2] and C[1,2].
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # B[0,2] is V[4] (j=0, k=2). B[1,2] is V[5] (j=1, k=2).
    # C[0,2] is W[4] (k=2, i=0). C[1,2] is W[5] (k=2, i=1).

    U_school = np.zeros((nm, 4))
    V_school = np.zeros((mp, 4))
    W_school = np.zeros((pn, 4))

    # Term 8: A[0,0] * B[0,2] -> C[0,2]
    U_school[0, 0] = 1
    V_school[4, 0] = 1
    W_school[4, 0] = 1

    # Term 9: A[0,1] * B[1,2] -> C[0,2]
    U_school[1, 1] = 1
    V_school[5, 1] = 1
    W_school[4, 1] = 1

    # Term 10: A[1,0] * B[0,2] -> C[1,2]
    U_school[2, 2] = 1
    V_school[4, 2] = 1
    W_school[5, 2] = 1

    # Term 11: A[1,1] * B[1,2] -> C[1,2]
    U_school[3, 3] = 1
    V_school[5, 3] = 1
    W_school[5, 3] = 1

    # Combine
    U_hybrid = np.hstack([U_strassen, U_school])
    V_hybrid = np.hstack([V_strassen, V_school])
    W_hybrid = np.hstack([W_strassen, W_school])

    if is_valid(U_hybrid, V_hybrid, W_hybrid):
        return {"U": U_hybrid.tolist(), "V": V_hybrid.tolist(), "W": W_hybrid.tolist()}

    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
