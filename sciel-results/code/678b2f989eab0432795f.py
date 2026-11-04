import numpy as np
import time

def construct():
    """
    Returns a rank-49 exact tensor decomposition for 4x4 matrix multiplication
    using the Kronecker product of Strassen's rank-7 2x2 algorithm.
    """
    n = 4
    m = 4
    p = 4

    # Strassen 2x2 decomposition (rank 7)
    # Indices: U is 4x7 (i,j), V is 4x7 (j,k), W is 4x7 (k,i)
    # Convention: U[i*2+j], V[j*2+k], W[k*2+i]
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # i=0, j=0
        [0, 0, 0, 0, 1, 0, 1],   # i=0, j=1
        [0, 1, 0, 0, 0, 1, 0],   # i=1, j=0
        [1, 1, 0, 1, 0, 0, -1]   # i=1, j=1
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # j=0, k=0
        [0, 0, 1, 0, 0, 1, 0],   # j=0, k=1
        [0, 0, 0, 1, 0, 0, 1],   # j=1, k=0
        [1, 0, -1, 0, 1, 0, 1]   # j=1, k=1
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # k=0, i=0
        [0, 1, 0, 1, 0, 0, 0],   # k=0, i=1
        [0, 0, 1, 0, 1, 0, 0],   # k=1, i=0
        [1, -1, 1, 0, 0, 1, 0]   # k=1, i=1
    ], dtype=float)
    
    # Verify 2x2 Strassen
    # T_{i,j,k} = sum_c U[i,j,c] V[j,k,c] W[k,i,c] should be 1 if i==k else 0?
    # No, for matrix multiplication C = AB, the entry C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor is T_{i,j,k} where the output index is (i,k) and input indices are (i,j) and (j,k).
    # Wait, the standard tensor for matrix multiplication M_{n,m,p} has indices (i, j, k)
    # corresponding to A[i,j], B[j,k], C[i,k].
    # The decomposition is T_{i,j,k} = sum_c U_c[i,j] V_c[j,k] W_c[i,k].
    # Note: W index is (i,k), not (k,i) usually, but the task says W[k*n+i, r].
    # Task says: "output factor follows the official trace convention W[k*n+i,r]".
    # This implies the index for W is flattened as k*n + i.
    # So W_{k,i,c} corresponds to the coefficient for C[i,k]? Or C[k,i]?
    # Let's check the task description: "W[k*n+i,r]".
    # In standard matrix multiplication C = A B, C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # The term is A_{i,j} * B_{j,k} * (1 for C_{i,k}).
    # So the tensor component is 1 at (i,j,k) if we identify the output index as (i,k).
    # If the output index is stored as W[k*n+i], then the index (k,i) in W corresponds to the output position.
    # Does W[k*n+i] mean the output is C[k,i] or C[i,k]?
    # Usually, if we say "trace convention" or similar, it might imply specific ordering.
    # Let's look at the verification in the current program:
    # s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    # This sums over c. The result is the coefficient for the term A[i,j] B[j,k] C[?,?].
    # If this sum equals 1, it means the decomposition correctly captures the multiplication.
    # The indices in the sum are i (row of A), j (col of A / row of B), k (col of B).
    # The product A[i,j] * B[j,k] contributes to C[i,k].
    # So the "output" index in the tensor is (i,k).
    # The program accesses W[k*n+i]. This suggests the index (k,i) in the flattened W array
    # corresponds to the output component associated with (i,k) in the multiplication?
    # Or does it mean the output is C[k,i]?
    # If the output were C[k,i], then A[i,j]B[j,k] would not directly map to a single C entry in a simple way without transposes.
    # However, the standard matrix multiplication tensor T_{ijk} (where i,j,k are the three indices)
    # is defined such that T_{ijk} = 1 if the indices form a valid "path" i->j->k.
    # In the context of the tensor T = sum_c u_c \otimes v_c \otimes w_c,
    # u_c is a vector of size n*m (indexed by i,j),
    # v_c is a vector of size m*p (indexed by j,k),
    # w_c is a vector of size p*n (indexed by k,i).
    # The tensor component T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c}.
    # For matrix multiplication, T_{i,j,k} should be 1 if we consider the "identity" of the multiplication?
    # Actually, the matrix multiplication tensor is defined such that
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This can be written as C_{i,k} = sum_{j,c} (A_{i,j} U_{i,j,c}) (B_{j,k} V_{j,k,c}) W_{k,i,c} ? No.
    # It is C_{i,k} = sum_j A_{i,j} B_{j,k} * 1.
    # So the tensor T_{i,j,k} (where the third index is the output index pair) is 1.
    # But here the third factor W has indices (k,i).
    # So the sum is over j? No, the sum is over c (the rank).
    # The equation is: sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} = 1 for all i,j,k.
    # This implies that the tensor T_{i,j,k} (with output index k,i) is 1 everywhere?
    # No, that's not right. The tensor for matrix multiplication is sparse.
    # T_{i,j,k} (where k,i is the output index) is 1 if... wait.
    # The tensor T is a 3-way array of size (n*m) x (m*p) x (p*n).
    # T_{(i,j), (j,k), (k,i)} = 1.
    # All other components are 0.
    # So the verification loop:
    # for i, j, k: sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1
    # This checks if the specific component T_{(i,j), (j,k), (k,i)} is 1.
    # This is correct for the matrix multiplication tensor.
    
    ok2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = np.sum(U2[i*2+j, :] * V2[j*2+k, :] * W2[k*2+i, :])
                if abs(s - 1.0) > 1e-10:
                    ok2 = False
                    break
            if not ok2:
                break
        if not ok2:
            break
    
    if not ok2:
        # Fallback: schoolbook 4x4 (rank 64)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build 4x4 rank-49 from Kronecker product
    rank49 = 49
    U49 = np.zeros((16, 49), dtype=float)
    V49 = np.zeros((16, 49), dtype=float)
    W49 = np.zeros((16, 49), dtype=float)
    
    for i in range(4):
        for j in range(4):
            i_hi, i_lo = i // 2, i % 2
            j_hi, j_lo = j // 2, j % 2
            idx_u = i * 4 + j
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    U49[idx_u, c] = U2[i_hi * 2 + j_hi, c1] * U2[i_lo * 2 + j_lo, c2]
    
    for j in range(4):
        for k in range(4):
            j_hi, j_lo = j // 2, j % 2
            k_hi, k_lo = k // 2, k % 2
            idx_v = j * 4 + k
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    V49[idx_v, c] = V2[j_hi * 2 + k_hi, c1] * V2[j_lo * 2 + k_lo, c2]
    
    for k in range(4):
        for i in range(4):
            k_hi, k_lo = k // 2, k % 2
            i_hi, i_lo = i // 2, i % 2
            idx_w = k * 4 + i
            for c1 in range(7):
                for c2 in range(7):
                    c = c1 * 7 + c2
                    W49[idx_w, c] = W2[k_hi * 2 + i_hi, c1] * W2[k_lo * 2 + i_lo, c2]
    
    # Verify rank-49
    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify_decomposition(U49, V49, W49, n, m, p):
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to lists with integer/half-integer entries
    def to_half_integer_list(arr):
        """Convert array to list of lists with half-integer (multiple of 0.5) entries"""
        result = []
        for row in arr:
            new_row = []
            for val in row:
                # Round to nearest 0.5
                rounded = round(val * 2) / 2
                # If it's very close to an integer, use integer
                if abs(rounded - round(rounded)) < 1e-10:
                    new_row.append(int(round(rounded)))
                else:
                    # Represent as fraction with denominator 2
                    new_row.append(rounded)
            result.append(new_row)
        return result
    
    U_list = to_half_integer_list(U49)
    V_list = to_half_integer_list(V49)
    W_list = to_half_integer_list(W49)
    
    # Final verification with the converted lists
    def verify_lists(U, V, W, n, m, p, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(len(U[0])))
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    if not verify_lists(U_list, V_list, W_list, n, m, p):
        # If conversion caused issues, fall back to original floats (which are exact integers here)
        U_list = [[float(x) for x in row] for row in U49.tolist()]
        V_list = [[float(x) for x in row] for row in V49.tolist()]
        W_list = [[float(x) for x in row] for row in W49.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
