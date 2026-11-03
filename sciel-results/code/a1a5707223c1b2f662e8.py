import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    def verify(U, V, W, tol=1e-8):
        rank = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k2*n+i2, r]
                            exp = 1.0 if (i == i2 and k == k2) else 0.0
                            if abs(s - exp) > tol:
                                return False
        return True

    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Try to find rank-47 decomposition by solving linear system for null space
    # In schoolbook, term r=(i,j,k) has U at (i*m+j), V at (j*p+k), W at (k*n+i)
    # Need c[i,j,k] such that:
    #   sum_k c[i,j,k] = 0 for all (i,j)  [12 eqs]
    #   sum_i c[i,j,k] = 0 for all (j,k)  [16 eqs]
    #   sum_j c[i,j,k] = 0 for all (k,i)  [12 eqs]
    # Build linear system
    A_mat = []
    b_vec = []
    # sum_k c[i,j,k] = 0
    for i in range(n):
        for j in range(m):
            row = [0.0] * (n*m*p)
            for k in range(p):
                row[i*m*p + j*p + k] = 1.0
            A_mat.append(row)
            b_vec.append(0.0)
    # sum_i c[i,j,k] = 0
    for j in range(m):
        for k in range(p):
            row = [0.0] * (n*m*p)
            for i in range(n):
                row[i*m*p + j*p + k] = 1.0
            A_mat.append(row)
            b_vec.append(0.0)
    # sum_j c[i,j,k] = 0
    for k in range(p):
        for i in range(n):
            row = [0.0] * (n*m*p)
            for j in range(m):
                row[i*m*p + j*p + k] = 1.0
            A_mat.append(row)
            b_vec.append(0.0)

    A_arr = np.array(A_mat)
    b_arr = np.array(b_vec)
    
    # Find null space of A_arr
    try:
        _, S, Vt = np.linalg.svd(A_arr)
        # Null space is the last (48 - rank) right singular vectors
        rank_A = np.sum(S > 1e-10)
        null_dim = n*m*p - rank_A
        if null_dim > 0:
            null_space = Vt[rank_A:]  # shape (null_dim, 48)
            # Try each null space vector
            for ns_vec in null_space:
                if time.time() > deadline - 10:
                    break
                # Check if ns_vec has half-integer entries
                # Scale to make entries half-integers
                # Find a scaling factor such that all entries are half-integers
                # Try small integer scalings
                found = False
                for scale in range(1, 100):
                    c_scaled = ns_vec * scale
                    # Check if all entries are close to half-integers
                    c_rounded = np.round(2 * c_scaled) / 2.0
                    if np.max(np.abs(c_scaled - c_rounded)) < 1e-6:
                        # Try this c
                        # Remove term 0 and adjust others
                        U_new = np.array(U_sb, dtype=float)
                        V_new = np.array(V_sb, dtype=float)
                        W_new = np.array(W_sb, dtype=float)
                        
                        # c[i,j,k] is the coefficient for term (i,j,k)
                        # New decomposition: for term r != 0, add c[r] * term_0
                        # Term 0 corresponds to (i=0, j=0, k=0)
                        # U_0 has 1 at (0*m+0) = 0
                        # V_0 has 1 at (0*p+0) = 0
                        # W_0 has 1 at (0*n+0) = 0
                        
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    r = i*m*p + j*p + k
                                    if r == 0:
                                        continue
                                    c_val = c_rounded[r]
                                    if abs(c_val) > 1e-10:
                                        U_new[i*m+j, r] += c_val * U_new[0, 0]  # U_0[0]=1
                                        V_new[j*p+k, r] += c_val * V_new[0, 0]  # V_0[0]=1
                                        W_new[k*n+i, r] += c_val * W_new[0, 0]  # W_0[0]=1
                        
                        # Zero out term 0
                        U_new[:, 0] = 0
                        V_new[:, 0] = 0
                        W_new[:, 0] = 0
                        
                        # Remove term 0
                        U_final = np.delete(U_new, 0, axis=1)
                        V_final = np.delete(V_new, 0, axis=1)
                        W_final = np.delete(W_new, 0, axis=1)
                        
                        if verify(U_final.tolist(), V_final.tolist(), W_final.tolist()):
                            return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                        found = True
                        break
                if found:
                    break
    except Exception:
        pass

    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
