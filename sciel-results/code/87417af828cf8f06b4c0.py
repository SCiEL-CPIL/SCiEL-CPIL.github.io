import numpy as np
import time

def construct():
    _t0 = time.time()
    n, m, p = 2, 4, 4
    
    def verify(U, V, W):
        """Verify the decomposition is exact."""
        # U: (n*m, r), V: (m*p, r), W: (p*n, r)
        # C[i,j] = sum_s U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s]
        # Need to check for all i,k,j: T[i,k,j]=1 = sum_s U[i,k,s]*V[k,j,s]*W[i,j,s]
        # But this is a 3-way sum, hard to check directly.
        # Instead, check: for all i,j: C[i,j] = sum_k A[i,k]*B[k,j]
        # with A[i,k] = U[i*m+k, :], B[k,j] = V[k*p+j, :], etc.
        # Actually the decomposition means:
        # For any A, B: C[i,j] = sum_s (sum_k U[i*m+k,s]*A[i,k]) * (sum_k V[k*p+j,s]*B[k,j]) * W[j*n+i,s]
        # Wait no. The standard tensor decomposition:
        # T_{ikj} = sum_s U_{ik,s} V_{kj,s} W_{ij,s}
        # means: C[i,j] = sum_{k,s} A[i,k] B[k,j] W[i,j,s] U[i,k,s] V[k,j,s]
        # Hmm, that's not right either.
        
        # The correct interpretation: the multiplication tensor T has T[i,k,j] = 1.
        # A rank-r decomposition: T[i,k,j] = sum_s U[i,k,s] * V[k,j,s] * W[i,j,s]
        # This means: for all i,k,j, sum_s U[i,k,s]*V[k,j,s]*W[i,j,s] = 1
        
        r = U.shape[1]
        # Check all (i,k,j) triples
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    s = np.sum(U[i*m+k, :] * V[k*p+j, :] * W[j*n+i, :])
                    if s != 1:
                        return False
        return True

    # Try rank 24 with a structured search
    # Use the 4-block Strassen as base and try to merge factors
    
    # Actually, let me try a completely different approach:
    # Treat the problem as 2*(4x4) but with shared B structure.
    
    # For now, return the known rank-28 solution
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U[X_rows[0], col_offset + 0] += 1
        U[X_rows[3], col_offset + 0] += 1
        V[Y_rows[0], col_offset + 0] += 1
        V[Y_rows[3], col_offset + 0] += 1
        W[Z_rows[0], col_offset + 0] += 1
        W[Z_rows[3], col_offset + 0] += 1
        U[X_rows[2], col_offset + 1] += 1
        U[X_rows[3], col_offset + 1] += 1
        V[Y_rows[0], col_offset + 1] += 1
        W[Z_rows[2], col_offset + 1] += 1
        W[Z_rows[3], col_offset + 1] -= 1
        U[X_rows[0], col_offset + 2] += 1
        V[Y_rows[1], col_offset + 2] += 1
        V[Y_rows[3], col_offset + 2] -= 1
        W[Z_rows[1], col_offset + 2] += 1
        W[Z_rows[3], col_offset + 2] += 1
        U[X_rows[3], col_offset + 3] += 1
        V[Y_rows[2], col_offset + 3] += 1
        V[Y_rows[0], col_offset + 3] -= 1
        W[Z_rows[0], col_offset + 3] += 1
        W[Z_rows[2], col_offset + 3] += 1
        U[X_rows[0], col_offset + 4] += 1
        U[X_rows[1], col_offset + 4] += 1
        V[Y_rows[3], col_offset + 4] += 1
        W[Z_rows[0], col_offset + 4] -= 1
        W[Z_rows[1], col_offset + 4] += 1
        U[X_rows[2], col_offset + 5] += 1
        U[X_rows[0], col_offset + 5] -= 1
        V[Y_rows[0], col_offset + 5] += 1
        V[Y_rows[1], col_offset + 5] += 1
        W[Z_rows[3], col_offset + 5] += 1
        U[X_rows[1], col_offset + 6] += 1
        U[X_rows[3], col_offset + 6] -= 1
        V[Y_rows[2], col_offset + 6] += 1
        V[Y_rows[3], col_offset + 6] += 1
        W[Z_rows[0], col_offset + 6] += 1

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)
    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)
    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)
    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
