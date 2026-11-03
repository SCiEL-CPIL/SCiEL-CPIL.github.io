import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Rank 42 solution:
    # 1. Strassen core for i in {0,1}, j in {0..3}, k in {0..3} (Rank 28)
    # 2. Right edge for i in {0,1}, j=4, k in {0..4} (Rank 5)
    # 3. Bottom edge for i in {0,1}, j in {0..3}, k=4 (Rank 4)
    # 4. Bottom row for i=2, j in {0..4}, k in {0..4} (Rank 5)
    
    total_rank = 42
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    def add_strassen_block(i0, j0, k0, col_offset):
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    # 1. Strassen Core: i in {0,1}, j in {0..3}, k in {0..3}
    # Blocks: (0,0,0), (0,2,0), (0,0,2), (0,2,2)
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    # 2. Right Edge: i in {0,1}, j=4, k in {0..4}
    # For each k, we have a column
    for k in range(5):
        c = col
        # U: i=0, j=4 and i=1, j=4
        U[0*m + 4][c] += 1
        U[1*m + 4][c] += 1
        # V: j=4, k=k
        V[4*p + k][c] += 1
        # W: k=k, i=0 and i=1
        W[k*n + 0][c] += 1
        W[k*n + 1][c] += 1
        col += 1
        
    # 3. Bottom Edge: i in {0,1}, j in {0..3}, k=4
    # For each j, we have a column
    for j in range(4):
        c = col
        # U: i=0, j=j and i=1, j=j
        U[0*m + j][c] += 1
        U[1*m + j][c] += 1
        # V: j=j, k=4
        V[j*p + 4][c] += 1
        # W: k=4, i=0 and i=1
        W[4*n + 0][c] += 1
        W[4*n + 1][c] += 1
        col += 1
        
    # 4. Bottom Row: i=2, j in {0..4}, k in {0..4}
    # For each j, we have a column
    for j in range(5):
        c = col
        # U: i=2, j=j
        U[2*m + j][c] += 1
        # V: j=j, k=k (for all k? No, V is fixed per column)
        # Wait, the term is A_{2j} B_{jk}.
        # We want T_{2jk} = 1 for all k.
        # So for column c_j, we need V_{jk, c_j} to be 1 for all k?
        # And W_{k2, c_j} to be 1 for all k?
        # Let's check:
        # Sum_c U_{2j', c} V_{j'k', c} W_{k'2, c}
        # For c = j: U_{2j, j}=1.
        # We need V_{jk, j}=1 for all k, and W_{k2, j}=1 for all k.
        # Then for any k', V_{jk', j}=1 and W_{k'2, j}=1.
        # Product is 1.
        # For c != j: U_{2j, c}=0.
        # So yes, set V and W to 1 for all valid indices.
        for k in range(5):
            V[j*p + k][c] += 1
            W[k*n + 2][c] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    
    # Verification
    # T_ijk = sum_c U_{im+j, c} V_{jp+k, c} W_{kn+i, c}
    # We want T_ijk = 1 for all i,j,k
    T = np.zeros((n, m, p))
    for c in range(U.shape[1]):
        u = U[:, c].reshape(n, m)
        v = V[:, c].reshape(m, p)
        w = W[:, c].reshape(p, n)
        # Outer product? No, the contraction is element-wise multiplication of the 3 vectors?
        # No, the tensor is U \otimes V \otimes W.
        # T_ijk = sum_c U_{ij,c} V_{jk,c} W_{ki,c}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u[i, j] * v[j, k] * w[k, i]
                    
    # Check if T is all ones
    if np.allclose(T, 1.0):
        print("Decomposition is correct.")
    else:
        print("Decomposition is INCORRECT.")
        print("Min:", np.min(T), "Max:", np.max(T))
        # Print differences
        diff = T - 1.0
        if np.any(diff != 0):
             indices = np.argwhere(diff != 0)
             for idx in indices[:10]:
                 i, j, k = idx
                 print(f"Error at {i},{j},{k}: {T[i,j,k]}")
