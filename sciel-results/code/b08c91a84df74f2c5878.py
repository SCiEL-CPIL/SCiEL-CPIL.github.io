import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define partitions for Strassen blocks
    # n: [0, 1], [2]
    # m: [0, 1], [2, 3], [4]
    # p: [0, 1], [2, 3], [4, 5], [6]
    
    n_blocks = [(0, 2)]
    m_blocks = [(0, 2), (2, 4)]
    p_blocks = [(0, 2), (2, 4), (4, 6)]
    
    # We will generate 6 Strassen blocks
    blocks = []
    for n_start, n_end in n_blocks:
        for m_start, m_end in m_blocks:
            for p_start, p_end in p_blocks:
                blocks.append((n_start, m_start, p_start))
                
    # Track covered (i, j, k) indices
    covered = set()
    for n0, m0, p0 in blocks:
        for i in range(n0, n0 + 2):
            for j in range(m0, m0 + 2):
                for k in range(p0, p0 + 2):
                    covered.add((i, j, k))
    
    # Initialize U, V, W as lists of columns (sparse representation initially)
    # Each column is a dict {index: value}
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Add Strassen blocks
    for n0, m0, p0 in blocks:
        # Indices
        i0, i1 = n0, n0 + 1
        j0, j1 = m0, m0 + 1
        k0, k1 = p0, p0 + 1
        
        # U indices (i * m + j)
        u00 = i0 * m + j0
        u01 = i0 * m + j1
        u10 = i1 * m + j0
        u11 = i1 * m + j1
        
        # V indices (j * p + k)
        v00 = j0 * p + k0
        v01 = j0 * p + k1
        v10 = j1 * p + k0
        v11 = j1 * p + k1
        
        # W indices (k * n + i)
        w00 = k0 * n + i0
        w01 = k1 * n + i0
        w10 = k0 * n + i1
        w11 = k1 * n + i1
        
        # M1 = (A00 + A11)(B00 + B11)
        U_cols.append({u00: 1, u11: 1})
        V_cols.append({v00: 1, v11: 1})
        W_cols.append({w00: 1, w11: 1})
        
        # M2 = (A10 + A11)B00
        U_cols.append({u10: 1, u11: 1})
        V_cols.append({v00: 1})
        W_cols.append({w10: 1, w11: -1})
        
        # M3 = A00(B01 - B11)
        U_cols.append({u00: 1})
        V_cols.append({v01: 1, v11: -1})
        W_cols.append({w01: 1, w11: 1})
        
        # M4 = A11(B10 - B00)
        U_cols.append({u11: 1})
        V_cols.append({v10: 1, v00: -1})
        W_cols.append({w00: 1, w10: 1})
        
        # M5 = (A00 + A01)B11
        U_cols.append({u00: 1, u01: 1})
        V_cols.append({v11: 1})
        W_cols.append({w00: -1, w01: 1})
        
        # M6 = (A10 - A00)(B00 + B01)
        U_cols.append({u10: 1, u00: -1})
        V_cols.append({v00: 1, v01: 1})
        W_cols.append({w11: 1})
        
        # M7 = (A01 - A11)(B10 + B11)
        U_cols.append({u01: 1, u11: -1})
        V_cols.append({v10: 1, v11: 1})
        W_cols.append({w00: 1})
        
    # Add schoolbook for remaining (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in covered:
                    continue
                U_cols.append({i * m + j: 1})
                V_cols.append({j * p + k: 1})
                W_cols.append({k * n + i: 1})
                
    # Convert sparse columns to dense matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for idx, val in U_cols[c].items():
            U[idx][c] = val
        for idx, val in V_cols[c].items():
            V[idx][c] = val
        for idx, val in W_cols[c].items():
            W[idx][c] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result['U'][0])
    
    assert rank == 99, f"Expected rank 99, got {rank}"
    
    import random
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    C_decomp = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(p):
            val = 0
            for r in range(rank):
                ua = 0
                for jj in range(m):
                    ua += result['U'][i * m + jj][r] * A[i][jj]
                vb = 0
                for jj in range(m):
                    vb += result['V'][jj * p + k][r] * B[jj][k]
                val += result['W'][k * n + i][r] * ua * vb
            C_decomp[i][k] = val
            
    if C_decomp != C:
        print("Verification failed!", file=sys.stderr)
    else:
        print("Verification passed.", file=sys.stderr)
