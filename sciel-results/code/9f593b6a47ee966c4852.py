"""Matrix multiplication 2,5,6 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Target Rank 54 based on Strassen decomposition of 2x2 blocks
    rank = 54
    # Use NumPy arrays for efficiency
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        """
        Adds a rank-7 Strassen decomposition for a 2x2 * 2x2 block.
        Indices:
        i: i_base, i_base+1
        j: j_base, j_base+1
        k: k_base, k_base+1
        """
        nonlocal col
        
        if col + 7 > rank:
            return False

        # M0: (A00+A11)(B00+B11)
        U[i_base * m + j_base, col] += 1
        U[(i_base + 1) * m + (j_base + 1), col] += 1
        V[j_base * p + k_base, col] += 1
        V[(j_base + 1) * p + (k_base + 1), col] += 1
        W[k_base * n + i_base, col] += 1
        W[(k_base + 1) * n + (i_base + 1), col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        U[(i_base + 1) * m + j_base, col] += 1
        U[(i_base + 1) * m + (j_base + 1), col] += 1
        V[j_base * p + k_base, col] += 1
        W[k_base * n + (i_base + 1), col] += 1
        W[(k_base + 1) * n + (i_base + 1), col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        U[i_base * m + j_base, col] += 1
        V[j_base * p + (k_base + 1), col] += 1
        V[(j_base + 1) * p + (k_base + 1), col] -= 1
        W[(k_base + 1) * n + i_base, col] += 1
        W[(k_base + 1) * n + (i_base + 1), col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        U[(i_base + 1) * m + (j_base + 1), col] += 1
        V[(j_base + 1) * p + k_base, col] += 1
        V[j_base * p + k_base, col] -= 1
        W[k_base * n + i_base, col] += 1
        W[k_base * n + (i_base + 1), col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        U[i_base * m + j_base, col] += 1
        U[i_base * m + (j_base + 1), col] += 1
        V[(j_base + 1) * p + (k_base + 1), col] += 1
        W[k_base * n + i_base, col] -= 1
        W[(k_base + 1) * n + i_base, col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        U[(i_base + 1) * m + j_base, col] += 1
        U[i_base * m + j_base, col] -= 1
        V[j_base * p + k_base, col] += 1
        V[j_base * p + (k_base + 1), col] += 1
        W[(k_base + 1) * n + (i_base + 1), col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        U[i_base * m + (j_base + 1), col] += 1
        U[(i_base + 1) * m + (j_base + 1), col] -= 1
        V[(j_base + 1) * p + k_base, col] += 1
        V[(j_base + 1) * p + (k_base + 1), col] += 1
        W[k_base * n + i_base, col] += 1
        col += 1
        
        return True

    def add_naive_2112_block(i_base, j_base, k_base):
        """
        Adds a rank-4 naive decomposition for a 2x1 * 1x2 block.
        Indices:
        i: i_base, i_base+1
        j: j_base
        k: k_base, k_base+1
        """
        nonlocal col
        
        if col + 4 > rank:
            return False

        # 1: A00 B00 -> C00
        U[i_base * m + j_base, col] += 1
        V[j_base * p + k_base, col] += 1
        W[k_base * n + i_base, col] += 1
        col += 1
        
        # 2: A00 B01 -> C01
        U[i_base * m + j_base, col] += 1
        V[j_base * p + (k_base + 1), col] += 1
        W[(k_base + 1) * n + i_base, col] += 1
        col += 1
        
        # 3: A10 B00 -> C10
        U[(i_base + 1) * m + j_base, col] += 1
        V[j_base * p + k_base, col] += 1
        W[k_base * n + (i_base + 1), col] += 1
        col += 1
        
        # 4: A10 B01 -> C11
        U[(i_base + 1) * m + j_base, col] += 1
        V[j_base * p + (k_base + 1), col] += 1
        W[(k_base + 1) * n + (i_base + 1), col] += 1
        col += 1
        
        return True

    # Apply Strassen blocks for j pairs (0,1) and (2,3)
    # j_base = 0
    if not all([
        add_strassen_block(0, 0, 0),
        add_strassen_block(0, 0, 2),
        add_strassen_block(0, 0, 4)
    ]):
        raise RuntimeError("Strassen block 0 failed")
        
    # j_base = 2
    if not all([
        add_strassen_block(0, 2, 0),
        add_strassen_block(0, 2, 2),
        add_strassen_block(0, 2, 4)
    ]):
        raise RuntimeError("Strassen block 2 failed")
        
    # Apply naive blocks for j=4
    if not all([
        add_naive_2112_block(0, 4, 0),
        add_naive_2112_block(0, 4, 2),
        add_naive_2112_block(0, 4, 4)
    ]):
        raise RuntimeError("Naive block 4 failed")
        
    # Verify col == 54
    if col != 54:
        # Fallback to schoolbook if logic error
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Verify the decomposition using NumPy
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
                
    # Compute T_dec = sum_r U[:,r] V[:,r] W[:,r]
    # T_dec[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T_dec = np.einsum('ar,br,cr->abc', U, V, W)
    
    if not np.allclose(T_dec, T, atol=1e-10, rtol=1e-10):
        # If verification fails, return schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int64)
        V = np.zeros((m * p, rank), dtype=np.int64)
        W = np.zeros((p * n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
