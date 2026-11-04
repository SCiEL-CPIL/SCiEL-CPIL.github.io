"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
DEADLINE = START_TIME + 115  # Safety margin before 120s limit

def _time_remaining():
    return DEADLINE - time.time()

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 54 decomposition using Strassen's algorithm for 2x2 blocks
    # A is 2x5, B is 5x6, C is 2x6
    # Partition: j-pairs (0,1) and (2,3) use Strassen (7 mults each per k-pair)
    # j=4 uses naive (4 mults per k-pair)
    # k-pairs: (0,1), (2,3), (4,5)
    # Total: 2 * 3 * 7 + 1 * 3 * 4 = 42 + 12 = 54
    
    rank = 54
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = [0]  # Use list for mutability in nested function
    
    def add_strassen_block(i_base, j_base, k_base):
        """Add 7 rank-1 terms for a 2x2x2 Strassen block."""
        c = col[0]
        
        # M0: (A00+A11)(B00+B11) -> C00 + C11
        U[i_base * m + j_base, c] += 1
        U[(i_base+1) * m + (j_base+1), c] += 1
        V[j_base * p + k_base, c] += 1
        V[(j_base+1) * p + (k_base+1), c] += 1
        W[k_base * n + i_base, c] += 1
        W[(k_base+1) * n + (i_base+1), c] += 1
        c += 1
        
        # M1: (A10+A11)B00 -> C10 + C11 (C11 gets -1)
        U[(i_base+1) * m + j_base, c] += 1
        U[(i_base+1) * m + (j_base+1), c] += 1
        V[j_base * p + k_base, c] += 1
        W[k_base * n + (i_base+1), c] += 1
        W[(k_base+1) * n + (i_base+1), c] -= 1
        c += 1
        
        # M2: A00(B01-B11) -> C01 + C11
        U[i_base * m + j_base, c] += 1
        V[j_base * p + (k_base+1), c] += 1
        V[(j_base+1) * p + (k_base+1), c] -= 1
        W[(k_base+1) * n + i_base, c] += 1
        W[(k_base+1) * n + (i_base+1), c] += 1
        c += 1
        
        # M3: A11(B10-B00) -> C00 + C10
        U[(i_base+1) * m + (j_base+1), c] += 1
        V[(j_base+1) * p + k_base, c] += 1
        V[j_base * p + k_base, c] -= 1
        W[k_base * n + i_base, c] += 1
        W[k_base * n + (i_base+1), c] += 1
        c += 1
        
        # M4: (A00+A01)B11 -> C00 (-1) + C01
        U[i_base * m + j_base, c] += 1
        U[i_base * m + (j_base+1), c] += 1
        V[(j_base+1) * p + (k_base+1), c] += 1
        W[k_base * n + i_base, c] -= 1
        W[(k_base+1) * n + i_base, c] += 1
        c += 1
        
        # M5: (A10-A00)(B00+B01) -> C11
        U[(i_base+1) * m + j_base, c] += 1
        U[i_base * m + j_base, c] -= 1
        V[j_base * p + k_base, c] += 1
        V[j_base * p + (k_base+1), c] += 1
        W[(k_base+1) * n + (i_base+1), c] += 1
        c += 1
        
        # M6: (A01-A11)(B10+B11) -> C00
        U[i_base * m + (j_base+1), c] += 1
        U[(i_base+1) * m + (j_base+1), c] -= 1
        V[(j_base+1) * p + k_base, c] += 1
        V[(j_base+1) * p + (k_base+1), c] += 1
        W[k_base * n + i_base, c] += 1
        c += 1
        
        col[0] = c
    
    def add_naive_2112_block(i_base, j_base, k_base):
        """Add 4 rank-1 terms for a 2x1 * 1x2 multiplication block."""
        c = col[0]
        
        # A00 * B00 -> C00
        U[i_base * m + j_base, c] += 1
        V[j_base * p + k_base, c] += 1
        W[k_base * n + i_base, c] += 1
        c += 1
        
        # A00 * B01 -> C01
        U[i_base * m + j_base, c] += 1
        V[j_base * p + (k_base+1), c] += 1
        W[(k_base+1) * n + i_base, c] += 1
        c += 1
        
        # A10 * B00 -> C10
        U[(i_base+1) * m + j_base, c] += 1
        V[j_base * p + k_base, c] += 1
        W[k_base * n + (i_base+1), c] += 1
        c += 1
        
        # A10 * B01 -> C11
        U[(i_base+1) * m + j_base, c] += 1
        V[j_base * p + (k_base+1), c] += 1
        W[(k_base+1) * n + (i_base+1), c] += 1
        c += 1
        
        col[0] = c
    
    # Apply Strassen blocks for j-pairs (0,1) and (2,3)
    # k-pairs are (0,1), (2,3), (4,5)
    
    # j_pair (0,1)
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # j_pair (2,3)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # Apply naive blocks for j=4
    # k-pairs are (0,1), (2,3), (4,5)
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)
    
    # Trim to actual rank used
    actual_rank = col[0]
    if actual_rank < rank:
        U = U[:, :actual_rank]
        V = V[:, :actual_rank]
        W = W[:, :actual_rank]
    
    # Verify the decomposition with a quick check (only if time permits)
    if _time_remaining() > 10:
        try:
            r = U.shape[1]
            
            # Verify by checking that the decomposition produces correct results
            # for random matrices
            np.random.seed(42)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            # Compute C using the tensor decomposition
            # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
            C_computed = np.zeros((n, p))
            for r in range(r):
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            C_computed[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A_test[i,j] * B_test[j,k]
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                # Decomposition is incorrect; fall back to schoolbook
                rank_school = n * m * p
                U_school = np.zeros((n * m, rank_school), dtype=int)
                V_school = np.zeros((m * p, rank_school), dtype=int)
                W_school = np.zeros((p * n, rank_school), dtype=int)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx = i * m * p + j * p + k
                            U_school[i*m+j, idx] = 1
                            V_school[j*p+k, idx] = 1
                            W_school[k*n+i, idx] = 1
                return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
        except Exception:
            pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
