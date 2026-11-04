import time
import numpy as np

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Strategy:
    # A is 2x5, B is 5x6, C is 2x6
    # Split A columns: (0,1), (2,3), (4)
    # Split B rows: (0,1), (2,3), (4)
    # Split C columns: (0,1), (2,3), (4,5)
    #
    # For each (A_block, B_block) pair:
    #   - If both are 2x2 (A cols j,j+1 and B rows j,j+1): use Strassen (7 mults) per C column pair
    #   - If A is 2x1 and B is 1x6: use naive (2 mults per C col)
    #
    # Blocks:
    #   A(0,1) x B(0,1): Strassen for C cols (0,1), (2,3), (4,5) -> 3*7 = 21
    #   A(2,3) x B(2,3): Strassen for C cols (0,1), (2,3), (4,5) -> 3*7 = 21
    #   A(4) x B(4): naive for all 6 C cols -> 6*2 = 12
    #   Total: 21 + 21 + 12 = 54
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        """Add Strassen decomposition for A[i_base:i_base+2, j_base:j_base+2] * B[j_base:j_base+2, k_base:k_base+2]"""
        nonlocal col
        # A indices: U[i*m + j] for i in {i_base, i_base+1}, j in {j_base, j_base+1}
        # B indices: V[j*p + k] for j in {j_base, j_base+1}, k in {k_base, k_base+1}
        # C indices: W[k*n + i] for k in {k_base, k_base+1}, i in {i_base, i_base+1}
        
        a00 = i_base * m + j_base
        a01 = i_base * m + (j_base + 1)
        a10 = (i_base + 1) * m + j_base
        a11 = (i_base + 1) * m + (j_base + 1)
        
        b00 = j_base * p + k_base
        b01 = j_base * p + (k_base + 1)
        b10 = (j_base + 1) * p + k_base
        b11 = (j_base + 1) * p + (k_base + 1)
        
        c00 = k_base * n + i_base
        c01 = k_base * n + (i_base + 1)
        c10 = (k_base + 1) * n + i_base
        c11 = (k_base + 1) * n + (i_base + 1)
        
        # M0: (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M1: (a10+a11)(b00) -> c01 += 1, c11 -= 1
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[c01][col] += 1
        W[c11][col] -= 1
        col += 1
        
        # M2: (a00)(b01-b11) -> c10 += 1, c11 += 1
        U[a00][col] += 1
        V[b01][col] += 1
        V[b11][col] -= 1
        W[c10][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M3: (a11)(b10-b00) -> c00 += 1, c01 += 1
        U[a11][col] += 1
        V[b10][col] += 1
        V[b00][col] -= 1
        W[c00][col] += 1
        W[c01][col] += 1
        col += 1
        
        # M4: (a00+a01)(b11) -> c00 -= 1, c10 += 1
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[c00][col] -= 1
        W[c10][col] += 1
        col += 1
        
        # M5: (a10-a00)(b00+b01) -> c11 += 1
        U[a10][col] += 1
        U[a00][col] -= 1
        V[b00][col] += 1
        V[b01][col] += 1
        W[c11][col] += 1
        col += 1
        
        # M6: (a01-a11)(b10+b11) -> c00 += 1
        U[a01][col] += 1
        U[a11][col] -= 1
        V[b10][col] += 1
        V[b11][col] += 1
        W[c00][col] += 1
        col += 1
    
    def add_naive_block(i_base, j_base, k_base):
        """Add naive decomposition for A[i_base:i_base+2, j_base] * B[j_base, k_base:k_base+2]"""
        nonlocal col
        # A indices: U[i*m + j_base] for i in {i_base, i_base+1}
        # B indices: V[j_base*p + k] for k in {k_base, k_base+1}
        # C indices: W[k*n + i] for k in {k_base, k_base+1}, i in {i_base, i_base+1}
        
        a0 = i_base * m + j_base
        a1 = (i_base + 1) * m + j_base
        
        b0 = j_base * p + k_base
        b1 = j_base * p + (k_base + 1)
        
        c00 = k_base * n + i_base
        c01 = k_base * n + (i_base + 1)
        c10 = (k_base + 1) * n + i_base
        c11 = (k_base + 1) * n + (i_base + 1)
        
        # a0 * b0 -> c00
        U[a0][col] += 1
        V[b0][col] += 1
        W[c00][col] += 1
        col += 1
        
        # a0 * b1 -> c10
        U[a0][col] += 1
        V[b1][col] += 1
        W[c10][col] += 1
        col += 1
        
        # a1 * b0 -> c01
        U[a1][col] += 1
        V[b0][col] += 1
        W[c01][col] += 1
        col += 1
        
        # a1 * b1 -> c11
        U[a1][col] += 1
        V[b1][col] += 1
        W[c11][col] += 1
        col += 1
    
    # Block 1: A cols (0,1) x B rows (0,1) -> Strassen for C col pairs (0,1), (2,3), (4,5)
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Block 2: A cols (2,3) x B rows (2,3) -> Strassen for C col pairs (0,1), (2,3), (4,5)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # Block 3: A col (4) x B row (4) -> naive for C col pairs (0,1), (2,3), (4,5)
    add_naive_block(0, 4, 0)
    add_naive_block(0, 4, 2)
    add_naive_block(0, 4, 4)
    
    # Verify col == 54
    if col != 54:
        # Fallback: this shouldn't happen, but if it does, we have a bug
        # For now, just return what we have (should be caught by verification)
        pass
    
    # Self-verification: check that the decomposition is correct
    # Convert to numpy arrays for verification
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Let's verify this for a few random points
    
    # Create random test matrices
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute C using the decomposition
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * A[i,j] * B[j,k]
    # Wait, that's not right. The decomposition is:
    # C[i,k] = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k]) * W[k*n+i,r]
    # No, that's not right either. Let me think again.
    
    # The standard tensor decomposition for matrix multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_r (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k]) * W[k*n+i,r]
    
    # Actually, the correct form is:
    # C[i,k] = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r] * A[i,j]) * (sum_j V[j*p+k,r] * B[j,k])
    
    # Let's verify
    C_decomposed = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                u_val = 0.0
                for j in range(m):
                    u_val += U_arr[i * m + j, r] * A_test[i, j]
                v_val = 0.0
                for j in range(m):
                    v_val += V_arr[j * p + k, r] * B_test[j, k]
                C_decomposed[i, k] += W_arr[k * n + i, r] * u_val * v_val
    
    # Check if C_decomposed matches C_expected
    if not np.allclose(C_decomposed, C_expected, atol=1e-10):
        # If verification fails, we have a bug in the decomposition
        # Return a naive decomposition as fallback
        rank_naive = n * m * p
        U_naive = [[0] * rank_naive for _ in range(n * m)]
        V_naive = [[0] * rank_naive for _ in range(m * p)]
        W_naive = [[0] * rank_naive for _ in range(p * n)]
        
        col_naive = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_naive[i * m + j][col_naive] = 1
                    V_naive[j * p + k][col_naive] = 1
                    W_naive[k * n + i][col_naive] = 1
                    col_naive += 1
        
        return {"U": U_naive, "V": V_naive, "W": W_naive}
    
    return {"U": U, "V": V, "W": W}
