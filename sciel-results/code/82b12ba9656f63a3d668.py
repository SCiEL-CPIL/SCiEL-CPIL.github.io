import numpy as np
import time
# EVOLVE-BLOCK-START
def _construct_strassen_54():
    """
    Construct a rank-54 decomposition for 2x5x6 matrix multiplication.
    Strategy: 
    - Split the 5 columns of A into two 2-column blocks (j=0,1 and j=2,3) and one 1-column block (j=4).
    - Split the 6 rows of B into three 2-row blocks (k=0,1; k=2,3; k=4,5).
    - Use Strassen's algorithm (rank 7) for each 2x2 block multiplication (A_block 2x2, B_block 2x2).
    - Use schoolbook (rank 4) for the 2x1 block multiplication (A_block 2x1, B_block 1x2).
    
    Total rank: 3 blocks * 7 (Strassen) + 1 block * 4 (Schoolbook) = 21 + 4 = 25? 
    Wait, the dimensions are 2x5 * 5x6.
    A is 2x5. B is 5x6.
    We can group the 'j' index (inner dimension of A, outer of B).
    j in {0,1,2,3,4}.
    
    If we group j={0,1} and j={2,3} and j={4}:
    For j={0,1}: A_sub is 2x2, B_sub is 2x6.
    We can split B_sub into 3 blocks of 2x2: B_{0,1}, B_{2,3}, B_{4,5}.
    For each 2x2 * 2x2 multiplication, we use Strassen (rank 7).
    There are 3 such blocks for the j={0,1} case. Total rank 3*7 = 21.
    
    For j={2,3}: A_sub is 2x2, B_sub is 2x6.
    Similarly, split B into 3 blocks of 2x2.
    For each 2x2 * 2x2 multiplication, we use Strassen (rank 7).
    Total rank 3*7 = 21.
    
    For j={4}: A_sub is 2x1, B_sub is 1x6.
    Split B into 3 blocks of 1x2.
    For each 2x1 * 1x2 multiplication, it is a rank 4 operation (schoolbook for 2x2 with one zero column/row? No, 2x1 * 1x2 is outer product of col and row, which is rank 4? 
    C = a * b^T. C is 2x2. C_{00}=a0b0, C_{01}=a0b1, C_{10}=a1b0, C_{11}=a1b1.
    This requires 4 multiplications.
    So for the 3 blocks of B, we have 3 * 4 = 12 rank.
    
    Total Rank = 21 + 21 + 12 = 54.
    """
    n, m, p = 2, 5, 6
    nm, mp, pn = n*m, m*p, p*n
    rank = 54
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    # Strassen coefficients for 2x2 matmul:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Indices for A (2x2):
    # A[0,0] is at i=0, j=0 -> index 0*5+0 = 0
    # A[0,1] is at i=0, j=1 -> index 0*5+1 = 1
    # A[1,0] is at i=1, j=0 -> index 1*5+0 = 5
    # A[1,1] is at i=1, j=1 -> index 1*5+1 = 6
    
    # Indices for B (2x2):
    # B[0,0] is at j=0, k=0 -> index 0*6+0 = 0
    # B[0,1] is at j=0, k=1 -> index 0*6+1 = 1
    # B[1,0] is at j=1, k=0 -> index 1*6+0 = 6
    # B[1,1] is at j=1, k=1 -> index 1*6+1 = 7
    
    # Indices for C (2x2) in W (shape 6x2, index k*n+i):
    # C[0,0] is at k=0, i=0 -> index 0*2+0 = 0
    # C[0,1] is at k=0, i=1 -> index 0*2+1 = 1
    # C[1,0] is at k=1, i=0 -> index 1*2+0 = 2
    # C[1,1] is at k=1, i=1 -> index 1*2+1 = 3
    
    def add_strassen_block(j_offset, k_offset):
        nonlocal col
        # A indices
        a00 = 0 * m + (j_offset + 0)
        a01 = 0 * m + (j_offset + 1)
        a10 = 1 * m + (j_offset + 0)
        a11 = 1 * m + (j_offset + 1)
        
        # B indices
        b00 = (j_offset + 0) * p + (k_offset + 0)
        b01 = (j_offset + 0) * p + (k_offset + 1)
        b10 = (j_offset + 1) * p + (k_offset + 0)
        b11 = (j_offset + 1) * p + (k_offset + 1)
        
        # C indices
        c00 = (k_offset + 0) * n + 0
        c01 = (k_offset + 0) * n + 1
        c10 = (k_offset + 1) * n + 0
        c11 = (k_offset + 1) * n + 1
        
        # M1
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
        # M2
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # M3
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1
        col += 1
        
        # M4
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        
        # M5
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        
        # M6
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        
        # M7
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        
    def add_schoolbook_block(j_offset, k_offset):
        nonlocal col
        # A is 2x1: indices a00, a10 (using j_offset)
        a00 = 0 * m + (j_offset + 0)
        a10 = 1 * m + (j_offset + 0)
        
        # B is 1x2: indices b00, b01 (using j_offset, k_offset)
        b00 = (j_offset + 0) * p + (k_offset + 0)
        b01 = (j_offset + 0) * p + (k_offset + 1)
        
        # C is 2x2: indices c00, c01, c10, c11
        c00 = (k_offset + 0) * n + 0
        c01 = (k_offset + 0) * n + 1
        c10 = (k_offset + 1) * n + 0
        c11 = (k_offset + 1) * n + 1
        
        # C00 = A00 * B00
        U[a00, col] += 1
        V[b00, col] += 1
        W[c00, col] += 1
        col += 1
        
        # C01 = A00 * B01
        U[a00, col] += 1
        V[b01, col] += 1
        W[c01, col] += 1
        col += 1
        
        # C10 = A10 * B00
        U[a10, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1
        col += 1
        
        # C11 = A10 * B01
        U[a10, col] += 1
        V[b01, col] += 1
        W[c11, col] += 1
        col += 1

    # j=0,1 block (A is 2x2)
    for k_off in range(0, 6, 2):
        add_strassen_block(0, k_off)
        
    # j=2,3 block (A is 2x2)
    for k_off in range(0, 6, 2):
        add_strassen_block(2, k_off)
        
    # j=4 block (A is 2x1)
    for k_off in range(0, 6, 2):
        add_schoolbook_block(4, k_off)
        
    return U, V, W

def _linear_polish(U, V, W, T, n, m, p, max_iters=10, tol=1e-9):
    """
    Alternating Linear Least Squares for tensor decomposition.
    This is much faster and more stable than gradient descent for exact recovery.
    """
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    # Pre-compute target slices
    # T[a, :, :] is a matrix of size mp x pn
    # T[:, b, :] is a matrix of size nm x pn
    # T[:, :, c] is a matrix of size nm x mp
    
    for _ in range(max_iters):
        # Update U
        for a in range(nm):
            # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # Let M[r, b, c] = V[b, r] W[c, r]
            # We want U[a, :] such that U[a, :] @ M_flat = T[a, :, :].flatten()
            # M_flat is (r, mp*pn)
            # Actually, it's easier to form the system:
            # sum_r U[a, r] * (V[b, r] W[c, r]) = T[a, b, c]
            # This is a linear system in U[a, :].
            # Let X = U[a, :].
            # sum_r X[r] * A[r, b, c] = T[a, b, c]
            # where A[r, b, c] = V[b, r] * W[c, r]
            # We can flatten b,c into a single index.
            # A_flat[r, bc] = V[b, r] * W[c, r]
            # T_flat[bc] = T[a, b, c]
            # Solve A_flat.T @ X = T_flat ? No.
            # sum_r X[r] * A_flat[r, bc] = T_flat[bc]
            # This is A_flat.T @ X = T_flat?
            # Let's check dimensions.
            # A_flat is (r, mp*pn).
            # X is (r,).
            # A_flat.T is (mp*pn, r).
            # A_flat.T @ X is (mp*pn,).
            # T_flat is (mp*pn,).
            # So we solve A_flat.T @ X = T_flat.
            
            A_flat = np.zeros((r, mp * pn))
            for b in range(mp):
                for c in range(pn):
                    A_flat[:, b * pn + c] = V[b, :] * W[c, :]
            T_flat = T[a, :, :].flatten()
            
            try:
                U[a, :], _, _, _ = np.linalg.lstsq(A_flat.T, T_flat, rcond=None)
            except:
                pass

        # Update V
        for b in range(mp):
            A_flat = np.zeros((r, nm * pn))
            for a in range(nm):
                for c in range(pn):
                    A_flat[:, a * pn + c] = U[a, :] * W[c, :]
            T_flat = T[:, b, :].flatten()
            
            try:
                V[b, :], _, _, _ = np.linalg.lstsq(A_flat.T, T_flat, rcond=None)
            except:
                pass

        # Update W
        for c in range(pn):
            A_flat = np.zeros((r, nm * mp))
            for a in range(nm):
                for b in range(mp):
                    A_flat[:, a * mp + b] = U[a, :] * V[b, :]
            T_flat = T[:, :, c].flatten()
            
            try:
                W[c, :], _, _, _ = np.linalg.lstsq(A_flat.T, T_flat, rcond=None)
            except:
                pass
                
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    r = U.shape[1]
    nm, mp, pn = n*m, m*p, p*n
    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    # Check zero entries (sample)
    np.random.seed(0)
    for _ in range(100):
        a = np.random.randint(0, nm)
        b = np.random.randint(0, mp)
        c = np.random.randint(0, pn)
        i_a, j_a = divmod(a, m)
        j_b, k_b = divmod(b, p)
        k_c, i_c = divmod(c, n)
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue
        val = 0.0
        for cc in range(r):
            val += U[a, cc] * V[b, cc] * W[c, cc]
        if abs(val) > tol:
            return False
    return True

def construct():
    n, m, p = 2, 5, 6
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # 1. Construct Rank 54 Solution
    U54, V54, W54 = _construct_strassen_54()
    
    # Verify it
    if not _verify(U54, V54, W54, n, m, p):
        # Fallback to schoolbook if construction failed (should not happen)
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": [[float(x) for x in row] for row in U], 
                "V": [[float(x) for x in row] for row in V], 
                "W": [[float(x) for x in row] for row in W]}
    
    best_rank = 54
    best_U, best_V, best_W = U54, V54, W54
    
    # 2. Try to improve to rank < 54
    # Lower bound is max(nm, mp, pn) = 30.
    # Try ranks 53 down to 40 (or until time runs out)
    # Use Linear Polish with random restarts
    
    for rank in range(53, 30, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        # Fewer restarts for higher ranks? Or fixed?
        max_restarts = 5
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            np.random.seed(100 + rank * 10 + restart)
            scale = 1.0 / np.sqrt(rank)
            U = np.random.randn(nm, rank) * scale
            V = np.random.randn(mp, rank) * scale
            W = np.random.randn(pn, rank) * scale
            
            # Polish
            U, V, W = _linear_polish(U, V, W, T, n, m, p, max_iters=20)
            
            # Verify
            if _verify(U, V, W, n, m, p, tol=1e-7):
                found = True
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
        
        if found:
            # If we found a lower rank, we can try to go lower immediately
            # But the loop structure handles this by continuing to rank-1
            pass

    # Return best result
    return {"U": [[float(x) for x in row] for row in best_U], 
            "V": [[float(x) for x in row] for row in best_V], 
            "W": [[float(x) for x in row] for row in best_W]}
# EVOLVE-BLOCK-END
