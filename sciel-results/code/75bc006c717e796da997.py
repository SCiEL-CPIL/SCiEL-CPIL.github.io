"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W each of shape (4, 7) with integer entries.
    The convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    such that sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] = delta(i,j,k) for the tensor.
    """
    # Strassen's 7 products for 2x2 matrices
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # U: maps (i,j) -> contributions to M1..M7
    # V: maps (j,k) -> contributions to M1..M7
    # W: maps (k,i) -> contributions to M1..M7
    
    # U[i*2+j, r] where i=0,1 and j=0,1
    # A00 is (0,0), A01 is (0,1), A10 is (1,0), A11 is (1,1)
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (A00+A11)(B00+B11)
    U[0*2+0, 0] = 1  # A00
    U[1*2+1, 0] = 1  # A11
    V[0*2+0, 0] = 1  # B00
    V[1*2+1, 0] = 1  # B11
    W[0*2+0, 0] = 1  # C00 gets +M1
    W[1*2+1, 0] = 1  # C11 gets +M1
    
    # M2 = (A10+A11)B00
    U[1*2+0, 1] = 1  # A10
    U[1*2+1, 1] = 1  # A11
    V[0*2+0, 1] = 1  # B00
    W[1*2+0, 1] = 1  # C10 gets +M2
    
    # M3 = A00(B10-B11)
    U[0*2+0, 2] = 1  # A00
    V[1*2+0, 2] = 1  # B10
    V[1*2+1, 2] = -1 # B11
    W[0*2+1, 2] = 1  # C01 gets +M3
    
    # M4 = A11(B01-B10)
    U[1*2+1, 3] = 1  # A11
    V[0*2+1, 3] = 1  # B01
    V[1*2+0, 3] = -1 # B10
    W[1*2+1, 3] = 1  # C11 gets +M4
    
    # M5 = (A00+A01)B11
    U[0*2+0, 4] = 1  # A00
    U[0*2+1, 4] = 1  # A01
    V[1*2+1, 4] = 1  # B11
    W[0*2+0, 4] = 1  # C00 gets +M5
    
    # M6 = (A10-A00)(B00+B10)
    U[1*2+0, 5] = 1  # A10
    U[0*2+0, 5] = -1 # A00
    V[0*2+0, 5] = 1  # B00
    V[1*2+0, 5] = 1  # B10
    W[1*2+0, 5] = 1  # C10 gets +M6
    
    # M7 = (A01-A11)(B01+B11)
    U[0*2+1, 6] = 1  # A01
    U[1*2+1, 6] = -1 # A11
    V[0*2+1, 6] = 1  # B01
    V[1*2+1, 6] = 1  # B11
    W[0*2+1, 6] = 1  # C01 gets +M7
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                coeff = 0.0
                for r in range(rank):
                    coeff += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(coeff - 1) > tol:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using vectorized operations."""
    rank = U.shape[1]
    # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should be 1
    # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # = sum_r U3[i,j,r] * (V3[j,k,r] * W3[k,i,r])
    
    # Use einsum
    # First compute V3[j,k,r] * W3[k,i,r] -> sum over k? No, we need to keep k for the final sum
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    
    # Let's do it step by step
    # For fixed i,j: sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r] for each k
    # = sum_r U3[i,j,r] * (V3[j,:,r] * W3[:,i,r])
    
    # V3[j,:,r] has shape (p, rank) for fixed j
    # W3[:,i,r] has shape (p, rank) for fixed i
    # Product has shape (p, rank)
    # Then dot with U3[i,j,:] (shape rank)
    
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            # sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r] for all k
            # = sum_r U3[i,j,r] * (V3[j,:,r] * W3[:,i,r])
            # V3[j,:,r] shape (p,), W3[:,i,r] shape (p,)
            prod = V3[j, :, :] * W3[:, i, :]  # shape (p, rank)
            T[i, j, :] = np.sum(U3[i, j, :] * prod, axis=1)
    
    return np.all(np.abs(T - 1) <= tol)

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _block_decomposition_3x4x6():
    """
    Construct a decomposition for 3x4x6 matrix multiplication.
    
    Strategy: Split the 3x4x6 problem into blocks.
    We can write the 4x6 matrix B as [B1; B2] where B1 is 2x6 and B2 is 2x6.
    Similarly A is 3x4 = [A1 A2] where A1 is 3x2 and A2 is 3x2.
    
    Then C = A1*B1 + A2*B2.
    
    Each term is 3x2 * 2x6. We can further decompose this.
    
    Alternative: Use the known optimal rank for 2x2x3 which is 11? No, 2x2x3 has rank at most 12 (schoolbook) and it's known to be 11 or 12.
    Actually, the rank of 2x2x3 is 11 (Landsberg).
    
    Let's try a different approach. Split m=4 into 2+2 and p=6 into 3+3.
    Then we have four subproblems of type 3x2x3.
    The rank of 3x2x3 is at most 18 (schoolbook). But we can do better.
    
    Actually, let's use a simpler construction.
    Split the 4x6 matrix into 2x3 blocks: there are 2*(6/3) = 2*2 = 4 blocks of size 2x3.
    And the 3x4 matrix into 3x2 blocks: there are 3*(4/2) = 3*2 = 6 blocks of size 3x2.
    
    Hmm, this is getting complicated. Let me try a direct approach.
    
    We know that rank(2x2x2) = 7 (Strassen).
    
    For 3x4x6, we can use the following:
    - The minimum rank is at least max(n*m, m*p, p*n) = max(12, 24, 18) = 24.
    - The schoolbook rank is 3*4*6 = 72.
    
    Let's try to use a block construction.
    Write A (3x4) as [A_left (3x2), A_right (3x2)]
    Write B (4x6) as [B_top (2x6); B_bottom (2x6)]
    
    C = A_left * B_top + A_right * B_bottom
    
    Now, A_left * B_top is 3x2 * 2x6.
    We can further split B_top into 2x3 blocks: B_top = [B_top1 (2x3), B_top2 (2x3)]
    Then A_left * B_top = A_left * B_top1 + A_left * B_top2
    
    Each term is 3x2 * 2x3. The rank of 3x2x3 is at most 18, but we can do better.
    
    Actually, let's just use a known result. The rank of the 3x4x6 tensor is at most 48.
    This can be achieved by splitting m=4 into 2+2 and using rank 12 for each 3x2x6 subproblem.
    But 3x2x6 has rank at most 36 (schoolbook), so that's not helpful.
    
    Let me try a different splitting. Split p=6 into 3+3.
    Then C = A * B_left + A * B_right where B_left is 4x3 and B_right is 4x3.
    Each term is 3x4 * 4x3. The rank of 3x4x3 is at most 36 (schoolbook).
    
    This is not leading anywhere productive. Let me just use the search approach but with better initialization.
    
    Actually, let's try to construct a rank-48 decomposition explicitly.
    We can split the 4x6 matrix B into two 2x6 matrices: B = [B1; B2].
    And the 3x4 matrix A into [A1 A2] where A1, A2 are 3x2.
    
    C = A1*B1 + A2*B2
    
    Now, for each term A1*B1 (3x2 * 2x6), we can use the following:
    Split the 2x6 matrix B1 into three 2x2 blocks: B1 = [B1a B1b B1c] where each is 2x2.
    Then A1*B1 = A1*B1a + A1*B1b + A1*B1c
    
    Each term is 3x2 * 2x2. We can use Strassen-like for this.
    Actually, 3x2 * 2x2: we can write this as (3x2) * (2x2) = 3x2 result.
    The rank of 3x2x2 is at most 12 (schoolbook). But we can do better.
    
    Let's use the fact that we can absorb one dimension. 
    For 3x2x2: we can think of it as 3 copies of 1x2x2.
    The rank of 1x2x2 is 2 (since it's just scaling).
    So rank(3x2x2) <= 3*2 = 6? No, that's not right either.
    
    Actually, for 3x2x2, the schoolbook rank is 3*2*2 = 12.
    But we can do better by grouping.
    
    Let me just try the iterative refinement approach with better parameters.
    """
    pass

def _try_rank_decomposition_v2(n, m, p, rank, time_budget=30.0):
    """
    Try to find a rank-`rank` decomposition using structured search with better initialization.
    """
    deadline = time.time() + time_budget
    best_result = None
    
    # Try different random seeds
    max_restarts = 30
    
    for restart in range(max_restarts):
        if time.time() > deadline - 3:
            break
        
        np.random.seed(1000 + restart)
        
        # Initialize with random values
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Iterative refinement
        for iteration in range(100):
            if time.time() > deadline - 3:
                break
            
            # Solve for W given U, V
            # For each (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Variables: W[k*n+i, r] for all k, i, r
            num_constraints = n * m * p
            num_vars_w = p * n * rank
            
            A_w = np.zeros((num_constraints, num_vars_w))
            b_w = np.ones(num_constraints)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            A_w[idx, (k * n + i) * rank + r] = U[i*m+j, r] * V[j*p+k, r]
                        idx += 1
            
            W_flat, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
            W = W_flat.reshape(p * n, rank)
            
            # Solve for U given V, W
            num_vars_u = n * m * rank
            A_u = np.zeros((num_constraints, num_vars_u))
            b_u = np.ones(num_constraints)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            A_u[idx, (i * m + j) * rank + r] = V[j*p+k, r] * W[k*n+i, r]
                        idx += 1
            
            U_flat, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
            U = U_flat.reshape(n * m, rank)
            
            # Solve for V given U, W
            num_vars_v = m * p * rank
            A_v = np.zeros((num_constraints, num_vars_v))
            b_v = np.ones(num_constraints)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            A_v[idx, (j * p + k) * rank + r] = U[i*m+j, r] * W[k*n+i, r]
                        idx += 1
            
            V_flat, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
            V = V_flat.reshape(m * p, rank)
            
            # Check convergence
            if _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-8):
                best_result = (U, V, W)
                break
        
        if best_result is not None:
            break
    
    return best_result

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    start_time = time.time()
    
    # Start with schoolbook as fallback
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions
    # The theoretical lower bound is max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # Let's try ranks from 30 up to 72
    
    ranks_to_try = [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 64, 68]
    
    for r in ranks_to_try:
        elapsed = time.time() - start_time
        if elapsed > 130:  # Safety margin
            break
        
        # Allocate time based on remaining budget
        remaining = 130 - elapsed
        time_budget = min(15.0, remaining * 0.5)
        
        if time_budget < 5:
            break
        
        result = _try_rank_decomposition_v2(n, m, p, r, time_budget=time_budget)
        if result is not None:
            best_rank = r
            best_U, best_V, best_W = result
            break
    
    # Final verification
    if not _verify_decomposition_fast(best_U, best_V, best_W, n, m, p, tol=1e-8):
        # Fallback to schoolbook
        best_U, best_V, best_W = _schoolbook_decomposition(n, m, p)
        best_rank = n * m * p
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
