import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Store best result found so far
    best_rank = schoolbook_rank
    best_solution = None
    
    # Generate schoolbook solution
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Strategy: Decompose 2x3x6 into two 2x3x3 problems.
    # If we find a rank R_sub solution for 2x3x3, we can get rank 2*R_sub for 2x3x6.
    # Schoolbook for 2x3x3 is 18. So 2*18=36 (same as schoolbook).
    # We need R_sub < 18 to improve.
    # Known lower bound for 2x3x3 is 13.
    
    n_sub, m_sub, p_sub = 2, 3, 3
    
    # Try to find a solution for 2x3x3 with rank < 18
    for target_rank in range(17, 12, -1):
        if time.time() > deadline - 20.0:
            break
            
        sol_sub = _search_decomp(n_sub, m_sub, p_sub, target_rank, time_limit=10.0)
        if sol_sub is not None:
            # Assemble 2x3x6 solution
            U_full, V_full, W_full = _assemble_split_solution(n, m, p, sol_sub)
            # Verify the full solution
            if _verify(n, m, p, U_full, V_full, W_full):
                best_rank = 2 * target_rank
                best_solution = {"U": U_full, "V": V_full, "W": W_full}
                return best_solution

    return best_solution

def _search_decomp(n, m, p, target_rank, time_limit=5.0):
    """
    Searches for a decomposition of rank <= target_rank for n x m x p multiplication.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    
    # Initialize with schoolbook and try to reduce rank?
    # Or random search.
    # Random search is slow.
    # Let's try a simple local search on random sparse matrices.
    
    # We can also try to use a known structure if target_rank is close to known values.
    # For now, use random search with integer entries {-1, 0, 1}.
    
    # To speed up, we can use the fact that we only need to check a few random matrices A, B.
    # But we need EXACT decomposition.
    
    # Let's try a few random candidates.
    max_attempts = 5000
    attempts = 0
    
    while attempts < max_attempts and time.time() - start_time < time_limit:
        attempts += 1
        
        # Generate random sparse matrices
        # Density 0.3
        U = np.zeros((n * m, target_rank), dtype=int)
        V = np.zeros((m * p, target_rank), dtype=int)
        W = np.zeros((p * n, target_rank), dtype=int)
        
        for r in range(target_rank):
            # Pick random non-zero entries
            # U
            idx = np.random.choice(n * m, size=max(1, int(0.3 * n * m)), replace=False)
            U[idx, r] = np.random.randint(-1, 2, size=len(idx))
            
            # V
            idx = np.random.choice(m * p, size=max(1, int(0.3 * m * p)), replace=False)
            V[idx, r] = np.random.randint(-1, 2, size=len(idx))
            
            # W
            idx = np.random.choice(p * n, size=max(1, int(0.3 * p * n)), replace=False)
            W[idx, r] = np.random.randint(-1, 2, size=len(idx))
            
        if _verify_fast(n, m, p, U, V, W, num_tests=2):
            # Final verification
            if _verify(n, m, p, U.tolist(), V.tolist(), W.tolist(), num_tests=10):
                return U.tolist(), V.tolist(), W.tolist()
                
    return None

def _assemble_split_solution(n, m, p, sol_sub):
    """
    Assembles a 2x3x6 solution from two 2x3x3 solutions.
    sol_sub is (U_sub, V_sub, W_sub) for 2x3x3.
    """
    U_sub, V_sub, W_sub = sol_sub
    R_sub = len(U_sub[0])
    
    # U_full: (n*m, 2*R_sub)
    # V_full: (m*p, 2*R_sub)
    # W_full: (p*n, 2*R_sub)
    
    # For 2x3x6, p=6. We split k into 0..2 and 3..5.
    # U is the same for both parts?
    # In the tensor decomposition T = sum u_r v_r w_r.
    # T1 is non-zero for k=0..2. T2 is non-zero for k=3..5.
    # U1, V1, W1 are for T1.
    # U2, V2, W2 are for T2.
    # We can just reuse the same U_sub, V_sub for both, but shifted W.
    # Wait, V_sub is for B (3x3).
    # For the first half, B is B[0:3, 0:3].
    # For the second half, B is B[0:3, 3:6].
    # So V1 should correspond to B1 and V2 to B2.
    # But the decomposition is for the tensor.
    # The tensor T1 has support on B indices j, k where k < 3.
    # The tensor T2 has support on B indices j, k where k >= 3.
    # So V1 is non-zero only for rows j*p+k where k < 3.
    # V2 is non-zero only for rows j*p+k where k >= 3.
    
    # If we use the same U_sub for both, that's fine.
    # We need to construct V_full and W_full carefully.
    
    U_full = []
    for i in range(n * m):
        row = U_sub[i] + U_sub[i] # Duplicate U_sub
        U_full.append(row)
        
    V_full = []
    # V_sub is 9 x R_sub. Rows correspond to j, k in 0..2.
    # V_full is 18 x 2*R_sub.
    # Rows 0..8 correspond to k=0..2 (first block).
    # Rows 9..17 correspond to k=3..5 (second block).
    
    for r in range(R_sub):
        # First block V (k=0..2)
        # V_full[j, 0:3, r] = V_sub[j, r]
        # V_full[j, 3:6, r] = 0
        # Second block V (k=0..2)
        # V_full[j, 0:3, R_sub+r] = 0
        # V_full[j, 3:6, R_sub+r] = V_sub[j, r]
        
        pass # We will build row by row
    
    V_full = [[0] * (2 * R_sub) for _ in range(m * p)]
    for j in range(m):
        for k in range(p_sub): # 0..2
            idx1 = j * p_sub + k
            idx2 = j * p + k
            V_full[idx2] = V_sub[idx1] + [0] * R_sub
            idx3 = j * p + (k + p_sub)
            V_full[idx3] = [0] * R_sub + V_sub[idx1]
            
    W_full = [[0] * (2 * R_sub) for _ in range(p * n)]
    # W_sub is 6 x R_sub. Rows correspond to k, i in 0..2, 0..1.
    # W_full rows 0..5 (k=0..2) get W_sub.
    # W_full rows 6..11 (k=3..5) get W_sub shifted.
    
    for k in range(p_sub):
        for i in range(n):
            idx1 = k * n + i
            idx2 = k * n + i
            W_full[idx2] = W_sub[idx1] + [0] * R_sub
            idx3 = (k + p_sub) * n + i
            W_full[idx3] = [0] * R_sub + W_sub[idx1]
            
    return U_full, V_full, W_full

def _verify_fast(n, m, p, U, V, W, num_tests=1):
    """
    Fast verification using numpy arrays.
    U, V, W are numpy arrays.
    """
    for _ in range(num_tests):
        A = np.random.randint(-2, 3, size=(n, m))
        B = np.random.randint(-2, 3, size=(m, p))
        
        C_direct = A @ B
        
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        LA = np.einsum('ij,ijr->ir', A, U_reshaped)
        LB = np.einsum('jk,jkr->kr', B, V_reshaped)
        
        C_decomp = np.einsum('kir,ir,kr->ik', W_reshaped, LA, LB)
        
        if not np.allclose(C_direct, C_decomp, atol=1e-9):
            return False
    return True

def _verify(n, m, p, U, V, W, num_tests=10):
    """
    Verifies if the decomposition U, V, W correctly computes A * B.
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    """
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    return _verify_fast(n, m, p, U_np, V_np, W_np, num_tests=num_tests)
# EVOLVE-BLOCK-END
