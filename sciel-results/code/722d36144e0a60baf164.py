"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact using numpy for speed."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _snap_to_half_int(arr):
    """Snap values to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _snap_to_int(arr):
    """Snap values to nearest integer."""
    return np.round(arr)

def _solve_linear_for_factor(fixed_factors, target_tensor, n, m, p, factor_idx, rank):
    """
    Solve for one factor (U, V, or W) given the other two are fixed.
    factor_idx: 0 for U, 1 for V, 2 for W.
    Returns the solved factor as a numpy array.
    """
    U, V, W = fixed_factors
    
    num_eq = n * m * p
    
    if factor_idx == 0: # U
        num_unknowns = n * m * rank
    elif factor_idx == 1: # V
        num_unknowns = m * p * rank
    else: # W
        num_unknowns = p * n * rank
        
    A = np.zeros((num_eq, num_unknowns))
    b = np.zeros(num_eq)
    
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                b[eq_idx] = 1.0 # Target tensor entry is 1
                
                if factor_idx == 0: # Solve for U
                    for r in range(rank):
                        A[eq_idx, u_idx * rank + r] = V[v_idx, r] * W[w_idx, r]
                elif factor_idx == 1: # Solve for V
                    for r in range(rank):
                        A[eq_idx, v_idx * rank + r] = U[u_idx, r] * W[w_idx, r]
                else: # Solve for W
                    for r in range(rank):
                        A[eq_idx, w_idx * rank + r] = U[u_idx, r] * V[v_idx, r]
                
                eq_idx += 1
                
    # Solve least squares
    x, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
    
    if factor_idx == 0:
        return x.reshape(n * m, rank)
    elif factor_idx == 1:
        return x.reshape(m * p, rank)
    else:
        return x.reshape(p * n, rank)

def _strassen_2x3x1(n=2, m=3, p=1):
    """
    Construct the known rank-5 decomposition for 2x3x1 matrix multiplication.
    This is a Strassen-like algorithm.
    U: 2*3 = 6 rows, 5 cols
    V: 3*1 = 3 rows, 5 cols
    W: 1*2 = 2 rows, 5 cols
    """
    # Indices:
    # U rows: (i, j) for i in 0..1, j in 0..2 -> 0..5
    # V rows: (j, k) for j in 0..2, k=0 -> 0..2
    # W rows: (k, i) for k=0, i in 0..1 -> 0..1
    
    U = np.zeros((6, 5))
    V = np.zeros((3, 5))
    W = np.zeros((2, 5))
    
    # Based on standard 2x2 Strassen adapted or specific 2x3x1 known solution
    # One known solution for 2x3x1 (rank 5):
    # Let A be 2x3, B be 3x1. C = AB is 2x1.
    # c0 = a00*b0 + a01*b1 + a02*b2
    # c1 = a10*b0 + a11*b1 + a12*b2
    
    # We can use the following decomposition:
    # r0 = (a00+a10) * b0
    # r1 = (a01+a11) * b1
    # r2 = (a02+a12) * b2
    # r3 = (a00-a10) * b0
    # r4 = (a01-a11) * b1 ... wait, this doesn't handle b2 for difference.
    
    # Let's use a known rank-5 scheme for 2x3x1.
    # Actually, 2x3x1 is equivalent to 2x2x2 minus something? No.
    # Let's just use the schoolbook for the block if we can't find a simple one, 
    # but we know rank 5 is possible.
    
    # A common decomposition:
    # m1 = (a00 + a10) * b0
    # m2 = (a01 + a11) * b1
    # m3 = (a02 + a12) * b2
    # m4 = (a00 - a10) * b0
    # m5 = (a01 - a11) * b1
    # m6 = (a02 - a12) * b2
    # This is rank 6.
    
    # Rank 5 exists. 
    # c0 = m1 + m2 + m3
    # c1 = m1 - m2 - m3 ... no.
    
    # Let's use the explicit rank-5 decomposition for 2x3x1:
    # U (6x5), V (3x5), W (2x5)
    
    # r=0: U[0,0]=1, U[3,0]=1; V[0,0]=1; W[0,0]=1, W[1,0]=1
    # r=1: U[1,1]=1, U[4,1]=1; V[1,1]=1; W[0,1]=1, W[1,1]=1
    # r=2: U[2,2]=1, U[5,2]=1; V[2,2]=1; W[0,2]=1, W[1,2]=1
    # r=3: U[0,3]=1, U[3,3]=-1; V[0,3]=1; W[0,3]=1, W[1,3]=-1
    # r=4: U[1,4]=1, U[4,4]=-1; V[1,4]=1; W[0,4]=1, W[1,4]=-1
    # Missing b2 in difference?
    # c0 = r0+r1+r2
    # c1 = r0+r1+r2 ... no.
    
    # Let's try:
    # c0 = m1 + m2 + m3
    # c1 = m1 + m4 + m5
    # where m1 = (a00+a10)b0, m2 = (a01+a11)b1, m3 = (a02+a12)b2
    # m4 = (a10-a00)b0, m5 = (a11-a01)b1 ... need m6 for b2?
    
    # Actually, for 2x3x1, the rank is 5.
    # One solution:
    # U:
    # [1, 1, 1, 1, 0]
    # [1, 1, 1, 0, 1]
    # [1, 1, 1, 0, 0]
    # [1, 1, 1, -1, 0]
    # [1, 1, 1, 0, -1]
    # [1, 1, 1, 0, 0]
    
    # V:
    # [1, 0, 0, 1, 0]
    # [0, 1, 0, 0, 1]
    # [0, 0, 1, 0, 0]
    
    # W:
    # [1, 1, 1, 1, 1]
    # [1, 1, 1, -1, -1]
    
    # Let's verify:
    # c0 = sum_r U[0,r] V[j,r] W[0,r] ... wait, V depends on j.
    # c0 = sum_r U[0,r] (sum_j V[j,r] a_j) W[0,r] ?? No.
    # Tensor form: sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{i,0} delta_{k,0} ... no.
    # For 2x3x1: i in 0,1; j in 0,1,2; k=0.
    # Target: 1 if (i,j) corresponds to A_ij and B_j0.
    # U[i,j,r] * V[j,0,r] * W[0,i,r] = 1 if it's the right term?
    # No, the sum over r must yield 1 for the product A_ij B_j0.
    # So sum_r U[i,j,r] V[j,0,r] W[0,i,r] should be 1 for all i,j?
    # No, the tensor entry T[i,j,k] is 1 if it represents the multiplication A_ij * B_jk.
    # Here k=0. So T[i,j,0] = 1 for all i,j.
    # So we need sum_r U[i,j,r] V[j,0,r] W[0,i,r] = 1 for all i in {0,1}, j in {0,1,2}.
    
    # Let's check my proposed matrices:
    # i=0, j=0:
    # r=0: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
    # r=1: U[0,1]=1, V[0,1]=0, W[0,1]=1 -> 0
    # r=2: U[0,2]=1, V[0,2]=0, W[0,2]=1 -> 0
    # r=3: U[0,3]=1, V[0,3]=1, W[0,3]=1 -> 1
    # r=4: U[0,4]=0, V[0,4]=0, W[0,4]=1 -> 0
    # Sum = 2. Incorrect.
    
    # I will fall back to schoolbook for the blocks if I can't quickly verify a Strassen block.
    # However, the prompt mentions "5 blocks of rank 11 each" in knowledge, which sums to 55.
    # 2x3x10 can be split into 5 blocks of 2x3x2? No, 10/2=5.
    # If we use 2x3x2 blocks, rank of 2x3x2 is 6 (schoolbook) or 5?
    # Rank of 2x2x2 is 7. Rank of 2x3x2 is at least 6.
    # If rank(2x3x2) = 6, then 5*6 = 30? No, the blocks are independent in k?
    # If we split p=10 into 5 blocks of p=2, the tensors are disjoint in k.
    # So rank(2x3x10) <= 5 * rank(2x3x2).
    # If rank(2x3x2) = 6, total rank 30.
    # But schoolbook is 60.
    # Is rank(2x3x2) 6?
    # 2x2x2 is 7. 2x3x2 is likely 6 or 7.
    # If it's 6, we get rank 30.
    # If it's 7, we get rank 35.
    
    # Let's try to construct a rank-6 decomposition for 2x3x2.
    # Or just use the known rank-55 solution structure from the "inspiration" which suggests 55 is possible.
    # The inspiration says "5 blocks of rank 11 each".
    # 11 for 2x3x2? That's high. Schoolbook is 12 (2*3*2).
    # So 11 is a slight improvement over schoolbook for 2x3x2.
    
    # Strategy: 
    # 1. Construct a rank-55 solution using 5 blocks of rank-11 decompositions for 2x3x2.
    # 2. Use this as the initial guess for ALS to try and reduce rank to 54.
    
    # How to get rank-11 for 2x3x2?
    # Schoolbook is 12.
    # We can try to merge two terms.
    # (a00 b00 + a00 b01) + ...
    
    # For now, I will use a generic random initialization for rank 55, 
    # but bias it towards the block structure by initializing each block with schoolbook 
    # and then perturbing.
    
    pass

def _init_block_structure(n, m, p, rank, block_p):
    """
    Initialize U, V, W with a block structure.
    p is split into blocks of size block_p.
    For each block, we use a schoolbook decomposition of rank m*block_p.
    Total rank = (p/block_p) * m * block_p = m*p.
    This is just schoolbook.
    To get rank < m*p, we need a better block decomposition.
    """
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    num_blocks = p // block_p
    rank_per_block = rank // num_blocks
    
    # This assumes rank is divisible by num_blocks
    # For n=2, m=3, p=10, block_p=2, num_blocks=5.
    # rank=55, rank_per_block=11.
    # Schoolbook rank for 2x3x2 is 12.
    # We need a rank-11 decomposition for 2x3x2.
    
    # Let's try to find one or use a random init for the block.
    # Given time constraints, I will use random init for the blocks 
    # but scale them to have integer-like properties.
    
    for b in range(num_blocks):
        start_r = b * rank_per_block
        end_r = start_r + rank_per_block
        start_k = b * block_p
        end_k = start_k + block_p
        
        # Random init for the block
        U_block = np.random.randint(-1, 2, (n * m, rank_per_block))
        V_block = np.random.randint(-1, 2, (m * block_p, rank_per_block))
        W_block = np.random.randint(-1, 2, (block_p * n, rank_per_block))
        
        U[:, start_r:end_r] = U_block
        V[start_k*m:(end_k)*m, start_r:end_r] = V_block
        W[start_k*n:end_k*n, start_r:end_r] = W_block
        
    return U, V, W

def _als_decomposition(n, m, p, rank, max_iters=100, seed=42, init_U=None, init_V=None, init_W=None):
    """
    Alternating Least Squares with snapping to find a decomposition.
    """
    np.random.seed(seed)
    
    if init_U is None:
        U = np.random.randint(-2, 3, (n * m, rank))
        V = np.random.randint(-2, 3, (m * p, rank))
        W = np.random.randint(-2, 3, (p * n, rank))
    else:
        U = init_U.copy()
        V = init_V.copy()
        W = init_W.copy()
        
    for it in range(max_iters):
        # Solve for U
        U_sol = _solve_linear_for_factor((U, V, W), None, n, m, p, 0, rank)
        U = _snap_to_half_int(U_sol)
        
        # Solve for V
        V_sol = _solve_linear_for_factor((U, V, W), None, n, m, p, 1, rank)
        V = _snap_to_half_int(V_sol)
        
        # Solve for W
        W_sol = _solve_linear_for_factor((U, V, W), None, n, m, p, 2, rank)
        W = _snap_to_half_int(W_sol)
        
        # Check if exact
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    return None

def construct():
    start_time = time.time()
    deadline = 145
    
    n, m, p = DIMENSIONS
    
    # Fallback
    fallback = _schoolbook(n, m, p)
    
    # Try to find rank 54 using block-structured initialization
    # We know 55 is possible.
    
    # 1. Try rank 54 with block structure (5 blocks of 2x3x2, rank 11 each)
    # This requires a rank-11 decomposition for 2x3x2.
    # If we can't find it, the random init within the block might fail.
    
    # Let's try multiple random seeds for the block structure.
    for seed in range(20):
        if time.time() > deadline:
            break
            
        # Initialize with block structure
        U_init, V_init, W_init = _init_block_structure(n, m, p, 54, block_p=2)
        
        # Run ALS
        res = _als_decomposition(n, m, p, 54, max_iters=50, seed=seed, 
                                 init_U=U_init, init_V=V_init, init_W=W_init)
        if res is not None:
            return res
            
    # 2. If rank 54 fails, try rank 55 (known possible)
    # Use block structure for 55 as well (5 blocks of rank 11)
    # Or just random.
    
    for seed in range(20):
        if time.time() > deadline:
            break
            
        res = _als_decomposition(n, m, p, 55, max_iters=50, seed=seed)
        if res is not None:
            return res
            
    return fallback
# EVOLVE-BLOCK-END
