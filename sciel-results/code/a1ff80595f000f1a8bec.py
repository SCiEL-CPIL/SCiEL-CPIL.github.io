"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    C = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
    return bool(np.allclose(C, 1.0, atol=tol))

def _schoolbook_decomposition(n, m, p):
    """Returns the schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Return Strassen's rank-7 decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    products = [
        (np.array([[1,0],[0,1]], dtype=np.float64), np.array([[1,1],[0,0]], dtype=np.float64), np.array([[1,0],[0,-1]], dtype=np.float64)),
        (np.array([[0,0],[1,1]], dtype=np.float64), np.array([[1,0],[0,0]], dtype=np.float64), np.array([[0,0],[0,1]], dtype=np.float64)),
        (np.array([[1,0],[0,0]], dtype=np.float64), np.array([[0,0],[1,-1]], dtype=np.float64), np.array([[0,1],[0,0]], dtype=np.float64)),
        (np.array([[0,0],[0,1]], dtype=np.float64), np.array([[0,1],[-1,0]], dtype=np.float64), np.array([[0,0],[1,0]], dtype=np.float64)),
        (np.array([[1,1],[0,0]], dtype=np.float64), np.array([[0,0],[0,1]], dtype=np.float64), np.array([[-1,0],[0,1]], dtype=np.float64)),
        (np.array([[-1,0],[1,0]], dtype=np.float64), np.array([[1,1],[0,0]], dtype=np.float64), np.array([[0,0],[0,1]], dtype=np.float64)),
        (np.array([[0,1],[0,-1]], dtype=np.float64), np.array([[0,0],[1,1]], dtype=np.float64), np.array([[1,0],[0,1]], dtype=np.float64)),
    ]
    
    for r, (Ur, Vr, Wr) in enumerate(products):
        U[:, r] = Ur.flatten()
        V[:, r] = Vr.flatten()
        W[:, r] = Wr.flatten()
    
    return U, V, W

def _combine_blocks_mid(U1, V1, W1, U2, V2, W2, n, m1, m2, p):
    """
    Combine two block decompositions split on the middle dimension:
    Block 1: n x m1 x p, Block 2: n x m2 x p
    Total: n x (m1+m2) x p
    """
    r1 = U1.shape[1]
    r2 = U2.shape[1]
    total_rank = r1 + r2
    m_total = m1 + m2
    
    U = np.zeros((n * m_total, total_rank), dtype=np.float64)
    V = np.zeros((m_total * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    for i in range(n):
        for j in range(m1):
            U[i * m_total + j, :r1] = U1[i * m1 + j, :]
        for j in range(m2):
            U[i * m_total + m1 + j, r1:] = U2[i * m2 + j, :]
    
    for j in range(m1):
        for k in range(p):
            V[j * p + k, :r1] = V1[j * p + k, :]
    for j in range(m2):
        for k in range(p):
            V[(m1 + j) * p + k, r1:] = V2[j * p + k, :]
    
    for k in range(p):
        for i in range(n):
            W[k * n + i, :r1] = W1[k * n + i, :]
            W[k * n + i, r1:] = W2[k * n + i, :]
    
    return U, V, W

def _build_subblock(n, m, p):
    """Build best known decomposition for sub-block n x m x p."""
    # For 2x2x2, use Strassen (rank 7 vs schoolbook 8)
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    # Otherwise use schoolbook
    return _schoolbook_decomposition(n, m, p)

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for matrix multiplication.
    Dimensions: n=3, m=5, p=6.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    # Try block decomposition: split middle dimension 5 into 2+3
    if time.time() < deadline - 30.0:
        U1, V1, W1 = _build_subblock(n, 2, p)  # 3x2x6
        U2, V2, W2 = _build_subblock(n, 3, p)  # 3x3x6
        U_block, V_block, W_block = _combine_blocks_mid(U1, V1, W1, U2, V2, W2, n, 2, 3, p)
        
        if _verify_decomposition(U_block, V_block, W_block, n, m, p):
            block_rank = U_block.shape[1]
            if block_rank < best_rank:
                best_rank = block_rank
                best_U, best_V, best_W = U_block.copy(), V_block.copy(), W_block.copy()
    
    # Try splitting middle dimension 5 into 1+4
    if time.time() < deadline - 20.0:
        U1, V1, W1 = _build_subblock(n, 1, p)  # 3x1x6
        U2, V2, W2 = _build_subblock(n, 4, p)  # 3x4x6
        U_block, V_block, W_block = _combine_blocks_mid(U1, V1, W1, U2, V2, W2, n, 1, 4, p)
        
        if _verify_decomposition(U_block, V_block, W_block, n, m, p):
            block_rank = U_block.shape[1]
            if block_rank < best_rank:
                best_rank = block_rank
                best_U, best_V, best_W = U_block.copy(), V_block.copy(), W_block.copy()
    
    # Try splitting both middle and output: 3x5x6 = (2x5x2 + 1x5x2) + (2x5x4 + 1x5x4)
    # This gets complex; skip for now and try simpler approaches
    
    # Try: split output p=6 into 2+4, and for each use Strassen on 2x2x2 sub-blocks
    if time.time() < deadline - 10.0:
        # For 3x5x2: can we do better than 30?
        # Split middle 5 into 2+3: 3x2x2 + 3x3x2
        U1a, V1a, W1a = _build_subblock(3, 2, 2)  # 3x2x2, schoolbook=12
        U2a, V2a, W2a = _build_subblock(3, 3, 2)  # 3x3x2, schoolbook=18
        U_a, V_a, W_a = _combine_blocks_mid(U1a, V1a, W1a, U2a, V2a, W2a, 3, 2, 3, 2)
        
        U1b, V1b, W1b = _build_subblock(3, 2, 4)  # 3x2x4, schoolbook=24
        U2b, V2b, W2b = _build_subblock(3, 3, 4)  # 3x3x4, schoolbook=36
        U_b, V_b, W_b = _combine_blocks_mid(U1b, V1b, W1b, U2b, V2b, W2b, 3, 2, 3, 4)
        
        # Now combine U_a (3x5x2) and U_b (3x5x4) on output dimension
        # W_a: (2*3, r_a), W_b: (4*3, r_b)
        r_a = U_a.shape[1]
        r_b = U_b.shape[1]
        total_r = r_a + r_b
        
        U_final = np.zeros((3*5, total_r), dtype=np.float64)
        V_final = np.zeros((5*6, total_r), dtype=np.float64)
        W_final = np.zeros((6*3, total_r), dtype=np.float64)
        
        U_final[:, :r_a] = U_a
        U_final[:, r_a:] = U_b
        
        V_final[:, :r_a] = V_a
        V_final[:, r_a:] = V_b
        
        # W: W_a has shape (2*3, r_a) for k in [0,2), W_b has shape (4*3, r_b) for k in [2,6)
        for k in range(2):
            for i in range(3):
                W_final[k*3+i, :r_a] = W_a[k*3+i, :]
        for k in range(4):
            for i in range(3):
                W_final[(2+k)*3+i, r_a:] = W_b[k*3+i, :]
        
        if _verify_decomposition(U_final, V_final, W_final, n, m, p):
            final_rank = U_final.shape[1]
            if final_rank < best_rank:
                best_rank = final_rank
                best_U, best_V, best_W = U_final.copy(), V_final.copy(), W_final.copy()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
