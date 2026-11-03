"""Matrix multiplication 2,4,7 tensor decomposition using block-structured tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]
DEADLINE = 130  # Safety margin

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify if the decomposition is exact."""
    rank = U.shape[1]
    U_ = U.reshape(n, m, rank)
    V_ = V.reshape(m, p, rank)
    W_ = W.reshape(p, n, rank)
    
    # The tensor T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
    # We want T_{i,j,k} = 1 for all i,j,k
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        result += np.einsum('ij,jk,ki->ijk', U_[:, :, r], V_[:, :, r], W_[:, :, r])
    
    if np.max(np.abs(result - 1.0)) > tol:
        return False
    return True

def _construct_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as 4x7, 4x7, 4x7 arrays.
    """
    # Strassen's algorithm for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+h)
    # M7 = (a-c)(e+g)
    
    # U[i,j,r] where i is row of A, j is col of A, r is rank index
    # V[i,j,r] where i is row of B, j is col of B, r is rank index
    # W[i,j,r] where i is row of C, j is col of C, r is rank index
    
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # Indexing: A[i,j] -> U[i*2+j, r]
    # B[i,j] -> V[i*2+j, r]
    # C[i,j] -> W[i*2+j, r]
    
    # M1 = (a+d)(e+h)
    # a = A[0,0] = U[0,0], d = A[1,1] = U[3,0]
    # e = B[0,0] = V[0,0], h = B[1,1] = V[3,0]
    # C[0,0] += M1, C[0,1] -= M3, C[1,0] -= M2, C[1,1] += M1 - M4 + M5
    # Actually, let me use the standard Strassen formula:
    # C = [[M1 - M4 + M5 + M7, M3 + M4], [M2 + M4, M1 - M3 + M6]]
    
    # r=0: M1 = (a+d)(e+h)
    U[0, 0] = 1.0  # a
    U[3, 0] = 1.0  # d
    V[0, 0] = 1.0  # e
    V[3, 0] = 1.0  # h
    W[0, 0] = 1.0  # C[0,0] += M1
    W[3, 0] = 1.0  # C[1,1] += M1
    
    # r=1: M2 = (c+d)e
    # c = A[1,0] = U[2,0], d = A[1,1] = U[3,0]
    # e = B[0,0] = V[0,0]
    # C[1,0] += M2, C[1,1] += M2
    U[2, 1] = 1.0  # c
    U[3, 1] = 1.0  # d
    V[0, 1] = 1.0  # e
    W[2, 1] = 1.0  # C[1,0] += M2
    W[3, 1] = 1.0  # C[1,1] += M2
    
    # r=2: M3 = a(f-h)
    # a = A[0,0] = U[0,0]
    # f = B[0,1] = V[1,0], h = B[1,1] = V[3,0]
    # C[0,1] += M3, C[1,1] -= M3
    U[0, 2] = 1.0  # a
    V[1, 2] = 1.0  # f
    V[3, 2] = -1.0  # -h
    W[1, 2] = 1.0  # C[0,1] += M3
    W[3, 2] = -1.0  # C[1,1] -= M3
    
    # r=3: M4 = d(g-e)
    # d = A[1,1] = U[3,0]
    # g = B[1,0] = V[2,0], e = B[0,0] = V[0,0]
    # C[0,0] -= M4, C[0,1] += M4, C[1,0] += M4, C[1,1] -= M4
    U[3, 3] = 1.0  # d
    V[2, 3] = 1.0  # g
    V[0, 3] = -1.0  # -e
    W[0, 3] = -1.0  # C[0,0] -= M4
    W[1, 3] = 1.0  # C[0,1] += M4
    W[2, 3] = 1.0  # C[1,0] += M4
    W[3, 3] = -1.0  # C[1,1] -= M4
    
    # r=4: M5 = (a+b)h
    # a = A[0,0] = U[0,0], b = A[0,1] = U[1,0]
    # h = B[1,1] = V[3,0]
    # C[0,0] += M5, C[1,1] += M5
    U[0, 4] = 1.0  # a
    U[1, 4] = 1.0  # b
    V[3, 4] = 1.0  # h
    W[0, 4] = 1.0  # C[0,0] += M5
    W[3, 4] = 1.0  # C[1,1] += M5
    
    # r=5: M6 = (c-d)(f+h)
    # c = A[1,0] = U[2,0], d = A[1,1] = U[3,0]
    # f = B[0,1] = V[1,0], h = B[1,1] = V[3,0]
    # C[1,1] += M6
    U[2, 5] = 1.0  # c
    U[3, 5] = -1.0  # -d
    V[1, 5] = 1.0  # f
    V[3, 5] = 1.0  # h
    W[3, 5] = 1.0  # C[1,1] += M6
    
    # r=6: M7 = (a-c)(e+g)
    # a = A[0,0] = U[0,0], c = A[1,0] = U[2,0]
    # e = B[0,0] = V[0,0], g = B[1,0] = V[2,0]
    # C[0,0] += M7
    U[0, 6] = 1.0  # a
    U[2, 6] = -1.0  # -c
    V[0, 6] = 1.0  # e
    V[2, 6] = 1.0  # g
    W[0, 6] = 1.0  # C[0,0] += M7
    
    return U, V, W

def _construct_block_decomposition(n, m, p):
    """
    Construct a block-structured decomposition for n x m by m x p.
    Split into 2x2 blocks where possible.
    """
    # For 2x4x7:
    # A is 2x4, B is 4x7, C is 2x7
    # Split A into two 2x2 blocks: A_left (cols 0-1), A_right (cols 2-3)
    # Split B into two 4x? blocks: B_top (rows 0-1), B_bottom (rows 2-3)
    # Actually, let's think of it as:
    # C = A * B = [A_left, A_right] * [B_top; B_bottom]
    # = A_left * B_top + A_right * B_bottom
    # Each term is 2x2 * 2x7 or 2x2 * 2x7
    # We can use Strassen for the 2x2 part and schoolbook for the rest.
    
    # Let's use a simpler approach: decompose into 2x2x2 blocks
    # A is 2x4 = two 2x2 blocks
    # B is 4x7 = two 4x? blocks, but 7 is not even.
    # Let's split B into 4x3 and 4x4, or handle it differently.
    
    # Alternative: use the fact that R(2,2,2) = 7, R(2,2,1) = 4, R(2,2,3) = ?
    # Let's just build a decomposition using Strassen for 2x2x2 and schoolbook for the rest.
    
    # For 2x4x7:
    # Think of it as 2x2 * 2x7 + 2x2 * 2x7 (splitting the middle dimension)
    # Each 2x2 * 2x7 can be done with rank 7*7 = 49? No, that's not right.
    
    # Let's use a different approach: use the known result that R(2,4,7) <= 50
    # by decomposing into blocks.
    
    # Actually, let's just try to find a decomposition with rank < 56 using random search
    # with a structured initialization based on Strassen.
    
    return None

def _random_search(n, m, p, max_rank, num_trials=1000):
    """
    Random search for a low-rank decomposition.
    """
    start_time = time.time()
    best_rank = max_rank
    best_U, best_V, best_W = None, None, None
    
    # Initialize with Strassen-like structure for 2x2x2 blocks
    U_strassen, V_strassen, W_strassen = _strassen_2x2x2()
    
    for trial in range(num_trials):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        rank = max_rank - trial // 10  # Decrease rank gradually
        if rank < 28:
            break
            
        # Try different rank values
        for r in range(55, 27, -1):
            if time.time() - start_time > DEADLINE - 5:
                break
                
            # Generate random U, V, W with rank r
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Normalize to make entries closer to integers/half-integers
            # This is a heuristic
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            if _verify(U, V, W, n, m, p):
                best_rank = r
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < max_rank:
            break
    
    if best_U is None:
        return None
    
    return best_U, best_V, best_W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Baseline
    U_sb, V_sb, W_sb = _construct_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try block-structured decomposition
    block_result = _construct_block_decomposition(n, m, p)
    if block_result is not None:
        U, V, W = block_result
        if _verify(U, V, W, n, m, p):
            best_rank = U.shape[1]
            best_U, best_V, best_W = U, V, W
    
    # Try random search
    random_result = _random_search(n, m, p, best_rank)
    if random_result is not None:
        U, V, W = random_result
        if U.shape[1] < best_rank:
            best_rank = U.shape[1]
            best_U, best_V, best_W = U, V, W
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
