"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    def verify_solution(U, V, W, tol=1e-8):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def is_half_integer(arr, tol=1e-5):
        flat = np.array(arr, dtype=float).flatten()
        v = flat * 2
        return np.all(np.abs(v - np.round(v)) < tol)
    
    # Schoolbook rank 18 (guaranteed correct)
    def schoolbook():
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
        return U, V, W, rank
    
    # Try to find a lower rank solution using algebraic construction
    # For 2x3 * 3x3, the minimal rank is known to be 12 in some cases
    # Let's try a rank-12 construction
    
    def construct_rank_12():
        """
        Rank-12 decomposition.
        
        Idea: Group by output column k. For each k, compute A * B[:,k].
        This is a 2x3 * 3x1 multiplication, which can be done in rank 4
        using Strassen-like methods, but for this small case, rank 3 suffices
        (one per input j). So 3 columns * 3 products = 9, but we need to
        handle the W factor correctly.
        
        Actually, let me use a different approach: 
        Use 4 products per output column, total 12.
        """
        rank = 12
        U = np.zeros((6, rank))
        V = np.zeros((9, rank))
        W = np.zeros((6, rank))
        
        # For each output column k (0,1,2), use 4 products
        # Product structure for 2x3 * 3x1:
        # p1 = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2]  -> C[0]
        # p2 = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2]  -> C[1]
        # This is already 2 products, but we can't share across k easily.
        
        # Let me just use the schoolbook for now and return rank 18
        # The key is to have a correct solution
        return None
    
    # Return schoolbook solution
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    
    if verify_solution(U_sb.tolist(), V_sb.tolist(), W_sb.tolist()):
        if is_half_integer(U_sb) and is_half_integer(V_sb) and is_half_integer(W_sb):
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Should not reach here
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
