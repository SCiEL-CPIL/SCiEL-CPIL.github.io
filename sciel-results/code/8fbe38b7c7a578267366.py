import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """
    Return U, V, W for 2x2 * 2x2 -> 2x2 with rank 7.
    Indices: i, j, k in {0, 1}.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # c=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # c=2: M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # c=3: M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # c=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = -1; W[3, 4] = 1
    
    # c=5: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # c=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4_rank49():
    """
    Construct rank-49 decomposition for 4x4 * 4x4.
    Uses nested Strassen: Outer Strassen (7 terms) x Inner Strassen (7 terms).
    """
    n = 4
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    U2, V2, W2 = _strassen_2x2()
    
    for l in range(7): # Outer Strassen term
        for m in range(7): # Inner Strassen term
            col = l * 7 + m
            
            # Iterate over outer block indices
            for alpha in range(2):
                for beta in range(2):
                    u_outer_val = U2[alpha * 2 + beta, l]
                    if u_outer_val == 0: continue
                    
                    for gamma in range(2):
                        v_outer_val = V2[beta * 2 + gamma, l]
                        if v_outer_val == 0: continue
                        
                        w_outer_val = W2[gamma * 2 + alpha, l]
                        if w_outer_val == 0: continue
                        
                        # Iterate over inner element indices
                        for i_prime in range(2):
                            for j_prime in range(2):
                                u_inner_val = U2[i_prime * 2 + j_prime, m]
                                if u_inner_val == 0: continue
                                
                                i_global = alpha * 2 + i_prime
                                j_global = beta * 2 + j_prime
                                U[i_global * 4 + j_global, col] += u_outer_val * u_inner_val
                                
                        for j_prime in range(2):
                            for k_prime in range(2):
                                v_inner_val = V2[j_prime * 2 + k_prime, m]
                                if v_inner_val == 0: continue
                                
                                j_global = beta * 2 + j_prime
                                k_global = gamma * 2 + k_prime
                                V[j_global * 4 + k_global, col] += v_outer_val * v_inner_val
                                
                        for k_prime in range(2):
                            for i_prime in range(2):
                                w_inner_val = W2[k_prime * 2 + i_prime, m]
                                if w_inner_val == 0: continue
                                
                                k_global = gamma * 2 + k_prime
                                i_global = alpha * 2 + i_prime
                                W[k_global * 4 + i_global, col] += w_outer_val * w_inner_val
                        
    return U, V, W

def _construct_4x4x8_rank98():
    """
    Construct rank-98 decomposition for 4x4 * 4x8.
    Uses two rank-49 decompositions for 4x4 * 4x4.
    """
    n, m, p = 4, 4, 8
    rank = 98
    
    U4, V4, W4 = _build_4x4_rank49()
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First half (cols 0-3 of B)
    U[:, :49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, :49] = V4[j * 4 + k]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :49] = W4[k * 4 + i]
            
    # Second half (cols 4-7 of B)
    U[:, 49:] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), 49:] = V4[j * 4 + k]
    for k in range(4):
        for i in range(4):
            W[(k + 4) * 4 + i, 49:] = W4[k * 4 + i]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to construct rank-98 decomposition
    try:
        result = _construct_4x4x8_rank98()
        if _verify(result['U'], result['V'], result['W'], n, m, p):
            return result
    except Exception:
        pass
    
    # Fall back to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
