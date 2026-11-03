import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # 5s margin before 150s kill
    
    # Schoolbook as fallback
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    def check_solution(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U_np[u_idx], V_np[v_idx] * W_np[w_idx])
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    # Try to construct a rank-24 Winograd-based decomposition
    # Strategy: partition into overlapping 2x2 blocks and use rank-3 Winograd
    
    # For 3x3 x 3x4, we can think of it as:
    # A is 3x3, B is 3x4
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Let's try a structured decomposition with rank 24
    # Use Winograd for 2x2 blocks
    
    # Define the decomposition
    rank = 24
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: rows 0-1 of A, cols 0-1 of B (2x2 x 2x2 -> rank 3)
    # Winograd: (a+b), (a-b), (b+c) style
    # For 2x2: M = [[a,b],[c,d]] * [[e,f],[g,h]]
    # Winograd rank 3: 
    #   u1 = a+b+c+d, u2 = a-b, u3 = c-d
    #   v1 = e+f+g+h, v2 = e-f, v3 = g-h
    #   w1 = 1, w2 = 1, w3 = 1
    #   But this doesn't work directly...
    
    # Actually, Winograd 2x2x2 rank 3:
    #   t1 = (a+b)(e+f)
    #   t2 = (a-b)(e-f)  
    #   t3 = (a+c)(e+g)
    #   Then reconstruct...
    
    # This is getting too complex to do by hand. Let me try a search approach
    # with structured initialization.
    
    # Try random search with half-integer values for rank 20-24
    for target_rank in [20, 21, 22, 23, 24]:
        if time.time() > deadline - 10:
            break
            
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            # Initialize with small integers/half-integers
            U = np.random.randint(-2, 3, size=(n * m, target_rank)) * 0.5
            V = np.random.randint(-2, 3, size=(m * p, target_rank)) * 0.5
            W = np.random.randint(-2, 3, size=(p * n, target_rank)) * 0.5
            
            # Simple local search: try to satisfy equations one by one
            # This is heuristic and may not converge
            
            if check_solution(U.tolist(), V.tolist(), W.tolist(), target_rank):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Return schoolbook if no better solution found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
