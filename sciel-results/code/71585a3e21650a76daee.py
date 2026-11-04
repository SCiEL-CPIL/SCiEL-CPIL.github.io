import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Build the target tensor's non-zero structure
    # T[I, J, K] = 1 where:
    # I = i*m + j  (i in [0,n), j in [0,m))
    # J = j*p + k  (j in [0,m), k in [0,p))
    # K = k*n + i  (k in [0,p), i in [0,n))
    
    I_to_JK = [[] for _ in range(n*m)]
    J_to_IK = [[] for _ in range(m*p)]
    K_to_IJ = [[] for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                I_to_JK[I].append((J, K))
                J_to_IK[J].append((I, K))
                K_to_IJ[K].append((I, J))
    
    # Helper: verify a candidate decomposition (U, V, W as integer arrays)
    def verify(U, V, W):
        R = U.shape[1]
        # Check all non-zero entries
        for I in range(n*m):
            for J, K in I_to_JK[I]:
                val = int(np.sum(U[I, :] * V[J, :] * W[K, :]))
                if val != 1:
                    return False
        # Check a sample of zero entries
        rng = np.random.RandomState(42)
        for _ in range(200):
            I = rng.randint(0, n*m)
            J = rng.randint(0, m*p)
            K = rng.randint(0, p*n)
            is_nonzero = any((J, K) == jk for jk in I_to_JK[I])
            if not is_nonzero:
                val = int(np.sum(U[I, :] * V[J, :] * W[K, :]))
                if val != 0:
                    return False
        return True
    
    # Helper: build schoolbook decomposition (rank = n*m*p)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    I = i * m + j
                    J = j * p + k
                    K = k * n + i
                    U[I, col] = 1
                    V[J, col] = 1
                    W[K, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Helper: build a block decomposition using Strassen for 2x2x2 sub-blocks
    # This is a structured approach that should give better rank than schoolbook.
    # Strategy: decompose the 4x5x5 multiplication into blocks.
    # A is 4x5, B is 5x5, C is 4x5.
    # Split A into [A1 | A2] where A1 is 4x2, A2 is 4x3
    # Split B into [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    # C = A1*B1 + A2*B2 where B1 = [B11; B21] (5x2), B2 = [B12; B22] (5x3)
    # Actually, let's think of it differently.
    # C = A * B
    # Split columns of A and rows of B:
    # A = [A_left | A_right] where A_left is 4x2, A_right is 4x3
    # B = [B_top; B_bottom] where B_top is 2x5, B_bottom is 3x5
    # C = A_left * B_top + A_right * B_bottom
    # A_left * B_top: 4x2 * 2x5 -> use Strassen-like or just schoolbook (rank 4*2*5=40? No, rank is 2*4*5=40)
    # Actually for matrix multiplication n x m * m x p, schoolbook rank is n*m*p.
    # A_left * B_top: 4x2 * 2x5, schoolbook rank = 4*2*5 = 40
    # A_right * B_bottom: 4x3 * 3x5, schoolbook rank = 4*3*5 = 60
    # Total = 100, same as schoolbook.
    
    # Let's try a different split: use Strassen on the 2x2 block part.
    # Split A into 2x2 blocks of size 2x2 and 2x3:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 2x2, A22 is 2x3
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 2x2, B22 is 2x3
    # C = [[C11, C12], [C21, C22]] where C11 is 2x2, C12 is 2x3, C21 is 2x2, C22 is 2x3
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For C11: 2x2 * 2x2 + 2x3 * 2x2... this is getting complicated.
    
    # Let me try a simpler approach: use the known Strassen decomposition for 2x2x2
    # and combine it with schoolbook for the remaining parts.
    
    # Actually, let me just use the ALS approach but with better initialization and more careful handling.
    # The key insight is that we should use a structured initialization based on known low-rank decompositions.
    
    # Let's try a different approach: use the fact that for small matrices, we can enumerate
    # or use a more systematic search.
    
    # For now, let's use the ALS approach with better parameters.
    
    start_time = time.time()
    deadline = start_time + 110
    
    best_rank = n * m * p
    best_U, best_V, best_W = schoolbook()
    
    # Try ranks from a reasonable lower bound upwards
    # For 4x5x5, the rank is at least max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # But realistically it's much higher. Let's try from 50 to 100.
    
    for R in range(50, 100):
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(123 + R * 100 + restart)
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # ALS iterations
            for it in range(300):
                if time.time() > deadline:
                    break
                
                # Update U
                for I in range(n*m):
                    A_mat = np.zeros((R, R))
                    b_vec = np.zeros(R)
                    for J, K in I_to_JK[I]:
                        vw = V[J, :] * W[K, :]
                        A_mat += np.outer(vw, vw)
                        b_vec += vw
                    A_mat += 1e-8 * np.eye(R)
                    try:
                        U[I, :] = np.linalg.solve(A_mat, b_vec)
                    except:
                        pass
                
                # Update V
                for J in range(m*p):
                    A_mat = np.zeros((R, R))
                    b_vec = np.zeros(R)
                    for I, K in J_to_IK[J]:
                        uw = U[I, :] * W[K, :]
                        A_mat += np.outer(uw, uw)
                        b_vec += uw
                    A_mat += 1e-8 * np.eye(R)
                    try:
                        V[J, :] = np.linalg.solve(A_mat, b_vec)
                    except:
                        pass
                
                # Update W
                for K in range(p*n):
                    A_mat = np.zeros((R, R))
                    b_vec = np.zeros(R)
                    for I, J in K_to_IJ[K]:
                        uv = U[I, :] * V[J, :]
                        A_mat += np.outer(uv, uv)
                        b_vec += uv
                    A_mat += 1e-8 * np.eye(R)
                    try:
                        W[K, :] = np.linalg.solve(A_mat, b_vec)
                    except:
                        pass
                
                # Check if we've converged to an integer solution
                if it % 20 == 19:
                    U_round = np.round(U).astype(int)
                    V_round = np.round(V).astype(int)
                    W_round = np.round(W).astype(int)
                    
                    # Quick check: max deviation from integer
                    max_dev_U = np.max(np.abs(U - U_round))
                    max_dev_V = np.max(np.abs(V - V_round))
                    max_dev_W = np.max(np.abs(W - W_round))
                    
                    if max_dev_U < 0.1 and max_dev_V < 0.1 and max_dev_W < 0.1:
                        if verify(U_round, V_round, W_round):
                            best_rank = R
                            best_U = U_round
                            best_V = V_round
                            best_W = W_round
                            found = True
                            break
            
            if found:
                break
        
        if found:
            break
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
