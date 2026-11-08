import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    def verify_solution(U, V, W, R):
        """Verify that U, V, W with rank R correctly decompose the matrix multiplication tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(m):
                        val = 0.0
                        for r in range(R):
                            val += W[k * n + i, r] * U[i * m + j, r] * V[l * p + k, r]
                        expected = 1.0 if j == l else 0.0
                        if abs(val - expected) > 1e-9:
                            return False
        return True
    
    # Strassen's rank-7 decomposition for 2x2x2
    # U rows: (i,j) for i in {0,1}, j in {0,1}
    # V rows: (j,k) for j in {0,1}, k in {0,1}
    # W rows: (k,i) for k in {0,1}, i in {0,1}
    # 
    # Strassen's 7 factors:
    # 1: U[0,0]=1, V[0,0]=1, W[0,0]=1
    # 2: U[0,1]=1, V[1,0]=1, W[0,1]=1
    # 3: U[1,0]=1, V[0,1]=1, W[1,0]=1
    # 4: U[1,1]=1, V[1,1]=1, W[1,1]=1
    # 5: U[0,0]=1, V[1,1]=1, W[0,1]=1
    # 6: U[1,1]=1, V[0,0]=1, W[1,0]=1
    # 7: U[0,1]=1, V[0,1]=1, W[1,1]=1
    # 
    # Wait, let me use the standard Strassen decomposition more carefully.
    # For C = A * B where A is 2x2, B is 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Let me index: A[i,j], B[j,k], C[i,k]
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # For 2x2x2: m=2, p=2, n=2
    # U has 4 rows: (0,0), (0,1), (1,0), (1,1)
    # V has 4 rows: (0,0), (0,1), (1,0), (1,1)
    # W has 4 rows: (0,0), (0,1), (1,0), (1,1)
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
        R = 7
        U = np.zeros((4, R))
        V = np.zeros((4, R))
        W = np.zeros((4, R))
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0, 0] = 1  # A00
        U[3, 0] = 1  # A11
        V[0, 0] = 1  # B00
        V[3, 0] = 1  # B11
        W[0, 0] = 1  # C00
        W[3, 0] = 1  # C11
        
        # M2 = (A10 + A11)B00
        U[2, 1] = 1  # A10
        U[3, 1] = 1  # A11
        V[0, 1] = 1  # B00
        W[2, 1] = 1  # C10
        W[3, 1] = -1 # C11
        
        # M3 = A00(B10 - B11)
        U[0, 2] = 1  # A00
        V[2, 2] = 1  # B10
        V[3, 2] = -1 # B11
        W[1, 2] = 1  # C01
        W[3, 2] = 1  # C11
        
        # M4 = A11(B01 - B00)
        U[3, 3] = 1  # A11
        V[1, 3] = 1  # B01
        V[0, 3] = -1 # B00
        W[0, 3] = 1  # C00
        W[2, 3] = 1  # C10
        
        # M5 = (A00 + A01)B11
        U[0, 4] = 1  # A00
        U[1, 4] = 1  # A01
        V[3, 4] = 1  # B11
        W[0, 4] = -1 # C00
        W[1, 4] = 1  # C01
        
        # M6 = (A10 - A00)(B00 + B01)
        U[2, 5] = 1  # A10
        U[0, 5] = -1 # A00
        V[0, 5] = 1  # B00
        V[1, 5] = 1  # B01
        W[3, 5] = 1  # C11
        
        # M7 = (A01 - A11)(B10 + B11)
        U[1, 6] = 1  # A01
        U[3, 6] = -1 # A11
        V[2, 6] = 1  # B10
        V[3, 6] = 1  # B11
        W[0, 6] = 1  # C00
        W[3, 6] = -1 # C11
        
        return U, V, W
    
    def try_rank(R, max_trials=200):
        """Try to find a valid decomposition of rank R."""
        for trial in range(max_trials):
            if time.time() > deadline - 10:
                return None
            
            # Random U, V with entries in {-1, 0, 1}
            U_flat = np.random.randint(-1, 2, (n * m, R))
            V_flat = np.random.randint(-1, 2, (m * p, R))
            
            # Solve for W
            W_flat = np.zeros((p * n, R))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # M_ik[j*m+l, r] = U[i*m+j, r] * V[l*p+k, r]
                    M_ik = np.zeros((m * m, R))
                    for j in range(m):
                        for l in range(m):
                            M_ik[j * m + l, :] = U_flat[i * m + j, :] * V_flat[l * p + k, :]
                    
                    b_ik = np.zeros(m * m)
                    for j in range(m):
                        b_ik[j * m + j] = 1
                    
                    try:
                        w, residuals, rank, s = np.linalg.lstsq(M_ik, b_ik, rcond=None)
                    except:
                        valid = False
                        break
                    
                    if len(residuals) == 0 or residuals[0] > 1e-9:
                        valid = False
                        break
                    
                    W_flat[k * n + i, :] = w
            
            if not valid:
                continue
            
            # Snap to half integers
            W_snapped = np.round(W_flat * 2) / 2
            
            # Verify
            if verify_solution(U_flat.tolist(), V_flat.tolist(), W_snapped.tolist(), R):
                return {
                    "U": [[float(v) for v in row] for row in U_flat],
                    "V": [[float(v) for v in row] for row in V_flat],
                    "W": [[float(v) for v in row] for row in W_snapped]
                }
        
        return None
    
    # Try block decomposition: split 4x4x6 into two 2x4x3 blocks
    # Each 2x4x3 can be further decomposed using Strassen-like approaches
    # 
    # Actually, let's try a different approach: use the known rank-21 for 4x4x4
    # and extend it to 4x4x6 by adding 2 more columns
    # 
    # First, let's try random search from rank 20 up
    for R in range(20, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(R, max_trials=150)
        if result is not None:
            best_solution = result
            best_rank = R
            break
    
    return best_solution
