import numpy as np
import random
import time

DIMENSIONS = [2, 3, 3]
GLOBAL_START_TIME = time.time()
TIME_LIMIT = 135.0  # Safety margin before 150s wall-clock kill

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def verify(U, V, W):
        R = len(U[0]) if isinstance(U[0], list) else U.shape[1]
        if isinstance(U, np.ndarray):
            U = U.tolist()
        if isinstance(V, np.ndarray):
            V = V.tolist()
        if isinstance(W, np.ndarray):
            W = W.tolist()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = 0.0
                    for r in range(R):
                        val += U[idx_u][r] * V[idx_v][r] * W[idx_w][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def rank_17_decomposition():
        """
        Construct a rank-17 decomposition for 2x3 * 3x3 matrix multiplication.
        Uses Strassen for 2x2x2 block and schoolbook for remaining terms.
        """
        rank = 17
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        # Decompose A (2x3) and B (3x3) into blocks:
        # A = [A11 A12] where A11 is 2x1, A12 is 2x2
        # B = [[B11, B12], [B21, B22]] where each is 2x2 or 1x2 or 2x1
        # Actually, let's use a different block structure.
        # A is 2x3, B is 3x3
        # Let's split B into columns: B = [B1 | B2 | B3] where B1,B2,B3 are 3x1
        # Or split B into rows: B = [B_row1; B_row2; B_row3] where each is 1x3
        
        # Better approach: use the known rank-17 construction.
        # The tensor T_2,3,3 has rank 17.
        # One construction: 
        #   Use 7 terms for a 2x2x2 Strassen on a subblock
        #   Use 10 schoolbook terms for the rest
        # Total: 7 + 10 = 17
        
        # Let me use a more systematic approach.
        # A[i,j] for i in {0,1}, j in {0,1,2}
        # B[j,k] for j in {0,1,2}, k in {0,1,2}
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # Schoolbook would use 6 terms (2*3).
        # Strassen on 2x2 blocks can save terms.
        
        # Let's use this construction:
        # Split A into A[:,0] (2x1) and A[:,1:3] (2x2)
        # Split B into B[0,:] (1x3) and B[1:3,:] (2x3)
        
        # C = A[:,0] @ B[0,:] + A[:,1:3] @ B[1:3,:]
        
        # First term: (2x1)(1x3) = 2x3, schoolbook needs 2*1*3 = 6 terms
        # Second term: (2x2)(2x3) = 2x3, can use Strassen-like for 2x2x2 part
        
        # Actually, let me just use a known rank-17 decomposition.
        # From literature, the rank of M_2,3,3 is 17.
        
        # I'll construct it using a combination approach:
        # For the 2x2x2 subproblem (using A[:,1:3] and B[1:3,:]), use 7 Strassen terms
        # For the remaining terms involving A[:,0] and B[0,:], use schoolbook
        
        # Strassen for 2x2 matrices:
        # P1 = (A11 + A22)(B11 + B22)
        # P2 = (A21 + A22)B11
        # P3 = A11(B21 - B22)
        # P4 = A22(B12 - B11)
        # P5 = (A11 + A12)B22
        # P6 = (A21 - A11)(B11 + B12)
        # P7 = (A12 - A22)(B21 + B22)
        # C11 = P1 + P4 - P5 - P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P7
        
        # But we need to handle the 2x2x3 case.
        
        # Let me use a simpler approach: just find a rank-17 solution numerically.
        
        return None
    
    def try_find_rank(R, time_budget=10.0):
        start_time = time.time()
        max_attempts = 3000
        for attempt in range(max_attempts):
            if time.time() - start_time > time_budget or time.time() - GLOBAL_START_TIME > TIME_LIMIT:
                break
            
            # Generate random U and V with controlled sparsity
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            
            # For each column r, pick a structured pattern
            for r in range(R):
                # Each column gets 1-3 non-zeros in U and V
                n_u_nonzero = random.randint(1, 3)
                n_v_nonzero = random.randint(1, 3)
                
                u_indices = random.sample(range(nm), min(n_u_nonzero, nm))
                v_indices = random.sample(range(mp), min(n_v_nonzero, mp))
                
                for idx in u_indices:
                    U[idx, r] = random.choice([-1.0, 1.0])
                for idx in v_indices:
                    V[idx, r] = random.choice([-1.0, 1.0])
            
            # Build coefficient matrix C for W
            # C[e, col] where e = i*m*p + j*p + k, col = r*pn + (k*n+i)
            C = np.zeros((n*m*p, pn*R))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = i*m*p + j*p + k
                        for r in range(R):
                            u_val = U[i*m+j, r]
                            v_val = V[j*p+k, r]
                            w_idx = k*n + i
                            col = r * pn + w_idx
                            C[e, col] = u_val * v_val
            
            b = np.ones(n*m*p)
            
            # Solve linear system
            try:
                w_flat, residuals, rank_c, sv = np.linalg.lstsq(C, b, rcond=None)
            except:
                continue
                
            if len(residuals) > 0 and residuals[0] > 1e-6:
                continue
            
            W = w_flat.reshape(pn, R)
            
            if verify(U, V, W):
                return U.tolist(), V.tolist(), W.tolist()
                
        return None
    
    # Try rank 17 first (known to be optimal)
    for R in [17, 16, 15, 14, 13, 12]:
        remaining = TIME_LIMIT - (time.time() - GLOBAL_START_TIME)
        if remaining < 5.0:
            break
        time_budget = min(20.0, remaining - 5.0)
        result = try_find_rank(R, time_budget=time_budget)
        if result:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
            
    # Fallback to schoolbook
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
