import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 110 second budget, safety margin
    
    # Schoolbook as fallback
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    def verify(U_r, V_r, W_r, rank):
        """Verify decomposition is exact"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                    if s != 1:
                        return False
        return True
    
    # Known 3x3x3 rank-21 decomposition (Strassen-based)
    # For 3x3x3: indices i,j in 0..2 for A, k in 0..2 for B
    # We'll build a structured candidate for 3x3x4
    
    # Strategy: Try to find a rank-r decomposition where the structure 
    # is based on 3x3x3 rank-21 plus additional terms for column 3
    
    # First, try a structured approach: use rank 24 (21 + 3 extra for 4th col)
    # with the constraint that terms 0-20 handle cols 0-2 and terms 21-23 handle col 3
    
    target_ranks = [24, 25, 26, 27, 28, 30, 32, 36]
    
    for target_rank in target_ranks:
        if time.time() > deadline:
            break
        
        found = False
        max_trials = 2000 if target_rank <= 27 else 5000
        
        for trial in range(max_trials):
            if time.time() > deadline:
                break
            
            # Generate random integer/half-integer entries
            # Use small values to make exact solutions more likely
            scale = 2 if target_rank <= 25 else 3
            U_r = np.random.randint(-scale, scale+1, size=(n*m, target_rank)).astype(float)
            V_r = np.random.randint(-scale, scale+1, size=(m*p, target_rank)).astype(float)
            W_r = np.random.randint(-scale, scale+1, size=(p*n, target_rank)).astype(float)
            
            # For structured search: ensure first 21 terms can handle 3x3x3 part
            # and remaining terms handle the 4th column
            if target_rank >= 24:
                # Zero out W entries for k=3 in first 21 terms (they should only affect k=0,1,2)
                # And zero out W entries for k=0,1,2 in last terms (they should only affect k=3)
                if target_rank > 21:
                    num_base = min(21, target_rank)
                    num_extra = target_rank - num_base
                    
                    # First num_base terms: W should be 0 for k=3 (index 3*n + i = 9+i)
                    for c in range(num_base):
                        for i in range(n):
                            W_r[3*n + i, c] = 0
                    
                    # Last num_extra terms: W should be 0 for k=0,1,2
                    for c in range(num_base, target_rank):
                        for i in range(n):
                            W_r[0*n + i, c] = 0
                            W_r[1*n + i, c] = 0
                            W_r[2*n + i, c] = 0
            
            # Check validity
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(target_rank):
                            s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_r[k*n+i, c]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # If random search didn't work, try a more structured approach for lower ranks
        if target_rank <= 27:
            # Try fixing some entries and solving for others
            if time.time() > deadline:
                break
            
            # Structured: fix U and V to have sparse structure, solve for W
            for struct_trial in range(500):
                if time.time() > deadline:
                    break
                
                # Create sparse U and V
                U_r = np.zeros((n*m, target_rank))
                V_r = np.zeros((m*p, target_rank))
                
                for c in range(target_rank):
                    # Each term has 2 non-zero entries in U and 2 in V
                    u_idx1 = np.random.randint(0, n*m)
                    u_idx2 = np.random.randint(0, n*m)
                    v_idx1 = np.random.randint(0, m*p)
                    v_idx2 = np.random.randint(0, m*p)
                    
                    u_val1 = np.random.choice([-1, 1, -0.5, 0.5])
                    u_val2 = np.random.choice([-1, 1, -0.5, 0.5])
                    v_val1 = np.random.choice([-1, 1, -0.5, 0.5])
                    v_val2 = np.random.choice([-1, 1, -0.5, 0.5])
                    
                    U_r[u_idx1, c] = u_val1
                    U_r[u_idx2, c] = u_val2
                    V_r[v_idx1, c] = v_val1
                    V_r[v_idx2, c] = v_val2
                
                # Solve for W: for each (k,i), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                # This is a system of linear equations in W
                # Number of equations: n*m*p = 36
                # Number of unknowns: p*n*target_rank = 12*target_rank
                # We need target_rank >= 3 for overdetermined system to potentially work
                
                # Set up linear system: A * W_flat = b
                # where A has shape (n*m*p, p*n*target_rank)
                num_eq = n * m * p
                num_unk = p * n * target_rank
                
                A = np.zeros((num_eq, num_unk))
                b = np.ones(num_eq)
                
                eq_idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for c in range(target_rank):
                                w_idx = (k*n + i) * target_rank + c
                                A[eq_idx, w_idx] = U_r[i*m+j, c] * V_r[j*p+k, c]
                            eq_idx += 1
                
                # Solve least squares
                try:
                    W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_r = W_flat.reshape(p*n, target_rank)
                    
                    # Check if solution is close to integer/half-integer
                    # Round to nearest half-integer and verify
                    W_rounded = np.round(W_r * 2) / 2
                    
                    # Verify with rounded values
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for c in range(target_rank):
                                    s += U_r[i*m+j, c] * V_r[j*p+k, c] * W_rounded[k*n+i, c]
                                if abs(s - 1) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_rounded.tolist()}
                except:
                    pass
    
    return {"U": U, "V": V, "W": W}
