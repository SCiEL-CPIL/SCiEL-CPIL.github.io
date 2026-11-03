"""Program for 3x3 x 3x5 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0  # seconds before deadline to stop

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    def verify(U, V, W, rank):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    def make_schoolbook():
        """Return the schoolbook decomposition of rank n*m*p = 45."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = n * m * p  # 45
    
    # Strategy: Use a structured search with random perturbations
    # The tensor has a specific structure we can exploit
    
    def try_rank(R, max_attempts=200):
        """Try to find a rank-R decomposition using structured search."""
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            return None
        
        for attempt in range(max_attempts):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                return None
            
            # Create sparse U, V, W with structured patterns
            U_mat = np.zeros((n*m, R))
            V_mat = np.zeros((m*p, R))
            W_mat = np.zeros((p*n, R))
            
            # For each column r, assign non-zero entries with structured patterns
            for r in range(R):
                # U: choose rows to be non-zero
                num_u = random.randint(1, 4)
                u_rows = random.sample(range(n*m), min(num_u, n*m))
                for row in u_rows:
                    U_mat[row, r] = random.choice([-1, 1])
                
                # V: choose rows to be non-zero  
                num_v = random.randint(1, 4)
                v_rows = random.sample(range(m*p), min(num_v, m*p))
                for row in v_rows:
                    V_mat[row, r] = random.choice([-1, 1])
                
                # W: choose rows to be non-zero
                num_w = random.randint(1, 4)
                w_rows = random.sample(range(p*n), min(num_w, p*n))
                for row in w_rows:
                    W_mat[row, r] = random.choice([-1, 1])
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {
                    "U": U_mat.tolist(),
                    "V": V_mat.tolist(),
                    "W": W_mat.tolist()
                }
        
        return None
    
    # Try rank 43 and 44 with structured search
    for target_rank in [43, 44]:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        result = try_rank(target_rank)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
    
    # Check deadline
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
