"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 4, 4, 7
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p  # 112
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j][idx] = 1
                V_sb[j*p + k][idx] = 1
                W_sb[k*n + i][idx] = 1
                idx += 1
    
    def verify(U, V, W):
        """Verify that the decomposition is correct."""
        n, m, p = 4, 4, 7
        rank = len(U[0]) if U else 0
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find a lower rank decomposition using structured search
    # Strategy: use known fast matrix multiplication patterns
    
    # For 4x4, we can use the Coppersmith-Winograd type approach
    # Let's try a structured decomposition with small integer entries
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try rank 21-40 with structured patterns
    # Use a greedy approach: build up terms and check if we can satisfy all constraints
    
    # Alternative: use the fact that we can group terms
    # For the 4x4x7 case, let's try to use block structure
    
    # Try using half-integer and integer entries with a structured pattern
    # Based on the idea that T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    # Let's try a systematic search over small ranks with structured patterns
    # We'll use a simplified version of the tensor decomposition search
    
    # For small cases, we can try to find a decomposition by solving a system
    # of equations. Each (i,j,k) gives one equation.
    # With r terms, we have r*3 unknowns (per term, 3 values) and n*m*p equations.
    
    # This is a nonlinear system (trilinear), so we need a different approach.
    
    # Let's try: fix U and V, then W is determined (if the system is consistent)
    # Or use iterative refinement.
    
    # Practical approach: try known decompositions for small matrix multiplication
    
    # For 4x4 matrix multiplication, one known approach uses rank ~21-25
    # Let's try to construct a valid decomposition with rank < 112
    
    # Use a random search with verification
    np.random.seed(42)
    
    for target_rank in [30, 25, 22, 21, 20, 18, 16, 14, 12, 10]:
        if time.time() > deadline - 30:
            break
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            # Try random structured decomposition
            # Use small integer entries: -1, 0, 1
            U_try = np.zeros((n*m, target_rank), dtype=float)
            V_try = np.zeros((m*p, target_rank), dtype=float)
            W_try = np.zeros((p*n, target_rank), dtype=float)
            
            # Fill with random -1, 0, 1
            for r in range(target_rank):
                for _ in range(3):  # Each term has 3 non-zero entries
                    # Pick random positions
                    ui = np.random.randint(0, n*m)
                    vj = np.random.randint(0, m*p)
                    wk = np.random.randint(0, p*n)
                    U_try[ui, r] = np.random.choice([-1, 1])
                    V_try[vj, r] = np.random.choice([-1, 1])
                    W_try[wk, r] = np.random.choice([-1, 1])
            
            # Convert to lists
            U_list = U_try.tolist()
            V_list = V_try.tolist()
            W_list = W_try.tolist()
            
            if verify(U_list, V_list, W_list):
                best_U, best_V, best_W = U_list, V_list, W_list
                best_rank = target_rank
                break
        if best_rank < rank_sb:
            break
    
    # If no better found, return schoolbook
    if best_rank < rank_sb:
        return {'U': best_U, 'V': best_V, 'W': best_W}
    else:
        return {'U': U_sb, 'V': V_sb, 'W': W_sb}

# EVOLVE-BLOCK-END
