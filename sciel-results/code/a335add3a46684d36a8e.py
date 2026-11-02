"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    # T[i, j, k] where:
    # i = i_A * m + j_A, i_A in [0,1], j_A in [0,1] -> i in [0,3]
    # j = j_A * p + k_B, j_A in [0,1], k_B in [0,8] -> j in [0,17]
    # k = k_B * n + i_A, k_B in [0,8], i_A in [0,1] -> k in [0,17]
    # T[i,j,k] = 1 iff i_A*2+j_A=i, j_A*9+k_B=j, k_B*2+i_A=k
    # i.e., T[i_A*2+j_A, j_A*9+k_B, k_B*2+i_A] = 1
    
    # Build the target tensor
    T = np.zeros((4, 18, 18), dtype=np.int64)
    for i_A in range(2):
        for j_A in range(2):
            for k_B in range(9):
                i = i_A * 2 + j_A
                j = j_A * 9 + k_B
                k = k_B * 2 + i_A
                T[i, j, k] = 1
    
    # Try to find a low-rank decomposition
    # Start with schoolbook rank 36 and try to reduce
    start_time = time.time()
    deadline = start_time + 120
    
    best_U = None
    best_V = None
    best_W = None
    best_rank = 36  # schoolbook
    
    # Schoolbook initialization
    U_school = np.zeros((4, 36), dtype=np.int64)
    V_school = np.zeros((18, 36), dtype=np.int64)
    W_school = np.zeros((18, 36), dtype=np.int64)
    col = 0
    for i_A in range(2):
        for j_A in range(2):
            for k_B in range(9):
                i = i_A * 2 + j_A
                j = j_A * 9 + k_B
                k = k_B * 2 + i_A
                U_school[i, col] = 1
                V_school[j, col] = 1
                W_school[k, col] = 1
                col += 1
    
    # Verify schoolbook
    T_recon = np.einsum('ir,jr,kr->ijk', U_school, V_school, W_school)
    assert np.array_equal(T_recon, T), "Schoolbook failed"
    
    # Try to reduce rank by finding linear dependencies
    # Strategy: try to express some schoolbook terms as combinations of others
    # This is hard in general, so let's try a heuristic search
    
    # Alternative: use a random initialization and optimize
    # But for exact integer decomposition, this is tricky
    
    # Let's try a structured approach based on Strassen
    # For M(2,2,9), we can use the following idea:
    # Group the 9 columns into groups and apply Strassen-like formulas
    
    # Actually, let me try to directly search for rank < 36
    # by trying to find r such that T can be decomposed with r terms
    
    # Use a simple iterative approach: start with rank 32 and try to solve
    for target_rank in [32, 33, 34, 35]:
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            # Random initialization with small integers
            U = np.zeros((4, target_rank), dtype=np.int64)
            V = np.zeros((18, target_rank), dtype=np.int64)
            W = np.zeros((18, target_rank), dtype=np.int64)
            
            # Initialize with random signs and small values
            for c in range(target_rank):
                U[:, c] = np.random.randint(-2, 3, size=4)
                V[:, c] = np.random.randint(-2, 3, size=18)
                W[:, c] = np.random.randint(-2, 3, size=18)
            
            # Check if this works (unlikely for random init)
            T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.array_equal(T_recon, T):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If we couldn't find a better decomposition, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
