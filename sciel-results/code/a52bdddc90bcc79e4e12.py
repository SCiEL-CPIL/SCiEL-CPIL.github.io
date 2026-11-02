"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s to be safe
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14
    
    # Build the target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose T."""
        rank = U.shape[1]
        T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol
    
    def to_list(mat):
        """Convert numpy array to list of lists with clean values."""
        return [[0 if abs(x) < 1e-12 else (1 if abs(x-1) < 1e-12 else (-1 if abs(x+1) < 1e-12 else (0.5 if abs(x-0.5) < 1e-12 else (-0.5 if abs(x+0.5) < 1e-12 else x)))) for x in row] for row in mat]
    
    # Try to find lower rank decompositions
    # Strategy: try ranks from 41 down, but this is expensive
    # Instead, use random initialization and optimization
    
    # For small tensors, we can try specific constructions
    # Let's try a few different approaches
    
    # Approach 1: Try to find a rank-41 decomposition by modifying schoolbook
    # This is hard, so let's try random search for lower ranks
    
    # Approach 2: Use the fact that for small matrices, we can try
    # various algebraic identities
    
    # Let's try a more systematic approach: search for rank r decompositions
    # by trying to fit random initializations
    
    current_rank = rank_school - 1
    while current_rank >= 20 and time.time() < deadline:
        # Try to find a decomposition of rank current_rank
        found = False
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(nm, current_rank)
            V = np.random.randn(mp, current_rank)
            W = np.random.randn(pn, current_rank)
            
            # Try to optimize: minimize ||U⊗V⊗W - T||_F
            # Use a simple gradient-free approach or alternating least squares
            
            # Alternating least squares
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Update U: fix V, W
                # T_recon[i,j,k] = sum_c U[i,c] V[j,c] W[k,c]
                # For each i, solve for U[i,:] given V, W
                # Residual R[i,j,k] = T[i,j,k] - sum_{c'} U[i,c'] V[j,c'] W[k,c']
                # We want to minimize sum_{j,k} (T[i,j,k] - sum_c U[i,c] V[j,c] W[k,c])^2
                
                # Build the system: for each i, we have a linear system in U[i,:]
                # Let's use a vectorized approach
                # For fixed i, define M_i[j,k,c] = V[j,c] * W[k,c]
                # Then we want U[i,:] to minimize ||T[i,:,:] - U[i,:] @ M_i_flat||^2
                
                # This is expensive, so let's use a simpler approach
                # Just do a few iterations of a simple update
                
                break  # Skip ALS for now, too slow
            
            # Check if random init happens to work (unlikely but possible for small cases)
            if check_decomposition(U, V, W):
                best_rank = current_rank
                best_U = U
                best_V = V
                best_W = W
                found = True
                break
        
        if found:
            break
        
        current_rank -= 1
    
    # Try a few more specific constructions
    # For 2x3 x 3x7, maybe there's a known lower rank
    
    # Let's try rank 40, 39, etc. with more careful search
    for target_rank in range(min(best_rank - 1, 41), 15, -1):
        if time.time() > deadline:
            break
        
        # Try structured random search
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Initialize with sparsity
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            # Random sparse initialization
            for c in range(target_rank):
                # Each column has a few non-zero entries
                u_nonzero = random.sample(range(nm), min(3, nm))
                v_nonzero = random.sample(range(mp), min(5, mp))
                w_nonzero = random.sample(range(pn), min(4, pn))
                
                for idx in u_nonzero:
                    U[idx, c] = random.choice([1, -1, 0.5, -0.5])
                for idx in v_nonzero:
                    V[idx, c] = random.choice([1, -1, 0.5, -0.5])
                for idx in w_nonzero:
                    W[idx, c] = random.choice([1, -1, 0.5, -0.5])
            
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break
    
    # Convert to lists
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
