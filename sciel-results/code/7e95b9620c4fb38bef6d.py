"""Combined program: structured anchor-based search for lower rank matrix multiplication tensor."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 148.0  # Safety margin before 150s kill
    
    # 1. Baseline: Schoolbook Decomposition (Rank 60)
    rank_baseline = n * m * p
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_base[i * m + j][col] = 1
                V_base[j * p + k][col] = 1
                W_base[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = rank_baseline
    deadline = start_time + time_limit - 3.0
    
    def verify_solution(U, V, W, R):
        """Verify that the tensor equation holds exactly."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(R):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def try_find_solution(R, entry_set, num_trials, deadline):
        """Try to find a valid decomposition of rank R with given entry set."""
        for t in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Generate random U and V
            U = np.random.choice(entry_set, size=(n * m, R))
            V = np.random.choice(entry_set, size=(m * p, R))
            
            W_candidate = np.zeros((p * n, R), dtype=float)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build A matrix: A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    A = np.zeros((m, R), dtype=float)
                    for j in range(m):
                        A[j, :] = U[i * m + j, :].astype(float) * V[j * p + k, :].astype(float)
                    
                    # Solve A * w = ones(m) for w in half-integers
                    b = np.ones(m)
                    
                    # First check consistency
                    try:
                        rank_A = np.linalg.matrix_rank(A)
                        A_aug = np.hstack((A, b.reshape(-1, 1)))
                        rank_aug = np.linalg.matrix_rank(A_aug)
                        
                        if rank_A != rank_aug:
                            valid = False
                            break
                        
                        # Get particular solution via least squares
                        w_ls, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        
                        # Check if w_ls is already close to half-integers
                        w_half = np.round(w_ls * 2) / 2.0
                        if np.allclose(A.dot(w_half), b, atol=1e-8):
                            W_candidate[k * n + i, :] = w_half
                        else:
                            # Try null space perturbations
                            if rank_A < R:
                                # Get null space basis from SVD
                                _, _, Vt = np.linalg.svd(A, full_matrices=True)
                                null_basis = Vt[rank_A:].T  # shape (R, R-rank_A)
                                
                                found = False
                                for _ in range(50):
                                    y = np.random.randn(null_basis.shape[1])
                                    w_try = w_ls + null_basis.dot(y)
                                    w_try_half = np.round(w_try * 2) / 2.0
                                    if np.allclose(A.dot(w_try_half), b, atol=1e-8):
                                        W_candidate[k * n + i, :] = w_try_half
                                        found = True
                                        break
                                
                                if not found:
                                    valid = False
                                    break
                            else:
                                valid = False
                                break
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Full verification
                if verify_solution(U, V, W_candidate, R):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_candidate.tolist()
                    }
        
        return None
    
    # Search strategy: try ranks from high to low
    # Use different entry sets for different rank ranges
    entry_sets = [
        [-1, 0, 1],           # Small entries, most likely to work
        [-2, -1, 0, 1, 2],    # Medium entries
        [-1, -0.5, 0, 0.5, 1], # Include half-integers in U,V
        [-2, -1, -0.5, 0, 0.5, 1, 2],  # Mixed
    ]
    
    # Allocate time budget
    total_search_time = time_limit - 5.0
    num_ranks_to_try = min(20, rank_baseline - 40)  # Try ranks 59 down to 40
    time_per_rank = total_search_time / num_ranks_to_try
    
    for rank_offset in range(1, num_ranks_to_try + 1):
        R = rank_baseline - rank_offset
        if time.time() > deadline:
            break
        
        # Allocate time for this rank
        rank_deadline = min(time.time() + time_per_rank, deadline)
        
        # Choose entry set based on rank
        if R >= 55:
            sets_to_try = [entry_sets[0], entry_sets[1]]
        elif R >= 50:
            sets_to_try = [entry_sets[0], entry_sets[2], entry_sets[1]]
        else:
            sets_to_try = [entry_sets[1], entry_sets[3], entry_sets[0]]
        
        # Number of trials per entry set
        trials_per_set = max(100, int(2000 * time_per_rank / 10.0))
        
        found = False
        for entry_set in sets_to_try:
            if time.time() > rank_deadline:
                break
            
            result = try_find_solution(R, entry_set, trials_per_set, rank_deadline)
            if result is not None:
                best_solution = result
                best_rank = R
                found = True
                break
        
        if found:
            # Continue searching for even lower rank if time permits
            if time.time() > deadline:
                break
    
    return best_solution
