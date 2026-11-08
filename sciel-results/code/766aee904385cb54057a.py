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
    
    # Try structured rank-29 approach:
    # Use rank-21 for 4x4x4 sub-block, rank-8 for remaining 4x2
    # This gives total rank 29
    
    # First, let's try to find a good rank-21 decomposition for 4x4x4
    # using Strassen-like or Winograd patterns
    
    # For now, let's try random search with the constraint that we want
    # integer/half-integer entries and rank around 29
    
    # Search for lower rank
    for R in range(20, best_rank):
        if time.time() > deadline - 15:
            break
        
        found = False
        for trial in range(50):  # Reduce trials if time is tight
            if time.time() > deadline - 15:
                break
            
            # Random U, V with entries in {-1, 0, 1}
            # But we want to bias towards structured solutions
            # Let's try a mix of random and structured
            
            if trial % 5 == 0:
                # Structured: try to use Winograd/Strassen patterns
                # For 4x4x4, we can use Strassen with rank 7 for 2x2,
                # then extend to 4x4
                U_flat = np.zeros((n * m, R))
                V_flat = np.zeros((m * p, R))
                
                # Initialize with some structure
                for r in range(R):
                    # Random pattern
                    U_flat[:, r] = np.random.choice([-1, 0, 1], size=n * m)
                    V_flat[:, r] = np.random.choice([-1, 0, 1], size=m * p)
            else:
                # Pure random
                U_flat = np.random.randint(-1, 2, (n * m, R))
                V_flat = np.random.randint(-1, 2, (m * p, R))
            
            # Check if solution exists
            valid = True
            W_flat = np.zeros((p * n, R))
            
            for i in range(n):
                for k in range(p):
                    # Construct M_ik
                    # M_ik[j, l, r] = U[i, j, r] * V[l, k, r]
                    M_ik = np.zeros((m * m, R))
                    for j in range(m):
                        for l in range(m):
                            for r in range(R):
                                M_ik[j * m + l, r] = U_flat[i * m + j, r] * V_flat[l * p + k, r]
                    
                    b_ik = np.zeros(m * m)
                    for j in range(m):
                        b_ik[j * m + j] = 1
                    
                    # Solve M_ik w = b_ik
                    # Use lstsq
                    try:
                        w, residuals, rank, s = np.linalg.lstsq(M_ik, b_ik, rcond=None)
                    except:
                        valid = False
                        break
                    
                    # Check residual
                    if len(residuals) == 0 or residuals[0] > 1e-9:
                        valid = False
                        break
                    
                    W_flat[k * n + i, :] = w
            
            if valid:
                # Snap to half integers
                W_snapped = np.round(W_flat * 2) / 2
                
                # Verify snapped solution
                valid_snapped = True
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            for l in range(m):
                                val = 0
                                for r in range(R):
                                    val += W_snapped[k * n + i, r] * U_flat[i * m + j, r] * V_flat[l * p + k, r]
                                if abs(val - (1 if j == l else 0)) > 1e-9:
                                    valid_snapped = False
                                    break
                            if not valid_snapped:
                                break
                        if not valid_snapped:
                            break
                    if not valid_snapped:
                        break
                
                if valid_snapped:
                    # Convert to list of lists
                    def convert(mat):
                        return [[float(v) for v in row] for row in mat]
                    
                    best_solution = {
                        "U": convert(U_flat),
                        "V": convert(V_flat),
                        "W": convert(W_snapped)
                    }
                    best_rank = R
                    found = True
                    break
        
        if found:
            break
    
    return best_solution
