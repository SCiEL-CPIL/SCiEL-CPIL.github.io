import time
import numpy as np

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 145.0  # Leave 5s safety margin for 150s limit

def construct():
    n, m, p = DIMENSIONS
    
    # 1. Generate the base Strassen decomposition (Rank 54)
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # 7 columns for one 2x2 * 2x2 block
        
        # m0: (x00+x11)(y00+y11) -> Z00+=1, Z11+=1
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # m1: (x10+x11)y00 -> Z10+=1, Z11-=1
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        # m2: x00(y01-y11) -> Z01+=1, Z11+=1
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # m3: x11(y10-y00) -> Z00+=1, Z10+=1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # m4: (x00+x01)y11 -> Z00-=1, Z01+=1
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # m5: (x10-x00)(y00+y01) -> Z11+=1
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # m6: (x01-x11)(y10+y11) -> Z00+=1
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    # Blocks for A_R (j=3,4) * B_R (j=3,4)
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)
    
    # Blocks for A_L1 (j=0,1) * B_L1 (j=0,1)
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Outer product for A_L2 (j=2) * B_L2 (j=2)
    # A_L2 is 2x1, B_L2 is 1x6
    for k in range(p):
        # C[0, k] += A[0, 2] * B[2, k]
        U[0 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # C[1, k] += A[1, 2] * B[2, k]
        U[1 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1

    # 2. Convert to numpy for efficient checking
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # 3. Rank Reduction via Column Elimination
    # We want to find a subset of columns of size r < 54 such that
    # sum_{c in subset} U[:,c] V[:,c] W[:,c] = T
    # This is equivalent to finding a sparse solution to a linear system.
    # Let's use a greedy approach:
    # 1. Start with all 54 columns.
    # 2. Randomly permute columns.
    # 3. Iterate through columns. Try to remove column c.
    #    If removing it breaks the identity, try to compensate by adjusting other columns?
    #    Simpler: just try to remove. If it fails, keep it.
    #    This is very weak.
    
    # Better approach:
    # The tensor T is fixed.
    # We have a set of vectors t_c = U[:,c] \otimes V[:,c] \otimes W[:,c].
    # We want sum c_c * t_c = T.
    # We know sum 1 * t_c = T.
    # We want to find a sparse vector c such that sum c_c t_c = T.
    # This is a linear system M c = T, where M is the matrix with columns t_c.
    # Since we have a known solution c=1, we are looking for a sparse solution.
    # This is generally hard.
    
    # Heuristic:
    # Try to remove columns in a random order.
    # When we remove column c, we have an error e = -t_c.
    # We need to find a combination of remaining columns that sums to e.
    # If the remaining columns span the space, we can solve for the coefficients.
    # But we want to keep them sparse.
    
    # Let's try a different heuristic:
    # Merge columns.
    # If we have two columns c1, c2, can we replace them with one column c?
    # U_c = U_c1 + U_c2, V_c = V_c1 + V_c2, W_c = W_c1 + W_c2?
    # (U1+U2)(V1+V2)(W1+W2) = U1V1W1 + U1V1W2 + U1V2W1 + U1V2W2 + ...
    # This introduces cross terms. It only works if cross terms cancel out or are zero.
    
    # Let's stick to a simple search for dependent columns.
    # If column c is a linear combination of other columns in the tensor space, we can remove it.
    # Actually, we just need the sum to be T.
    
    # Let's try a simple random restart search for a subset of columns of size r.
    # For r=50, we need to find 50 columns out of 54 that sum to T.
    # This is equivalent to finding 4 columns that sum to 0 (since sum of all 54 is T).
    # So we are looking for a linear dependency among the 54 vectors t_c.
    # If such a dependency exists, we can remove those columns and adjust?
    # No, if sum_{c in S} t_c = 0, then sum_{c not in S} t_c = T.
    # So we just need to find a non-trivial linear combination of t_c's that sums to 0.
    
    # Let's check for linear dependencies in the tensor space.
    # The dimension of the tensor space is n*m*p = 60.
    # We have 54 vectors. They are likely independent.
    # However, we are not restricted to the span of the 54 vectors.
    # We can use ANY vectors in the tensor space.
    
    # So the problem is: find a set of rank-1 tensors {t_c} of size r such that sum t_c = T.
    # This is the matrix multiplication problem.
    
    # Let's try to use the fact that we can mix columns.
    # We can perform Gaussian elimination on the columns of U, V, W simultaneously?
    # No, the coupling is in the product.
    
    # Let's try a simple optimization:
    # Minimize the L1 norm of the coefficients c in M c = T.
    # M is 60 x 54.
    # This is underdetermined.
    # We can use scipy.optimize or simple gradient descent on c.
    # But we want c to be sparse (0/1).
    
    # Let's try a random search for a subset of columns.
    # Since 54 is small, we can try to remove 1, 2, 3 columns.
    
    # First, check if any single column is 0. (No)
    # Check if any two columns sum to 0. (No)
    # Check if any three columns sum to 0.
    
    # Let's do a random search for a subset of size 50.
    # We pick 4 columns to remove. There are C(54,4) ~ 300k combinations.
    # For each, we check if the sum of the remaining 50 is T.
    # This is fast enough.
    
    best_rank = 54
    best_U, best_V, best_W = U, V, W
    
    # Precompute tensor T
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    T_flat = T.flatten()
    
    # Precompute t_c for all c
    t_c_list = []
    for c in range(54):
        t_c = (U[:, c] * V[:, c] * W[:, c]).reshape(n, m, p).flatten()
        t_c_list.append(t_c)
    
    # Check for 1 column removal (sum of 53 = T => removed column = 0)
    # Check for 2 column removal (sum of 52 = T => sum of 2 removed = 0)
    # Check for 3 column removal (sum of 51 = T => sum of 3 removed = 0)
    # Check for 4 column removal (sum of 50 = T => sum of 4 removed = 0)
    
    # We can just iterate over subsets of columns to remove.
    # Since we want to minimize rank, we want to remove as many as possible.
    
    # Let's try to remove 4 columns.
    # We need sum_{c in R} t_c = 0.
    
    # This is a subset sum problem in a high-dimensional space.
    # Random search might work if there are many solutions.
    
    # Let's do a random search for 4 columns.
    import itertools
    
    found_50 = False
    # Try random subsets of 4
    for _ in range(10000):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        R = np.random.choice(54, 4, replace=False)
        s = sum(t_c_list[c] for c in R)
        if np.allclose(s, 0, atol=1e-9):
            found_50 = True
            # Construct new U, V, W
            keep = [c for c in range(54) if c not in R]
            best_U = U[:, keep]
            best_V = V[:, keep]
            best_W = W[:, keep]
            best_rank = 50
            break
            
    if not found_50:
        # Try 3 columns
        found_51 = False
        for _ in range(10000):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            R = np.random.choice(54, 3, replace=False)
            s = sum(t_c_list[c] for c in R)
            if np.allclose(s, 0, atol=1e-9):
                found_51 = True
                keep = [c for c in range(54) if c not in R]
                best_U = U[:, keep]
                best_V = V[:, keep]
                best_W = W[:, keep]
                best_rank = 51
                break

    # Convert back to lists
    best_U = best_U.tolist()
    best_V = best_V.tolist()
    best_W = best_W.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
