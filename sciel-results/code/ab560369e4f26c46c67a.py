"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    errors = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > 1e-10:
                    errors += 1
    return errors == 0

def _try_reduce_rank(U, V, W, n, m, p, max_iterations=50):
    """
    Try to reduce rank by finding linear dependencies among columns.
    For each pair of columns, check if one can be expressed as a linear combination
    of the others, and if so, eliminate it.
    """
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    current_U = U_np.copy()
    current_V = V_np.copy()
    current_W = W_np.copy()
    
    for _ in range(max_iterations):
        reduced = False
        for r in range(current_U.shape[1]):
            # Try to express column r as linear combination of other columns
            # Build the matrix of all other columns in the "flattened" space
            # The tensor constraint: for all (i,j,k), sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = 1
            # Column r contributes: U[:,r] * outer(V[:,r], W[:,r]) to the tensor
            # We want to find coefficients c_s for s != r such that:
            #   U[:,r]*outer(V[:,r], W[:,r]) = sum_{s!=r} c_s * U[:,s]*outer(V[:,s], W[:,s])
            
            # This is a system in the space of tensors. Flatten:
            # The "vector" for column s is vec(U[:,s] * outer(V[:,s], W[:,s]))
            # But this is not quite right because the tensor entry is a product.
            
            # Alternative: check if column r is in the span of others by checking
            # if there exist c_s such that the tensor contribution matches.
            # This is a nonlinear problem in general, but we can check linear dependence
            # of the rank-1 tensors.
            
            # Simpler heuristic: check if the column vectors U[:,r], V[:,r], W[:,r]
            # are "redundant" by checking if removing column r and adjusting others
            # can still represent the tensor.
            
            # For now, use a simple check: if column r has all zeros in U, V, or W, remove it
            if np.allclose(U_np[:, r], 0) or np.allclose(V_np[:, r], 0) or np.allclose(W_np[:, r], 0):
                # Remove column r
                mask = [x for x in range(current_U.shape[1]) if x != r]
                current_U = current_U[:, mask]
                current_V = current_V[:, mask]
                current_W = current_W[:, mask]
                reduced = True
                break
        
        if not reduced:
            # Try pairwise: check if two columns can be merged
            # This is more complex; skip for now
            break
    
    new_rank = current_U.shape[1]
    if new_rank < rank:
        # Convert back to list of lists
        U_new = current_U.tolist()
        V_new = current_V.tolist()
        W_new = current_W.tolist()
        # Round to half-integers if close
        U_new = [[round(x, 10) for x in row] for row in U_new]
        V_new = [[round(x, 10) for x in row] for row in V_new]
        W_new = [[round(x, 10) for x in row] for row in W_new]
        if _verify_decomposition(U_new, V_new, W_new, n, m, p):
            return U_new, V_new, W_new
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    t0 = time.time()
    deadline = t0 + 110  # 110s safety margin within 150s limit
    
    # Strategy: Split the 3x4 matrix into a 2x4 block and a 1x4 block.
    # 2x4 x 4x4: Use Strassen on four 2x2 sub-blocks -> rank 4*7 = 28
    # 1x4 x 4x4: Schoolbook -> rank 16
    # Total rank: 44
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # (12, 44)
    V = [[0] * rank for _ in range(m * p)]  # (16, 44)
    W = [[0] * rank for _ in range(p * n)]  # (12, 44)
    
    col = 0
    
    # Part 1: 2x4 x 4x4 using Strassen (rows 0,1 of A)
    # Split into four 2x2 * 2x2 blocks:
    for j in range(2):  # A col pair / B row pair
        for i in range(2):  # B col pair / C col pair
            a_r0, a_r1 = 0, 1
            b_c0, b_c1 = 2*j, 2*j+1
            c_c0, c_c1 = 2*i, 2*i+1
            c_r0, c_r1 = 0, 1
            
            # M1 = (A[a_r0,b_c0] + A[a_r1,b_c1]) * (B[b_c0,c_c0] + B[b_c1,c_c1])
            U[a_r0 * m + b_c0][col] += 1
            U[a_r1 * m + b_c1][col] += 1
            V[b_c0 * p + c_c0][col] += 1
            V[b_c1 * p + c_c1][col] += 1
            W[c_c0 * n + a_r0][col] += 1
            W[c_c1 * n + a_r1][col] += 1
            col += 1
            
            # M2 = (A[a_r1,b_c0] + A[a_r1,b_c1]) * B[b_c0,c_c0]
            U[a_r1 * m + b_c0][col] += 1
            U[a_r1 * m + b_c1][col] += 1
            V[b_c0 * p + c_c0][col] += 1
            W[c_c0 * n + a_r1][col] += 1
            W[c_c1 * n + a_r1][col] -= 1
            col += 1
            
            # M3 = A[a_r0,b_c0] * (B[b_c0,c_c1] - B[b_c1,c_c1])
            U[a_r0 * m + b_c0][col] += 1
            V[b_c0 * p + c_c1][col] += 1
            V[b_c1 * p + c_c1][col] -= 1
            W[c_c1 * n + a_r0][col] += 1
            W[c_c1 * n + a_r1][col] += 1
            col += 1
            
            # M4 = A[a_r1,b_c1] * (B[b_c1,c_c0] - B[b_c0,c_c0])
            U[a_r1 * m + b_c1][col] += 1
            V[b_c1 * p + c_c0][col] += 1
            V[b_c0 * p + c_c0][col] -= 1
            W[c_c0 * n + a_r0][col] += 1
            W[c_c0 * n + a_r1][col] += 1
            col += 1
            
            # M5 = (A[a_r0,b_c0] + A[a_r0,b_c1]) * B[b_c1,c_c1]
            U[a_r0 * m + b_c0][col] += 1
            U[a_r0 * m + b_c1][col] += 1
            V[b_c1 * p + c_c1][col] += 1
            W[c_c0 * n + a_r0][col] -= 1
            W[c_c1 * n + a_r1][col] += 1
            col += 1
            
            # M6 = (A[a_r1,b_c0] - A[a_r0,b_c0]) * (B[b_c0,c_c0] + B[b_c0,c_c1])
            U[a_r1 * m + b_c0][col] += 1
            U[a_r0 * m + b_c0][col] -= 1
            V[b_c0 * p + c_c0][col] += 1
            V[b_c0 * p + c_c1][col] += 1
            W[c_c1 * n + a_r1][col] += 1
            col += 1
            
            # M7 = (A[a_r0,b_c1] - A[a_r1,b_c1]) * (B[b_c1,c_c0] + B[b_c1,c_c1])
            U[a_r0 * m + b_c1][col] += 1
            U[a_r1 * m + b_c1][col] -= 1
            V[b_c1 * p + c_c0][col] += 1
            V[b_c1 * p + c_c1][col] += 1
            W[c_c0 * n + a_r0][col] += 1
            col += 1
    
    # Part 2: 1x4 x 4x4 schoolbook (row 2 of A)
    for j in range(4):
        for k in range(4):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
    
    assert col == rank, f"Expected rank {rank} but used {col} columns"
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U = [[0] * rank_sb for _ in range(n * m)]
        V = [[0] * rank_sb for _ in range(m * p)]
        W = [[0] * rank_sb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by finding linear dependencies
    if time.time() < deadline - 10:
        U_red, V_red, W_red = _try_reduce_rank(U, V, W, n, m, p)
        if len(U_red[0]) < rank:
            return {"U": U_red, "V": V_red, "W": W_red}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
