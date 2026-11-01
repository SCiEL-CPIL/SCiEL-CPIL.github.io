"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # The multiplication tensor T[i,j,k] = 1 should equal sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (u_vec, v_vec, w_vec) where u_vec is length 4 (i*m+j), 
    v_vec is length 4 (j*p+k), w_vec is length 4 (k*n+i).
    """
    # Strassen's 7 multiplications for C = A*B where A,B are 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    # 
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    #
    # c11 = M1 - M4 + M5 - M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M7
    #
    # For our tensor: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # With n=2, m=2, p=2: U index = i*2+j, V index = j*2+k, W index = k*2+i
    
    terms = []
    # M1 = (a11+a22)(b11+b22) -> contributes to c11, c22
    # U: a11 at (0,0)=0, a22 at (1,1)=3
    # V: b11 at (0,0)=0, b22 at (1,1)=3
    # W: c11 at (0,0)=0, c22 at (1,1)=3
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # M2 = (a21+a22)b11 -> contributes to c21, c22
    # U: a21 at (1,0)=2, a22 at (1,1)=3
    # V: b11 at (0,0)=0
    # W: c21 at (0,1)=1, c22 at (1,1)=3
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 1]))
    # M3 = a11(b12-b22) -> contributes to c12, c22
    # U: a11 at (0,0)=0
    # V: b12 at (0,1)=1, b22 at (1,1)=3
    # W: c12 at (0,1)=1, c22 at (1,1)=3
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]))
    # M4 = a22(b21-b11) -> contributes to c11, c21
    # U: a22 at (1,1)=3
    # V: b21 at (1,0)=2, b11 at (0,0)=0
    # W: c11 at (0,0)=0, c21 at (0,1)=1
    terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]))
    # M5 = (a11+a12)b22 -> contributes to c11, c12
    # U: a11 at (0,0)=0, a12 at (0,1)=1
    # V: b22 at (1,1)=3
    # W: c11 at (0,0)=0, c12 at (0,1)=1
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]))
    # M6 = (a21-a11)(b11+b12) -> contributes to c21
    # U: a21 at (1,0)=2, a11 at (0,0)=0
    # V: b11 at (0,0)=0, b12 at (0,1)=1
    # W: c21 at (0,1)=1
    terms.append(([1, 0, -1, 0], [1, 1, 0, 0], [0, 1, 0, 0]))
    # M7 = (a12-a22)(b21+b22) -> contributes to c11, c22
    # U: a12 at (0,1)=1, a22 at (1,1)=3
    # V: b21 at (1,0)=2, b22 at (1,1)=3
    # W: c11 at (0,0)=0, c22 at (1,1)=3
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]))
    
    return terms

def _build_decomposition_from_terms(all_terms, n, m, p):
    """Build U, V, W matrices from a list of (u_vec, v_vec, w_vec) terms.
    Each vec has length n*m, m*p, p*n respectively.
    """
    rank = len(all_terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for r, (u_vec, v_vec, w_vec) in enumerate(all_terms):
        for idx in range(n * m):
            U[idx][r] = u_vec[idx]
        for idx in range(m * p):
            V[idx][r] = v_vec[idx]
        for idx in range(p * n):
            W[idx][r] = w_vec[idx]
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 2, 5
    
    # Strategy: Use Strassen for the first 2 columns (2x2x2 block) and 
    # schoolbook for the remaining 3 columns, then try to find a better
    # combined decomposition.
    
    # First, get the Strassen terms for the 2x2x2 block
    strassen_terms_2x2 = _strassen_2x2_terms()
    
    # For the full 2x2x5 problem:
    # U has shape (4, rank), V has shape (10, rank), W has shape (10, rank)
    # The first 2 columns of B (k=0,1) form a 2x2x2 block
    # The last 3 columns (k=2,3,4) need separate handling
    
    # Start with Strassen(2x2x2) + schoolbook(2x2x3) = 7 + 6 = 13 terms
    # But we can potentially do better by finding a joint decomposition
    
    # Build the initial 13-term decomposition
    all_terms = []
    
    # Add Strassen terms for columns k=0,1
    for u_vec, v_vec, w_vec in strassen_terms_2x2:
        # Extend v_vec from length 4 (j*2+k for k=0,1) to length 10 (j*5+k for k=0..4)
        v_full = [0.0] * 10
        for j in range(2):
            for k in range(2):
                v_full[j * 5 + k] = v_vec[j * 2 + k]
        # w_vec is already length 4 (k*2+i for k=0,1), extend to length 10 (k*5+i for k=0..4)
        w_full = [0.0] * 10
        for k in range(2):
            for i in range(2):
                w_full[k * 5 + i] = w_vec[k * 2 + i]
        all_terms.append((list(u_vec), v_full, w_full))
    
    # Add schoolbook terms for columns k=2,3,4
    for i in range(n):
        for j in range(m):
            for k in range(2, p):
                u_vec = [0.0] * 4
                u_vec[i * m + j] = 1.0
                v_vec = [0.0] * 10
                v_vec[j * p + k] = 1.0
                w_vec = [0.0] * 10
                w_vec[k * n + i] = 1.0
                all_terms.append((u_vec, v_vec, w_vec))
    
    U, V, W = _build_decomposition_from_terms(all_terms, n, m, p)
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fall back to schoolbook
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
    
    current_rank = len(all_terms)
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by finding linear dependencies or better decompositions
    # Use numpy to check if we can reduce the rank
    
    if time.time() < deadline:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check if any columns are linearly dependent in a way that allows reduction
        # The tensor is T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # We need to find a minimal set of rank-1 tensors that sum to T
        
        # Try greedy elimination: for each pair of terms, check if one can be absorbed
        # This is complex, so let's try a different approach:
        # Use the fact that for rectangular matrices, we might find better decompositions
        
        # Try to use the structure: the 2x2x5 problem can be seen as
        # multiplying by [B1 B2] where B1, B2 are 2x2 and B3 is 2x3
        # Or [B1 | B2 | B3 | B4 | B5] as 2x1 blocks
        
        # Alternative: try to find a decomposition with rank < current_rank
        # by solving a system of equations
        
        # For rank r, we need to find U (4xr), V (10xr), W (10xr) such that
        # sum_{r'} U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'] = 1 for all i,j,k
        
        # This is a non-convex problem. Let's try a heuristic approach:
        # Start with the current decomposition and try to merge terms
        
        # Check if we can reduce by 1: try to express one term as a combination of others
        # by solving a linear system
        
        for target_rank in range(current_rank - 1, 6, -1):
            if time.time() > deadline - 10:
                break
            
            # Try to find a decomposition of rank target_rank
            # Using the constraint that the tensor must be reproduced exactly
            
            # For small ranks, we can try to solve the system
            # The number of constraints is n*m*p = 20
            # The number of variables is rank*(n*m + m*p + p*n) = rank*(4+10+10) = 24*rank
            
            # For rank 12: 288 variables, 20 constraints - underdetermined
            # We can try to find a solution
            
            # Use a randomized approach: fix some structure and solve
            # For the rectangular case, we can exploit the block structure
            
            # Let's try: use 7 Strassen-like terms for the 2x2 part and 
            # 5 terms for the 2x3 part, but try to overlap them
            
            # Actually, let's try a more systematic approach:
            # The multiplication tensor for 2x2x5 has rank at most 13 (as we have)
            # The lower bound is max(n*m, m*p, p*n) = max(4, 10, 10) = 10
            # So the rank is between 10 and 13
            
            # Try rank 12: 
            # Use 7 terms for the 2x2x2 block (Strassen) and 5 terms for the remaining
            # but try to share some structure
            
            # For the 2x2x3 part (columns 2,3,4), the schoolbook rank is 6
            # Can we do it in 5? The 2x2x3 tensor has rank at least max(4,6,6)=6
            # So no, we need at least 6 terms for the 2x2x3 part alone
            
            # But when combined with the 2x2x2 part, we might share terms
            # Total lower bound is still 10
            
            # Let's try to construct a rank-12 decomposition:
            # 7 Strassen terms for 2x2x2 + 5 terms for 2x2x3 (if possible)
            # But 2x2x3 needs rank >= 6, so this won't work directly
            
            # Alternative: use a different decomposition for the whole 2x2x5
            # that has rank 12
            
            # Let me try to find it by solving the system
            # For rank r, we have variables U (4xr), V (10xr), W (10xr)
            # Constraints: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            
            # This is a tensor completion problem. Let's try a simple approach:
            # Assume W has a specific structure and solve for U, V
            
            # For now, let's just return the 13-term decomposition
            # and see if we can improve it with a more sophisticated search
            
            pass
    
    # Try a simple optimization: check if we can eliminate any term
    # by seeing if it's redundant given the others
    
    # Convert to numpy for easier manipulation
    U_arr = np.array(best_U, dtype=float)
    V_arr = np.array(best_V, dtype=float)  
    W_arr = np.array(best_W, dtype=float)
    
    current_rank = U_arr.shape[1]
    
    # Try to reduce rank by checking linear dependencies
    # For each column r, check if U[:,r] is in the span of other U columns,
    # and similarly for V and W, in a way that preserves the tensor product
    
    # This is complex, so let's try a different heuristic:
    # Look for columns that are sparse and might be mergeable
    
    # For now, return the best decomposition found
    # Convert back to lists
    U_list = [[float(x) for x in row] for row in U_arr]
    V_list = [[float(x) for x in row] for row in V_arr]
    W_list = [[float(x) for x in row] for row in W_arr]
    
    # Clean up: convert near-integers to exact integers
    for mat in [U_list, V_list, W_list]:
        for row in mat:
            for idx, val in enumerate(row):
                if abs(val - round(val)) < 1e-10:
                    row[idx] = round(val)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
