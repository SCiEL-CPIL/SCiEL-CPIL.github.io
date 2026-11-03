import numpy as np
import time
import random

DIMENSIONS = [2, 3, 3]
TIME_LIMIT = 120  # seconds
SAFETY_MARGIN = 10  # seconds before limit to stop search

def _verify_decomposition(U, V, W, n, m, p, R, num_tests=5, seed=42):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rng = np.random.RandomState(seed)
    for _ in range(num_tests):
        A = rng.randint(-3, 4, size=(n, m)).astype(float)
        B = rng.randint(-3, 4, size=(m, p)).astype(float)
        C_true = A @ B
        A_flat = A.flatten()
        B_flat = B.flatten()
        
        C_approx = np.zeros((n, p))
        for r in range(R):
            u_val = np.dot(U[:, r], A_flat)
            v_val = np.dot(V[:, r], B_flat)
            w_vec = W[:, r]
            for i in range(n):
                for k in range(p):
                    C_approx[i, k] += w_vec[k * n + i] * u_val * v_val
                        
        if not np.allclose(C_approx, C_true, atol=1e-9):
            return False
    return True

def _build_rank17(n, m, p):
    """Build the known rank-17 decomposition using Strassen on 2x2x2 block + schoolbook."""
    rank = 17
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    U = np.zeros((dim_U, rank), dtype=float)
    V = np.zeros((dim_V, rank), dtype=float)
    W = np.zeros((dim_W, rank), dtype=float)
    
    # --- Strassen's Algorithm for A[0:2, 0:2] * B[0:2, 0:2] ---
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    W[3, 1] = -1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1
    W[3, 2] = 1
    
    # M4 = A11 * (B10 - B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1
    W[1, 3] = 1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[2, 4] = 1
    W[0, 4] = -1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # --- Schoolbook for remaining terms ---
    # A[:, 2] * B[2, :]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    
    # A[:, 0:2] * B[:, 2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
    U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
    
    # A[:, 2] * B[2, 2]
    U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    return U, V, W, rank

def _try_find_rank16(n, m, p, deadline):
    """
    Attempt to find a rank-16 decomposition using a randomized approach.
    
    Strategy: We know that for 2x3x3 matrix multiplication, the tensor rank
    is at most 17 (Strassen + schoolbook). We try to find rank 16 by
    using a structured parameterization.
    
    Key insight: The matrix multiplication tensor T has entries T[i,j,k] = 1
    if the (i,j) element of A multiplies with (j,k) element of B to produce
    (i,k) element of C. 
    
    For rank 16, we need 16 terms. The idea is to use a "border rank" style
    approach or to find a clever combination that saves one multiplication.
    
    One approach: Try to combine some of the schoolbook terms with the
    Strassen terms to reduce the total count.
    
    We'll use a randomized search over sparse structures.
    """
    R = 16
    dim_U = n * m  # 6
    dim_V = m * p  # 9
    dim_W = p * n  # 6
    
    # Generate the target tensor
    # T[i*m+j, j*p+k] contributes to output index k*n+i
    # We represent the tensor as a 3D array T[i, j, k] = 1 for all valid i,j,k
    # where i in [0,n), j in [0,m), k in [0,p)
    
    # For each rank-1 term r, we have:
    # U_r (vector of length n*m), V_r (vector of length m*p), W_r (vector of length p*n)
    # The tensor decomposes as: sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i] = T[i,j,k]
    
    # For small dimensions, we can set up equations.
    # T[i,j,k] = sum_{r=0}^{R-1} U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
    
    # There are n*m*p = 18 equations.
    # Unknowns: R*(n*m + m*p + p*n) = 16*(6+9+6) = 16*21 = 336
    # So the system is heavily overdetermined in terms of equations but
    # underdetermined in terms of unknowns. We need to find a solution.
    
    # Approach: Fix V to a generic sparse pattern, then solve for U and W.
    # But U*V*W is trilinear, so we can't directly linearize.
    
    # Better approach: Use the fact that for each fixed pair (i,j), the equation
    # T[i,j,k] = sum_r U_r[i*m+j] * (sum_k' V_r[j*p+k'] * W_r[k'*n+i] delta_{k,k'})
    # Actually, let's think differently.
    
    # For each (i,j), define the vector v_{i,j} in R^p:
    # v_{i,j}[k] = T[i,j,k] = 1
    # This must equal sum_r U_r[i*m+j] * (V_r[j,:] * W_r[:,i])
    # where V_r[j,:] is the j-th row of V_r (length p) and W_r[:,i] is the
    # i-th column of W_r reshaped... 
    
    # Let me re-index. W_r has length p*n. W_r[k*n+i] is the coefficient for
    # output (i,k). So for fixed i, the vector w_r_i of length p has entries
    # w_r_i[k] = W_r[k*n+i].
    
    # Then T[i,j,k] = sum_r U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
    # = sum_r U_r[i*m+j] * (V_r[j,:] elementwise W_r_col_i)[k]
    
    # Let M_r[j, i] = U_r[i*m+j] (so M_r is n x m matrix)
    # Let N_r[j, k] = V_r[j*p+k] (so N_r is m x p matrix)  
    # Let P_r[k, i] = W_r[k*n+i] (so P_r is p x n matrix)
    
    # Then T[i,j,k] = sum_r M_r[i,j] * N_r[j,k] * P_r[k,i]
    
    # This is exactly: T = sum_r (M_r ⊗ N_r ⊗ P_r) in the appropriate sense.
    # More precisely, the (i,j,k) entry of T is sum_r M_r[i,j]*N_r[j,k]*P_r[k,i].
    
    # We need: for all i,j,k: sum_r M_r[i,j]*N_r[j,k]*P_r[k,i] = 1
    
    # This is a system of 18 equations in 16*9 = 144 unknowns (each M_r, N_r, P_r
    # has 6, 9, 6 entries respectively, total 21 per r, times 16 = 336 unknowns).
    
    # Randomized approach: Randomly initialize, then use iterative refinement.
    # Or: use the structure of the known rank-17 solution and try to merge
    # two terms into one.
    
    # Let's try a different approach: start from the rank-17 solution and try
    # to eliminate one term by adjusting others.
    
    # Actually, let's try a direct randomized search with a time limit.
    # We'll use a simple iterative method:
    # 1. Randomly initialize U, V, W with small random values
    # 2. Compute the residual tensor R[i,j,k] = 1 - sum_r U_r[i*m+j]*V_r[j*p+k]*W_r[k*n+i]
    # 3. Update one factor at a time using a greedy approach
    
    # But this is unlikely to converge in the time available for such a non-convex problem.
    
    # Alternative: Use the known rank-17 and try to find a linear dependency
    # among the 17 terms that allows merging.
    
    # For now, let's try a limited structured search.
    # Key idea: In the rank-17 solution, the last 10 terms (indices 7-16) are
    # all "schoolbook" terms with single 1s in U, V, W. Maybe we can combine
    # some of these.
    
    # Let's try: can we express two of the schoolbook terms as a single
    # Strassen-like term?
    
    # Terms 7-10: A[:,2] * B[2,:]
    # Terms 11-14: A[:,0:2] * B[:,2]  
    # Terms 15-16: A[:,2] * B[2,2]
    
    # The term A02*B20 and A12*B20 both involve B20.
    # Can we write A02*B20 + A12*B20 = (A02+A12)*B20 - A12*B20? No, that's circular.
    
    # Let's try a completely different structured approach.
    # For 2x3x3, we can think of it as:
    # C = A * B where A is 2x3, B is 3x3
    # Split A = [A_left (2x2), A_right (2x1)]
    # Split B = [B_top (2x3); B_bottom (1x3)]
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top is 2x2 * 2x3 = 2x3, needs rank at most 7*3/2... 
    # Actually 2x2x3 tensor has rank at most... let's think.
    # The 2x2x3 matrix multiplication tensor: we need to compute 2x3 output.
    # Each output (i,k) = sum_j A[i,j]*B[j,k] for j=0,1.
    # So 6 outputs, each is a dot product of length 2.
    # Schoolbook: 6 terms. Can we do better?
    # 2x2x3: The tensor has 2*2*3=12 entries, all 1.
    # Rank of 2x2x3 multiplication tensor: I believe it's 6 (schoolbook is optimal
    # for this small case, but let me verify).
    
    # Actually, for 2x2x2, Strassen gives rank 7 vs schoolbook 8.
    # For 2x2x3, we can apply Strassen to the 2x2x2 part and add one more term
    # for the third column, giving rank 7+1=8? Or can we do better?
    
    # Let me just try to brute-force search for rank 16 with a time limit.
    # Given the complexity, I'll use a heuristic approach.
    
    # Actually, given the time constraints and the difficulty of finding rank 16
    # through randomized search, let me try a more targeted approach.
    
    # The key question is: is the rank of the 2x3x3 matrix multiplication tensor
    # 16 or 17? 
    
    # From the literature, I recall that for small matrix multiplication, the
    # rank is often equal to the schoolbook rank or slightly less due to
    # Strassen-type improvements. For 2x3x3:
    # - Schoolbook: 18
    # - Strassen on 2x2x2 + schoolbook for rest: 17
    # - Can we do 16?
    
    # Let me try a specific construction that might achieve rank 16.
    
    # Idea: Use a "mixed" Strassen approach where we overlap the Strassen
    # blocks with the schoolbook terms more cleverly.
    
    # Actually, let me try to use numpy to solve a least-squares problem
    # with a fixed V structure.
    
    # Fix V to have a specific sparse pattern, then solve for U and W.
    # The equation is: for each (i,j,k):
    #   1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is nonlinear in U, V, W. But if we fix V, it becomes:
    #   1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Which is still bilinear in U and W (for fixed V).
    
    # Let's try: fix V to be a random sparse matrix, then for each (i,j,k),
    # we have: 1 = sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
    
    # Define A_r[i,j] = U[i*m+j, r] and B_r[j,k] = V[j*p+k, r]
    # Then the equation is: 1 = sum_r A_r[i,j] * B_r[j,k] * W_r[k,i]
    
    # For fixed i,j: sum_r (A_r[i,j] * B_r[j,:]) * W_r[:,i] = 1 (vector in R^p)
    # This is: sum_r c_r * d_r = 1_vector, where c_r = A_r[i,j] and d_r = B_r[j,:] * W_r[:,i]
    
    # This is a system of p equations in the unknowns {A_r[i,j], W_r[k,i]} for r=0..R-1.
    
    # This is getting complex. Let me just try a simple randomized search
    # with a strict time limit and see if we get lucky.
    
    start_time = time.time()
    rng = np.random.RandomState(12345)
    
    # Try several random initializations and do a few iterations of
    # alternating minimization.
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random initialization
        U = rng.randn(dim_U, R) * 0.5
        V = rng.randn(dim_V, R) * 0.5
        W = rng.randn(dim_W, R) * 0.5
        
        # Alternating minimization: fix two, solve for the third
        # For fixed V, W: the equations are linear in U
        # For fixed U, W: the equations are linear in V
        # For fixed U, V: the equations are linear in W
        
        for iteration in range(20):
            if time.time() > deadline:
                break
            
            # Update U: for each (i,j), we need
            # 1 = sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) for all k
            # This gives p equations for R unknowns U[i*m+j, :]
            # Least squares solution
            
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Build the system: for each k, sum_r U[i*m+j,r] * coeff[r,k] = 1
                    # coeff[r,k] = V[j*p+k, r] * W[k*n+i, r]
                    A_sys = np.zeros((p, R))
                    for k in range(p):
                        A_sys[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    # Solve least squares
                    try:
                        U_new[i*m+j, :] = np.linalg.lstsq(A_sys, b, rcond=None)[0]
                    except:
                        pass
            
            U = U_new
            
            # Update V: for each (j,k), we need
            # 1 = sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) for all i
            # This gives n equations for R unknowns V[j*p+k, :]
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A_sys = np.zeros((n, R))
                    for i in range(n):
                        A_sys[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[j*p+k, :] = np.linalg.lstsq(A_sys, b, rcond=None)[0]
                    except:
                        pass
            
            V = V_new
            
            # Update W: for each (k,i), we need
            # 1 = sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) for all j
            # This gives m equations for R unknowns W[k*n+i, :]
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A_sys = np.zeros((m, R))
                    for j in range(m):
                        A_sys[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[k*n+i, :] = np.linalg.lstsq(A_sys, b, rcond=None)[0]
                    except:
                        pass
            
            W = W_new
        
        # Check if we found a solution
        if _verify_decomposition(U, V, W, n, m, p, R, num_tests=5, seed=42+attempt):
            return U, V, W, R
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # First, build the known rank-17 solution
    U17, V17, W17, rank17 = _build_rank17(n, m, p)
    
    # Verify the rank-17 solution
    if not _verify_decomposition(U17, V17, W17, n, m, p, rank17, num_tests=10, seed=42):
        # Should not happen, but fallback to schoolbook
        rank = 18
        dim_U = n * m
        dim_V = m * p
        dim_W = p * n
        U = np.zeros((dim_U, rank), dtype=float)
        V = np.zeros((dim_V, rank), dtype=float)
        W = np.zeros((dim_W, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find rank 16 if we have time
    if time.time() < deadline:
        result = _try_find_rank16(n, m, p, deadline)
        if result is not None:
            U, V, W, rank = result
            # Final verification
            if _verify_decomposition(U, V, W, n, m, p, rank, num_tests=20, seed=999):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank 17
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
