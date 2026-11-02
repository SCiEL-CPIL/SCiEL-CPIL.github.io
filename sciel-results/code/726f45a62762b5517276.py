"""Program to find a rank-minimal exact tensor decomposition for 2x4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    def verify(U, V, W, rank):
        """Verify tensor decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition, each of shape (4, 7)."""
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W
    
    def build_strassen_28():
        """Build the rank-28 Strassen-based decomposition."""
        U_full = np.zeros((8, 28))
        V_full = np.zeros((16, 28))
        W_full = np.zeros((8, 28))
        
        U1, V1, W1 = strassen_2x2x2()
        U_full[0:4, 0:7] = U1
        V_full[0:4, 0:7] = V1
        W_full[0:4, 0:7] = W1
        
        U2, V2, W2 = strassen_2x2x2()
        U_full[4:8, 7:14] = U2
        V_full[8:12, 7:14] = V2
        W_full[0:4, 7:14] = W2
        
        U3, V3, W3 = strassen_2x2x2()
        U_full[0:4, 14:21] = U3
        V_full[4:8, 14:21] = V3
        W_full[4:8, 14:21] = W3
        
        U4, V4, W4 = strassen_2x2x2()
        U_full[4:8, 21:28] = U4
        V_full[12:16, 21:28] = V4
        W_full[4:8, 21:28] = W4
        
        return U_full, V_full, W_full
    
    def try_rank_27_randomized(max_iter=200):
        """
        Try to find a rank-27 decomposition using randomized search.
        
        Strategy: Start from the rank-28 Strassen decomposition and try to
        merge/eliminate one rank-1 term by solving for adjustments.
        
        For M_{2,4,4}, we can exploit the structure:
        C[0,k] = sum_j A[0,j]*B[j,k]
        C[1,k] = sum_j A[1,j]*B[j,k]
        
        We can try to find a decomposition where one of the Strassen terms
        is shared or eliminated.
        """
        if time.time() > deadline:
            return None
        
        U28, V28, W28 = build_strassen_28()
        
        # Try to find a rank-27 by removing one column and adjusting others
        # The tensor T[i,j,k] = 1 for all i,j,k
        # We need: sum_{r=0}^{26} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        
        # Approach: Use the fact that we have 32 constraints (i,j,k triples)
        # and 27*(8+16+8) = 27*32 = 864 variables (with redundancy)
        # Actually, the tensor has rank 27 means we need 27 rank-1 terms.
        
        # Let's try a direct approach: fix U and V to be sparse (mostly from Strassen)
        # and solve for W, then check.
        
        # Better approach: try to find a rank-27 by perturbing the rank-28
        # and solving the resulting system.
        
        # The key insight: for n=2, we can write:
        # C[0,:] = A[0,:] * B  (row 0 of C is row 0 of A times B)
        # C[1,:] = A[1,:] * B  (row 1 of C is row 1 of A times B)
        
        # Each row of C is a linear combination of rows of B.
        # C[0,k] = sum_j A[0,j] B[j,k]
        # C[1,k] = sum_j A[1,j] B[j,k]
        
        # We can try to use 27 bilinear forms that span the space.
        
        # Let's try: use 14 terms for the "first half" and 13 for the "second half"
        # with one shared term.
        
        # Actually, let's try a more systematic approach.
        # The tensor T has shape (2,4,4) with all entries 1.
        # We want T[i,j,k] = sum_r u[i,j,r] * v[j,k,r] * w[k,i,r]
        
        # Note: w[k,i,r] depends on k and i (output index).
        
        # For a given r, define:
        # a_r[i,j] = u[i,j,r]  (2x4 matrix)
        # b_r[j,k] = v[j,k,r]  (4x4 matrix)  
        # c_r[k,i] = w[k,i,r]  (4x2 matrix)
        
        # Then T[i,j,k] = sum_r a_r[i,j] * b_r[j,k] * c_r[k,i]
        
        # For rank 27, we need 27 such triples.
        
        # Let's try to construct this by taking the Strassen rank-28 and
        # trying to eliminate one term.
        
        # Remove column 27 (the last one) and see if we can adjust the others.
        # This is a system of 32 nonlinear equations in 864-32=832 unknowns
        # (after fixing one scaling).
        
        # This is too hard to solve directly. Let's try a different approach.
        
        # Approach 2: Use the structure of n=2.
        # Write T[i,j,k] = 1 = sum_r u[i,j,r] * v[j,k,r] * w[k,i,r]
        
        # For i=0: 1 = sum_r u[0,j,r] * v[j,k,r] * w[k,0,r]
        # For i=1: 1 = sum_r u[1,j,r] * v[j,k,r] * w[k,1,r]
        
        # Let's define for each r:
        # f_r(j,k) = v[j,k,r]  (4x4)
        # g_r(j) = u[0,j,r] (row 0 of A part)
        # h_r(j) = u[1,j,r] (row 1 of A part)
        # p_r(k) = w[k,0,r] (column 0 of W part)
        # q_r(k) = w[k,1,r] (column 1 of W part)
        
        # Then:
        # 1 = sum_r g_r(j) * f_r(j,k) * p_r(k)  for all j,k
        # 1 = sum_r h_r(j) * f_r(j,k) * q_r(k)  for all j,k
        
        # This gives 32 equations.
        
        # Let's try to find a solution with rank 27.
        # We'll use a randomized initialization and Newton's method.
        
        rng = np.random.default_rng(42)
        
        for attempt in range(max_iter):
            if time.time() > deadline:
                break
            
            # Initialize with Strassen structure but random perturbations
            U = U28[:, :27].copy()
            V = V28[:, :27].copy()
            W = W28[:, :27].copy()
            
            # Add small random perturbations
            U += rng.normal(0, 0.1, U.shape)
            V += rng.normal(0, 0.1, V.shape)
            W += rng.normal(0, 0.1, W.shape)
            
            # Solve for W given U, V using the tensor equation
            # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # For each (i,k), we have: 1 = sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # This is: 1 = sum_r (sum_j U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r]
            
            # Let M_ik[r] = sum_j U[i*m+j,r] * V[j*p+k,r]
            # Then: 1 = sum_r M_ik[r] * W[k*n+i,r]
            
            # For each (i,k), this is a linear equation in W[k*n+i,:]:
            # M_ik @ W[k*n+i,:] = 1
            
            # There are 8 such equations (2*4=8), each of dimension 27.
            
            W_solved = np.zeros((8, 27))
            valid = True
            for i in range(2):
                for k in range(4):
                    # M_ik[r] = sum_j U[i*4+j, r] * V[j*4+k, r]
                    M_ik = np.zeros(27)
                    for j in range(4):
                        M_ik += U[i*4+j, :] * V[j*4+k, :]
                    # Solve M_ik @ W[k*2+i, :] = 1
                    # This is underdetermined (27 unknowns, 1 equation)
                    # Use minimum norm solution
                    if abs(np.sum(M_ik**2)) < 1e-12:
                        valid = False
                        break
                    W_solved[k*2+i, :] = M_ik / np.sum(M_ik**2)
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check if this solution actually works
            if verify(U, V, W_solved, 27):
                return U, V, W_solved
        
        return None
    
    # First, try to find rank 27
    best_result = None
    
    result_27 = try_rank_27_randomized(100)
    if result_27 is not None:
        U27, V27, W27 = result_27
        best_result = (U27, V27, W27, 27)
    else:
        # Fall back to rank 28
        U28, V28, W28 = build_strassen_28()
        if verify(U28, V28, W28, 28):
            best_result = (U28, V28, W28, 28)
        else:
            # Schoolbook fallback
            rank = n * m * p
            U_fb = np.zeros((n*m, rank))
            V_fb = np.zeros((m*p, rank))
            W_fb = np.zeros((p*n, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j, col] = 1
                        V_fb[j*p+k, col] = 1
                        W_fb[k*n+i, col] = 1
                        col += 1
            best_result = (U_fb, V_fb, W_fb, rank)
    
    U_best, V_best, W_best, rank_best = best_result
    
    return {
        "U": U_best.tolist(),
        "V": V_best.tolist(),
        "W": W_best.tolist()
    }
# EVOLVE-BLOCK-END
