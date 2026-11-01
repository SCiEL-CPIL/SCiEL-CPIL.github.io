import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin for 150s limit

    n, m, p = 2, 4, 4
    dim_U = n * m  # 8
    dim_V = m * p  # 16
    dim_W = p * n  # 8
    total_entries = n * m * p  # 64

    # Build the full matrix-multiplication tensor T as a 3D array T[i,j,k]
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to C[i,k]
    # In the standard convention: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] = 1 for all valid (i,j,k) where j is the
    # shared index. Actually, the tensor is:
    # T_{i,j,k} = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
    # This is always 1 for valid indices. So T is the "matrix multiplication tensor"
    # with T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait - that's not right either. The tensor encodes the bilinear map:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # As a tensor: T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C * delta(j connects i,k)
    # Actually the standard matrix multiplication tensor <n x m x p> is:
    # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i ⊗ e_j ⊗ e_k
    # where the bilinear form is: <T, A⊗B⊗C> = C^T (AB)
    # More precisely, the tensor has T[i,j,k] = 1 for all valid indices.
    # The decomposition T = sum_r U_r ⊗ V_r ⊗ W_r means:
    # sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k] = 1 for all i,j,k
    # But that's not right. Let me reconsider.
    
    # The correct formulation: The matrix multiplication tensor T satisfies:
    # For the bilinear map (A, B) -> C where C = A*B:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is defined such that:
    # T[i, j, k] = 1 if and only if the (i,j) element of A and (j,k) element of B
    # multiply to contribute to C[i,k].
    # In the standard tensor decomposition framework:
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where the bilinear form evaluation is:
    # C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) * (sum_k V[j,k,r] B[j,k]) * W[k,i,r]
    # Hmm, this is getting confused. Let me use the standard formulation.
    
    # Standard: The matrix multiplication tensor <n x m x p> is a 3-tensor T of size
    # n x m x p with T[i,j,k] = 1 for all valid indices. A rank-R decomposition is:
    # T[i,j,k] = sum_{r=1}^R U[i,r] * V[j,r] * W[k,r]
    # where U is n x R, V is m x R, W is p x R.
    # But wait, that would mean T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r] = 1 for ALL i,j,k.
    # That's only possible if the tensor is all-ones, which it's not.
    
    # I'm confusing myself. Let me use the correct formulation from the task:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition satisfies:
    # For all A (n x m) and B (m x p), C = A @ B:
    # C[i,k] = sum_r (sum_{i',j} U[i*m+j, r] * A[i',j]) * (sum_{j',k'} V[j'*p+k', r] * B[j',k']) * W[k*n+i, r]
    # Wait, that's not dimensionally right either.
    
    # Let me re-read the task: "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # And the check in the current program:
    # A_proj = np.sum(U[:, r] * A.flatten())  # scalar
    # B_proj = np.sum(V[:, r] * B.flatten())  # scalar
    # C_calc += W[:, r].reshape(p, n).T * (A_proj * B_proj)
    
    # So the decomposition is:
    # C = sum_r (U[:,r]^T A_flat) * (V[:,r]^T B_flat) * W[:,r].reshape(p,n).T
    # where A_flat is A flattened (row-major), B_flat is B flattened (row-major)
    # and W[:,r].reshape(p,n).T is an n x p matrix.
    
    # This means: C[i,k] = sum_r U_flat[i*m+j indexed] ... 
    # Let me just verify with the check function.
    
    def check_decomposition(U, V, W, trials=5):
        """Verify that U, V, W correctly decompose the matrix multiplication."""
        for _ in range(trials):
            A = np.random.randint(-10, 10, (n, m)).astype(float)
            B = np.random.randint(-10, 10, (m, p)).astype(float)
            C_expected = A @ B
            
            C_calc = np.zeros((n, p))
            for r in range(U.shape[1]):
                A_proj = np.dot(U[:, r], A.flatten())
                B_proj = np.dot(V[:, r], B.flatten())
                C_calc += (A_proj * B_proj) * W[:, r].reshape(p, n).T
            
            if not np.allclose(C_calc, C_expected, atol=1e-9):
                return False
        return True

    # First, construct the rank-28 Strassen solution as baseline
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 matrix multiplication, each 4x7 (flattened)."""
        # U: 2x2 flattened to 4x7
        # V: 2x2 flattened to 4x7  
        # W: 2x2 flattened to 4x7 (with k*n+i convention, n=2)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen's 7 products for 2x2:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b10)
        # M7 = (a01-a11)(b01+b11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # U index: i*m+j = i*2+j (for 2x2)
        # V index: j*p+k = j*2+k (for 2x2)
        # W index: k*n+i = k*2+i (for 2x2, n=2)
        
        # M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
        U[0, 0] = 1; U[3, 0] = 1  # A[0,0], A[1,1]
        V[0, 0] = 1; V[3, 0] = 1  # B[0,0], B[1,1]
        # C00 += 1, C11 -= 1
        W[0*2+0, 0] = 1   # k=0,i=0 -> C[0,0]
        W[1*2+1, 0] = -1  # k=1,i=1 -> C[1,1]
        
        # M2: (A[1,0]+A[1,1]) * B[0,0]
        U[2, 1] = 1; U[3, 1] = 1  # A[1,0], A[1,1]
        V[0, 1] = 1  # B[0,0]
        # C10 += 1, C11 -= 1
        W[0*2+1, 1] = 1   # k=0,i=1 -> C[1,0]
        W[1*2+1, 1] = -1  # k=1,i=1 -> C[1,1]
        
        # M3: A[0,0] * (B[1,0]-B[1,1])
        U[0, 2] = 1  # A[0,0]
        V[2, 2] = 1; V[3, 2] = -1  # B[1,0], B[1,1]
        # C01 += 1, C11 += 1
        W[1*2+0, 2] = 1   # k=1,i=0 -> C[0,1]
        W[1*2+1, 2] = 1   # k=1,i=1 -> C[1,1]
        
        # M4: A[1,1] * (B[0,1]-B[0,0])
        U[3, 3] = 1  # A[1,1]
        V[1, 3] = 1; V[0, 3] = -1  # B[0,1], B[0,0]
        # C00 += 1, C10 += 1
        W[0*2+0, 3] = 1   # k=0,i=0 -> C[0,0]
        W[0*2+1, 3] = 1   # k=0,i=1 -> C[1,0]
        
        # M5: (A[0,0]+A[0,1]) * B[1,1]
        U[0, 4] = 1; U[1, 4] = 1  # A[0,0], A[0,1]
        V[3, 4] = 1  # B[1,1]
        # C00 -= 1, C01 += 1
        W[0*2+0, 4] = -1  # k=0,i=0 -> C[0,0]
        W[1*2+0, 4] = 1   # k=1,i=0 -> C[0,1]
        
        # M6: (A[1,0]-A[0,0]) * (B[0,0]+B[1,0])
        U[2, 5] = 1; U[0, 5] = -1  # A[1,0], A[0,0]
        V[0, 5] = 1; V[2, 5] = 1  # B[0,0], B[1,0]
        # C11 += 1
        W[1*2+1, 5] = 1   # k=1,i=1 -> C[1,1]
        
        # M7: (A[0,1]-A[1,1]) * (B[0,1]+B[1,1])
        U[1, 6] = 1; U[3, 6] = -1  # A[0,1], A[1,1]
        V[1, 6] = 1; V[3, 6] = 1  # B[0,1], B[1,1]
        # C00 += 1
        W[0*2+0, 6] = 1   # k=0,i=0 -> C[0,0]
        
        return U, V, W

    U2, V2, W2 = strassen_2x2x2()
    
    # Build rank-28 solution by tiling the 2x2x2 Strassen into 2x4x4
    # Split j (middle dim, size 4) into j1={0,1}, j2={2,3}
    # Split k (output dim, size 4) into k1={0,1}, k2={2,3}
    # Four blocks: (jb, kb) for jb in {0,1}, kb in {0,1}
    
    R_base = 28
    U_base = np.zeros((dim_U, R_base))
    V_base = np.zeros((dim_V, R_base))
    W_base = np.zeros((dim_W, R_base))
    
    for jb in range(2):
        for kb in range(2):
            r_off = (jb * 2 + kb) * 7
            j_off = jb * 2
            k_off = kb * 2
            
            # U: index i*m + j = i*4 + (j_off + j)
            for i in range(2):
                for j in range(2):
                    U_base[i * 4 + j_off + j, r_off:r_off+7] = U2[i * 2 + j, :]
            
            # V: index j*p + k = (j_off + j)*4 + (k_off + k)
            for j in range(2):
                for k in range(2):
                    V_base[(j_off + j) * 4 + k_off + k, r_off:r_off+7] = V2[j * 2 + k, :]
            
            # W: index k*n + i = (k_off + k)*2 + i
            for k in range(2):
                for i in range(2):
                    W_base[(k_off + k) * 2 + i, r_off:r_off+7] = W2[k * 2 + i, :]
    
    # Verify base solution
    if not check_decomposition(U_base, V_base, W_base):
        # Fallback: schoolbook rank 32
        R_school = n * m * p
        U_school = np.zeros((dim_U, R_school))
        V_school = np.zeros((dim_V, R_school))
        W_school = np.zeros((dim_W, R_school))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i * m * p + j * p + k
                    U_school[i * m + j, col] = 1
                    V_school[j * p + k, col] = 1
                    W_school[k * n + i, col] = 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Now attempt rank 27 via global sparse linear algebra approach
    # Key insight: Instead of treating the four 2x2x2 blocks independently,
    # we model the full tensor and look for linear dependencies.
    
    # The tensor equation is:
    # For all i in [0,2), j in [0,4), k in [0,4):
    # sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*2+i, r] = 1
    
    # This is a system of 32 equations (2*4*4=32) in the bilinear products.
    # For rank R, we have R bilinear terms, each contributing to all 32 equations.
    
    # Strategy: Try to find a rank-27 solution by:
    # 1. Start with the rank-28 solution
    # 2. Try to express one term as a linear combination of others
    # 3. Use alternating minimization to refine
    
    # Actually, let's try a different approach: 
    # Use the fact that the tensor can be viewed as a linear map.
    # Fix A, then C = A*B is linear in B.
    # The map B -> C is a linear map from R^16 to R^8.
    # Its matrix representation (for fixed A) is an 8x16 matrix.
    # The rank of this linear map is at most 8 (since C is 2x4).
    
    # But we need a bilinear decomposition, not a linear one.
    
    # Let's try: Can we find rank 27 by exploiting that some Strassen terms
    # across different blocks might be linearly dependent?
    
    # For rank 27, we need:
    # T[i,j,k] = sum_{r=1}^{27} U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
    
    # This is a system of 32 equations. With 27 terms, we have 27*8 + 27*16 + 27*8 = 216+432+216 = 864 unknowns.
    # 32 equations, 864 unknowns - highly underdetermined.
    # But the equations are nonlinear (products of U, V, W).
    
    # Let's try a structured approach:
    # Assume the rank-27 solution has the same block structure as rank-28
    # but with one term shared across blocks.
    
    # Specifically, in the rank-28 solution, we have 4 groups of 7 terms.
    # For rank 27, we need to save one term. 
    # Idea: Make one of the 28 terms a linear combination of others.
    
    # Let's try: Can term (jb=0, kb=0, r=0) [first Strassen term of first block]
    # be expressed as a combination of terms from other blocks?
    
    # This is unlikely to work with simple linear combinations, but let's try
    # a more systematic approach: alternating minimization.
    
    # Alternating minimization for rank-R tensor decomposition:
    # 1. Initialize U, V, W randomly (or from rank-28 solution with one term removed)
    # 2. Fix V and W, solve for U (linear system)
    # 3. Fix U and W, solve for V (linear system)
    # 4. Fix U and V, solve for W (linear system)
    # 5. Repeat until convergence
    
    # The linear systems:
    # For fixed V, W: U[i,j,r] * (V[j,k,r] * W[k,i,r]) = T[i,j,k] for all i,j,k
    # This is: for each r, U[:,r] * (V[:,r] * W[:,r] reshaped) = T flattened
    # Actually, let's set it up properly.
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For fixed V, W: 
    #   sum_r U[i,j,r] * (V[j,k,r] * W[k,i,r]) = T[i,j,k]
    #   Let M_r[i,j] = V[j,k,r] * W[k,i,r] ... no, this doesn't work because
    #   V and W depend on k which is also an index of T.
    
    # Let me reformulate. Flatten the tensor:
    # T_flat[i*m*p + j*p + k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For fixed V, W:
    # T_flat = sum_r U[:,r] ⊙ (V[:,r] ⊙ W_reshaped[:,r])
    # where ⊙ is element-wise and W_reshaped maps W[k*n+i, r] to the right position.
    
    # Actually, for each r, define:
    # S_r[i,j,k] = V[j*p+k, r] * W[k*n+i, r]
    # Then T[i,j,k] = sum_r U[i*m+j, r] * S_r[i,j,k]
    # This is a linear system in U for fixed V, W.
    
    # Similarly for V and W.
    
    def solve_for_U(V, W, T_flat, R):
        """Solve for U given V, W, T_flat, R."""
        # T_flat[i*m*p + j*p + k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For each (i,j): T_flat[i*m*p + j*p + k] for all k
        # = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a system of p equations in R unknowns (U[i*m+j, 0..R-1])
        # for each (i,j) pair. There are n*m such pairs, each with p equations.
        
        U = np.zeros((dim_U, R))
        for i in range(n):
            for j in range(m):
                # Unknowns: U[i*m+j, r] for r = 0..R-1
                # Equations: for k = 0..p-1:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (T=1 for all valid)
                # Note: T[i,j,k] = 1 for all valid i,j,k in matrix multiplication tensor
                
                # Build the system: A_sys * u = b
                # A_sys[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # b[k] = 1 (since T[i,j,k] = 1)
                
                A_sys = np.zeros((p, R))
                b = np.ones(p)
                for k in range(p):
                    for r in range(R):
                        A_sys[k, R-1-r] = V[j*p+k, r] * W[k*n+i, r]
                    # Wait, I need to be careful with column ordering.
                    # Let me just use the natural ordering.
                
                # Rebuild properly
                A_sys = np.zeros((p, R))
                for k in range(p):
                    for r in range(R):
                        A_sys[k, r] = V[j*p+k, r] * W[k*n+i, r]
                
                # Solve A_sys * u = b (least squares if overdetermined)
                if R <= p:
                    # Underdetermined or square - use lstsq
                    u, _, _, _ = np.linalg.lstsq(A_sys, b, rcond=None)
                else:
                    u, _, _, _ = np.linalg.lstsq(A_sys, b, rcond=None)
                
                U[i*m+j, :] = u
        
        return U
    
    def solve_for_V(U, W, T_flat, R):
        """Solve for V given U, W, T_flat, R."""
        V = np.zeros((dim_V, R))
        for j in range(m):
            for k in range(p):
                # Unknowns: V[j*p+k, r] for r = 0..R-1
                # Equations: for i = 0..n-1:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                
                A_sys = np.zeros((n, R))
                b = np.ones(n)
                for i in range(n):
                    for r in range(R):
                        A_sys[i, r] = U[i*m+j, r] * W[k*n+i, r]
                
                u, _, _, _ = np.linalg.lstsq(A_sys, b, rcond=None)
                V[j*p+k, :] = u
        
        return V
    
    def solve_for_W(U, V, T_flat, R):
        """Solve for W given U, V, T_flat, R."""
        W = np.zeros((dim_W, R))
        for k in range(p):
            for i in range(n):
                # Unknowns: W[k*n+i, r] for r = 0..R-1
                # Equations: for j = 0..m-1:
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                
                A_sys = np.zeros((m, R))
                b = np.ones(m)
                for j in range(m):
                    for r in range(R):
                        A_sys[j, r] = U[i*m+j, r] * V[j*p+k, r]
                
                u, _, _, _ = np.linalg.lstsq(A_sys, b, rcond=None)
                W[k*n+i, :] = u
        
        return W
    
    def alternating_minimization(U_init, V_init, W_init, R, max_iters=200, tol=1e-10):
        """Run alternating minimization to find rank-R decomposition."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        for it in range(max_iters):
            if time.time() - start_time > deadline - 10:
                break
            
            U_old = U.copy()
            V_old = V.copy()
            W_old = W.copy()
            
            U = solve_for_U(V, W, None, R)
            V = solve_for_V(U, W, None, R)
            W = solve_for_W(U, V, None, R)
            
            # Check convergence
            change = (np.linalg.norm(U - U_old) + np.linalg.norm(V - V_old) + 
                     np.linalg.norm(W - W_old)) / (np.linalg.norm(U) + np.linalg.norm(V) + 
                     np.linalg.norm(W) + 1e-15)
            
            if change < tol:
                break
        
        return U, V, W
    
    # Try rank 27
    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = 28
    
    R_try = 27
    if time.time() - start_time < deadline - 30:
        # Initialize from rank-28 solution: drop the last term
        U_init = U_base[:, :R_try].copy()
        V_init = V_base[:, :R_try].copy()
        W_init = W_base[:, :R_try].copy()
        
        # Add small random perturbation to escape local minima
        U_init += 0.01 * np.random.randn(dim_U, R_try)
        V_init += 0.01 * np.random.randn(dim_V, R_try)
        W_init += 0.01 * np.random.randn(dim_W, R_try)
        
        U27, V27, W27 = alternating_minimization(U_init, V_init, W_init, R_try, max_iters=100)
        
        if check_decomposition(U27, V27, W27, trials=3):
            best_U, best_V, best_W = U27, V27, W27
            best_rank = 27
            print(f"Found rank 27 solution!")
    
    # Try rank 26
    R_try = 26
    if best_rank > 26 and time.time() - start_time < deadline - 20:
        U_init = U_base[:, :R_try].copy()
        V_init = V_base[:, :R_try].copy()
        W_init = W_base[:, :R_try].copy()
        
        U_init += 0.01 * np.random.randn(dim_U, R_try)
        V_init += 0.01 * np.random.randn(dim_V, R_try)
        W_init += 0.01 * np.random.randn(dim_W, R_try)
        
        U26, V26, W26 = alternating_minimization(U_init, V_init, W_init, R_try, max_iters=80)
        
        if check_decomposition(U26, V26, W26, trials=3):
            best_U, best_V, best_W = U26, V26, W26
            best_rank = 26
            print(f"Found rank 26 solution!")
    
    # If we found a lower rank, try to clean up to integer/half-integer values
    if best_rank < 28:
        # Round to nearest half-integer
        best_U = np.round(best_U * 2) / 2
        best_V = np.round(best_V * 2) / 2
        best_W = np.round(best_W * 2) / 2
        
        if not check_decomposition(best_U, best_V, best_W, trials=5):
            # Rounding broke it, keep the original
            pass
    
    # Final verification
    if not check_decomposition(best_U, best_V, best_W, trials=5):
        # Fall back to rank 28
        best_U, best_V, best_W = U_base, V_base, W_base
        best_rank = 28
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
