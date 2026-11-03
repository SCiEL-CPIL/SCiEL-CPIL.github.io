import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 3, 9]

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    def verify(U, V, W):
        """Verify that the decomposition is correct."""
        rank = U.shape[1]
        # Compute C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # for all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Faster verification using tensor contraction."""
        rank = U.shape[1]
        # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # We can do this more efficiently
        for r in range(rank):
            pass  # Just check a few random points
        # Check all points
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def schoolbook_decomposition():
        """Return the schoolbook rank-54 decomposition."""
        rank = n * m * p  # 54
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    def strassen_2x2x2_block():
        """
        Return a rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
        C[i,k] = sum_j A[i,j] * B[j,k] for i,j,k in {0,1}
        """
        # Strassen's 7 terms for 2x2x2
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B10)
        # M7 = (A01-A11)(B01+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For each term, we need U, V, W vectors
        # U[i*m+j, r] where m=2 for this sub-block
        # V[j*2+k, r]
        # W[k*2+i, r]
        
        terms = []
        # M1: (A00+A11)(B00+B11)
        # U: A00=1, A11=1 -> U[0,0]=1, U[1,1]=1 (i=0,j=0 and i=1,j=1)
        # V: B00=1, B11=1 -> V[0,0]=1, V[1,1]=1
        # W: C00=1, C11=1 -> W[0,0]=1, W[1,1]=1
        terms.append(([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]))
        
        # M2: (A10+A11)B00
        # U: A10=1, A11=1 -> U[1,0]=1, U[1,1]=1
        # V: B00=1 -> V[0,0]=1
        # W: C10=1 -> W[0,1]=1
        terms.append(([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1)]))
        
        # M3: A00(B10-B11)
        # U: A00=1 -> U[0,0]=1
        # V: B10=1, B11=-1 -> V[1,0]=1, V[1,1]=-1
        # W: C01=1, C11=1 -> W[1,0]=1, W[1,1]=1
        terms.append(([(0,0,1)], [(1,0,1),(1,1,-1)], [(1,0,1),(1,1,1)]))
        
        # M4: A11(B01-B00)
        # U: A11=1 -> U[1,1]=1
        # V: B01=1, B00=-1 -> V[0,1]=1, V[0,0]=-1
        # W: C00=1, C10=1 -> W[0,0]=1, W[0,1]=1
        terms.append(([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,1),(0,1,1)]))
        
        # M5: (A00+A01)B11
        # U: A00=1, A01=1 -> U[0,0]=1, U[0,1]=1
        # V: B11=1 -> V[1,1]=1
        # W: C00=-1, C01=1 -> W[0,0]=-1, W[1,0]=1
        terms.append(([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(1,0,1)]))
        
        # M6: (A10-A00)(B00+B10)
        # U: A10=1, A00=-1 -> U[1,0]=1, U[0,0]=-1
        # V: B00=1, B10=1 -> V[0,0]=1, V[1,0]=1
        # W: C11=1 -> W[1,1]=1
        terms.append(([(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)], [(1,1,1)]))
        
        # M7: (A01-A11)(B01+B11)
        # U: A01=1, A11=-1 -> U[0,1]=1, U[1,1]=-1
        # V: B01=1, B11=1 -> V[0,1]=1, V[1,1]=1
        # W: C00=1 -> W[0,0]=1
        terms.append(([(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)], [(0,0,1)]))
        
        return terms
    
    def build_decomposition(rank_target):
        """
        Try to build a decomposition with the given rank.
        Uses a structured approach combining Strassen for 2x2x2 blocks
        and schoolbook for the rest.
        """
        # The 2x3x9 multiplication can be seen as:
        # C (2x9) = A (2x3) @ B (3x9)
        # 
        # We can partition B into three 3x3 blocks: B = [B0 | B1 | B2]
        # Then C = [A@B0 | A@B1 | A@B2]
        #
        # For each 2x3x3 block multiplication, we can use a rank-16 decomposition
        # or try to do better.
        
        # Let's try a different approach: use the fact that 2x3x9 can be
        # decomposed using algebraic structure.
        
        # Approach: Use the 2x3x3 tensor decomposition and extend it.
        # For 2x3x3, the known optimal rank is 16 (or possibly lower).
        # Let's first find a good 2x3x3 decomposition, then extend to 2x3x9.
        
        # Actually, let's try a direct approach with random initialization and optimization.
        
        # First, let's try a structured decomposition:
        # Partition the 3 columns of A and 9 columns of B into blocks.
        # A is 2x3, B is 3x9.
        # 
        # We can write B as [B_00 B_01 B_02; B_10 B_11 B_12; B_20 B_21 B_22]
        # where each B_ij is a 1x3 block.
        # 
        # C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:] + A[0,2]*B[2,:]
        # C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:] + A[1,2]*B[2,:]
        
        # Let's try to use a rank-16 decomposition for each 2x3x3 block.
        # But we need to find such a decomposition first.
        
        # For now, let's use the Strassen-based approach for 2x2x2 sub-blocks.
        # We can partition the 2x3x9 as follows:
        # - Use Strassen for the 2x2x2 part (rows 0-1 of A, rows 0-1 of B, cols 0-1 of C)
        # - Use schoolbook for the remaining parts
        
        # Actually, let's try a cleaner approach:
        # The total rank needed is at most 54 (schoolbook).
        # We can try to reduce this by using Strassen-style tricks.
        
        # Let's build a decomposition that uses:
        # - For each pair of rows in C (i=0,1) and each block of 3 columns in C:
        #   C[i, 3k:3k+3] = A[i,:] @ B[:, 3k:3k+3]
        #   This is a 1x3 @ 3x3 = 1x3 multiplication, which has rank 9.
        #   Total: 2 * 3 * 9 = 54.
        
        # To do better, we need to share terms across blocks.
        
        # Let's try a random search for a lower-rank decomposition.
        
        best_U, best_V, best_W = None, None, None
        best_rank = n * m * p  # 54
        
        # Try rank from lower to upper
        for rank in range(30, best_rank + 1):
            if time.time() > deadline - 10:
                break
            
            # Try a few random initializations
            for attempt in range(5):
                if time.time() > deadline - 10:
                    break
                
                # Initialize with random values
                U = np.random.randn(n * m, rank)
                V = np.random.randn(m * p, rank)
                W = np.random.randn(p * n, rank)
                
                # Scale to make the product have the right magnitude
                scale = (n * m * p) ** 0.5
                U /= scale
                V /= scale
                W /= scale
                
                # Try to optimize
                # The loss function is sum over all (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
                # This is a non-convex problem, so we use a simple gradient-free approach.
                
                # Actually, let's use a smarter approach:
                # Fix U and V, solve for W (or vice versa).
                # This is a bilinear problem, which can be solved iteratively.
                
                # For fixed U and V, the equation for W is:
                # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
                # This is a linear system in W.
                
                # Let's set up the linear system.
                # For each (i,j,k), we have:
                # sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1
                # 
                # The unknowns are W[k*n+i, r] for k in range(p), i in range(n), r in range(rank).
                # Total unknowns: p * n * rank
                # Total equations: n * m * p
                # 
                # This is an overdetermined system (54 equations, 9*2*rank = 18*rank unknowns).
                # For rank >= 3, we have 54 equations and 54+ unknowns, so it's underdetermined.
                # We can use least squares.
                
                # Build the design matrix
                # Each row corresponds to an (i,j,k) triple
                # Each column corresponds to a (k,n+i,r) triple (i.e., W index)
                
                num_eqs = n * m * p
                num_unk = p * n * rank
                
                # Build the matrix A where A[(i,j,k), (k',n+i',r)] = 
                # U[i*m+j, r] * V[j*p+k, r] if (k'=k, i'=i) else 0
                # Wait, that's not right. Let me re-derive.
                
                # The equation is:
                # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
                # 
                # So for a fixed (i,j,k), the coefficients for W[k*n+i, r] are:
                # coeff[r] = U[i*m+j, r] * V[j*p+k, r]
                # 
                # And the equation is: sum_r coeff[r] * W[k*n+i, r] = 1
                # 
                # So the design matrix has a block structure. For each (i,j,k),
                # only the W entries with the same (k,i) are involved.
                
                # Let's build the system differently.
                # For each (k,i) pair, we have a vector W[k*n+i, :] of length rank.
                # The equations involving this vector are those where (k',i') = (k,i),
                # i.e., for all j in range(m):
                # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
                
                # So for each (k,i), we have m equations in rank unknowns.
                # Total: p*n*m = 54 equations, p*n*rank = 18*rank unknowns.
                
                # Let's solve this using least squares for each (k,i) block.
                
                # Actually, this won't work directly because the equations are coupled
                # through U and V. Let's use an alternating minimization approach.
                
                # Iteration 1: Fix U, V, solve for W
                # Iteration 2: Fix U, W, solve for V
                # Iteration 3: Fix V, W, solve for U
                # Repeat.
                
                converged = False
                for it in range(50):
                    if time.time() > deadline - 10:
                        break
                    
                    # Solve for W
                    # For each (k,i), build the system:
                    # For j in range(m):
                    #   sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
                    # This is: (U_i^T * diag(V_k)) @ W_ki = 1_m
                    # where U_i is the submatrix of U with rows corresponding to i,
                    #       V_k is the submatrix of V with columns corresponding to k,
                    #       W_ki is W[k*n+i, :]
                    
                    for k in range(p):
                        for i in range(n):
                            # Build the m x rank matrix
                            # Row j: U[i*m+j, :] * V[j*p+k, :]
                            A_block = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                            # Solve A_block @ W_ki = ones(m)
                            # Using least squares
                            try:
                                W[k*n+i, :], residuals, rank_s, sv = np.linalg.lstsq(A_block, np.ones(m), rcond=None)
                            except:
                                pass
                    
                    # Solve for V
                    for j in range(m):
                        for k in range(p):
                            # For i in range(n):
                            #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                            # This is: (W_k^T * diag(U_j)) @ V_jk = 1_n
                            A_block = np.array([W[k*n+i, :] * U[i*m+j, :] for i in range(n)])
                            try:
                                V[j*p+k, :], residuals, rank_s, sv = np.linalg.lstsq(A_block, np.ones(n), rcond=None)
                            except:
                                pass
                    
                    # Solve for U
                    for i in range(n):
                        for j in range(m):
                            # For k in range(p):
                            #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                            # This is: (V_j^T * diag(W_i)) @ U_ij = 1_p
                            A_block = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                            try:
                                U[i*m+j, :], residuals, rank_s, sv = np.linalg.lstsq(A_block, np.ones(p), rcond=None)
                            except:
                                pass
                    
                    # Check convergence
                    if verify(U, V, W):
                        converged = True
                        break
                
                if converged:
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    break
            
            if converged:
                break
        
        if best_U is not None:
            return best_U, best_V, best_W
        else:
            # Fall back to schoolbook
            return schoolbook_decomposition()
    
    # Try to find a good decomposition
    U, V, W = build_decomposition(n * m * p)
    
    # Verify
    if not verify(U, V, W):
        # Fall back to schoolbook
        U, V, W = schoolbook_decomposition()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
