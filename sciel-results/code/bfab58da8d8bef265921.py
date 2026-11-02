import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = 110  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # M_{1,1,2}: rank 2, U=[[1,1]], V=[[1,0],[0,1]], W=[[1,0],[0,1]]
    # M_{3,4,4}: use Strassen on padded 4x4 * 4x4, restrict to first 3 rows
    # Strassen rank-7 for 4x4*4x4, then tensor product with M_{1,1,2} gives rank 14
    
    # Strassen's 7 products for 2x2*2x2:
    # m1 = (a+b)(e+f)
    # m2 = (c+d)(e+f)
    # m3 = a(h-f)
    # m4 = d(g-e)
    # m5 = (a+e)(h+f)
    # m6 = (c+a)(f-h)
    # m7 = (b+d)(g-c)
    #
    # Result:
    # p11 = m1+m4-m5+m7
    # p12 = m5-m1+m2+m6
    # p21 = m1-m3+m4+m6
    # p22 = -m3+m2+m5+m7
    
    # For 4x4*4x4, we split into 2x2 blocks and apply Strassen recursively.
    # But actually, for 4x4*4x4, we can use Strassen directly on the 4x4 blocks
    # by viewing them as 2x2 blocks of 2x2 matrices, giving rank 7*7=49.
    # That's too high. Let me think differently.
    
    # Actually, the standard approach: M_{4,4,4} has rank 7 (Strassen for 2x2, 
    # but 4x4*4x4 requires more). Wait, Strassen is for 2x2*2x2 -> rank 7.
    # For 4x4*4x4, we need rank 7^2 = 49 using recursive Strassen.
    
    # Let me reconsider. The tensor M_{3,4,4} for 3x4 * 4x4 -> 3x4.
    # Schoolbook rank = 3*4*4 = 48.
    # 
    # Better approach: M_{3,4,8} can be decomposed as:
    # Split the 8 columns into two groups of 4: M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # 
    # For M_{3,4,4}: multiply 3x4 by 4x4 to get 3x4.
    # We can view this as: for each of 3 rows, compute (1x4) * (4x4) -> (1x4)
    # M_{1,4,4} has schoolbook rank 16.
    # 
    # Can we do better for M_{1,4,4}? 
    # M_{1,4,4} = M_{1,2,2} ⊗ M_{1,2,2} (splitting 4 into 2+2 for both input and output)
    # M_{1,2,2}: 1x2 * 2x2 -> 1x2, rank 4 (schoolbook)
    # So M_{1,4,4} rank = 4*4 = 16. No improvement.
    
    # Alternative: Use the fact that M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # and find the minimal rank for M_{3,4,4}.
    # 
    # M_{3,4,4}: A is 3x4, B is 4x4, C is 3x4.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Pad A to 4x4 by adding a zero row: A' = [A; 0_4]
    # Then A'*B gives 4x4, and C is the first 3 rows.
    # M_{4,4,4} has rank 7 (Strassen).
    # M_{3,4,4} is a sub-tensor, so rank <= 7? 
    # Actually no - the sub-tensor rank can be larger. But we can use the same
    # 7 rank-1 tensors from Strassen and just ignore the 4th row of output.
    # So M_{3,4,4} has rank <= 7.
    
    # Wait, that's not quite right either. The Strassen decomposition gives:
    # T4[i,j,k] = sum_c U4[i,j,c] * V4[j,k,c] * W4[k,c] for 4x4*4x4
    # For M_{3,4,4}, we need T3[i,j,k] = T4[i,j,k] for i<3, and we don't care about i=3.
    # We can use the same U4, V4, W4 but with U3[i,j,c] = U4[i,j,c] for i<3.
    # This gives a valid rank-7 decomposition for M_{3,4,4}!
    
    # So M_{3,4,4} has rank <= 7.
    # M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2} has rank <= 7 * 2 = 14.
    
    # Now let me construct the Strassen rank-7 decomposition for 4x4*4x4.
    # Actually, Strassen is for 2x2*2x2. For 4x4*4x4, we need to apply
    # Strassen at a higher level.
    
    # Let me use the recursive Strassen: M_{4,4,4} = M_{2,2,2} ⊗ M_{2,2,2}
    # No, that gives M_{4,4,4} with rank 7*7=49.
    
    # I think I'm confusing myself. Let me start fresh.
    
    # The multiplication tensor M_{n,m,p} has entries T[i,j,k] = 1 if this is
    # the (i,j,k) entry in the matrix multiplication A(n*m) * B(m*p) -> C(n*p).
    # T[i,j,k] = delta_{i,j,k} where the multiplication formula is C[i,k] = sum_j A[i,j]*B[j,k]
    
    # For M_{3,4,8}: T[i,j,k] = 1 for i in [0,2], j in [0,3], k in [0,7]
    # The tensor has 3*4*8 = 96 nonzero entries (one for each (i,j,k) triple).
    
    # Schoolbook: rank 96, each term picks one (i,j,k).
    
    # Strassen for 2x2*2x2: rank 7.
    # M_{2,2,2} = sum_{c=1}^7 u_c ⊗ v_c ⊗ w_c where:
    # u_c, v_c, w_c are specific linear combinations.
    
    # For our problem, let me try a direct construction.
    
    # M_{3,4,8}: 
    # Split j (the shared dimension, size 4) into j1 (size 2) + j2 (size 2)
    # Split k (output columns, size 8) into k1 (size 4) + k2 (size 4)
    # 
    # C[i,k1] = sum_{j1} A[i,j1]*B[j1,k1] + sum_{j2} A[i,j2]*B[j2,k1]
    # C[i,k2] = sum_{j1} A[i,j1]*B[j1,k2] + sum_{j2} A[i,j2]*B[j2,k2]
    
    # This is getting complex. Let me just implement a randomized search
    # for low-rank decompositions, starting from rank 14 upward.
    
    # Actually, let me try a smarter approach. Use the tensor product structure.
    
    # M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # 
    # M_{1,1,2}: U=[[1,1]], V=[[1,0],[0,1]], W=[[1,0],[0,1]]
    # 
    # For M_{3,4,4}, let me try to find a rank-7 decomposition using Strassen.
    # 
    # Actually, I realize the issue. Strassen gives rank 7 for M_{2,2,2}, not M_{4,4,4}.
    # M_{4,4,4} has minimum rank at most 49 (recursive Strassen) but likely less.
    
    # Let me just do a practical search. Start with schoolbook (96) and try to reduce.
    
    # Given time constraints, let me implement a hybrid approach:
    # 1. Try known structured decompositions
    # 2. Fall back to randomized search
    
    # Known: M_{3,4,8} can be done with rank at most 3*4*8/2 = 48 using 
    # simple blocking, but likely much better.
    
    # Let me try rank 24: split into 2x2 blocks in the j dimension.
    # M_{3,4,8} with j split into two 2s:
    # C[i,k] = sum_{j1=0}^1 A[i,j1]*B[j1,k] + sum_{j2=0}^1 A[i,j2+2]*B[j2+2,k]
    # 
    # For each (i,k), this is a sum of 4 products. But we can use Strassen-like
    # tricks to reduce the number of independent products.
    
    # Given the complexity, let me implement a practical randomized approach
    # with a good initial guess and refinement.
    
    best_rank = n * m * p  # 96
    best_result = None
    
    # First, let's try a structured rank-48 decomposition (blocking)
    # Split j into two halves of size 2
    # For each half, we have M_{3,2,8} which has schoolbook rank 3*2*8=48
    # But that's the same as before.
    
    # Let me try a different structured approach:
    # M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
    # If I can find M_{3,4,4} with rank R, total rank is 2R.
    
    # For M_{3,4,4}: 3x4 * 4x4 -> 3x4
    # Schoolbook: 3*4*4 = 48
    # 
    # Can I do better? Let's try rank 24 for M_{3,4,4}.
    # That would give rank 48 for M_{3,4,8}.
    
    # Actually, let me just do a well-optimized randomized search.
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    # Try structured decompositions first
    # Rank 48: simple blocking
    if time.time() - start_time < deadline:
        rank = 48
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i + j + k) % 2 == 0:
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
        # This doesn't work in general. Let me use proper schoolbook.
        
        # Proper schoolbook for rank 96
        U = np.zeros((n*m, best_rank))
        V = np.zeros((m*p, best_rank))
        W = np.zeros((p*n, best_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = (U, V, W)
    
    # Now try to find better decompositions
    # Use randomized approach with refinement
    
    for target_rank in [32, 24, 16, 14, 12, 10, 8]:
        if time.time() - start_time > deadline:
            break
        
        found = False
        for attempt in range(50):
            if time.time() - start_time > deadline:
                break
            
            # Random initialization with small integers
            np.random.seed(attempt * 1000 + target_rank)
            U = np.random.randint(-2, 3, size=(n*m, target_rank)).astype(float)
            V = np.random.randint(-2, 3, size=(m*p, target_rank)).astype(float)
            
            # Compute M[i,k,c] = sum_j U[i*m+j,c] * V[j*p+k,c]
            M = np.zeros((n, p, target_rank))
            for i in range(n):
                for k in range(p):
                    for c in range(target_rank):
                        M[i, k, c] = sum(U[i*m+j, c] * V[j*p+k, c] for j in range(m))
            
            # Solve for W: M[i,k,:] dot W[k*n+i,:] = 1
            W = np.zeros((p*n, target_rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    mv = M[i, k, :]
                    if np.linalg.norm(mv) < 1e-10:
                        valid = False
                        break
                    # Minimum norm solution: W = mv / (mv dot mv)
                    W[k*n+i, :] = mv / np.dot(mv, mv)
            if not valid:
                continue
            
            # Round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            
            if verify_decomposition(U_r, V_r, W_r, target_rank):
                best_rank = target_rank
                best_result = (U_r, V_r, W_r)
                found = True
                break
        
        if found:
            break
    
    # If no better found, use schoolbook
    if best_result is None:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
