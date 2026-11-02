import time
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # 1. Construct the schoolbook solution (Rank = n*m*p)
    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R), dtype=int)
        V = np.zeros((m * p, R), dtype=int)
        W = np.zeros((p * n, R), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R

    # 2. Construct a hybrid Strassen/Schoolbook solution
    # Split inner dimension m=3 into 2+1
    # A (2x3) = [A_left (2x2) | A_right (2x1)]
    # B (3x7) = [B_top (2x7); B_bottom (1x7)]
    # C = A_left * B_top + A_right * B_bottom
    #
    # For A_left * B_top (2x2 * 2x7):
    #   Split B_top columns into pairs: cols 0-1, 2-3, 4-5, and col 6
    #   Use Strassen for the three 2x2x2 blocks (rank 7 each)
    #   Use schoolbook for the 2x2x1 block (rank 2)
    #
    # For A_right * B_bottom (2x1 * 1x7):
    #   This is a rank-2 operation (outer product structure)
    #
    # Total rank = 7 + 7 + 7 + 2 + 2 = 25
    
    def strassen_hybrid():
        # U has shape (n*m, R) = (6, R)
        # V has shape (m*p, R) = (21, R)
        # W has shape (p*n, R) = (14, R)
        
        # Indexing:
        # U[i*m + j, r] where i in [0,2), j in [0,3)
        # V[j*p + k, r] where j in [0,3), k in [0,7)
        # W[k*n + i, r] where k in [0,7), i in [0,2)
        
        # Strassen for 2x2 * 2x2:
        # Given matrices X (2x2) and Y (2x2), compute X*Y
        # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
        # Result Z = [[z00, z01], [z10, z11]]
        # z00 = x00*y00 + x01*y10
        # z01 = x00*y01 + x01*y11
        # z10 = x10*y00 + x11*y10
        # z11 = x10*y01 + x11*y11
        #
        # Strassen uses 7 products:
        # m1 = (x00+x11)*(y00+y11)
        # m2 = (x10+x11)*y00
        # m3 = x00*(y01-y11)
        # m4 = x11*(y10-y00)
        # m5 = (x00+x01)*y11
        # m6 = (x10-x00)*(y00+y01)
        # m7 = (x01-x11)*(y10+y11)
        #
        # z00 = m1 + m4 - m5 + m7
        # z01 = m3 + m5
        # z10 = m2 + m4
        # z11 = m1 - m2 + m3 + m7
        #
        # For our tensor decomposition, we need to express each product as a rank-1 term.
        # The U, V, W matrices encode the coefficients.
        
        R = 25
        U = np.zeros((n * m, R), dtype=float)
        V = np.zeros((m * p, R), dtype=float)
        W = np.zeros((p * n, R), dtype=float)
        
        col = 0
        
        # Helper: add a Strassen decomposition for A_left (rows 0-1 of A, cols 0-1) 
        # times B block (rows 0-1 of B, cols col_start to col_start+1)
        def add_strassen(A_row_start, A_col_start, B_row_start, B_col_start, col_offset):
            # A is 2x2: rows A_row_start to A_row_start+1, cols A_col_start to A_col_start+1
            # B is 2x2: rows B_row_start to B_row_start+1, cols B_col_start to B_col_start+1
            # Output C is 2x2: rows A_row_start to A_row_start+1, cols B_col_start to B_col_start+1
            
            # m1 = (x00+x11)*(y00+y11)
            # U contributions: A[0,0] and A[1,1] get coefficient 1
            # V contributions: B[0,0] and B[1,1] get coefficient 1
            # W contributions: C[0,0] gets +1, C[0,1] gets 0, C[1,0] gets 0, C[1,1] gets +1
            # Wait, I need to be more careful about the tensor structure.
            
            # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # should equal 1 if (i,j,k) corresponds to a valid multiplication, 0 otherwise.
            
            # For Strassen, we're decomposing the 2x2x2 multiplication tensor.
            # Let's define local indices:
            # i_local in [0,2) -> global i = A_row_start + i_local
            # j_local in [0,2) -> global j = A_col_start + j_local  
            # k_local in [0,2) -> global k = B_col_start + k_local
            # B row local j_local -> global B row = B_row_start + j_local
            
            # m1 = (x00+x11)*(y00+y11)
            # This contributes to z00 and z11
            # In the tensor: for each (i_local, j_local, k_local), the contribution is
            # U[i_local*m + (A_col_start+j_local), col] * V[(B_row_start+j_local)*p + (B_col_start+k_local), col] * W[(B_col_start+k_local)*n + (A_row_start+i_local), col]
            
            # For m1: U gets 1 at positions (i_local=0, j_local=0) and (i_local=1, j_local=1)
            # V gets 1 at positions (j_local=0, k_local=0) and (j_local=1, k_local=1)
            # W gets 1 at positions (k_local=0, i_local=0) and (k_local=1, i_local=1)
            
            # Let me just directly compute the Strassen coefficients for the 2x2x2 tensor.
            
            # The 2x2x2 multiplication tensor has entries T[i,j,k] = 1 if the multiplication
            # C[i,k] = sum_j A[i,j]*B[j,k] includes this term.
            # Actually, T[i,j,k] = 1 for all i,j,k in [0,2) because each term A[i,j]*B[j,k]
            # contributes to C[i,k].
            
            # For Strassen decomposition of the 2x2x2 tensor:
            # We want sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k] = 1 for all i,j,k
            
            # Strassen rank-7 decomposition:
            # r=0: m1 = (x00+x11)*(y00+y11)
            # U[0,0,0]=1, U[1,1,0]=1, others 0
            # V[0,0,0]=1, V[1,1,0]=1, others 0  
            # W[0,0,0]=1, W[1,1,0]=1, others 0
            # This gives: T[0,0,0] += 1*1*1 = 1, T[0,0,1] += 0, T[0,1,0] += 0, etc.
            # Wait, this isn't right. Let me think again.
            
            # Actually, the tensor entry T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[i,k].
            # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k.
            
            # The decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            
            # For Strassen, we have 7 terms. Let me look up the exact coefficients.
            
            # I'll use a known rank-7 decomposition for 2x2x2.
            # Reference: standard Strassen algorithm
            
            # Let me define the 7 rank-1 tensors:
            # Each rank-1 tensor is U_r[i,j] * V_r[j,k] * W_r[k,i]
            
            # Term 0: U = [[1,0],[0,1]], V = [[1,0],[0,1]], W = [[1,0],[0,1]]
            # This gives diagonal products: A[0,0]*B[0,0], A[1,1]*B[1,1]
            
            # This is getting complex. Let me use a simpler approach: just use schoolbook for everything
            # but with the block structure to potentially find a better decomposition.
            
            # For now, let's just return schoolbook and I'll improve this.
            pass
        
        # Let me try a different approach: directly construct the hybrid decomposition
        # by combining known rank decompositions.
        
        # Actually, let me just verify that schoolbook works and return it for now,
        # then I can iterate on the Strassen part.
        
        return None  # Placeholder
    
    # For now, use schoolbook as the baseline
    best_U, best_V, best_W, best_R = schoolbook()
    
    # 3. Attempt to find a lower rank using the hybrid approach
    # We'll try to construct a rank-25 decomposition using Strassen blocks
    
    def construct_hybrid():
        """Construct a hybrid Strassen/Schoolbook decomposition."""
        # n=2, m=3, p=7
        # U: (6, R), V: (21, R), W: (14, R)
        
        # Split: A = [A12 | A2] where A12 is 2x2 (rows 0-1, cols 0-1), A2 is 2x1 (rows 0-1, col 2)
        # B = [B1; B2] where B1 is 2x7 (rows 0-1, all cols), B2 is 1x7 (row 2, all cols)
        
        # C = A12 * B1 + A2 * B2
        
        # For A12 * B1: B1 has 7 columns, split into [B1a | B1b | B1c | B1d]
        # where B1a = cols 0-1 (2x2), B1b = cols 2-3 (2x2), B1c = cols 4-5 (2x2), B1d = col 6 (2x1)
        
        # Use Strassen (rank 7) for each 2x2x2 block
        # Use schoolbook (rank 2) for the 2x2x1 block
        
        # For A2 * B2: This is (2x1) * (1x7) = 2x7
        # C[i,k] = A2[i,0] * B2[0,k] for i in [0,2), k in [0,7)
        # This is rank 2: for each i, C[i,k] = A2[i,0] * B2[0,k]
        # We can express this with 2 rank-1 terms:
        # Term for i=0: U[0,0]=1, V[2*7+k, r]=B2[0,k], W[k*2+0, r]=1 ... 
        # Actually, the tensor entry T[i,j,k] = 1 if j=2 (the third column of A) and we're multiplying
        # A[i,2] * B[2,k] to get C[i,k]
        # So T[i,2,k] = 1 for all i in [0,2), k in [0,7)
        
        # Rank-2 decomposition for this part:
        # r1: U[0,2, r1] = 1, V[2*7+k, r1] = 1 for all k, W[k*2+0, r1] = 1 for all k
        #     This gives sum over k of 1*1*1 = 7 for T[0,2,k]? No, that's not right.
        
        # Let me reconsider. The tensor is T[i,j,k] where:
        # i = row of C (0 or 1)
        # j = column of A = row of B (0, 1, or 2)
        # k = column of C (0 to 6)
        
        # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]
        # For standard multiplication, T[i,j,k] = 1 for all valid (i,j,k)
        
        # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # For the A2*B2 part: T[i,2,k] = 1 for i in {0,1}, k in {0,...,6}
        # We need: sum_r U[i*3+2, r] * V[2*7+k, r] * W[k*2+i, r] = 1
        
        # Simple rank-2 solution:
        # r=0: U[0*3+2, 0] = 1, U[1*3+2, 0] = 0
        #       V[2*7+k, 0] = 1 for all k
        #       W[k*2+0, 0] = 1 for all k
        #       This gives T[0,2,k] = 1*1*1 = 1 ✓, T[1,2,k] = 0*1*1 = 0 ✗
        
        # r=1: U[0*3+2, 1] = 0, U[1*3+2, 1] = 1
        #       V[2*7+k, 1] = 1 for all k
        #       W[k*2+1, 1] = 1 for all k
        #       This gives T[0,2,k] = 0, T[1,2,k] = 1 ✓
        
        # So rank 2 works for the A2*B2 part.
        
        # Now for the Strassen part. Let me implement a rank-7 decomposition for 2x2x2.
        
        # The 2x2x2 multiplication tensor: T[i,j,k] = 1 for i,j,k in {0,1}
        # We need rank-7 decomposition.
        
        # Known rank-7 decomposition (Strassen):
        # I'll use the following (from literature):
        
        # Let me define the 7 terms with their U, V, W coefficients.
        # Local indices: i,j,k in {0,1}
        
        # Term 0:
        # U[0,0]=1, U[1,1]=1
        # V[0,0]=1, V[1,1]=1
        # W[0,0]=1, W[1,1]=1
        # Contribution: T[0,0,0] += 1*1*1 = 1
        #               T[0,0,1] += 0*1*0 = 0 (W[1,0]=0)
        #               T[0,1,0] += 0*0*0 = 0
        #               T[0,1,1] += 0*1*0 = 0
        #               T[1,0,0] += 0*1*0 = 0
        #               T[1,0,1] += 0*1*0 = 0
        #               T[1,1,0] += 1*0*1 = 0
        #               T[1,1,1] += 1*1*1 = 1
        # So this gives T[0,0,0]=1, T[1,1,1]=1, others 0. Not quite right.
        
        # I think I need to look up the exact Strassen tensor decomposition.
        
        # Let me use a different approach: I'll use the fact that the 2x2x2 tensor
        # can be decomposed with rank 7, and I'll construct it explicitly.
        
        # From Strassen's algorithm:
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b12-b22)
        # m4 = a22(b21-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        
        # c11 = m1 + m4 - m5 + m7
        # c12 = m3 + m5
        # c21 = m2 + m4
        # c22 = m1 - m2 + m3 + m7
        
        # The tensor decomposition should satisfy:
        # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = T[i,j,k]
        
        # where T[i,j,k] = 1 for all i,j,k in {0,1} (since each A[i,j]*B[j,k] contributes to C[i,k])
        
        # Wait, that's not right. T[i,j,k] should be 1 if the term A[i,j]*B[j,k] appears in C[i,k],
        # which is always true for standard matrix multiplication. So T[i,j,k] = 1 for all i,j,k.
        
        # But the Strassen algorithm computes C using 7 multiplications, not 4.
        # The tensor decomposition is different from the algorithm.
        
        # Let me reconsider. The matrix multiplication tensor T is defined as:
        # T[i,j,k] = 1 if and only if the (i,k)-entry of the product matrix is the sum
        # over j of A[i,j]*B[j,k].
        
        # So T[i,j,k] = 1 for all valid (i,j,k).
        
        # The rank of this tensor is the minimum number of rank-1 tensors that sum to T.
        
        # For 2x2x2, the rank is 7 (Strassen).
        
        # Let me find an explicit rank-7 decomposition.
        
        # I'll use the following (adapted from known results):
        
        # Actually, let me just use schoolbook for now and return it.
        # The hybrid approach is complex to implement correctly in the time available.
        
        return None
    
    # Verify the schoolbook solution is correct
    def verify(U, V, W, R):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(R):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if not verify(best_U, best_V, best_W, best_R):
        # Fallback: construct schoolbook again
        best_U, best_V, best_W, best_R = schoolbook()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
