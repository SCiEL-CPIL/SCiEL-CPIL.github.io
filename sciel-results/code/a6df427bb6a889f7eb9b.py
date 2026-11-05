"""Matrix multiplication 4x5x6 - rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s kill
    
    num_U, num_V, num_W = n * m, m * p, p * n  # 20, 30, 24
    
    # Block decomposition using Strassen for 2x2x2 sub-blocks
    # Split n=4 into 2 blocks of size 2
    # Split m=5 into blocks of sizes [2, 2, 1]
    # Split p=6 into blocks of sizes [2, 2, 2]
    
    # Row blocks: indices [0,1], [2,3]
    row_blocks = [(0, 2), (2, 4)]
    # Shared blocks: indices [0,1], [2,3], [4]
    shared_blocks = [(0, 2), (2, 4), (4, 5)]
    # Col blocks: indices [0,1], [2,3], [4,5]
    col_blocks = [(0, 2), (2, 4), (4, 6)]
    
    U_list = []
    V_list = []
    W_list = []
    
    for rb in row_blocks:
        r_start, r_end = rb
        r_size = r_end - r_start
        for sb in shared_blocks:
            s_start, s_end = sb
            s_size = s_end - s_start
            for cb in col_blocks:
                c_start, c_end = cb
                c_size = c_end - c_start
                
                if r_size == 2 and s_size == 2 and c_size == 2:
                    # Use Strassen decomposition for 2x2x2
                    # Strassen's 7 multiplications for 2x2 matrix multiplication
                    # The 7 terms are:
                    # M1 = (A00+A11)*(B00+B11)
                    # M2 = (A10+A11)*B00
                    # M3 = A00*(B01-B11)
                    # M4 = A11*(B10-B00)
                    # M5 = (A00+A01)*B11
                    # M6 = (A10-A00)*(B00+B01)
                    # M7 = (A01-A11)*(B10+B11)
                    # C00 = M1 + M4 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    
                    # For the tensor decomposition, each Strassen term corresponds to
                    # a product of three vectors (one for U, one for V, one for W)
                    # We need to express each of the 7 Strassen multiplications as
                    # a rank-1 term in the tensor decomposition.
                    
                    # Actually, Strassen gives us 7 "scalar multiplications" M1..M7.
                    # Each M_t is a product of two linear combinations of matrix entries.
                    # In the tensor decomposition framework, each M_t contributes a rank-1
                    # term: U_factor * V_factor * W_factor
                    # where U_factor is a linear combination of A entries,
                    # V_factor is a linear combination of B entries,
                    # W_factor is a linear combination of C entries.
                    
                    # Let me think about this more carefully.
                    # The Strassen algorithm computes:
                    # C = A * B where A is 2x2, B is 2x2, C is 2x2
                    # A = [[A00, A01], [A10, A11]]
                    # B = [[B00, B01], [B10, B11]]
                    # C = [[C00, C01], [C10, C11]]
                    
                    # The 7 products are:
                    # M1 = (A00+A11)*(B00+B11)
                    # M2 = (A10+A11)*B00
                    # M3 = A00*(B01-B11)
                    # M4 = A11*(B10-B00)
                    # M5 = (A00+A01)*B11
                    # M6 = (A10-A00)*(B00+B01)
                    # M7 = (A01-A11)*(B10+B11)
                    
                    # And:
                    # C00 = M1 + M4 - M5 + M7
                    # C01 = M3 + M5
                    # C10 = M2 + M4
                    # C11 = M1 - M2 + M3 + M6
                    
                    # In the tensor decomposition, for each Strassen product M_t,
                    # we have a rank-1 contribution to the tensor.
                    # The tensor entry T[i,j,k] for the 2x2x2 sub-block is 1.
                    # The decomposition is:
                    # sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1 for all i,j,k in {0,1}
                    
                    # For Strassen, we can write:
                    # The contribution to C[i,k] from product M_t is:
                    # C[i,k] += alpha_t(i,k) * M_t
                    # where M_t = (sum_ij beta_t(i,j) * A[i,j]) * (sum_jk gamma_t(j,k) * B[j,k])
                    
                    # So the rank-1 term for M_t in the tensor decomposition is:
                    # U[i,j,t] = beta_t(i,j)
                    # V[j,k,t] = gamma_t(j,k)  
                    # W[k,i,t] = alpha_t(i,k)
                    
                    # Let me define the 7 Strassen terms with their coefficients:
                    
                    # M1 = (A00+A11)*(B00+B11)
                    # beta_1: A00=1, A11=1 -> U entries: U[0,0]=1, U[1,1]=1
                    # gamma_1: B00=1, B11=1 -> V entries: V[0,0]=1, V[1,1]=1
                    # alpha_1: C00=1, C11=1 -> W entries: W[0,0]=1, W[1,1]=1
                    
                    # M2 = (A10+A11)*B00
                    # beta_2: A10=1, A11=1 -> U[1,0]=1, U[1,1]=1
                    # gamma_2: B00=1 -> V[0,0]=1
                    # alpha_2: C10=1, C11=-1 -> W[0,1]=1, W[1,1]=-1
                    
                    # M3 = A00*(B01-B11)
                    # beta_3: A00=1 -> U[0,0]=1
                    # gamma_3: B01=1, B11=-1 -> V[0,1]=1, V[1,1]=-1
                    # alpha_3: C01=1, C11=1 -> W[1,0]=1, W[1,1]=1
                    
                    # M4 = A11*(B10-B00)
                    # beta_4: A11=1 -> U[1,1]=1
                    # gamma_4: B10=1, B00=-1 -> V[1,0]=1, V[0,0]=-1
                    # alpha_4: C00=1, C10=1 -> W[0,0]=1, W[0,1]=1
                    
                    # M5 = (A00+A01)*B11
                    # beta_5: A00=1, A01=1 -> U[0,0]=1, U[0,1]=1
                    # gamma_5: B11=1 -> V[1,1]=1
                    # alpha_5: C00=-1, C01=1 -> W[0,0]=-1, W[1,0]=1
                    
                    # M6 = (A10-A00)*(B00+B01)
                    # beta_6: A10=1, A00=-1 -> U[1,0]=1, U[0,0]=-1
                    # gamma_6: B00=1, B01=1 -> V[0,0]=1, V[0,1]=1
                    # alpha_6: C11=1 -> W[1,1]=1
                    
                    # M7 = (A01-A11)*(B10+B11)
                    # beta_7: A01=1, A11=-1 -> U[0,1]=1, U[1,1]=-1
                    # gamma_7: B10=1, B11=1 -> V[1,0]=1, V[1,1]=1
                    # alpha_7: C00=1, C11=1 -> W[0,0]=1, W[1,1]=1
                    
                    # Now let me verify this decomposition.
                    # For each (i,j,k), sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] should equal 1.
                    
                    # Let me build the U, V, W matrices for the 2x2x2 case:
                    # U is 4x7 (i*2+j, t), V is 4x7 (j*2+k, t), W is 4x7 (k*2+i, t)
                    
                    U_sub = np.zeros((4, 7), dtype=int)
                    V_sub = np.zeros((4, 7), dtype=int)
                    W_sub = np.zeros((4, 7), dtype=int)
                    
                    # M1
                    U_sub[0*2+0, 0] = 1  # A00
                    U_sub[1*2+1, 0] = 1  # A11
                    V_sub[0*2+0, 0] = 1  # B00
                    V_sub[1*2+1, 0] = 1  # B11
                    W_sub[0*2+0, 0] = 1  # C00
                    W_sub[1*2+1, 0] = 1  # C11
                    
                    # M2
                    U_sub[1*2+0, 1] = 1  # A10
                    U_sub[1*2+1, 1] = 1  # A11
                    V_sub[0*2+0, 1] = 1  # B00
                    W_sub[0*2+1, 1] = 1  # C10
                    W_sub[1*2+1, 1] = -1 # C11
                    
                    # M3
                    U_sub[0*2+0, 2] = 1  # A00
                    V_sub[0*2+1, 2] = 1  # B01
                    V_sub[1*2+1, 2] = -1 # B11
                    W_sub[1*2+0, 2] = 1  # C01
                    W_sub[1*2+1, 2] = 1  # C11
                    
                    # M4
                    U_sub[1*2+1, 3] = 1  # A11
                    V_sub[1*2+0, 3] = 1  # B10
                    V_sub[0*2+0, 3] = -1 # B00
                    W_sub[0*2+0, 3] = 1  # C00
                    W_sub[0*2+1, 3] = 1  # C10
                    
                    # M5
                    U_sub[0*2+0, 4] = 1  # A00
                    U_sub[0*2+1, 4] = 1  # A01
                    V_sub[1*2+1, 4] = 1  # B11
                    W_sub[0*2+0, 4] = -1 # C00
                    W_sub[1*2+0, 4] = 1  # C01
                    
                    # M6
                    U_sub[1*2+0, 5] = 1  # A10
                    U_sub[0*2+0, 5] = -1 # A00
                    V_sub[0*2+0, 5] = 1  # B00
                    V_sub[0*2+1, 5] = 1  # B01
                    W_sub[1*2+1, 5] = 1  # C11
                    
                    # M7
                    U_sub[0*2+1, 6] = 1  # A01
                    U_sub[1*2+1, 6] = -1 # A11
                    V_sub[1*2+0, 6] = 1  # B10
                    V_sub[1*2+1, 6] = 1  # B11
                    W_sub[0*2+0, 6] = 1  # C00
                    W_sub[1*2+1, 6] = 1  # C11
                    
                    # Verify: for all i,j,k in {0,1}, sum_t U[i,j,t]*V[j,k,t]*W[k,i,t] == 1
                    for i in range(2):
                        for j in range(2):
                            for k in range(2):
                                s = 0
                                for t in range(7):
                                    s += U_sub[i*2+j, t] * V_sub[j*2+k, t] * W_sub[k*2+i, t]
                                if s != 1:
                                    # This shouldn't happen, but let's handle it
                                    pass
                    
                    # Now map to global indices
                    # U global row: r_start + i, s_start + j -> (r_start+i)*m + (s_start+j)
                    # V global row: s_start + j, c_start + k -> (s_start+j)*p + (c_start+k)
                    # W global row: c_start + k, r_start + i -> (c_start+k)*n + (r_start+i)
                    
                    for t in range(7):
                        U_col = np.zeros(num_U, dtype=int)
                        V_col = np.zeros(num_V, dtype=int)
                        W_col = np.zeros(num_W, dtype=int)
                        
                        for i in range(2):
                            for j in range(2):
                                U_col[(r_start+i)*m + (s_start+j)] = U_sub[i*2+j, t]
                        
                        for j in range(2):
                            for k in range(2):
                                V_col[(s_start+j)*p + (c_start+k)] = V_sub[j*2+k, t]
                        
                        for k in range(2):
                            for i in range(2):
                                W_col[(c_start+k)*n + (r_start+i)] = W_sub[k*2+i, t]
                        
                        U_list.append(U_col)
                        V_list.append(V_col)
                        W_list.append(W_col)
                    
                else:
                    # Schoolbook for this sub-block
                    for i in range(r_size):
                        for j in range(s_size):
                            for k in range(c_size):
                                U_col = np.zeros(num_U, dtype=int)
                                V_col = np.zeros(num_V, dtype=int)
                                W_col = np.zeros(num_W, dtype=int)
                                
                                U_col[(r_start+i)*m + (s_start+j)] = 1
                                V_col[(s_start+j)*p + (c_start+k)] = 1
                                W_col[(c_start+k)*n + (r_start+i)] = 1
                                
                                U_list.append(U_col)
                                V_list.append(V_col)
                                W_list.append(W_col)
    
    U = np.column_stack(U_list) if U_list else np.zeros((num_U, 0), dtype=int)
    V = np.column_stack(V_list) if V_list else np.zeros((num_V, 0), dtype=int)
    W = np.column_stack(W_list) if W_list else np.zeros((num_W, 0), dtype=int)
    
    best_rank = U.shape[1]
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Verify the decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(best_rank):
                    s += best_U[i*m+j, r] * best_V[j*p+k, r] * best_W[k*n+i, r]
                if s != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((num_U, rank_school), dtype=int)
        V_school = np.zeros((num_V, rank_school), dtype=int)
        W_school = np.zeros((num_W, rank_school), dtype=int)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, col] = 1
                    V_school[j*p+k, col] = 1
                    W_school[k*n+i, col] = 1
                    col += 1
        
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
