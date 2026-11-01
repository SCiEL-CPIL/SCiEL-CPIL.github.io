import numpy as np
import time
import random

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback: Schoolbook Rank 44
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school

    # Try to find a better rank
    # We know rank 43 is possible (Strassen 2x2 + Schoolbook 2x9)
    # We will try to find rank < 44
    
    def check_validity(U, V, W, rank):
        # Check if sum_c U[i,j1,c] V[j2,k,c] W[k,i,c] == delta(j1,j2)
        # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
        # Flatten U, V, W
        U_flat = U.reshape(n, m, rank)
        V_flat = V.reshape(m, p, rank)
        W_flat = W.reshape(p, n, rank)
        
        # Compute tensor T[i, j1, j2, k] = sum_c U[i, j1, c] V[j2, k, c] W[k, i, c]
        # Note: V index is j2*p + k -> V_flat[j2, k, c]
        # W index is k*n + i -> W_flat[k, i, c]
        
        # Efficient check using random matrices
        # C = A B
        # A (n x m), B (m x p)
        # C[i, k] = sum_j A[i, j] B[j, k]
        # Decomp: C[i, k] = sum_c W[k, i, c] (sum_j U[i, j, c] A[i, j]) (sum_j V[j, k, c] B[j, k])
        
        for _ in range(10):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_true = A @ B
            
            C_decomp = np.zeros((n, p))
            for c in range(rank):
                # L_c(A) = sum_j U[i, j, c] A[i, j]
                # This is a vector of length n? No, L_c depends on i?
                # In the formula: sum_c W[k, i, c] (sum_j U[i, j, c] A[i, j]) (sum_j V[j, k, c] B[j, k])
                # The term (sum_j U[i, j, c] A[i, j]) is a scalar for each i.
                # Let's call it LA[i].
                LA = np.zeros(n)
                for i in range(n):
                    LA[i] = np.sum(U_flat[i, :, c] * A[i, :])
                    
                MB = np.zeros(p)
                for k in range(p):
                    MB[k] = np.sum(V_flat[:, k, c] * B[:, k])
                    
                for i in range(n):
                    for k in range(p):
                        C_decomp[i, k] += W_flat[k, i, c] * LA[i] * MB[k]
                        
            if not np.allclose(C_true, C_decomp, atol=1e-5):
                return False
        return True

    def solve_rank(r):
        # Try to find a solution of rank r
        # Use random restarts and simple gradient descent?
        # Or just random sampling?
        # Given the non-convexity, random sampling is weak.
        # I will try to use the structure of the problem.
        # For now, I will just return None if I don't have a specific solver.
        # I will implement a simple search for r=43 explicitly.
        
        if r == 43:
            # Strassen for 2x2 block (cols 0,1) + Schoolbook for 2x9 block (cols 2..10)
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Part 1: Strassen for B[:, 0:2]
            # A is 2x2. B1 is 2x2.
            # P1 = (A00+A11)(B00+B11)
            # P2 = (A10+A11)(B00)
            # P3 = (A00-A10)(B11)
            # P4 = (A00+A11)(B10-B00)
            # P5 = (A00+A10)(B11-B01)
            # P6 = (A01+A11)(B00-B10)
            # P7 = (A00-A10)(B01+B11)
            
            # Mapping to U, V, W
            # C00 = P1 + P4 - P5 - P7
            # C01 = P3 + P5
            # C10 = P2 + P6
            # C11 = P1 - P2 + P3 + P4
            
            # We need to embed this into the general form.
            # The general form sums over c.
            # For c=0..6, we define U, V, W such that they produce P_c.
            # Then W for the final assembly?
            # No, the general form is C[i,k] = sum_c W[k,i,c] LA[i,c] MB[k,c].
            # This is not directly Strassen. Strassen is C = sum P_c * M_c where M_c are fixed matrices.
            # Here W[k,i,c] plays the role of the coefficient.
            
            # Let's just use the explicit Strassen coefficients.
            # Strassen P's:
            # P1: U: A00=1, A11=1. V: B00=1, B11=1.
            # P2: U: A10=1, A11=1. V: B00=1.
            # P3: U: A00=1, A10=-1. V: B11=1.
            # P4: U: A00=1, A11=1. V: B10=1, B00=-1.
            # P5: U: A00=1, A10=1. V: B11=1, B01=-1.
            # P6: U: A01=1, A11=1. V: B00=1, B10=-1.
            # P7: U: A00=1, A10=-1. V: B01=1, B11=1.
            
            # Indices for U (2x2): 0:A00, 1:A01, 2:A10, 3:A11
            # Indices for V (2x2 part): 0:B00, 1:B01, 2:B10, 3:B11 (cols 0,1)
            
            # For cols 2..10, use schoolbook.
            # There are 9 cols. 2 rows. 18 terms.
            # Total 7 + 18 = 25?
            # Wait, schoolbook for 2x2 by 2x9 is 2*2*9 = 36.
            # Why 18?
            # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k].
            # For each k, we need 2 multiplications?
            # No, the decomposition is sum_c W[k,i,c] LA[i,c] MB[k,c].
            # If we use schoolbook, we can set:
            # For each i, j, k, we have a term A[i,j] B[j,k].
            # This corresponds to c being unique for each (i,j,k)?
            # No, we can group them.
            
            # Let's just use the full schoolbook for the 2x9 part.
            # Rank 36.
            # Total 7 + 36 = 43.
            
            # Fill U, V, W for Strassen part (c=0..6)
            # Only affects k=0,1.
            
            # P1
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            # W for C00, C01, C10, C11
            # C00 += P1 -> W[0,0,0] += 1 (k=0, i=0)
            # C11 += P1 -> W[1,1,0] += 1 (k=1, i=1)
            W[0*2+0, 0] = 1
            W[1*2+1, 0] = 1
            
            # P2
            U[2, 1] = 1; U[3, 1] = 1
            V[0, 1] = 1
            W[0*2+1, 1] = 1 # C10 (i=1, k=0)
            W[1*2+1, 1] = -1 # C11 (i=1, k=1)
            
            # P3
            U[0, 2] = 1; U[2, 2] = -1
            V[3, 2] = 1
            W[0*2+1, 2] = 1 # C01 (i=0, k=1)
            W[1*2+1, 2] = 1 # C11 (i=1, k=1)
            
            # P4
            U[0, 3] = 1; U[3, 3] = 1
            V[2, 3] = 1; V[0, 3] = -1
            W[0*2+0, 3] = 1 # C00
            W[1*2+1, 3] = 1 # C11
            
            # P5
            U[0, 4] = 1; U[2, 4] = 1
            V[3, 4] = 1; V[1, 4] = -1
            W[0*2+0, 4] = -1 # C00
            W[0*2+1, 4] = 1 # C01
            
            # P6
            U[1, 5] = 1; U[3, 5] = 1
            V[0, 5] = 1; V[2, 5] = -1
            W[0*2+1, 5] = 1 # C10
            
            # P7
            U[0, 6] = 1; U[2, 6] = -1
            V[1, 6] = 1; V[3, 6] = 1
            W[0*2+0, 6] = -1 # C00
            W[0*2+1, 6] = -1 # C01
            
            # Schoolbook for k=2..10
            # For each k, i, j: term A[i,j] B[j,k]
            # We can assign one c for each (i,j,k) triple?
            # That would be 2*2*9 = 36 terms.
            col = 7
            for k in range(2, p):
                for i in range(n):
                    for j in range(m):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
                        
            return U, V, W, r

        return None

    # Try rank 43
    res = solve_rank(43)
    if res:
        U, V, W, r = res
        if check_validity(U, V, W, r):
            best_U, best_V, best_W = U, V, W
            best_rank = r

    # Try to search for lower rank?
    # Given time, I'll skip complex search and return 43 if valid.
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
