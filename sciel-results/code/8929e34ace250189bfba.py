import time
import numpy as np
from scipy.optimize import lsq_linear

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # 1. Find a rank-7 decomposition for 2x2 matrix multiplication
    # Tensor T[i,j,k] = 1 if i,j,k in {0,1}
    # We want U (4x7), V (4x7), W (4x7) such that sum_c U[i,c]V[j,c]W[k,c] = 1
    
    # Initialize with random values to avoid trivial zero solution
    np.random.seed(42)
    U_22 = np.random.randn(4, 7)
    V_22 = np.random.randn(4, 7)
    W_22 = np.random.randn(4, 7)
    
    # We need to solve for U, V, W.
    # This is a non-convex problem. However, we can use the known Strassen structure
    # or use a local search. Given the small size, we can try to optimize the cost function:
    # F = || sum_c u_c v_c w_c^T - T ||_F^2
    
    # Let's use a simple iterative refinement or just hardcode the known Strassen solution
    # to avoid optimization instability. The "solve" part is actually just ensuring we have
    # a valid set. Since I know Strassen works, I will construct the matrices explicitly
    # based on the standard Strassen formulas, but I will VERIFY them numerically.
    # If verification fails, I will fall back to a numerical search.
    
    # Standard Strassen for 2x2:
    # A = [[a, b], [c, d]] -> indices 0,1,2,3
    # B = [[e, f], [g, h]] -> indices 0,1,2,3
    # C = [[i, j], [k, l]] -> indices 0,1,2,3
    
    # p1 = (a+d)(e+h)
    # p2 = (c+d)(e+f)
    # p3 = (a+b)(g+h)
    # p4 = d(h-f)
    # p5 = a(e-g)
    # p6 = (c-a)(f+h)
    # p7 = (b-d)(e+g)
    
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 - p4
    # C11 = p1 - p2 + p3 + p6
    
    # Let's map this to U, V, W columns.
    # U row i (0..3), V row j (0..3), W row k (0..3)
    # Note: The problem statement says W[k*n+i, r]. For the 2x2 block, n=2.
    # So W index is k*2 + i.
    # But in the 2x2 sub-problem, we can just use local indices 0..3 for W too, 
    # as long as we map them correctly in the final assembly.
    
    # Let's define the local 2x2 Strassen matrices U2, V2, W2 (shape 4x7)
    # Local indices:
    # A: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # p1 = (a+d)(e+h)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1  # C00 += 1, C11 += 1
    
    # p2 = (c+d)(e+f)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1; V2[1, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1 # C10 += 1, C11 -= 1
    
    # p3 = (a+b)(g+h)
    U2[0, 2] = 1; U2[1, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = 1
    W2[1, 2] = 1; W2[3, 2] = 1  # C01 += 1, C11 += 1
    
    # p4 = d(h-f)
    U2[3, 3] = 1
    V2[3, 3] = 1; V2[1, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = -1 # C00 += 1, C10 -= 1
    
    # p5 = a(e-g)
    U2[0, 4] = 1
    V2[0, 4] = 1; V2[2, 4] = -1
    W2[1, 4] = 1; W2[0, 4] = -1 # C01 += 1, C00 -= 1
    
    # p6 = (c-a)(f+h)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[1, 5] = 1; V2[3, 5] = 1
    W2[3, 5] = 1  # C11 += 1
    
    # p7 = (b-d)(e+g)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[0, 6] = 1; V2[2, 6] = 1
    W2[0, 6] = 1  # C00 += 1
    
    # Verify the 2x2 block
    # T2[i,j,k] = sum_c U2[i,c] V2[j,c] W2[k,c] should be 1 for all i,j,k in 0..3
    # Note: W2 index here is local C index. In the full problem, C index is k*n+i.
    # For the 2x2 block embedded in the full problem, the local C indices map to global C indices.
    
    # Check:
    T2_calc = np.zeros((4, 4, 4))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0.0
                for c in range(7):
                    s += U2[i, c] * V2[j, c] * W2[k, c]
                T2_calc[i, j, k] = s
    
    # Expected T2:
    # A[i,j] * B[j,k] contributes to C[i,k]
    # In the tensor form T[i,j,k], the indices are:
    # i: row of A (0,1) -> local A indices 0,1,2,3? No.
    # The problem defines U as (n*m, rank). For 2x2, n=2, m=2.
    # U row index is i*m + j.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 2
    # i=1, j=1 -> 3
    # So U2 row 0 is A[0,0], row 1 is A[0,1], row 2 is A[1,0], row 3 is A[1,1].
    # Similarly V2 row 0 is B[0,0], row 1 is B[0,1], row 2 is B[1,0], row 3 is B[1,1].
    # W2 row index? The problem says W[k*n+i, r].
    # For 2x2, n=2.
    # C[0,0] -> k=0, i=0 -> 0
    # C[0,1] -> k=1, i=0 -> 1
    # C[1,0] -> k=0, i=1 -> 2
    # C[1,1] -> k=1, i=1 -> 3
    # So W2 row 0 is C[0,0], row 1 is C[0,1], row 2 is C[1,0], row 3 is C[1,1].
    
    # The tensor T[i,j,k] for matrix mult is 1 if the indices are valid.
    # Actually, the tensor T is defined such that T[i,j,k] = 1 for all valid i,j,k.
    # Wait, the matrix multiplication tensor T_{ijk} is 1 for all i,j,k?
    # Yes, because C[i,k] = sum_j A[i,j]B[j,k].
    # The decomposition is sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1.
    
    # So for the 2x2 block, we need sum_c U2[i*2+j, c] V2[j*2+k, c] W2[k*2+i, c] = 1.
    # My U2, V2, W2 above used local indices 0..3 directly as row indices.
    # Let's check if my manual assignment matches the i*m+j mapping.
    # U2 row 0: A[0,0] -> i=0,j=0. Correct.
    # U2 row 1: A[0,1] -> i=0,j=1. Correct.
    # U2 row 2: A[1,0] -> i=1,j=0. Correct.
    # U2 row 3: A[1,1] -> i=1,j=1. Correct.
    
    # V2 row 0: B[0,0] -> j=0,k=0. Correct.
    # V2 row 1: B[0,1] -> j=0,k=1. Correct.
    # V2 row 2: B[1,0] -> j=1,k=0. Correct.
    # V2 row 3: B[1,1] -> j=1,k=1. Correct.
    
    # W2 row 0: C[0,0] -> k=0,i=0. Correct.
    # W2 row 1: C[0,1] -> k=1,i=0. Correct.
    # W2 row 2: C[1,0] -> k=0,i=1. Correct.
    # W2 row 3: C[1,1] -> k=1,i=1. Correct.
    
    # Now let's verify the sum for a specific case, e.g., i=0, j=0, k=0.
    # This corresponds to A[0,0] * B[0,0] contributing to C[0,0].
    # U2[0, c] * V2[0, c] * W2[0, c]
    # c=0: 1*1*1 = 1
    # c=1: 0
    # c=2: 0
    # c=3: 0
    # c=4: 1*1*(-1) = -1 (Wait, U2[0,4]=1, V2[0,4]=1, W2[0,4]=-1)
    # c=5: 0
    # c=6: 0
    # Sum = 1 - 1 = 0. But it should be 1.
    
    # Let's re-evaluate Strassen.
    # C00 = p1 + p4 - p5 + p7
    # p1 = (a+d)(e+h). A[0,0]=a, A[1,1]=d. B[0,0]=e, B[1,1]=h.
    # p4 = d(h-f). A[1,1]=d. B[1,1]=h, B[0,1]=f.
    # p5 = a(e-g). A[0,0]=a. B[0,0]=e, B[1,0]=g.
    # p7 = (b-d)(e+g). A[0,1]=b, A[1,1]=d. B[0,0]=e, B[1,0]=g.
    
    # For i=0, j=0, k=0:
    # We are looking at the coefficient of A[0,0]B[0,0] in C[0,0].
    # This term comes from p1 (a*e), p5 (-a*e), p7 (b*e ... no, p7 is (b-d)(e+g) = be + bg - de - dg).
    # In p7, the term be is A[0,1]B[0,0]. This contributes to C[0,0]?
    # C[0,0] = sum_j A[0,j]B[j,0].
    # j=0: A[0,0]B[0,0]
    # j=1: A[0,1]B[1,0]
    # So A[0,1]B[0,0] does NOT contribute to C[0,0].
    
    # My Strassen mapping of W2 indices might be wrong or the formulas are mixed up.
    # Let's use a numerical solver to find U2, V2, W2. It's safer.
    
    def solve_strassen():
        # Random init
        U = np.random.randn(4, 7)
        V = np.random.randn(4, 7)
        W = np.random.randn(4, 7)
        
        # Gradient descent or similar?
        # Since it's small, we can try to minimize the error.
        # Error = sum_{i,j,k} (sum_c U[i,c]V[j,c]W[k,c] - 1)^2
        
        # This is a tensor decomposition problem.
        # We can use a simple iterative method:
        # Fix V, W, solve for U (linear LSQ)
        # Fix U, W, solve for V (linear LSQ)
        # Fix U, V, solve for W (linear LSQ)
        
        # Reshape to tensors
        # T[i,j,k] = 1
        
        for _ in range(1000):
            # Solve for U
            # For each i, U[i, :] is the solution to minimize sum_{j,k} (sum_c U[i,c] V[j,c] W[k,c] - 1)^2
            # This is a linear system for U[i, :].
            # Let M_i be a matrix where M_i[(j,k), c] = V[j, c] * W[k, c]
            # Then M_i @ U[i, :] = 1 (vector of ones of size 16)
            
            for i in range(4):
                M = np.zeros((16, 7))
                for j in range(4):
                    for k in range(4):
                        idx = j * 4 + k
                        M[idx, :] = V[j, :] * W[k, :]
                # Solve M @ u = 1
                # Use least squares
                u, _, _, _ = np.linalg.lstsq(M, np.ones(16), rcond=None)
                U[i, :] = u
                
            # Solve for V
            for j in range(4):
                M = np.zeros((16, 7))
                for i in range(4):
                    for k in range(4):
                        idx = i * 4 + k
                        M[idx, :] = U[i, :] * W[k, :]
                v, _, _, _ = np.linalg.lstsq(M, np.ones(16), rcond=None)
                V[j, :] = v
                
            # Solve for W
            for k in range(4):
                M = np.zeros((16, 7))
                for i in range(4):
                    for j in range(4):
                        idx = i * 4 + j
                        M[idx, :] = U[i, :] * V[j, :]
                w, _, _, _ = np.linalg.lstsq(M, np.ones(16), rcond=None)
                W[k, :] = w
                
            # Check convergence
            error = 0.0
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        s = np.dot(U[i, :], V[j, :] * W[k, :])
                        error += (s - 1.0)**2
            if error < 1e-6:
                return U, V, W, error
                
        return U, V, W, error

    U2, V2, W2, err = solve_strassen()
    
    if err > 1e-3:
        # If solver fails, fall back to schoolbook for the 2x2 block (rank 8)
        # Or just return schoolbook for the whole thing
        pass
        
    # 2. Construct the full 3x4x8 decomposition
    # Partition:
    # i (rows of A): 0, 1, 2
    # j (cols of A / rows of B): 0, 1, 2, 3
    # k (cols of B): 0, 1, 2, 3, 4, 5, 6, 7
    
    # We can partition j into {0,1} and {2,3}.
    # We can partition k into {0,1}, {2,3}, {4,5}, {6,7}.
    # We can partition i into {0,1} and {2}.
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2x2x2 (Strassen rank 7)
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3} -> 2x2x2x2 (Strassen rank 7)
    # Block 3: i in {0,1}, j in {0,1}, k in {4,5} -> 2x2x2x2 (Strassen rank 7)
    # Block 4: i in {0,1}, j in {0,1}, k in {6,7} -> 2x2x2x2 (Strassen rank 7)
    # Block 5: i in {0,1}, j in {2,3}, k in {0,1} -> 2x2x2x2 (Strassen rank 7)
    # Block 6: i in {0,1}, j in {2,3}, k in {2,3} -> 2x2x2x2 (Strassen rank 7)
    # Block 7: i in {0,1}, j in {2,3}, k in {4,5} -> 2x2x2x2 (Strassen rank 7)
    # Block 8: i in {0,1}, j in {2,3}, k in {6,7} -> 2x2x2x2 (Strassen rank 7)
    
    # Block 9: i in {2}, j in {0,1}, k in {0,1} -> 1x2x2x2 (Schoolbook rank 4)
    # Block 10: i in {2}, j in {0,1}, k in {2,3} -> 1x2x2x2 (Schoolbook rank 4)
    # Block 11: i in {2}, j in {0,1}, k in {4,5} -> 1x2x2x2 (Schoolbook rank 4)
    # Block 12: i in {2}, j in {0,1}, k in {6,7} -> 1x2x2x2 (Schoolbook rank 4)
    # Block 13: i in {2}, j in {2,3}, k in {0,1} -> 1x2x2x2 (Schoolbook rank 4)
    # Block 14: i in {2}, j in {2,3}, k in {2,3} -> 1x2x2x2 (Schoolbook rank 4)
    # Block 15: i in {2}, j in {2,3}, k in {4,5} -> 1x2x2x2 (Schoolbook rank 4)
    # Block 16: i in {2}, j in {2,3}, k in {6,7} -> 1x2x2x2 (Schoolbook rank 4)
    
    # Total Rank = 8 * 7 + 8 * 4 = 56 + 32 = 88.
    
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a block
    def add_strassen_block(i_start, j_start, k_start):
        # i_start, j_start, k_start are 0-based starts of 2x2 blocks
        # i in {i_start, i_start+1}
        # j in {j_start, j_start+1}
        # k in {k_start, k_start+1}
        
        for c in range(7):
            # U indices: i*m + j
            # i0 = i_start, i1 = i_start+1
            # j0 = j_start, j1 = j_start+1
            U[i_start * m + j_start, col + c] = U2[0, c]
            U[i_start * m + (j_start + 1), col + c] = U2[1, c]
            U[(i_start + 1) * m + j_start, col + c] = U2[2, c]
            U[(i_start + 1) * m + (j_start + 1), col + c] = U2[3, c]
            
            # V indices: j*p + k
            # j0, j1
            # k0 = k_start, k1 = k_start+1
            V[j_start * p + k_start, col + c] = V2[0, c]
            V[j_start * p + (k_start + 1), col + c] = V2[1, c]
            V[(j_start + 1) * p + k_start, col + c] = V2[2, c]
            V[(j_start + 1) * p + (k_start + 1), col + c] = V2[3, c]
            
            # W indices: k*n + i
            # k0, k1
            # i0, i1
            W[k_start * n + i_start, col + c] = W2[0, c]
            W[(k_start + 1) * n + i_start, col + c] = W2[1, c]
            W[k_start * n + (i_start + 1), col + c] = W2[2, c]
            W[(k_start + 1) * n + (i_start + 1), col + c] = W2[3, c]
            
    def add_schoolbook_block_1x2x2(i, j_start, k_start):
        # i is single row
        # j in {j_start, j_start+1}
        # k in {k_start, k_start+1}
        # Rank 4: (i,j0,k0), (i,j0,k1), (i,j1,k0), (i,j1,k1)
        
        # 1. (i, j0, k0)
        U[i * m + j_start, col] = 1
        V[j_start * p + k_start, col] = 1
        W[k_start * n + i, col] = 1
        col += 1
        
        # 2. (i, j0, k1)
        U[i * m + j_start, col] = 1
        V[j_start * p + (k_start + 1), col] = 1
        W[(k_start + 1) * n + i, col] = 1
        col += 1
        
        # 3. (i, j1, k0)
        U[i * m + (j_start + 1), col] = 1
        V[(j_start + 1) * p + k_start, col] = 1
        W[k_start * n + i, col] = 1
        col += 1
        
        # 4. (i, j1, k1)
        U[i * m + (j_start + 1), col] = 1
        V[(j_start + 1) * p + (k_start + 1), col] = 1
        W[(k_start + 1) * n + i, col] = 1
        col += 1
        
    # Add Strassen blocks
    # i in {0,1}, j in {0,1}
    for k_start in [0, 2, 4, 6]:
        add_strassen_block(0, 0, k_start)
        
    # i in {0,1}, j in {2,3}
    for k_start in [0, 2, 4, 6]:
        add_strassen_block(0, 2, k_start)
        
    # i in {2}, j in {0,1}
    for k_start in [0, 2, 4, 6]:
        add_schoolbook_block_1x2x2(2, 0, k_start)
        
    # i in {2}, j in {2,3}
    for k_start in [0, 2, 4, 6]:
        add_schoolbook_block_1x2x2(2, 2, k_start)
        
    # Verify
    def _verify(U, V, W, n, m, p):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True
    
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
                
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
