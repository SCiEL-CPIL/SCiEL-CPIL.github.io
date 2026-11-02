"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    t_start = time.time()
    DEADLINE = t_start + 140.0  # Leave 10s margin
    
    n, m, p = 2, 2, 5
    # Target rank: We know rank 18 is achievable (7 for 2x2x2 + 11 for 2x2x3? No, 2x2x3 is at most 12, so 19 max. Actually 2x2x3 rank is 11 or 12).
    # Standard 2x2x3 rank is 11 (Landsberg, 1978? No, 2x2x3 is 11? Let's check. 2x2x2 is 7. 2x2x3 is 11? 2x2x3 tensor rank is 11? 
    # Actually, the rank of 2x2x3 matrix multiplication is 11. 
    # So 7 (for first 2 cols of B) + 11 (for last 3 cols of B) = 18.
    # Can we do better? Maybe 17?
    # Let's try to find a rank 17 or 18 decomposition with small integers.
    # The schoolbook rank is 2*2*5 = 20.
    # So 18 is already better than 20.
    # Let's try to find a rank 17 solution.
    
    # We will use a simple iterative optimization to find a low-rank decomposition.
    # Since the space is small, we can use random initialization and gradient descent or simple projection.
    # However, exact integer/half-integer solutions are required.
    # A known rank-11 decomposition for 2x2x3 exists.
    # A known rank-7 decomposition for 2x2x2 exists (Strassen).
    # Combining them gives rank 18.
    
    # Let's try to find a rank 17 solution.
    # If we fail, we return the rank 18 solution constructed from Strassen + 2x2x3 block.
    
    # First, let's build the rank-18 solution to have a fallback.
    def build_rank_18():
        # 2x2x2 part (Strassen)
        # Indices: A is 2x2, B is 2x2, C is 2x2.
        # U_strassen: 4x7, V_strassen: 4x7, W_strassen: 4x7
        # Rows of U: (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3
        # Rows of V: (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3 (for B's first 2 cols)
        # Rows of W: (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3 (for C's first 2 cols)
        
        # Strassen's 7 terms for 2x2 matrix multiplication:
        # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
        # M1 = a00 * (b00 + b11)
        # M2 = (a10 + a11) * b00
        # M3 = a01 * (b10 + b11)
        # M4 = a11 * (b01 - b10)
        # M5 = (a00 + a01) * b10
        # M6 = (a10 - a00) * (b00 + b01)
        # M7 = (a11 - a01) * (b10 + b11)
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M7
        # C10 = M2 + M5
        # C11 = M1 - M2 + M3 + M6
        
        # U_strassen (rows: a00, a01, a10, a11; cols: M1..M7)
        U_s = np.array([
            [1, 0, 0, 0, 1, -1, 0],
            [0, 0, 1, 0, 0, 0, -1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 1, 0, 0, 1]
        ], dtype=float)
        
        # V_strassen (rows: b00, b01, b10, b11; cols: M1..M7)
        V_s = np.array([
            [1, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, -1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 1]
        ], dtype=float)
        
        # W_strassen (rows: c00, c01, c10, c11; cols: M1..M7)
        W_s = np.array([
            [1, 0, 0, 1, -1, 0, -1],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ], dtype=float)
        
        # 2x2x3 part (B is 2x3, C is 2x3)
        # Rows of U: a00, a01, a10, a11 -> 0,1,2,3
        # Rows of V: b00, b01, b02, b10, b11, b12 -> 4,5,6,7,8,9
        # Rows of W: c00, c01, c02, c10, c11, c12 -> 4,5,6,7,8,9
        
        # We need a rank-11 decomposition for 2x2x3.
        # One known decomposition (from literature, e.g., Landsberg or simpler constructions).
        # Let's try to find one or use a known one.
        # If I don't have a hardcoded rank-11 2x2x3, I can try to find it numerically.
        # But for now, let's assume we can find a rank-11 2x2x3.
        # Actually, let's just use schoolbook for 2x2x3 to be safe? Rank 12.
        # Then total rank = 7 + 12 = 19.
        # We want 18. So we need rank 11 for 2x2x3.
        
        # Let's try to construct a rank-11 2x2x3 decomposition.
        # This is a known result. I will try to recall or derive it.
        # If I can't, I will return the rank-19 solution (7+12) or even rank-20 (20).
        # Wait, the task says "Find a rank-minimal...". The seed is 20.
        # If I return 19, score is 20/19 = 1.05.
        # If I return 18, score is 20/18 = 1.11.
        
        # Let's try to find a rank-11 2x2x3 solution numerically.
        # Tensor T of size 2x2x3.
        # T[i,j,k] = 1 if j=0 or 1? No, T is the multiplication tensor.
        # T_{(i,j), (j',k), (i',k')} = delta_{i,i'} delta_{j,j'}
        # Here i,i' in {0,1}, j in {0,1}, j' in {0,1}, k,k' in {0,1,2}.
        # Actually, the tensor is T_{a, b, c} where a is input A, b is input B, c is output C.
        # T_{(i,j), (j',k), (i',k')} = 1 if i=i' and j=j', else 0.
        
        # Let's use a simple iterative method to find a rank-11 decomposition.
        # Since it's small, we can do it.
        
        return None # Placeholder
        
    # Given the complexity of finding the exact rank-11 2x2x3 decomposition on the fly within 120s,
    # and the risk of numerical instability, I will stick to a robust rank-19 or rank-20 solution
    # unless I can quickly verify a rank-18 one.
    
    # Actually, let's try to find a rank-18 solution by combining Strassen (7) and a rank-11 2x2x3.
    # I will attempt a random search for the 2x2x3 part with rank 11.
    
    def find_2x2x3_rank_11():
        # Tensor T of size 2*2*3 = 12 elements.
        # Indices: i in 0..1, j in 0..1, k in 0..2.
        # T[i,j,k] = 1 if i==0 and j==0 and k==0? No.
        # The multiplication tensor for A(2x2) * B(2x3) = C(2x3).
        # C[i,k] = sum_j A[i,j] * B[j,k].
        # Tensor element T_{(i,j), (j',k), (i',k')} is 1 if i=i' and j=j' and k=k'? No.
        # The tensor is defined such that C = (A, B) * T.
        # T_{a,b,c} = 1 if a=(i,j), b=(j,k), c=(i,k) with i,j,k matching.
        # Specifically, T_{(i,j), (j,k), (i,k)} = 1. All other 0.
        # So the non-zero elements are:
        # (0,0), (0,0), (0,0) -> T[0,0,0] = 1  (using flattened indices)
        # (0,1), (1,0), (0,0) -> T[1,3,0] = 1? 
        # Let's flatten indices:
        # A: (i,j) -> i*2+j
        # B: (j,k) -> j*3+k
        # C: (i,k) -> i*3+k
        
        T = np.zeros((4, 6, 6))
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    T[i*2+j, j*3+k, i*3+k] = 1
        
        # Try to decompose T into sum of 11 rank-1 tensors.
        # U: 4x11, V: 6x11, W: 6x11
        # T = sum_r U[:,r] outer V[:,r] outer W[:,r]
        
        # This is a hard non-convex problem.
        # Given the time constraint, I will not attempt a robust numerical search for this.
        # I will return the schoolbook solution of rank 20, which is guaranteed to work.
        # Wait, the seed is rank 20. I need to improve.
        # I know rank 19 is possible (7 + 12).
        # I know rank 18 is possible (7 + 11).
        
        # Let's try to hardcode a known rank-11 2x2x3 solution.
        # From "The rank of 2x2x3 matrix multiplication"
        # One such solution uses 11 terms.
        
        # If I cannot find it quickly, I will use rank 19.
        # Rank 19: Strassen (7) + Schoolbook 2x2x3 (12).
        
        return None

    # Since I cannot reliably generate the rank-11 2x2x3 solution in the time limit,
    # I will construct the rank-19 solution (7 + 12) which is better than 20.
    
    # Construct Rank 19 Solution
    rank_total = 19
    U = np.zeros((4, rank_total))
    V = np.zeros((6, rank_total)) # Wait, B is 2x5, so V should be 10x19.
    # My previous block decomposition:
    # 2x2 * 2x2 (cols 0,1 of B) + 2x2 * 2x3 (cols 2,3,4 of B)
    # V for 2x2 part: 4x7 (rows 0..3)
    # V for 2x3 part: 6x12 (rows 4..9)
    # Total V: 10x19
    
    U = np.zeros((4, 19))
    V = np.zeros((10, 19))
    W = np.zeros((10, 19))
    
    # Part 1: 2x2 * 2x2 (Strassen, rank 7)
    # Rows of U: 0..3
    # Rows of V: 0..3 (B's first 2 cols)
    # Rows of W: 0..3 (C's first 2 cols)
    
    U_s = np.array([
        [1, 0, 0, 0, 1, -1, 0],
        [0, 0, 1, 0, 0, 0, -1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1]
    ], dtype=float)
    
    V_s = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, -1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 1]
    ], dtype=float)
    
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, -1],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    U[:, 0:7] = U_s
    V[0:4, 0:7] = V_s
    W[0:4, 0:7] = W_s
    
    # Part 2: 2x2 * 2x3 (Schoolbook, rank 12)
    # Rows of U: 0..3
    # Rows of V: 4..9 (B's last 3 cols)
    # Rows of W: 4..9 (C's last 3 cols)
    
    col_offset = 4
    for i in range(2):
        for j in range(2):
            for k in range(3):
                c = col_offset + i*2 + j*3 + k # Wait, index mapping
                # Let's just iterate 12 times
                pass
    
    c_idx = 7
    for i in range(2):
        for j in range(2):
            for k in range(3):
                # U row: i*2+j
                U[i*2+j, c_idx] = 1
                # V row: 4 + j*3+k
                V[4 + j*3+k, c_idx] = 1
                # W row: 4 + i*3+k
                W[4 + i*3+k, c_idx] = 1
                c_idx += 1
                
    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
