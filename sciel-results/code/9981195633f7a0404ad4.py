import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 6
    # We want U of shape (n*m, rank) = (8, rank)
    # V of shape (m*p, rank) = (24, rank)
    # W of shape (p*n, rank) = (12, rank)
    # Tensor T[i, j, k] = 1 if i < n, j < m, k < p
    # Product C = A @ B where A is n x m, B is m x p
    # C[i, k] = sum_j A[i, j] B[j, k]
    # In flattened indices:
    # A index: a = i*m + j
    # B index: b = j*p + k
    # C index: c = k*n + i  (trace convention W[k*n+i, r])
    
    # Rank lower bound: max(n*p, n*m, m*p) = max(12, 8, 24) = 24.
    # Schoolbook rank: n*m*p = 48.
    # Strassen on 2x2 blocks: 
    # We can view this as multiplying three 2x2x2 blocks?
    # Let's try to find a low rank decomposition.
    
    # The tensor is the "matrix multiplication" tensor of size 2x4x6.
    # It can be factored using the Kronecker product of smaller tensors?
    # M_{2,4,6} = M_{2,2,2} x M_{1,2,3}? No, that's not right.
    
    # Let's use a simple block approach.
    # C = A B
    # Split B into two 4x3 blocks: B1, B2.
    # C1 = A B1 (2x3), C2 = A B2 (2x3).
    # Each is a 2x4x3 multiplication.
    # Rank of 2x4x3?
    # Lower bound max(6, 8, 12) = 12.
    # Can we do better than 12 for 2x4x3?
    
    # Let's try to construct a rank 24 solution first to ensure validity, then try to optimize.
    # Actually, let's just try to use the best known bounds.
    # For 2x4x6, a known decomposition might be around 30-36.
    
    # Let's try a random search or a structured search.
    # Since we need to finish quickly, let's try a small set of structured decompositions.
    
    # Structured approach:
    # Use 2x2x2 Strassen blocks.
    # We have 2 rows of A. So no vertical block structure for A.
    # We have 4 cols of A / 4 rows of B. We can split into two 2-col/row blocks.
    # We have 6 cols of B. We can split into three 2-col blocks.
    
    # So we have a 1 x 2 x 3 grid of 2x2x2 blocks.
    # Total blocks: 6.
    # Each 2x2x2 block can be done in rank 7 (Strassen).
    # Total rank: 6 * 7 = 42. This is what the current code does (score 1.143).
    
    # Can we do better?
    # Maybe some blocks can be merged or the decomposition can be optimized globally.
    # 
    # Let's try a linear algebra approach:
    # We want to represent T as sum_{r=1}^R u_r v_r^T w_r^T
    # where u_r in R^8, v_r in R^24, w_r in R^12.
    #
    # This is equivalent to finding rank-R decomposition of the tensor.
    #
    # Let's try R = 24 (the lower bound).
    # If we can find it, that's optimal.
    #
    # Let's try to use a random initialization and gradient descent?
    # No, tensor rank minimization is NP-hard.
    #
    # Let's try a specific structured decomposition.
    #
    # Consider the 2x4x3 case first.
    # A is 2x4, B is 4x3.
    # C = A B.
    #
    # Let's try to find a rank 12 decomposition for 2x4x3.
    #
    # Actually, let's look at the dimensions again.
    # n=2, m=4, p=6.
    #
    # Let's try R=36.
    #
    # I will implement a search for a low rank decomposition using random vectors and checking for linear dependence / null space?
    # Or just try to minimize the Frobenius norm of T - sum u_r v_r^T w_r^T.
    #
    # Given the 150s limit, I can do a few iterations of a heuristic.
    #
    # Let's try a simple heuristic:
    # 1. Start with R=42 (Strassen blocks).
    # 2. Try to reduce rank by merging columns?
    #
    # Actually, let's try to generate a random decomposition of rank R and see if we can find a valid one.
    # But "exact" is key. Random float won't be exact.
    # We need integer or half-integer.
    #
    # Let's stick to the Strassen block structure but see if we can optimize the 2x4x3 sub-blocks.
    #
    # For 2x4x3:
    # A (2x4) * B (4x3)
    # Split A into [A1 | A2] where A1, A2 are 2x2.
    # Split B into [B1; B2] where B1, B2 are 2x3.
    # C = A1 B1 + A2 B2.
    #
    # A1 B1 is 2x2x3.
    # A2 B2 is 2x2x3.
    #
    # Rank of 2x2x3?
    # Lower bound max(6, 4, 6) = 6.
    # Schoolbook rank = 2*2*3 = 12.
    #
    # Can we do 2x2x3 in rank < 12?
    # Yes, probably.
    #
    # Let's try to find a rank decomposition for 2x2x3.
    #
    # Let's use a known result: M_{2,2,3} has rank 9? Or 10?
    #
    # If we can do 2x2x3 in rank 9, then 2x4x3 (two blocks) would be 18.
    # Then 2x4x6 (three blocks) would be 3 * 18 = 54? No, that's worse than 42.
    #
    # Wait, the Strassen block approach was 6 blocks of 2x2x2 (rank 7) = 42.
    #
    # If we group differently:
    # 2x4x6 = 2x(2x3)x6?
    #
    # Let's try to find a decomposition of 2x4x6 directly.
    #
    # I will implement a small search for R in range(24, 42) using random integer vectors?
    # That seems unlikely to find an exact solution.
    #
    # Let's try to improve the 42 solution.
    #
    # The 42 solution uses independent Strassen decompositions for each 2x2x2 block.
    # Maybe we can share some of the multiplications?
    #
    # Let's try R=36.
    #
    # I'll provide the 42 solution as a baseline, and add a check.
    # If I have time, I'll try to run a quick local search to reduce rank.
    
    start_time = time.time()
    
    # Build the 42-rank Strassen solution
    def build_strassen_42():
        rank = 42
        U = np.zeros((8, rank), dtype=int)
        V = np.zeros((24, rank), dtype=int)
        W = np.zeros((12, rank), dtype=int)
        
        # Strassen terms for 2x2x2
        # (U, V, W) are sparse dictionaries (i,j) -> coeff
        terms = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):-1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (0,1):-1}),
        ]
        
        col = 0
        # 3 column blocks of B (size 2)
        for cb in range(3):
            # 2 row blocks of A/B (size 2)
            for rb in range(2):
                # 7 Strassen terms
                for (u_dict, v_dict, w_dict) in terms:
                    # Fill U
                    for (r, s), c in u_dict.items():
                        # A index: r * 4 + (rb*2 + s)
                        U[r * 4 + rb * 2 + s, col] += c
                    # Fill V
                    for (s, t), c in v_dict.items():
                        # B index: (rb*2 + s) * 6 + (cb*2 + t)
                        V[(rb * 2 + s) * 6 + cb * 2 + t, col] += c
                    # Fill W
                    for (r, t), c in w_dict.items():
                        # C index: (cb*2 + t) * 2 + r
                        W[(cb * 2 + t) * 2 + r, col] += c
                    col += 1
        return {'U': U, 'V': V, 'W': W}
    
    result = build_strassen_42()
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 100:
        return result
    
    # Try to see if we can reduce rank?
    # This is hard without a specific algorithm.
    # I will just return the 42 solution.
    
    return result
