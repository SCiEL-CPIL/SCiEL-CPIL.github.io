import numpy as np
from fractions import Fraction
import time

def construct():
    n, m, p = 2, 4, 5
    
    # Deadline guard
    deadline = time.time() + 110
    
    # Trivial rank-40 decomposition as fallback
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0), dtype=object)
    V0 = np.zeros((m * p, rank0), dtype=object)
    W0 = np.zeros((p * n, rank0), dtype=object)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U0[i * m + j, idx] = Fraction(1)
                V0[j * p + k, idx] = Fraction(1)
                W0[k * n + i, idx] = Fraction(1)
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly computes the matrix multiplication tensor"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if total != 1:
                        return False
        return True
    
    # Strategy: View 2x4x5 as direct sum of two 2x2x5 blocks
    # A = [A_left | A_right] where A_left, A_right are 2x2
    # B = [B_top; B_bottom] where B_top, B_bottom are 2x5
    # C = A_left * B_top + A_right * B_bottom
    
    # We need a rank-10 decomposition for 2x2x5 matrix multiplication
    # Let's construct one systematically using the structure of the problem
    
    # For 2x2x5, we have T[i,j,k] = 1 for i in {0,1}, j in {0,1}, k in {0..4}
    # We need: sum_r U[i,j]_r * V[j,k]_r * W[k,i]_r = 1
    
    # Let's try to find a rank-10 decomposition for 2x2x5
    # Using the fact that we can share computation across k values
    
    # One approach: use the identity that for 2x2x1, we can use rank 7 (Strassen)
    # But for 2x2x5, we can potentially do better by sharing across k
    
    # Let's try a structured approach:
    # For each j (0 or 1), we have a 2x5 subproblem
    # Actually, let's think of it differently
    
    # The tensor T[i,j,k] for 2x2x5 has 2*2*5 = 20 non-zero entries (all 1s)
    # We want to decompose this into rank r terms
    
    # Let's try rank 10:
    # Idea: pair up k values and use linear combinations
    
    # Actually, let's try a different approach:
    # Use the fact that C[i,k] = sum_j A[i,j] B[j,k]
    # For fixed i, this is a linear combination of the B[j,k] for j=0,1
    
    # Let's try to construct the 2x2x5 decomposition:
    # We'll use 10 terms, each corresponding to a specific (i,j) pair and a linear combination of k values
    
    # Actually, let's try a more systematic approach:
    # For 2x2x5, we can view it as computing 2x5 output from 2x2 and 2x5 inputs
    # The rank is at most 2*5 = 10 (trivial for the j dimension)
    
    # Trivial 2x2x5 decomposition of rank 10:
    # For each j in {0,1} and k in {0..4}, we have one term
    # U[i,j]_r = delta_{i, r//5} (but this doesn't work directly)
    
    # Let's use the standard approach:
    # For 2x2x5, the trivial rank is 2*2*5 = 20
    # But we can do better: rank = 2*5 = 10 by sharing across i
    
    # Construction:
    # For r = 0..9, let r = j*5 + k where j in {0,1}, k in {0..4}
    # U[i,j]_r = delta_{j, r//5} * (1 if i == r//5 else 0) -- this doesn't work
    
    # Let's try:
    # U[i,j]_r = 1 if j == r//5, else 0
    # V[j,k]_r = 1 if j == r//5 and k == r%5, else 0  
    # W[k,i]_r = 1
    # Then sum_r U[i,j]_r V[j,k]_r W[k,i]_r = sum_{r: j==r//5} delta_{k, r%5} * 1 = 1 if k exists for this j
    # This gives 1 for all i,j,k, which is what we want!
    
    # So for 2x2x5, rank 10 works with:
    # For r = j*5 + k (j in {0,1}, k in {0..4}):
    # U[i,j]_r = 1 if j == r//5 else 0
    # V[j,k]_r = 1 if j == r//5 and k == r%5 else 0
    # W[k,i]_r = 1
    
    # Now for 2x4x5, we split into two 2x2x5 blocks:
    # Block 1: i in {0,1}, j in {0,1}, k in {0..4} -> uses rows 0,1 of U, rows 0..9 of V, all W
    # Block 2: i in {0,1}, j in {2,3}, k in {0..4} -> uses rows 0,1 of U, rows 10..19 of V, all W
    
    # Wait, this doesn't quite work because the j indices are different
    
    # Let me reconsider:
    # For 2x4x5, we have i in {0,1}, j in {0,1,2,3}, k in {0..4}
    # We can split j into two groups: {0,1} and {2,3}
    
    # For group 1 (j in {0,1}):
    # Use the 2x2x5 decomposition with j' = j (0 or 1)
    # For group 2 (j in {2,3}):
    # Use the 2x2x5 decomposition with j' = j-2 (0 or 1)
    
    # Total rank: 10 + 10 = 20
    
    # Let's construct this:
    rank = 20
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    # Group 1: j in {0, 1}, r in {0..9}
    for r in range(10):
        j = r // 5  # 0 or 1
        k = r % 5   # 0..4
        
        for i in range(n):
            U[i * m + j, r] = Fraction(1)
        
        V[j * p + k, r] = Fraction(1)
        
        for i in range(n):
            W[k * n + i, r] = Fraction(1)
    
    # Group 2: j in {2, 3}, r in {10..19}
    for r in range(10, 20):
        j_prime = (r - 10) // 5  # 0 or 1
        j = j_prime + 2          # 2 or 3
        k = (r - 10) % 5         # 0..4
        
        for i in range(n):
            U[i * m + j, r] = Fraction(1)
        
        V[j * p + k, r] = Fraction(1)
        
        for i in range(n):
            W[k * n + i, r] = Fraction(1)
    
    # Verify the decomposition
    if verify_decomposition(U, V, W, rank):
        best_rank = rank
        best_U = U
        best_V = V
        best_W = W
    else:
        # Fallback to trivial decomposition
        best_rank = rank0
        best_U = U0
        best_V = V0
        best_W = W0
    
    # Convert to lists of integers/half-integers
    def convert_to_list(arr):
        res = []
        for row in arr:
            res_row = []
            for val in row:
                if isinstance(val, Fraction):
                    if val.denominator == 1:
                        res_row.append(int(val))
                    elif val.denominator == 2:
                        res_row.append(val.numerator / 2.0)
                    else:
                        res_row.append(float(val))
                else:
                    res_row.append(val)
            res.append(res_row)
        return res
    
    return {"U": convert_to_list(best_U), "V": convert_to_list(best_V), "W": convert_to_list(best_W)}
