"""Find rank-minimal tensor decomposition for 2x5 matrix times 5x5 matrix."""
import numpy as np
from itertools import product
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _verify_np(U, V, W, n, m, p):
    """Fast verification using numpy with tolerance for float, then exact check"""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                s = prod.sum()
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find lower rank using structured/randomized approach
    # The tensor T_{i,j,k} = 1 for all i,j,k
    # We need sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1
    
    # Strategy: Use the fact that n=2 is small.
    # For each column c, U[i*m+j,c] for i in {0,1}, j in {0..4} gives 10 values
    # V[j*p+k,c] for j in {0..4}, k in {0..4} gives 25 values
    # W[k*n+i,c] for k in {0..4}, i in {0,1} gives 10 values
    
    # Key insight: If we can make the decomposition separable in some way...
    
    # Try rank from lower bounds upward
    # Lower bound: max(n*m, m*p, p*n) = max(10, 25, 10) = 25
    # Also, the tensor has a certain structure.
    
    # Let's try a greedy construction:
    # For each (i,j,k), we need the sum over c to be 1.
    # If we set up columns such that each column "covers" certain (i,j,k) tuples...
    
    # Alternative: Use the fact that for fixed j, we need:
    # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for all i,k
    # This is a bilinear equation in i,k for each j.
    
    # Let's try: For each j, define matrices A_j (n x rank) where A_j[i,c] = U[i*m+j,c]
    # B_j (rank x p) where B_j[c,k] = V[j*p+k,c]  
    # C_j (p x n) where C_j[k,i] = W[k*n+i,c]
    # Then we need A_j @ B_j @ C_j = I_{n x p}... wait, that's not right dimensionally.
    
    # Actually: (A_j @ B_j) is n x p, and we need (A_j @ B_j)[i,k] * something = 1?
    # No: sum_c A_j[i,c] * B_j[c,k] * C_j[k,i]... C_j depends on k and i.
    
    # Let me re-index: W[k*n+i, c] so for fixed c, W_c is a vector of length p*n.
    # Reshape W_c as a p x n matrix: Wc[k,i] = W[k*n+i, c]
    # Similarly U_c reshaped: Uc[i,j] = U[i*m+j, c], size n x m
    # V_c reshaped: Vc[j,k] = V[j*p+k, c], size m x p
    
    # Then the condition is: for all i,j,k:
    # sum_c Uc[i,j] * Vc[j,k] * Wc[k,i] = 1
    
    # This is: sum_c trace(Uc * Vc * Wc) ... no, it's an element-wise product summed.
    
    # For fixed i,k: sum_j sum_c Uc[i,j] * Vc[j,k] * Wc[k,i] = 1
    # Let M_c = Uc @ Vc (n x p matrix), then M_c[i,k] = sum_j Uc[i,j] * Vc[j,k]
    # Then condition: sum_c M_c[i,k] * Wc[k,i] = 1 for all i,k
    
    # So: sum_c (Uc @ Vc)[i,k] * Wc[k,i] = 1
    
    # This is a constraint on the triple (Uc, Vc, Wc) for each c.
    
    # Strategy: Try to find rank r such that we can write 1 = sum of r terms of the form
    # (Uc @ Vc)[i,k] * Wc[k,i]
    
    # For a single term: if Uc @ Vc = E_{i,k} (matrix unit) and Wc[k,i] = 1, that works for one (i,k).
    # But we need it for ALL (i,k) simultaneously with the same Uc, Vc, Wc.
    
    # If Uc @ Vc = I (identity, n x p... but n=2, p=5, so not square). 
    # Hmm, Uc is 2x5, Vc is 5x5, so Uc@Vc is 2x5.
    # Wc is 5x2.
    # We need sum_c (Uc@Vc)[i,k] * Wc[k,i] = 1 for all i in {0,1}, k in {0..4}.
    
    # If we set Uc@Vc = some matrix M_c (2x5) and Wc = some matrix N_c (5x2),
    # then we need sum_c M_c[i,k] * N_c[k,i] = 1.
    
    # Note M_c[i,k] * N_c[k,i] is not a standard trace. It's sum over i,k of M_c[i,k]*N_c[k,i] 
    # but we need it element-wise = 1.
    
    # Let's try: if M_c = all-ones 2x5 matrix, N_c = all-ones 5x2 matrix,
    # then M_c[i,k]*N_c[k,i] = 1*1 = 1 for each (i,k). So one term works!
    
    # Can we realize M_c = J (2x5 all ones) as Uc @ Vc with Uc (2x5) and Vc (5x5)?
    # Yes! Set Uc = J (2x5) and Vc = I_5 (5x5 identity). Then Uc@Vc = J@I = J. 
    # Wait, J is 2x5, I is 5x5, J@I = J. Yes!
    # And N_c = Wc should be all-ones 5x2.
    
    # So: Uc[i,j] = 1 for all i,j (2x5 all ones)
    #     Vc[j,k] = delta_{j,k} (5x5 identity)
    #     Wc[k,i] = 1 for all k,i (5x2 all ones)
    
    # Check: (Uc@Vc)[i,k] = sum_j Uc[i,j]*Vc[j,k] = sum_j 1 * delta_{j,k} = 1.
    # Wc[k,i] = 1.
    # Product = 1*1 = 1. 
    
    # So RANK 1 WORKS?! Let me verify more carefully.
    
    # U has shape (n*m, rank) = (10, 1). U[i*m+j, 0] = Uc[i,j] = 1 for all i,j.
    # V has shape (m*p, rank) = (25, 1). V[j*p+k, 0] = Vc[j,k] = delta_{j,k}.
    # W has shape (p*n, rank) = (10, 1). W[k*n+i, 0] = Wc[k,i] = 1 for all k,i.
    
    # Verify: for all i,j,k: U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0]
    # = 1 * delta_{j,k} * 1 = delta_{j,k}
    
    # But we need this to equal 1 for ALL i,j,k! Not just when j=k!
    
    # Oh no, I made an error. The condition is sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # for ALL i,j,k, not summed over j. The j is a free index here!
    
    # So for a given (i,j,k), the value must be 1. If j != k, delta_{j,k} = 0, which is wrong.
    
    # I confused myself. Let me re-derive.
    
    # The matrix multiplication: (A·B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is defined by T[i,j,k] such that (A·B)[i,k] = sum_{i',j,k'} T[i',j,k'] A[i',j] B[j,k']
    # Comparing: T[i,j,k] = 1 if i'=i and k'=k, 0 otherwise. But since we're summing over j,
    # T[i,j,k] = 1 for all valid i,j,k.
    
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # So we need: for ALL i in {0,1}, j in {0..4}, k in {0..4}:
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # This is 50 independent equations (one for each (i,j,k) triple).
    
    # With rank 1: U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1 for all i,j,k.
    # Let a_{i,j} = U[i*m+j,0], b_{j,k} = V[j*p+k,0], c_{k,i} = W[k*n+i,0].
    # Need a_{i,j} * b_{j,k} * c_{k,i} = 1 for all i,j,k.
    
    # For fixed j: a_{0,j}*b_{j,0}*c_{0,0} = 1, a_{0,j}*b_{j,1}*c_{1,0} = 1, etc.
    # From a_{0,j}*b_{j,k}*c_{k,0} = 1 for all k: b_{j,k}*c_{k,0} is constant in k.
    # Similarly a_{1,j}*b_{j,k}*c_{k,1} = 1 for all k.
    
    # Let's try a_{i,j} = 1, c_{k,i} = 1, then b_{j,k} = 1 for all j,k.
    # Check: 1*1*1 = 1. Yes! This works!
    
    # So U[i*m+j,0] = 1 for all i,j
    #    V[j*p+k,0] = 1 for all j,k
    #    W[k*n+i,0] = 1 for all k,i
    
    # Rank 1 decomposition! Let me double-check with the original formulation.
    
    U_test = [[1] * 1 for _ in range(n * m)]  # 10x1, all ones
    V_test = [[1] * 1 for _ in range(m * p)]  # 25x1, all ones
    W_test = [[1] * 1 for _ in range(p * n)]  # 10x1, all ones
    
    if _verify(U_test, V_test, W_test, n, m, p):
        return {"U": U_test, "V": V_test, "W": W_test}
    
    # If rank 1 doesn't work (it should), try other approaches
    # But based on the math above, rank 1 should work.
    
    # Fallback: return schoolbook
    return best_result

