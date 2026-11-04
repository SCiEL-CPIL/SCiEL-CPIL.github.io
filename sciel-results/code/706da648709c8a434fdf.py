import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = 4, 4, 5
    
    # We'll try to build a decomposition with rank < 80
    # Strategy: Use Strassen for the 4x4 part (rank 49) and handle the 5th column separately
    
    # For now, let's return a known good solution.
    # The schoolbook is rank 80.
    # Can we do better?
    # Let's try a simple approach: use the schoolbook but see if we can merge terms.
    # Actually, let's just return the schoolbook to be safe, but the task asks for rank-minimal.
    # The current score is 1.0 (seed). If I return 80, score is 1.0.
    # I need to find a rank < 80 to improve.
    
    # Let's attempt a small search for a lower rank decomposition.
    # This is a hard problem. Let's try to verify if rank 79 is possible? Unlikely.
    # Let's try to construct a rank 72 solution?
    
    # Actually, let's look at the structure.
    # A (4x4) * B (4x5) = C (4x5)
    # We can split B into B1 (4x4) and B2 (4x1).
    # C1 = A * B1 (4x4)
    # C2 = A * B2 (4x1)
    
    # Rank(C1) <= 49 (Strassen)
    # Rank(C2) <= 4 (Schoolbook: each output element is a dot product of 4 terms)
    # Total rank = 49 + 4 = 53.
    
    # Let's implement the Strassen 4x4 decomposition.
    # This is complex. Let's use a known Strassen 4x4 algorithm.
    # Or simpler: use the fact that 4x4x4 multiplication has rank <= 49.
    # I will hardcode the Strassen 4x4 factors if I can find them, or use a simpler bound.
    
    # Alternative: Use the schoolbook for the 4x4 part (rank 64) and 4 for the 4x1 part.
    # Total rank = 68.
    # 68 < 80, so score = 80/68 = 1.176.
    # This is a safe improvement.
    
    # Let's construct the rank 68 solution.
    # Part 1: A (4x4) * B1 (4x4) -> C1 (4x4)
    # Schoolbook: 16*4 = 64 terms.
    # Part 2: A (4x4) * B2 (4x1) -> C2 (4x1)
    # Schoolbook: 4*4 = 16 terms? No, 4 output elements, each is sum of 4 products.
    # So 4 terms if we can reuse? No, schoolbook is 4*4=16 terms.
    # Wait, C2_i = sum_j A_ij B2_j.
    # We can use 4 terms: for each i, one term? No, the term must be a product of U,V,W.
    # Term r: U[i,j] V[j,k] W[k,i].
    # For C2 (which is k=4 in the 4x5 output? No, C is 4x5. Indices i in 0..3, k in 0..4).
    # C2 corresponds to k=4.
    # We need C[i,4] = sum_j A[i,j] B[j,4].
    # We can use 4 terms:
    # For j in 0..3:
    #   Term r_j:
    #   U[i,j] = delta_{i,i}? No.
    #   We need to isolate A[i,j] and B[j,4].
    #   If we use a term for each (i,j) pair, that's 16 terms.
    #   Can we do better?
    #   If we fix j, we need to compute sum_j A[i,j] B[j,4] for all i.
    #   This is a matrix-vector product. Rank 4.
    #   So yes, 4 terms?
    #   Let's check.
    #   Term t (t=0..3):
    #   We want to compute the contribution of B[t,4] to all C[i,4].
    #   C[i,4] += A[i,t] * B[t,4].
    #   So for each t, we have a term that multiplies A[i,t] and B[t,4] and adds to C[i,4].
    #   U[i,j] for this term: 1 if j==t, else 0. (Selects column t of A)
    #   V[j,k] for this term: 1 if j==t and k==4, else 0. (Selects element B[t,4])
    #   W[k,i] for this term: 1 if k==4, else 0. (Adds to C[i,4])
    #   Check: Sum over j of U[i,j] V[j,4] W[4,i] = U[i,t] * V[t,4] * W[4,i] = 1 * 1 * 1 = 1?
    #   Wait, the sum in the tensor decomposition is over the rank index r.
    #   For a fixed (i, k=4), the value is sum_r U[i,j_r] V[j_r, 4] W[4, i].
    #   Here j_r is the row index of V, which corresponds to the column index of U.
    #   In my term t:
    #   U[i,j] is non-zero only at j=t. So U[i,t]=1.
    #   V[j,k] is non-zero only at j=t, k=4. So V[t,4]=1.
    #   W[k,i] is non-zero only at k=4. So W[4,i]=1.
    #   So the term contributes 1 * 1 * 1 = 1 to the sum?
    #   No, the term contributes U[i,t] * V[t,4] * W[4,i].
    #   But the actual value we want is A[i,t] * B[t,4].
    #   The U, V, W matrices are constant coefficients. They don't depend on A and B.
    #   The decomposition is: C[i,k] = sum_r U[i,j]_r * A[i,j] * V[j,k]_r * B[j,k] * W[k,i]_r.
    #   Wait, the standard form is:
    #   C[i,k] = sum_r sum_j U[i,j]_r * A[i,j] * V[j,k]_r * B[j,k] * W[k,i]_r ?
    #   No, the tensor T_{i,j,k} = C_{i,k} = A_{i,j} B_{j,k}.
    #   The decomposition is T_{i,j,k} = sum_r U[i,j]_r V[j,k]_r W[k,i]_r.
    #   Wait, the indices are:
    #   U has shape (n*m, rank). Index (i*m+j, r).
    #   V has shape (m*p, rank). Index (j*p+k, r).
    #   W has shape (p*n, rank). Index (k*n+i, r).
    #   The condition is: For all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if it's the identity?
    #   No, the tensor for matrix multiplication is T[i,j,k] = delta_{i,i} delta_{j,j} delta_{k,k}?
    #   The tensor T is defined such that <T, A, B> = <C, I>.
    #   T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    #   So T[i,j,k] = 1 for all valid i,j,k? No.
    #   The multiplication tensor has entries T[i,j,k] = 1 if we are computing C[i,k] from A[i,j] and B[j,k].
    #   Actually, T[i,j,k] is 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    #   Wait, no. The tensor is the sum of all terms A[i,j]B[j,k] that form C[i,k].
    #   For a specific (i,k), the term A[i,j]B[j,k] appears.
    #   So the tensor entry T[i,j,k] is 1.
    #   We want to decompose T such that T[i,j,k] = sum_r U[i,j]_r V[j,k]_r W[k,i]_r.
    #   Here U[i,j]_r is the r-th component of U at index (i,j).
    #   So for the 4x1 part (k=4):
    #   We need T[i,j,4] = 1 for all i,j.
    #   We proposed 4 terms.
    #   Term t:
    #   U[i,j]_t = 1 if j==t, else 0.
    #   V[j,k]_t = 1 if j==t and k==4, else 0.
    #   W[k,i]_t = 1 if k==4, else 0.
    #   Let's check the sum for a fixed (i, j, k=4).
    #   Sum_t U[i,j]_t V[j,4]_t W[4,i]_t.
    #   If j != t, U[i,j]_t = 0.
    #   If j == t, U[i,j]_t = 1. V[j,4]_t = 1 (since k=4). W[4,i]_t = 1.
    #   So the sum is 1 for the term where t=j.
    #   So T[i,j,4] = 1. Correct.
    #   So we need 4 terms for the 4x1 part.
    
    # Total rank = 64 (for 4x4) + 4 (for 4x1) = 68.
    
    rank = 68
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    r = 0
    # Part 1: 4x4 multiplication (k in 0..3)
    for i in range(n):
        for j in range(m):
            for k in range(4):  # Only first 4 columns
                U[i*m + j, r] = 1.0
                V[j*p + k, r] = 1.0
                W[k*n + i, r] = 1.0
                r += 1
    
    # Part 2: 4x1 multiplication (k=4)
    for t in range(4):  # t is the column index of A / row index of B
        # Term t
        for i in range(n):
            U[i*m + t, r] = 1.0
        for j in range(m):
            if j == t:
                V[j*p + 4, r] = 1.0
        for k in range(p):
            if k == 4:
                for i in range(n):
                    W[k*n + i, r] = 1.0
        r += 1
    
    # Verify
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, r] = 1.0
                    V[j*p + k, r] = 1.0
                    W[k*n + i, r] = 1.0
                    r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
