import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # The multiplication tensor T has shape (n*m, m*p, p*n) = (4, 22, 22)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
    # We want T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    
    # Key insight: For 2x2 * 2x11, we can use Strassen on pairs of columns.
    # 11 columns = 5 pairs + 1 single
    # Each pair uses 7 Strassen terms, single uses 4 schoolbook terms
    # Total: 5*7 + 4 = 39
    
    # But can we do better? Let's think about whether the single column
    # can share structure with adjacent pairs.
    
    # Actually, let me reconsider. The standard Strassen for 2x2*2x2 gives rank 7.
    # For 2x2*2x1 (a single column), the schoolbook rank is 4.
    # Can we do better for a single column? No, because it's essentially
    # computing two independent dot products of length 2, which has tensor rank 4
    # (it's the sum of two rank-2 tensors, each being an outer product).
    
    # Wait, let me reconsider. For a single column k:
    # M[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # M[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # This is a 2x2x2 tensor (indices i, j, and the single k).
    # The tensor is: T[i,j] for the (i,k) output where j is the B row.
    # Actually, for fixed k, the relevant sub-tensor has indices:
    #   U index: i*m+j (4 values: 0,1,2,3)
    #   V index: j*p+k (2 values: p+k, p+1+k)  
    #   W index: k*n+i (2 values: k*n, k*n+1)
    
    # T[i*m+j, j*p+k, k*n+i] = 1
    # So: T[0, p+k, k*n] = 1  (i=0,j=0)
    #     T[1, p+1+k, k*n] = 1  (i=0,j=1)
    #     T[2, p+k, k*n+1] = 1  (i=1,j=0)
    #     T[3, p+1+k, k*n+1] = 1  (i=1,j=1)
    
    # This is a 4x2x2 tensor with 4 nonzero entries. Its rank is 4 (each nonzero
    # entry is in a different "corner" and they can't be combined).
    
    # So 39 seems tight. But wait - can we use a different decomposition?
    
    # Let me try a completely different approach: use the fact that the problem
    # is essentially computing A*B where A is 2x2 and B is 2x11.
    
    # We can write this as: for each of the 11 columns of B, compute A * b_col.
    # A * b_col is a 2-vector. The map (A, b_col) -> A*b_col is bilinear.
    
    # The tensor for A*b_col (where b_col is 2x1) has rank 4 as shown above.
    # So 11 columns * 4 = 44 for schoolbook.
    
    # With Strassen on pairs: 5*7 + 4 = 39.
    
    # Can we go below 39? Let's think about whether we can merge the single
    # column with a pair to get a 3-column block with rank < 7+4=11.
    
    # For 2x2 * 2x3, what's the minimal rank?
    # Schoolbook: 12. Strassen on first 2 + schoolbook on last: 7+4=11.
    # Can we do 10 or less?
    
    # The 2x2*2x3 tensor has 12 nonzero entries. The question is whether
    # it can be decomposed with rank < 11.
    
    # Actually, I think for 2x2*2xp, the rank is:
    # - p even: 7*(p/2) 
    # - p odd: 7*((p-1)/2) + 4
    
    # So for p=11: 7*5 + 4 = 39.
    
    # Let me verify this is correct by constructing the rank-39 decomposition
    # and checking it.
    
    def build_rank39():
        """Build a rank-39 decomposition using Strassen for 5 column pairs and schoolbook for 1 column."""
        rank = 39
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        
        col = 0
        
        # Process 5 pairs of columns: (0,1), (2,3), (4,5), (6,7), (8,9)
        for pair_idx in range(5):
            k0 = 2 * pair_idx
            k1 = 2 * pair_idx + 1
            
            # Strassen's 7 products for C = A * B_pair where B_pair is 2x2
            # A = [[A00, A01], [A10, A11]]
            # B_pair = [[B0k0, B0k1], [B1k0, B1k1]]
            
            # M1 = (A00 + A11) * (B0k0 + B1k1)
            U[0, col] = 1; U[3, col] = 1
            V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
            W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
            col += 1
            
            # M2 = (A10 + A11) * B0k0
            U[2, col] = 1; U[3, col] = 1
            V[0*p + k0, col] = 1
            W[k0*n + 1, col] = 1
            col += 1
            
            # M3 = A00 * (B0k1 - B1k1)
            U[0, col] = 1
            V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
            W[k1*n + 0, col] = 1
            col += 1
            
            # M4 = A11 * (B1k0 - B0k0)
            U[3, col] = 1
            V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
            W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
            col += 1
            
            # M5 = (A00 + A01) * B1k1
            U[0, col] = 1; U[1, col] = 1
            V[1*p + k1, col] = 1
            W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
            col += 1
            
            # M6 = (A10 - A00) * (B0k0 + B0k1)
            U[2, col] = 1; U[0, col] = -1
            V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
            W[k1*n + 1, col] = 1
            col += 1
            
            # M7 = (A01 - A11) * (B1k0 + B1k1)
            U[1, col] = 1; U[3, col] = -1
            V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
            W[k0*n + 0, col] = 1
            col += 1
        
        # Process the single column k=10 with schoolbook (4 terms)
        k = 10
        
        # Term 1: A[0,0] * B[0,k] -> M[0,k] += 1
        U[0, col] = 1
        V[0*p + k, col] = 1
        W[k*n + 0, col] = 1
        col += 1
        
        # Term 2: A[0,1] * B[1,k] -> M[0,k] += 1
        U[1, col] = 1
        V[1*p + k, col] = 1
        W[k*n + 0, col] = 1
        col += 1
        
        # Term 3: A[1,0] * B[0,k] -> M[1,k] += 1
        U[2, col] = 1
        V[0*p + k, col] = 1
        W[k*n + 1, col] = 1
        col += 1
        
        # Term 4: A[1,1] * B[1,k] -> M[1,k] += 1
        U[3, col] = 1
        V[1*p + k, col] = 1
        W[k*n + 1, col] = 1
        col += 1
        
        return U, V, W
    
    def verify(U, V, W):
        """Verify the decomposition is exact."""
        n, m, p = DIMENSIONS
        r = U.shape[1]
        
        # Check nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        
        # Check zero entries (sample some to be fast)
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    
                    if i == i2 and j == j2 and k == k2:
                        continue
                    
                    s = 0.0
                    for cc in range(r):
                        s += U[a, cc] * V[b, cc] * W[c, cc]
                    if abs(s) > 1e-9:
                        return False
        
        return True
    
    # Build and verify rank-39
    U, V, W = build_rank39()
    valid = verify(U, V, W)
    
    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 44
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, column] = 1
                V[j*p+k, column] = 1
                W[k*n+i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
