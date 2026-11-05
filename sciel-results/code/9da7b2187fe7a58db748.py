"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START

def _strassen_2x2_decomposition():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where:
    - U maps input (a,b;c,d) -> 7 products
    - V maps (e,f;g,h) -> 7 products
    - W maps 7 products -> output (ae+bg, af+bh; ce+dg, cf+dh)
    
    Standard Strassen:
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(f-h)
    M4 = d(g-e)
    M5 = (a+b)h
    M6 = (c-a)(e+f)
    M7 = (b-d)(g+h)
    
    Output:
    p = M1 + M4 - M5 + M7
    q = M3 + M5
    r = M2 + M4
    s = M1 - M2 + M3 + M6
    """
    n = 2
    rank = 7
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Input ordering: [a, b; c, d] -> [a, b, c, d]
    # Output ordering (trace convention W[k*n+i, r]):
    # For 2x2: output indices are k*2+i where k is row of B, i is row of A
    # Actually for 2x2*2x2: A is 2x2, B is 2x2, C = A*B is 2x2
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # The tensor T[i,j,k] = A[i,k]*B[k,j]*C[i,j]
    # W index: k*n+i where n=2, so W[k*2+i]
    # For output C[i,j], the relevant indices in W are for each (i,j) pair
    # Actually W has shape (p*n, rank) = (2*2, 7) = (4, 7)
    # W[k*n+i, r] where k indexes column of A (row of B), i indexes row of A
    # So W[0] = k=0,i=0, W[1] = k=0,i=1, W[2] = k=1,i=0, W[3] = k=1,i=1
    
    # U: input A flattened as [a, b, c, d] (row-major)
    # V: input B flattened as [e, f, g, h] (row-major)
    
    # M1 = (a+d)(e+h): U contributions: a->1, d->1; V: e->1, h->1
    U[0, 0] = 1  # a
    U[3, 0] = 1  # d
    V[0, 0] = 1  # e
    V[3, 0] = 1  # h
    # M1 contributes to p (C[0,0]) and s (C[1,1])
    # p = C[0,0], W index for (i=0,j=0): we need to think about this differently
    # 
    # Let me reconsider. The tensor T[i,j,k] where C[i,j] = sum_k A[i,k]B[k,j]
    # U has shape (n*m, rank) = (2*2, 7) = (4, 7), indexed by [i*m+j] = [i*2+j]
    # V has shape (m*p, rank) = (2*2, 7) = (4, 7), indexed by [j*p+k] = [j*2+k]
    # W has shape (p*n, rank) = (2*2, 7) = (4, 7), indexed by [k*n+i] = [k*2+i]
    #
    # So for each rank-1 term r:
    # C[i,j] gets contribution W[k*2+i, r] * U[i*2+k, r] * V[k*2+j, r]
    # summed over k
    #
    # For C[i,j] = sum_k A[i,k]B[k,j]:
    # We need: sum_r W[k*2+i, r] * U[i*2+k, r] * V[k*2+j, r] = delta-like
    # Actually the full tensor is T[i,j,k] = 1 for all i,j,k (it's the identity tensor)
    # The decomposition satisfies: T[i,j,k] = sum_r U[i,k,r] * V[k,j,r] * W[i,j,r]
    # Wait, let me re-read the convention.
    
    # The task says: "output factor follows the official trace convention W[k*n+i,r]"
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # For n=2, m=2, p=2: U is (4,rank), V is (4,rank), W is (4,rank)
    # U indexed by [i*m+j] = [i*2+j], V indexed by [j*p+k] = [j*2+k], W indexed by [k*n+i] = [k*2+i]
    #
    # The matrix multiplication tensor T[i,j,k] satisfies:
    # (A*B)[i,j] = sum_k A[i,k] * B[k,j]
    # The tensor decomposition: T[i,j,k] = sum_r U[i*2+k, r] * V[k*2+j, r] * W[k*2+i, r]
    # Wait, that doesn't look right dimensionally. Let me think again.
    
    # Actually, the standard formulation:
    # T is a 3-tensor with indices (i, j, k) where T[i,j,k] = 1 (the multiplication tensor)
    # We want: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] (cyclic)
    # But the task uses a specific indexing:
    # U has n*m = 4 entries, indexed as U[i*m+j, r] = U[i*2+j, r]
    # V has m*p = 4 entries, indexed as V[j*p+k, r] = V[j*2+k, r]
    # W has p*n = 4 entries, indexed as W[k*n+i, r] = W[k*2+i, r]
    #
    # So the decomposition is: for all i in [0,n), j in [0,m), k in [0,p):
    # 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # i.e., 1 = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    
    # This is the standard matrix multiplication tensor.
    
    # For Strassen on 2x2:
    # The 7 products are linear combinations of A and B entries.
    # Let me define them properly.
    
    # Reset
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # A indices: U[0]=a (i=0,j=0), U[1]=b (i=0,j=1), U[2]=c (i=1,j=0), U[3]=d (i=1,j=1)
    # B indices: V[0]=e (j=0,k=0), V[1]=f (j=0,k=1), V[2]=g (j=1,k=0), V[3]=h (j=1,k=1)
    # W indices: W[0] (k=0,i=0), W[1] (k=0,i=1), W[2] (k=1,i=0), W[3] (k=1,i=1)
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # M1 contributes to C[0,0] and C[1,1]
    # C[0,0] = ae+bg: needs W[k*2+0] for k=0,1
    # C[1,1] = cf+dh: needs W[k*2+1] for k=0,1
    # M1 = (a+d)(e+h) = ae+ah+de+dh
    # For C[0,0]: the ae term comes from k=0, i=0 -> W[0,0]
    # For C[1,1]: the dh term comes from k=1, i=1 -> W[3,0]
    # But M1 also has ah and de which are not in C[0,0] or C[1,1] directly
    # This is where the linear combinations come in - we need to combine M1 with other terms
    
    # Actually, in Strassen, the output is:
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # So W for each M_r gives the coefficient in the output.
    # W[k*2+i, r] is the coefficient of M_r in C[i, ?]
    # Wait, W is indexed by [k*n+i] and the output C[i,j] = sum_k sum_r U[i*2+k,r]V[k*2+j,r]W[k*2+i,r]
    # Hmm, this is getting confusing. Let me just verify with the schoolbook.
    
    # Schoolbook: for each (i,j,k), set U[i*2+k, col]=1, V[k*2+j, col]=1, W[k*2+i, col]=1
    # Then sum_r U[i*2+k,r]V[k*2+j,r]W[k*2+i,r] = 1 when r corresponds to (i,j,k), else 0
    # So the sum over r gives 1 for each (i,j,k). Correct.
    
    # For Strassen, we need to find U,V,W such that for all i,j,k:
    # sum_r U[i*2+k,r] * V[k*2+j,r] * W[k*2+i,r] = 1
    
    # This is a system of 8 equations (2*2*2) with 7*4*3 = 84 unknowns.
    # It's underdetermined, so we can use Strassen's specific structure.
    
    # Let me just hardcode the known Strassen decomposition.
    # Reference: the 7 products and their contributions.
    
    # I'll define it by specifying which (i,k) pairs contribute to each U,
    # which (k,j) pairs to each V, and which (k,i) pairs to each W.
    
    # M1 = (a+d)(e+h)
    # U: a is (i=0,k=0), d is (i=1,k=1)
    U[0, 0] = 1  # (i=0,k=0) -> a
    U[3, 0] = 1  # (i=1,k=1) -> d
    # V: e is (k=0,j=0), h is (k=1,j=1)
    V[0, 0] = 1  # (k=0,j=0) -> e
    V[3, 0] = 1  # (k=1,j=1) -> h
    # W: contributes to C[0,0] (i=0) and C[1,1] (i=1)
    # C[0,0] uses k=0 and k=1: W[0,0] for (k=0,i=0), W[2,0] for (k=1,i=0)
    # C[1,1] uses k=0 and k=1: W[1,0] for (k=0,i=1), W[3,0] for (k=1,i=1)
    # From C[0,0] = M1 + M4 - M5 + M7, M1 has coefficient +1
    # From C[1,1] = M1 - M2 + M3 + M6, M1 has coefficient +1
    # But which k,i does it contribute to?
    # M1 = (a+d)(e+h) = ae + ah + de + dh
    # In C[0,0] = ae + bg: only ae from M1
    # In C[1,1] = cf + dh: only dh from M1
    # So W[0,0] should make ae appear in C[0,0]: U[0,0]=1 (a), V[0,0]=1 (e), W[0,0]=1 gives ae to C[0,0]
    # W[3,0] should make dh appear in C[1,1]: U[3,0]=1 (d), V[3,0]=1 (h), W[3,0]=1 gives dh to C[1,1]
    # But ah and de also appear in M1. They need to cancel with other terms.
    # ah: U[0,0]=1, V[3,0]=1, W[3,0]=1 gives ah to C[1,1] (since k=1,i=1 -> W[3,0])
    # Wait, W[k*2+i] with k=1,i=1 is W[3]. So ah contributes to C[1,1] via W[3,0].
    # de: U[3,0]=1, V[0,0]=1, W[0,0]=1 gives de to C[0,0] (since k=0,i=0 -> W[0,0])
    # So M1 contributes: ae to C[0,0], de to C[0,0], ah to C[1,1], dh to C[1,1]
    # We need: C[0,0] gets +ae, C[1,1] gets +dh, and de, ah cancel.
    # W[0,0] = 1: gives ae + de to C[0,0]
    # W[3,0] = 1: gives ah + dh to C[1,1]
    # W[2,0] = 0, W[1,0] = 0
    W[0, 0] = 1
    W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1  # c: (i=1,k=0)
    U[3, 1] = 1  # d: (i=1,k=1)
    V[0, 1] = 1  # e: (k=0,j=0)
    # M2 = ce + de
    # C[1,0] = ce + dg: M2 contributes ce
    # C[1,1] = cf + dh: M2 contributes -de (coefficient -1 in C[1,1])
    # ce: U[2,1]=1, V[0,1]=1, W[0,1]=1 gives ce to C[1,0] (k=0,i=1 -> W[1,1])
    # Wait, W[k*2+i]: k=0,i=1 -> W[1]. So W[1,1]=1 gives ce to C[1,0]
    # de: U[3,1]=1, V[0,1]=1, W[0,1]=1 gives de to C[1,0]
    # But we want de to cancel in C[1,1], not C[1,0]
    # Hmm, I think I'm making an error. Let me reconsider.
    
    # I'll take a different approach: just use numpy to verify and construct.
    # Actually, let me just use the known result and verify numerically.
    
    return None  # Placeholder, will implement properly below


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Start with schoolbook rank 64
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    # Try to reduce using Strassen-like blocks
    # For 2x4x8, we can think of A as 2x4 = 2x(2x2), B as 4x8 = (2x2)x(2x4)
    # Or use block structure.
    
    # A simpler approach: use the fact that 2x4x8 can be decomposed using
    # Strassen on the 2x2 blocks.
    
    # A is 2x4: rows i=0,1; columns j=0,1,2,3
    # B is 4x8: rows k=0,1,2,3; columns l=0,...,7
    # C is 2x8: C[i,l] = sum_k A[i,k] * B[k,l]
    
    # Split k into two groups: k in {0,1} and k in {2,3}
    # C[i,l] = sum_{k=0,1} A[i,k]B[k,l] + sum_{k=2,3} A[i,k]B[k,l]
    
    # For the first sum, A[i,0:2] is 2x2, B[0:2,l] is 2x8
    # We can use Strassen-like decomposition on the 2x2 part of A and B.
    
    # Actually, let me try a direct approach: use random search or known results.
    # The minimal rank for 2x4x8 is known to be at most 56 (via Strassen) and at least 40.
    
    # Let me try to construct a rank-56 decomposition using Strassen.
    # A = [[a00, a01, a02, a03],
    #      [a10, a11, a12, a13]]
    # B = [[b00, b01, ..., b07],
    #      [b10, b11, ..., b17],
    #      [b20, b21, ..., b27],
    #      [b30, b31, ..., b37]]
    
    # Split into two 2x2 blocks in the middle dimension:
    # A_left = [[a00, a01], [a10, a11]], A_right = [[a02, a03], [a12, a13]]
    # B_top = B[0:2, :], B_bottom = B[2:4, :]
    
    # C[i,l] = sum_{k=0,1} A_left[i,k] B_top[k,l] + sum_{k=0,1} A_right[i,k] B_bottom[k,l]
    
    # For each 2x2x8 multiplication, we can use a rank-? decomposition.
    # 2x2x8: A is 2x2, B is 2x8, C is 2x8
    # C[i,l] = sum_{k=0,1} A[i,k] B[k,l]
    # This is like multiplying 2x2 by 2x8.
    # We can think of B as two 2x4 matrices or use Strassen on the 2x2 part.
    
    # For 2x2x2, Strassen gives rank 7.
    # For 2x2x8, we can split the 8 columns into four 2-column groups:
    # C[i, 2t:2t+2] = A * B[:, 2t:2t+2]
    # Each is a 2x2x2 multiplication, rank 7 each, total 4*7 = 28 for one 2x2x8.
    # Two such (left and right) give 56 total.
    
    # So rank 56 is achievable. Let me construct it.
    
    # For each t in 0,1,2,3:
    #   C[i, 2t] = sum_k A_left[i,k] B_top[k, 2t]
    #   C[i, 2t+1] = sum_k A_left[i,k] B_top[k, 2t+1]
    # This is a 2x2x2 multiplication with B being a 2x2 matrix (columns 2t, 2t+1).
    
    # Similarly for A_right and B_bottom.
    
    # Total: 4 (t values) * 2 (left/right) * 7 (Strassen rank) = 56
    
    # Let me implement this.
    
    def strassen_2x2x2():
        """
        Return U (4x7), V (4x7), W (4x7) for 2x2x2 matrix multiplication.
        U indexed by [i*2+k], V by [k*2+j], W by [k*2+i]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # I'll use a known correct Strassen decomposition and verify.
        # Let me define it via the 7 products.
        
        # Product 0: M1 = (a+d)(e+h)
        # a = A[0,0] -> U[0*2+0] = U[0]
        # d = A[1,1] -> U[1*2+1] = U[3]
        # e = B[0,0] -> V[0*2+0] = V[0]
        # h = B[1,1] -> V[1*2+1] = V[3]
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        # M1 contributes to C[0,0] and C[1,1]
        # C[0,0] = M1 + M4 - M5 + M7
        # C[1,1] = M1 - M2 + M3 + M6
        # W[k*2+i]: for C[0,0], i=0; for C[1,1], i=1
        # M1's contribution to C[0,0]: via k=0 (ae) and k=1 (de, but de should cancel)
        # Actually, let me just set W and verify numerically.
        W[0, 0] = 1  # (k=0, i=0)
        W[3, 0] = 1  # (k=1, i=1)
        
        # Product 1: M2 = (c+d)e
        # c = A[1,0] -> U[1*2+0] = U[2]
        # d = A[1,1] -> U[3]
        # e = B[0,0] -> V[0]
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # M2 contributes to C[1,0] (+) and C[1,1] (-)
        W[1, 1] = 1  # (k=0, i=1) for C[1,0]
        W[3, 1] = -1  # (k=1, i=1) for C[1,1]
        
        # Product 2: M3 = a(f-h)
        # a = A[0,0] -> U[0]
        # f = B[0,1] -> V[0*2+1] = V[1]
        # h = B[1,1] -> V[3]
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        # M3 contributes to C[0,1] (+) and C[1,1] (+)
        W[2, 2] = 1  # (k=1, i=0) for C[0,1]
        W[3, 2] = 1  # (k=1, i=1) for C[1,1]
        
        # Product 3: M4 = d(g-e)
        # d = A[1,1] -> U[3]
        # g = B[1,0] -> V[1*2+0] = V[2]
        # e = B[0,0] -> V[0]
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        # M4 contributes to C[0,0] (+) and C[1,0] (+)
        W[0, 3] = 1  # (k=0, i=0) for C[0,0]
        W[1, 3] = 1  # (k=1, i=0) for C[1,0]
        
        # Product 4: M5 = (a+b)h
        # a = A[0,0] -> U[0]
        # b = A[0,1] -> U[0*2+1] = U[1]
        # h = B[1,1] -> V[3]
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # M5 contributes to C[0,1] (+) and C[0,0] (-)
        W[2, 4] = 1  # (k=1, i=0) for C[0,1]
        W[0, 4] = -1  # (k=0, i=0) for C[0,0]
        
        # Product 5: M6 = (c-a)(e+f)
        # c = A[1,0] -> U[2]
        # a = A[0,0] -> U[0]
        # e = B[0,0] -> V[0]
        # f = B[0,1] -> V[1]
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # M6 contributes to C[1,1] (+)
        W[3, 5] = 1  # (k=1, i=1) for C[1,1]
        
        # Product 6: M7 = (b-d)(g+h)
        # b = A[0,1] -> U[1]
        # d = A[1,1] -> U[3]
        # g = B[1,0] -> V[2]
        # h = B[1,1] -> V[3]
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        # M7 contributes to C[0,0] (+)
        W[0, 6] = 1  # (k=0, i=0) for C[0,0]
        
        return U, V, W
    
    # Verify Strassen decomposition
    U_s, V_s, W_s = strassen_2x2x2()
    # Check: for all i,j,k in {0,1}: sum_r U[i*2+k,r]*V[k*2+j,r]*W[k*2+i,r] should be 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.sum(U_s[i*2+k, :] * V_s[k*2+j, :] * W_s[k*2+i, :])
                if abs(val - 1) > 1e-10:
                    print(f"Strassen verification failed: i={i}, j={j}, k={k}, val={val}")
                    # Fallback to schoolbook
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Construct rank-56 decomposition for 2x4x8
    # Split into 8 blocks of 2x2x2
    # A: 2x4, B: 4x8
    # For t in 0,1,2,3 (column pair in B) and s in 0,1 (row pair in B / middle dimension):
    #   A_block = A[:, 2s:2s+2] (2x2)
    #   B_block = B[2s:2s+2, 2t:2t+2] (2x2)
    #   C_block = C[:, 2t:2t+2] += A_block * B_block (2x2)
    
    rank_56 = 56
    U56 = np.zeros((n * m, rank_56))  # (8, 56)
    V56 = np.zeros((m * p, rank_56))  # (32, 56)
    W56 = np.zeros((p * n, rank_56))  # (16, 56)
    
    col = 0
    for s in range(2):  # middle dimension block
        for t in range(4):  # output column block
            # A indices: i in {0,1}, k in {2s, 2s+1}
            # U index: i*m + k = i*4 + (2s + k_local)
            # V indices: k in {2s, 2s+1}, j in {2t, 2t+1}
            # V index: k*p + j = (2s + k_local)*8 + (2t + j_local)
            # W indices: k in {2s, 2s+1}, i in {0,1}
            # W index: k*n + i = (2s + k_local)*2 + i
            
            for r in range(7):
                # U: for each (i, k_local) in Strassen U
                for i in range(2):
                    for k_local in range(2):
                        U_val = U_s[i * 2 + k_local, r]
                        if U_val != 0:
                            U56[i * m + (2 * s + k_local), col] += U_val
                
                # V: for each (k_local, j_local) in Strassen V
                for k_local in range(2):
                    for j_local in range(2):
                        V_val = V_s[k_local * 2 + j_local, r]
                        if V_val != 0:
                            V56[(2 * s + k_local) * p + (2 * t + j_local), col] += V_val
                
                # W: for each (k_local, i) in Strassen W
                for k_local in range(2):
                    for i in range(2):
                        W_val = W_s[k_local * 2 + i, r]
                        if W_val != 0:
                            W56[(2 * s + k_local) * n + i, col] += W_val
                
                col += 1
    
    # Verify the rank-56 decomposition
    # Check: for all i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}:
    # sum_r U56[i*4+k, r] * V56[k*8+j, r] * W56[k*2+i, r] should be 1
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U56[i * m + k, :] * V56[k * p + j, :] * W56[k * n + i, :])
                if abs(val - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
    
    if valid:
        return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
    else:
        # Fallback to schoolbook
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
