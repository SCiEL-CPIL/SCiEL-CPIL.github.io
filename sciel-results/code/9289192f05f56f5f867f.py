"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W form a valid tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c] == W[k*n+i,c]
    """
    n, m, p = n, m, p
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]
    # And check it equals sum_c W[k*n+i,c] * (something)
    # Actually the convention is: (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: M[i,j,k] = A[i,j]*B[j,k] -> C[i,k]
    # Decomposition: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, let me re-read the convention.
    # The standard tensor decomposition for matmul:
    # (AB)_{ik} = sum_j A_{ij} B_{jk}
    # Tensor T_{ijk} = e_{ij} \otimes e_{jk} \otimes e_{ki} (in some basis)
    # Actually the matmul tensor is: T[i,j,k] such that sum_{i,j} T[i,j,k] A[i,j] B[j,k] = C[k]... 
    # No. Let me think again.
    # The bilinear form: C = AB, C[i,k] = sum_j A[i,j] B[j,k]
    # This is a tensor T of shape (n, m, p) where T[i,j,k] = 1 if... no.
    # The tensor is: for each triple (i,j,k), the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So T[i,j,k] = delta_{i,k}... no that's not right either.
    # Let's use the standard: the matmul tensor M has M[i,j,k] = 1 for all valid i,j,k
    # where i in [0,n), j in [0,m), k in [0,p), and the "output index" is (i,k).
    # The tensor decomposition says: M[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # with U indexed by (i,j) -> row i*m+j, V indexed by (j,k) -> row j*p+k, W indexed by (i,k) -> row i*n+k or k*n+i?
    # The task says: W[k*n+i, r]
    # So W row index = k*n + i, meaning W is indexed by (k, i) with k first, i second.
    # So the decomposition is: for all i,j,k:
    #   1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's verify this.
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k])
                # This should equal... wait, no.
                # The product U[i*m+j, c] * V[j*p+k, c] summed over c gives a scalar.
                # But we also have W[k*n+i, c]. So the full term is sum_c U*V*W.
                # That's not a simple dot product of U and V.
                # It's: sum_c (U[i*m+j, c] * V[j*p+k, c]) * W[k*n+i, c]
                # = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                
                # For the schoolbook case: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for the matching c
                # So sum_c = 1 for the matching c, 0 otherwise.
                # So the sum is 1. Correct.
                
                term = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                err = abs(term - 1.0)
                if err > max_err:
                    max_err = err
    
    return max_err < tol


def _strassen_2x2():
    """Return Strassen decomposition for 2x2x2 matmul.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Convention: W[k*n+i, c] with n=2, so W row = k*2+i
    """
    # Standard Strassen:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-d)(f+g)
    # m7 = (b-d)(e+h)
    # C11 = m1 - m4 + m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m3 + m2 - m6
    
    # U indexed by (i,j) -> row i*2+j:
    # (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # V indexed by (j,k) -> row j*2+k:
    # (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # W indexed by (k,i) -> row k*2+i:
    # (0,0)=C11, (0,1)=C12, (1,0)=C21, (1,1)=C22
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a+d)(e+h): U rows 0(a),3(d); V rows 0(e),3(h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1  # C11
    
    # m2 = (c+d)e: U rows 2(c),3(d); V row 0(e)
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # C21 (k=1,i=0 -> row 1*2+0=2)
    
    # m3 = a(f-h): U row 0(a); V rows 1(f),3(h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # C12 (k=0,i=1 -> row 0*2+1=1)
    W[3, 2] = -1  # C22 (k=1,i=1 -> row 1*2+1=3)
    
    # m4 = d(g-e): U row 3(d); V rows 2(g),0(e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1  # C11
    W[2, 3] = 1   # C21
    
    # m5 = (a+b)h: U rows 0(a),1(b); V row 3(h)
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1  # C11
    W[1, 4] = 1  # C12
    
    # m6 = (c-d)(f+g): U rows 2(c),3(d); V rows 1(f),2(g)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[2, 5] = 1
    W[3, 5] = -1  # C22
    
    # m7 = (b-d)(e+h): U rows 1(b),3(d); V rows 0(e),3(h)
    U[1, 6] = 1; U[3, 6] = -1
    V[0, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C11
    
    return U.tolist(), V.tolist(), W.tolist()


def _split_and_combine(n, m, p, n_split, p_split):
    """
    Split 4x4x6 into (n x m x p1) and (n x m x p2) where p1+p2=p.
    Use known decompositions for each part and combine.
    """
    n, m, p = n, m, p
    p1 = p_split
    p2 = p - p_split
    
    # For the first part: n x m x p1
    # For the second part: n x m x p2
    
    # We need decompositions for:
    # 4x4x4: rank 49 (Lader)
    # 4x4x2: rank 28
    
    # Let's build 4x4x4 using Lader's method or a known rank-49 decomposition.
    # Actually, let's use a simpler approach: build 4x4x4 from 2x2x2 blocks.
    # 4x4x4 can be done with rank 49 using Lader's result.
    # But implementing Lader is complex. Let's try a different approach.
    
    # Alternative: use the fact that 4x4x4 has rank 49 (Lader 1976).
    # One way to get rank 49: use a tensor decomposition based on group theory.
    # This is complex. Let me try a different strategy.
    
    # Strategy: Use a recursive/block approach.
    # 4x4x6: split p into 4+2.
    # 4x4x4: use a known rank-49 decomposition.
    # 4x4x2: use a known rank-28 decomposition.
    
    # For 4x4x2, we can think of it as two 4x4x1 problems.
    # 4x4x1: rank 16 (schoolbook, since each output is a dot product of length 4).
    # Wait, 4x4x1: C[i,0] = sum_j A[i,j] B[j,0]. This is 16 independent dot products.
    # Rank 16.
    # 4x4x2: C[i,0] = sum_j A[i,j] B[j,0], C[i,1] = sum_j A[i,j] B[j,1]
    # This is 32 independent dot products in schoolbook.
    # But we can do better. Rank 28 is known for 4x4x2.
    
    # Let me try a different approach: use a randomized/optimization approach.
    # Actually, let's just implement a known rank-49 decomposition for 4x4x4
    # and a rank-28 decomposition for 4x4x2.
    
    # For simplicity and reliability, let's use a constructive approach.
    # 
    # 4x4x4 rank 49 decomposition (Lader):
    # This is based on the group Z2 x Z2 x Z2 or similar.
    # 
    # Actually, let me try a simpler known decomposition.
    # 
    # For 4x4x4, one known rank-49 decomposition uses the following idea:
    # Use 49 terms. This is hard to write out by hand.
    # 
    # Let me try a different approach: use a numerical optimization to find
    # a low-rank decomposition, then round to integers/half-integers.
    
    # Actually, let's try a completely different strategy.
    # Use the "border rank" or "tensor rank" results.
    # 
    # For 4x4x6, the best known rank is 77 (from the split 49+28).
    # Let me try to implement the 4x4x4 rank-49 and 4x4x2 rank-28 decompositions.
    
    # 4x4x2 rank 28:
    # C[i,k] = sum_j A[i,j] B[j,k] for k=0,1
    # We can write this as:
    # C[i,0] = sum_j A[i,j] B[j,0]
    # C[i,1] = sum_j A[i,j] B[j,1]
    # 
    # Schoolbook: 32 terms.
    # To get 28, we need to save 4 terms.
    # 
    # One approach: use Strassen-like ideas on blocks.
    # 
    # Let me try a different approach: use a known decomposition.
    # 
    # Actually, let's use a numerical approach. I'll set up the problem as
    # finding matrices U, V, W with minimal rank such that the tensor equation holds.
    # This is a non-convex problem, but for small dimensions it might be tractable.
    
    # Let me try a simpler approach: use a greedy or randomized method to find
    # a low-rank decomposition.
    
    # Actually, the simplest reliable approach: use the known result that
    # 4x4x4 has rank 49 and 4x4x2 has rank 28, and implement these.
    
    # For 4x4x4 rank 49, I'll use a constructive method based on:
    # The 4x4x4 tensor can be decomposed using 49 terms.
    # One construction: use the fact that 4x4x4 = (2x2x2) \otimes (2x2x2) in some sense,
    # but that gives rank 7*7=49. Yes! The Kronecker product of two Strassen
    # decompositions gives a 4x4x4 decomposition of rank 49.
    
    # Wait, is that right? If we have a rank-r decomposition for 2x2x2,
    # then the Kronecker product gives a rank-r^2 decomposition for 4x4x4.
    # So 7^2 = 49. Yes!
    
    # Similarly, for 4x4x2: we can't directly Kronecker product.
    # But we can think of 4x4x2 as a 4x4x4 with the last 2 columns zero.
    # That doesn't directly help.
    
    # For 4x4x2, let's use a different approach.
    # 4x4x2: rank 28.
    # One way: use 4x4x4 rank 49 but restrict to the first 2 output columns.
    # That doesn't reduce the rank.
    
    # Alternative: 4x4x2 can be seen as two 4x4x1 problems.
    # 4x4x1: rank 16 (each output C[i,0] is a dot product of 4 elements).
    # Two of them: rank 32 in schoolbook.
    # To get 28, we need a more clever decomposition.
    
    # Let me try: 4x4x2 = (2x4x2) \oplus (2x4x2) in terms of output rows.
    # 2x4x2: C[i,k] = sum_j A[i,j] B[j,k] for i=0,1; j=0,1,2,3; k=0,1
    # This is 2*4*2 = 16 terms in schoolbook.
    # Can we do better? 2x4x2 has rank at most 16.
    # Actually, 2x4x2: the tensor has shape (2,4,2).
    # Rank of 2x4x2 tensor: at most 2*4=8? No, that's not right.
    # The matmul tensor 2x4x2: M[i,j,k] = 1 for all i,j,k.
    # Wait, no. M[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k].
    # So M[i,j,k] = 1 for all valid i,j,k.
    # The rank of this tensor: for 2x4x2, it's at most 2*4=8? 
    # No, the rank of the matmul tensor [n,m,p] is at most n*m*p.
    # For 2x4x2, schoolbook gives 16.
    # Can we do better? 
    # 2x4x2: C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k] + A[0,3]B[3,k]
    # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k] + A[1,3]B[3,k]
    # This is like two independent 1x4x2 problems.
    # 1x4x2: C[0,k] = sum_j A[0,j] B[j,k]
    # This is rank 4 (each term A[0,j]*B[j,k] for j=0..3, k=0,1).
    # Wait, 1x4x2: the tensor has shape (1,4,2).
    # M[0,j,k] = 1 for all j,k.
    # Rank: at most 4*2=8? No, at most 1*4*2=8.
    # But we can do: C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0] + A[0,3]B[3,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1] + A[0,3]B[3,1]
    # Schoolbook: 8 terms.
    # Can we do better? 
    # Let m1 = A[0,0]*(B[0,0]+B[0,1]), m2 = A[0,1]*(B[1,0]+B[1,1]), etc.
    # Then C[0,0] = ... this doesn't work directly because we need separate outputs.
    # 
    # Actually, for 1x4x2, the rank is 4.
    # Because: C[0,k] = sum_j A[0,j] B[j,k]
    # We can write: for each j, define m_j = A[0,j] * (B[j,0], B[j,1])
    # Then C[0,k] = sum_j m_j[k]
    # But this uses 4 "terms" where each term contributes to both outputs.
    # In the tensor decomposition framework:
    # U[0,j, c] * V[j,k, c] * W[k,0, c] = 1
    # For c = j (4 terms):
    # U[0,j,j] = 1, V[j,k,j] = 1, W[k,0,j] = 1
    # This works! Rank 4 for 1x4x2.
    # 
    # So 2x4x2: rank 8 (two independent 1x4x2).
    # 4x4x2: rank 16 (four independent 1x4x2).
    # That's the same as schoolbook (32/2 = 16... wait, schoolbook for 4x4x2 is 32).
    # 
    # Hmm, I'm confusing myself. Let me restart.
    # 
    # 4x4x2: n=4, m=4, p=2
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k] for i=0..3, k=0,1
    # Total outputs: 4*2 = 8
    # Schoolbook: 4*4*2 = 32 terms
    # 
    # Each output C[i,k] is a dot product of length 4.
    # We can't easily share computation between different (i,k) pairs
    # because they involve different rows of A and different columns of B.
    # 
    # The best known rank for 4x4x2 is 28, I believe.
    # 
    # Let me just use a numerical optimization approach.
    # I'll try to find a rank-r decomposition for 4x4x4 and 4x4x2
    # using a randomized approach with rounding.
    
    # Actually, let's take a step back. The simplest improvement over schoolbook (rank 96)
    # is the Kronecker product of two Strassen decompositions for 4x4x4 (rank 49).
    # For 4x4x2, I'll use a simple approach.
    
    # Let me implement the Kronecker product for 4x4x4.
    
    U2, V2, W2 = _strassen_2x2()
    U2 = np.array(U2)
    V2 = np.array(V2)
    W2 = np.array(W2)
    
    # Kronecker product: 
    # U4[i*4+j, c1*7+c2] = U2[i*2+j1, c1] * U2[j*2+j2, c2]
    # Wait, I need to be careful with the indexing.
    # 
    # For 2x2x2: U2[i*2+j, c], V2[j*2+k, c], W2[k*2+i, c]
    # For 4x4x4: U4[i*4+j, c], V4[j*4+k, c], W4[k*4+i, c]
    # 
    # The Kronecker product of two 2x2x2 decompositions:
    # Let the first 2x2x2 handle (i1, j1, k1) and the second handle (i2, j2, k2).
    # Then i = i1*2 + i2, j = j1*2 + j2, k = k1*2 + k2.
    # 
    # U4[i*4+j, c1*7+c2] = U2[i1*2+j1, c1] * U2[i2*2+j2, c2]
    # V4[j*4+k, c1*7+c2] = V2[j1*2+k1, c1] * V2[j2*2+k2, c2]
    # W4[k*4+i, c1*7+c2] = W2[k1*2+i1, c1] * W2[k2*2+i2, c2]
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            U4[i * 4 + j, c] = U2[i1 * 2 + j1, c1] * U2[i2 * 2 + j2, c2]
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            V4[j * 4 + k, c] = V2[j1 * 2 + k1, c1] * V2[j2 * 2 + k2, c2]
            
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1 * 2 + k2
                            i = i1 * 2 + i2
                            W4[k * 4 + i, c] = W2[k1 * 2 + i1, c1] * W2[k2 * 2 + i2, c2]
    
    # Verify 4x4x4
    assert _verify(U4.tolist(), V4.tolist(), W4.tolist(), 4, 4, 4), "4x4x4 decomposition failed"
    
    # Now for 4x4x2, let's use a simple approach.
    # 4x4x2: n=4, m=4, p=2
    # Let's use schoolbook for now (rank 32) and see if we can improve.
    # Actually, let's try a simple improvement.
    # 
    # 4x4x2: C[i,k] = sum_j A[i,j] B[j,k]
    # Split j into two groups of 2: j = 2*j1 + j2
    # C[i,k] = sum_{j1,j2} A[i, 2*j1+j2] B[2*j1+j2, k]
    # 
    # This doesn't directly help. Let's just use schoolbook for 4x4x2 (rank 32).
    # Total rank: 49 + 32 = 81.
    # That's better than 96 (schoolbook for 4x4x6).
    
    # Actually, let me try to do better for 4x4x2.
    # Use a 2x2x2 Strassen-like approach on blocks.
    # 
    # 4x4x2: think of A as 2x2 blocks of 2x2, B as 2x2 blocks of 2x1.
    # A = [[A11, A12], [A21, A22]] where each Ai1 is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bj1 is 2x1
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #      [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    # 
    # Each Ci1 is 2x1.
    # C11 = A11*B11 + A12*B21 (2x2 * 2x1 + 2x2 * 2x1 = 2x1)
    # 
    # This is getting complicated. Let me just use schoolbook for 4x4x2.
    
    U2p = np.zeros((16, 32))
    V2p = np.zeros((8, 32))
    W2p = np.zeros((8, 32))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U2p[i * 4 + j, col] = 1
                V2p[j * 2 + k, col] = 1
                W2p[k * 4 + i, col] = 1
                col += 1
    
    # Verify 4x4x2
    assert _verify(U2p.tolist(), V2p.tolist(), W2p.tolist(), 4, 4, 2), "4x4x2 decomposition failed"
    
    # Combine 4x4x4 and 4x4x2 into 4x4x6
    # 4x4x6: n=4, m=4, p=6
    # U: (16, rank), V: (24, rank), W: (24, rank)
    # 
    # The 4x4x4 part uses k=0..3, the 4x4x2 part uses k=4,5.
    # 
    # U: same for both parts (depends on i,j)
    # V: for 4x4x4, V[j*4+k, c] for k=0..3; for 4x4x2, V[j*2+k, c] for k=0,1
    #    In the combined V, the 4x4x4 part uses rows j*6+k for k=0..3,
    #    and the 4x4x2 part uses rows j*6+4+k' for k'=0,1 (i.e., k=4,5)
    # W: for 4x4x4, W[k*4+i, c] for k=0..3; for 4x4x2, W[k'*4+i, c] for k'=0,1
    #    In the combined W, the 4x4x4 part uses rows k*6+i for k=0..3,
    #    and the 4x4x2 part uses rows (4+k')*6+i for k'=0,1 (i.e., k=4,5)
    
    rank4 = 49
    rank2 = 32
    total_rank = rank4 + rank2
    
    U = np.zeros((16, total_rank))
    V = np.zeros((24, total_rank))
    W = np.zeros((24, total_rank))
    
    # 4x4x4 part
    U[:, :rank4] = U4
    for j in range(4):
        for k in range(4):
            V[j * 6 + k, :rank4] = V4[j * 4 + k]
    for k in range(4):
        for i in range(4):
            W[k * 6 + i, :rank4] = W4[k * 4 + i]
    
    # 4x4x2 part
    U[:, rank4:] = U2p
    for j in range(4):
        for k in range(2):
            V[j * 6 + 4 + k, rank4:] = V2p[j * 2 + k]
    for k in range(2):
        for i in range(4):
            W[(4 + k) * 6 + i, rank4:] = W2p[k * 4 + i]
    
    # Verify
    assert _verify(U.tolist(), V.tolist(), W.tolist(), 4, 4, 6), "4x4x6 decomposition failed"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    n, m, p = DIMENSIONS
    
    # Try the split approach: 4x4x4 (rank 49) + 4x4x2 (rank 32) = rank 81
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    try:
        result = _split_and_combine(n, m, p, 4, 4)
        elapsed = time.time() - start_time
        if elapsed < deadline:
            return result
    except Exception:
        pass
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
