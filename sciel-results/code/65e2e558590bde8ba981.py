import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120

    # The tensor for matrix multiplication C = AB, A in R^{n x m}, B in R^{m x p}, C in R^{n x p}.
    # T_{i,j,k} = 1 if C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # We want to find a rank-r decomposition: T = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
    # This is equivalent to finding matrices U (nm x R), V (mp x R), W (pn x R) such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    # Or, using the trace convention W[k*n+i, r]:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T_{i,j,k} = 1.

    # Strategy from Directive:
    # 1. Construct the 60x360 coefficient matrix? No, that's for checking linear dependence of monomials.
    #    The directive says: "Construct the 60x360 coefficient matrix of the bilinear map A⊗B→C and perform an exact rank factorization over the rationals... to determine if the tensor rank is strictly less than 60".
    #    Actually, the tensor rank is not the rank of a single matrix. It's the minimal R such that the tensor can be written as sum of R rank-1 tensors.
    #    However, if we fix two modes, the tensor defines a linear map from the third mode's space to the tensor product of the other two.
    #    Let's re-read the directive carefully: "identify the minimal linear subspace of rank-1 tensors that span the multiplication tensor".
    #    This is just the definition of tensor rank.
    #    The directive suggests using Bareiss to do exact rank factorization.
    #    Is it possible to determine the tensor rank via matrix rank?
    #    The border rank is related to the rank of catalecticant matrices.
    #    But we need exact rank.
    #    
    #    Let's look at the dimensions.
    #    n=3, m=4, p=5.
    #    Schoolbook rank = 3*4*5 = 60.
    #    
    #    If the tensor rank were 60, it would be "generic" in some sense, but 3x4x5 is small.
    #    Known bounds:
    #    R(3,4,5) <= R(3,4,4) + R(3,4,1) ? No, R(n,m,p) <= R(n,m,p1) + R(n,m,p2) if p=p1+p2.
    #    R(3,4,5) <= R(3,4,4) + 3*4*1 = R(3,4,4) + 12.
    #    R(3,4,4) <= R(4,4,4) = 49 (Strassen).
    #    So R(3,4,5) <= 49 + 12 = 61. This is worse than 60.
    #    
    #    Wait, R(3,4,4) is the rank of multiplying 3x4 by 4x4.
    #    Is R(3,4,4) < 48?
    #    If we pad 3x4 to 4x4, the rank is 49. But the 3x4 tensor is a sub-tensor.
    #    The rank of a sub-tensor is <= rank of the full tensor.
    #    So R(3,4,4) <= 49.
    #    Then R(3,4,5) <= 49 + 12 = 61.
    #    
    #    What if we split differently?
    #    R(3,4,5) <= R(3,3,5) + R(3,1,5).
    #    R(3,1,5) = 1*5 = 5 (dot product of length 1? No. A is 3x1, B is 1x5. C is 3x5. C_{i,k} = A_{i,0} B_{0,k}. Rank is 1*1*5 = 5? No, it's an outer product. Rank is 1.
    #    Wait. T_{i,j,k} for n=3, m=1, p=5.
    #    C_{i,k} = A_{i,0} B_{0,k}.
    #    This is a rank-1 tensor (outer product of vector A and vector B).
    #    So R(3,1,5) = 1.
    #    R(3,3,5) <= R(3,3,3) + R(3,3,2).
    #    R(3,3,3) = 23 (Lader).
    #    R(3,3,2) <= R(3,3,2).
    #    R(3,3,2) <= R(4,4,2) = 4*4*2 = 32? No.
    #    R(3,3,2) is multiplying 3x3 by 3x2.
    #    We can use Strassen?
    #    Pad to 4x4? R(4,4,4)=49.
    #    R(3,3,2) <= 49?
    #    Actually, R(3,3,2) <= R(3,3,1) + R(3,3,1) = 3*3*1 + 3*3*1 = 18.
    #    So R(3,3,5) <= 23 + 18 = 41.
    #    R(3,4,5) <= R(3,3,5) + R(3,1,5) = 41 + 1 = 42.
    #    
    #    Wait, is R(3,1,5) = 1?
    #    A is 3x1. B is 1x5.
    #    C_{i,k} = A_{i,0} B_{0,k}.
    #    Tensor T_{i,j,k} where j=0.
    #    T_{i,0,k} = A_{i,0} B_{0,k}.
    #    This is indeed a rank-1 tensor (separable).
    #    So R(3,1,5) = 1.
    #    
    #    Let's check R(3,3,2).
    #    Multiply 3x3 by 3x2.
    #    Split B into two 3x1 matrices.
    #    R(3,3,2) <= R(3,3,1) + R(3,3,1).
    #    R(3,3,1): A 3x3, B 3x1. C 3x1.
    #    C_i = sum_j A_{ij} B_{j,0}.
    #    This is 3 dot products of length 3.
    #    Rank of dot product of length 3 is 3.
    #    So R(3,3,1) = 3.
    #    So R(3,3,2) <= 3 + 3 = 6.
    #    
    #    So R(3,3,5) <= R(3,3,3) + R(3,3,2) = 23 + 6 = 29.
    #    R(3,4,5) <= R(3,3,5) + R(3,1,5) = 29 + 1 = 30.
    #    
    #    This is a massive improvement! 30 < 60.
    #    Can we do better?
    #    R(3,3,3) = 23 is known.
    #    Is R(3,3,2) < 6?
    #    R(3,3,2) is the rank of the 3x3x2 tensor.
    #    It is known that R(3,3,2) = 6?
    #    Let's check.
    #    The tensor is T_{i,j,k} for i,j in 0..2, k in 0..1.
    #    It consists of two 3x3 identity-like structures? No.
    #    C_{i,0} = sum_j A_{ij} B_{j,0}.
    #    C_{i,1} = sum_j A_{ij} B_{j,1}.
    #    This is equivalent to multiplying A (3x3) by B (3x2).
    #    This is the same as two independent matrix-vector products.
    #    Each matrix-vector product (3x3 * 3x1) has rank 3.
    #    Since the B indices are disjoint (k=0 and k=1), the tensors are independent.
    #    So R(3,3,2) = 3 + 3 = 6.
    #    
    #    So the bound is 23 + 6 + 1 = 30.
    #    
    #    Can we improve R(3,3,3)?
    #    Lader's algorithm is rank 23. It is known to be optimal for 3x3?
    #    Yes, R(3,3,3) = 23 is a known result (Lader 1975).
    #    
    #    So we have a decomposition of rank 30.
    #    
    #    Let's verify the logic.
    #    A is 3x4. Split A into A1 (3x3) and A2 (3x1).
    #    B is 4x5. Split B into B1 (3x5) and B2 (1x5).
    #    C = A1*B1 + A2*B2.
    #    A1*B1 is 3x3 * 3x5.
    #    A2*B2 is 3x1 * 1x5.
    #    
    #    Rank(A1*B1) = R(3,3,5).
    #    Rank(A2*B2) = R(3,1,5) = 1.
    #    
    #    R(3,3,5): Split B1 into B11 (3x3) and B12 (3x2).
    #    A1*B1 = A1*B11 + A1*B12.
    #    Rank(A1*B11) = R(3,3,3) = 23.
    #    Rank(A1*B12) = R(3,3,2) = 6.
    #    Total Rank = 23 + 6 + 1 = 30.
    
    # We need to construct the U, V, W matrices for this decomposition.
    # This requires:
    # 1. A rank-23 decomposition for 3x3 x 3x3 (Lader).
    # 2. A rank-6 decomposition for 3x3 x 3x2 (Schoolbook).
    # 3. A rank-1 decomposition for 3x1 x 1x5 (Outer Product).
    
    # Let's implement these.
    
    def lader_3x3():
        # Lader's algorithm for 3x3 matrix multiplication.
        # Rank 23.
        # Reference: "A note on the rank of the 3x3 matrix multiplication tensor" or similar.
        # The explicit coefficients are complex.
        # I will use a known list of the 23 terms.
        # From "Matrix Multiplication and its Applications" or online resources.
        # The 23 bilinear forms are:
        # (A00+A11+A22)(B00+B11+B22)
        # (A00-A11)(B00-B11)
        # (A00-A22)(B00-B22)
        # (A11-A22)(B11-B22)
        # (A01+A12)(B10+B21)
        # (A01-A12)(B10-B21)
        # (A02+A21)(B20+B12)
        # (A02-A21)(B20-B12)
        # (A10+A20)(B01+B02)
        # (A10-A20)(B01-B02)
        # (A01+A21)(B10+B20)
        # (A01-A21)(B10-B20)
        # (A02+A12)(B20+B10)
        # (A02-A12)(B20-B10)
        # (A10+A11)(B01+B11)
        # (A10-A11)(B01-B11)
        # (A20+A21)(B02+B12)
        # (A20-A21)(B02-B12)
        # (A00+A10)(B00+B10)
        # (A00-A10)(B00-B10)
        # (A01+A02)(B10+B20)
        # (A01-A02)(B10-B20)
        # (A11+A12)(B01+B21)
        # (A11-A12)(B01-B21)
        
        # Wait, this list might be incomplete or incorrect. Lader's algorithm is specific.
        # Let's use a verified source or derive it?
        # Actually, implementing Lader correctly from memory is risky.
        # Is there a simpler rank 23 decomposition?
        # Or can I use a rank 24 decomposition if I can't get 23?
        # If I use rank 24, total is 24+6+1 = 31.
        # If I use schoolbook for 3x3 (rank 9), total is 9+6+1 = 16? No, 9+6+1 is not how it adds up.
        # R(3,3,3) schoolbook is 9.
        # R(3,3,2) schoolbook is 6.
        # R(3,1,5) schoolbook is 1.
        # Total 9+6+1 = 16.
        # Wait. 16 < 60.
        # Why did I think Lader was needed?
        # Schoolbook rank for 3x3 is 3*3*3 = 27.
        # Oh, I confused dimensions.
        # 3x3 x 3x3 schoolbook rank is 27.
        # Lader reduces it to 23.
        # 3x3 x 3x2 schoolbook rank is 3*3*2 = 18.
        # My previous calculation: R(3,3,2) = 6?
        # Let's re-evaluate R(3,3,2).
        # A is 3x3. B is 3x2.
        # C is 3x2.
        # C_{i,k} = sum_j A_{ij} B_{jk}.
        # This is NOT just dot products.
        # It is matrix multiplication.
        # The schoolbook rank is 3*3*2 = 18.
        # Can we do better?
        # Strassen for 2x2 is 7.
        # Can we embed 3x3 x 3x2 into something smaller?
        # R(3,3,2) <= R(4,4,4) = 49? No, that's bigger.
        # R(3,3,2) <= R(3,3,3) = 23?
        # Since 2 < 3, the tensor is a sub-tensor.
        # So R(3,3,2) <= 23.
        # Is it lower?
        # The "dot product" argument I used earlier was for m=1.
        # Here m=3.
        # So R(3,3,2) is likely around 18 or slightly less.
        # Let's assume schoolbook 18 for safety, or try to find a better one.
        # If R(3,3,2) = 18, then total rank = 23 (Lader) + 18 + 1 = 42.
        # 42 < 60. This is still a huge improvement.
        # If I use schoolbook for 3x3 (27), total = 27 + 18 + 1 = 46.
        # 46 < 60.
        
        # So even with pure schoolbook for the sub-blocks, we get rank 46.
        # Let's verify the decomposition structure.
        # A (3x4) = [A1 (3x3) | A2 (3x1)]
        # B (4x5) = [B1 (3x5); B2 (1x5)]
        # C = A1*B1 + A2*B2.
        
        # A1*B1: A1 (3x3), B1 (3x5).
        # Split B1 = [B11 (3x3) | B12 (3x2)].
        # A1*B1 = A1*B11 + A1*B12.
        
        # Term 1: A1*B11 (3x3 x 3x3). Schoolbook rank 27.
        # Term 2: A1*B12 (3x3 x 3x2). Schoolbook rank 18.
        # Term 3: A2*B2 (3x1 x 1x5). Rank 1.
        
        # Total Rank = 27 + 18 + 1 = 46.
        
        # This is much easier to implement than Lader.
        # Let's try to implement this rank 46 decomposition.
        
        # Structure:
        # U: (3*4, 46) = (12, 46)
        # V: (4*5, 46) = (20, 46)
        # W: (5*3, 46) = (15, 46)
        
        # Indices:
        # U index: i*m + j. i in 0..2, j in 0..3.
        # V index: j*p + k. j in 0..3, k in 0..4.
        # W index: k*n + i. k in 0..4, i in 0..2.
        
        # We will build U, V, W by concatenating the blocks.
        
        U = np.zeros((n*m, 0))
        V = np.zeros((m*p, 0))
        W = np.zeros((p*n, 0))
        
        # Helper to add a block decomposition
        def add_block(u_block, v_block, w_block):
            nonlocal U, V, W
            if u_block.shape[1] == 0:
                return
            U = np.hstack([U, u_block])
            V = np.hstack([V, v_block])
            W = np.hstack([W, w_block])
            
        # Block 1: A1*B11 (3x3 x 3x3)
        # A1 is A[0:3, 0:3]. Indices i=0..2, j=0..2.
        # B11 is B[0:3, 0:3]. Indices j=0..2, k=0..2.
        # Result C11 is C[0:3, 0:3]. Indices i=0..2, k=0..2.
        
        # Schoolbook for 3x3:
        # Rank 27.
        u1 = np.zeros((12, 27))
        v1 = np.zeros((20, 27))
        w1 = np.zeros((15, 27))
        
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # A index: i*4 + j
                    # B index: j*5 + k
                    # C index: k*3 + i
                    u1[i*4 + j, col] = 1
                    v1[j*5 + k, col] = 1
                    w1[k*3 + i, col] = 1
                    col += 1
        add_block(u1, v1, w1)
        
        # Block 2: A1*B12 (3x3 x 3x2)
        # A1 is A[0:3, 0:3].
        # B12 is B[0:3, 3:5]. Indices j=0..2, k=3..4.
        # Result C12 is C[0:3, 3:5]. Indices i=0..2, k=3..4.
        
        # Schoolbook for 3x3 x 3x2:
        # Rank 18.
        u2 = np.zeros((12, 18))
        v2 = np.zeros((20, 18))
        w2 = np.zeros((15, 18))
        
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(2): # k is local 0..1, global k is 3..4
                    # A index: i*4 + j
                    # B index: j*5 + (3+k)
                    # C index: (3+k)*3 + i
                    u2[i*4 + j, col] = 1
                    v2[j*5 + (3+k), col] = 1
                    w2[(3+k)*3 + i, col] = 1
                    col += 1
        add_block(u2, v2, w2)
        
        # Block 3: A2*B2 (3x1 x 1x5)
        # A2 is A[0:3, 3:4]. Indices i=0..2, j=3.
        # B2 is B[3:4, 0:5]. Indices j=3, k=0..4.
        # Result C2 is C[0:3, 0:5]. Indices i=0..2, k=0..4.
        
        # Rank 1 (Outer product).
        # C_{i,k} = A_{i,3} * B_{3,k}.
        # This is a rank-1 tensor.
        # u3[i*4+3] = 1 for i=0..2.
        # v3[3*5+k] = 1 for k=0..4.
        # w3[k*3+i] = 1 for k=0..4, i=0..2.
        
        u3 = np.zeros((12, 1))
        v3 = np.zeros((20, 1))
        w3 = np.zeros((15, 1))
        
        for i in range(3):
            u3[i*4 + 3, 0] = 1
        for k in range(5):
            v3[3*5 + k, 0] = 1
        for k in range(5):
            for i in range(3):
                w3[k*3 + i, 0] = 1
        
        add_block(u3, v3, w3)
        
        # Total rank = 27 + 18 + 1 = 46.
        
        # Verify correctness with a random test.
        # If time permits, we can try to optimize further.
        # But 46 is already a significant improvement over 60.
        
        # Check deadline
        if time.time() > deadline - 5:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        # Optional: Try to improve using Lader for the 3x3 part.
        # If we can implement Lader (rank 23) correctly, we save 4 ranks.
        # Total would be 23 + 18 + 1 = 42.
        # Let's attempt to implement Lader if we have time.
        
        # For now, return the rank 46 solution.
        
        # Verification
        # A = random(3,4), B = random(4,5)
        # C = A @ B
        # C_calc = sum_r U[:,r] outer V[:,r] outer W[:,r] ... reshaped
        # Let's do a quick numpy check.
        
        A_test = np.random.rand(n, m)
        B_test = np.random.rand(m, p)
        C_expected = A_test @ B_test
        
        # Reconstruct C from U, V, W
        # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a bit slow to do element-wise in Python for verification, but let's try einsum.
        # U: (12, 46), V: (20, 46), W: (15, 46)
        # We need to map indices back.
        # U_flat[i*m+j, r]
        # V_flat[j*p+k, r]
        # W_flat[k*n+i, r]
        
        # Reshape U to (3,4,46), V to (4,5,46), W to (5,3,46)
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # C[i,k] = sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        C_calc = np.einsum('ijr,jkr,ki r->ik', U_reshaped, V_reshaped, W_reshaped)
        
        if not np.allclose(C_expected, C_calc):
            # Fallback to schoolbook if verification fails
            rank_schoolbook = n * m * p
            U_sb = np.zeros((n*m, rank_schoolbook))
            V_sb = np.zeros((m*p, rank_schoolbook))
            W_sb = np.zeros((p*n, rank_schoolbook))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i*m+j, col] = 1
                        V_sb[j*p+k, col] = 1
                        W_sb[k*n+i, col] = 1
                        col += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
