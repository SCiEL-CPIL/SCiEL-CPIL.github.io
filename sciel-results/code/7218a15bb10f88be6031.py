import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition using a recursive block approach.
    # Base cases:
    # 1x1x1: Rank 1
    # 2x2x2: Rank 7 (Strassen)
    # We represent a decomposition as a list of tuples (u_idx, v_idx, w_idx, coeff)
    # where U[u_idx, col] = coeff, V[v_idx, col] = coeff, W[w_idx, col] = coeff
    # Actually, the interface requires U, V, W matrices.
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    
    # We will generate the tensor decomposition terms.
    # Each term is (i, j, k, coeff) corresponding to U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    # Wait, the standard decomposition is sum_c U[:,c] V[:,c]^T W[:,c]^T?
    # No, the tensor is sum_c u_c \otimes v_c \otimes w_c.
    # The multiplication is C = A B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor T[i,j,k] = delta? No.
    # The bilinear map is (A, B) -> C.
    # C[i,k] = sum_{j} A[i,j] B[j,k].
    # This is a sum of rank 1 tensors?
    # Each term in the sum is A[i,j] * B[j,k] * 1 (for C[i,k]).
    # So the tensor is sum_{i,j,k} A_{ij} B_{jk} C_{ki}? No.
    # The tensor T has indices i, j, k.
    # T[i, j, k] = 1 if it contributes to C[i,k] from A[i,j] and B[j,k].
    # Actually, the matrix multiplication tensor is defined such that
    # <T, A \otimes B> = Tr(A B W)?
    # The problem states: "output factor follows the official trace convention W[k*n+i,r]".
    # This implies the contraction is sum_{i,j,k} A[i,j] B[j,k] W[k*n+i, r]?
    # No, usually it's sum_{i,j,k} A[i,j] B[j,k] C[k,i]?
    # Let's check the seed program.
    # U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    # This corresponds to the term A[i,j] B[j,k] contributing to C[i,k].
    # The index of W is k*n + i.
    # So W is a vector of length p*n.
    # The result C is reconstructed from W?
    # "output factor ... W[k*n+i,r]".
    # This implies C[i,k] is obtained from W[k*n+i, :].
    # So C = reshape(W, (p, n, rank)) -> sum over rank?
    # If W has shape (p*n, rank), then C[i,k] = sum_r W[k*n+i, r] * (something)?
    # No, the decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    # Then C[i,k] = sum_j A[i,j] B[j,k] is computed as sum_r (sum_j A[i,j] U[j, r]) (sum_k B[j,k] V[k, r])?
    # No.
    # Standard tensor decomposition for matmul:
    # C = sum_r (A U_r) (V_r B)^T?
    # Let's stick to the indices given in the seed.
    # U index: i*m + j.
    # V index: j*p + k.
    # W index: k*n + i.
    
    # We will generate the list of (i, j, k) tuples with coefficients.
    # For Strassen 2x2x2:
    # The 7 terms are known.
    
    def strassen_2x2x2_terms():
        # Returns list of (i, j, k, coeff) for 0<=i,j,k<2
        # Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22) B11
        # M3 = A11 (B12 - B22)
        # M4 = A22 (B21 - B11)
        # M5 = (A11 + A12) B22
        # M6 = (A21 - A11) (B11 + B12)
        # M7 = (A12 - A22) (B21 + B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to express this in terms of U, V, W indices.
        # Each M is a product of a linear combo of A and a linear combo of B.
        # M = (sum a_ij U_ij) (sum b_jk V_jk) ?
        # The rank 1 term is u \otimes v \otimes w.
        # u is a vector in R^4 (flattened A), v in R^4 (flattened B), w in R^4 (flattened C).
        # Actually, U is (n*m, rank). For 2x2x2, U is (4, 7).
        # The term corresponds to a specific (i, j, k) in the flattened space?
        # No, the decomposition is sum_r u_r \otimes v_r \otimes w_r.
        # u_r has non-zero entries at indices corresponding to A's elements.
        # v_r has non-zero entries at indices corresponding to B's elements.
        # w_r has non-zero entries at indices corresponding to C's elements.
        
        # For Strassen, we can define the 7 vectors u_r, v_r, w_r directly.
        # But it's easier to think in terms of the (i,j,k) contributions.
        # Actually, the "terms" are the rank 1 components.
        # Let's define the 7 components for 2x2x2.
        
        # Component 1: M1
        # A part: A11 + A22. Indices: (0,0) and (1,1).
        # B part: B11 + B22. Indices: (0,0) and (1,1).
        # C part: C11 + C22. Indices: (0,0) and (1,1).
        # Wait, C11 = M1 + ... and C22 = M1 + ...
        # So w_1 has 1 at (0,0) and 1 at (1,1).
        
        # This approach is tricky to map to the global indices.
        # I will implement a recursive function that returns U, V, W matrices for a given n, m, p.
        
        pass

    # Since I cannot easily code the recursive Strassen with correct index mapping in a short time 
    # and guarantee it beats 84 without the explicit rank 13/10 decompositions, 
    # I will return the schoolbook decomposition (rank 90) or the known 84 if I can implement it.
    # Given the risk of bugs in a complex recursive implementation, 
    # and the fact that the current best is 84, 
    # I will provide a robust implementation of the schoolbook (rank 90) 
    # but optimized for speed, or try to hardcode the 84 structure if possible.
    
    # Actually, the prompt says "score is the rank".
    # If I return rank 90, score is 1.0.
    # If I return rank 84, score is 1.071.
    # I should try to get 84.
    
    # The "Block Strassen" strategy:
    # Split 3x5x6 into blocks.
    # I will implement the specific 84-rank decomposition found in the "inspiration".
    # It uses rank-7 Strassen on 2x2x2 sub-blocks.
    
    # How to construct 84?
    # As derived: 3 * R(3,5,2) where R(3,5,2) = 28.
    # R(3,5,2) = R(3,2,2) + R(3,3,2) = 11 + 17 = 28.
    # R(3,2,2) = 11 (1x2x2 rank 4 + 2x2x2 rank 7).
    # R(3,3,2) = 17 (1x3x2 rank 6 + 2x3x2 rank 11).
    # R(2,3,2) = 11 (2x1x2 rank 4 + 2x2x2 rank 7).
    
    # I will implement this recursive structure.
    
    def get_decomposition(n, m, p):
        # Returns U, V, W for n x m x p
        if n == 1 and m == 1 and p == 1:
            U = [[1]]
            V = [[1]]
            W = [[1]]
            return U, V, W
        
        # Base case: 2x2x2 Strassen
        if n == 2 and m == 2 and p == 2:
            # Hardcode Strassen 2x2x2
            # U is 4x7, V is 4x7, W is 4x7
            # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
            
            # Strassen vectors (u, v, w)
            # u is in R^4 (A), v in R^4 (B), w in R^4 (C)
            # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
            # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
            # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3 (W[k*2+i])
            
            # M1: (A00+A11)(B00+B11) -> C00, C11
            u1 = [1, 0, 0, 1]
            v1 = [1, 0, 0, 1]
            w1 = [1, 0, 0, 1] # C00 (k=0,i=0 -> 0), C11 (k=1,i=1 -> 3)
            
            # M2: (A10+A11)B00 -> C10
            u2 = [0, 0, 1, 1]
            v2 = [1, 0, 0, 0]
            w2 = [0, 0, 1, 0] # C10 (k=0,i=1 -> 2)
            
            # M3: A00(B01-B11) -> C01
            u3 = [1, 0, 0, 0]
            v3 = [0, 1, 0, -1]
            w3 = [0, 1, 0, 0] # C01 (k=1,i=0 -> 1)
            
            # M4: A11(B10-B00) -> C10, C11?
            # C10 = M2 + M4. C11 = M1 - M2 + M3 + M6.
            # Wait, standard Strassen:
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            # My indices:
            # C[i,k]. W[k*n+i].
            # C00 (i=0,k=0) -> W[0]
            # C01 (i=0,k=1) -> W[1]
            # C10 (i=1,k=0) -> W[2]
            # C11 (i=1,k=1) -> W[3]
            
            # M4: A11(B10-B00).
            # A11 is index 3. B10 is index 2, B00 is index 0.
            u4 = [0, 0, 0, 1]
            v4 = [0, 0, 1, -1] # B10 - B00? No, B10 is (1,0), index 2. B00 is (0,0), index 0.
            # v4 should be [ -1, 0, 1, 0 ] for B10 - B00?
            # B[j,k]. B10 is j=1,k=0. Index 1*2+0=2.
            # B00 is j=0,k=0. Index 0.
            # So v4 = [ -1, 0, 1, 0 ].
            # Contributes to C10 (i=1,k=0 -> W[2]) and C11?
            # C10 = M2 + M4.
            # C11 does not use M4 in standard Strassen?
            # C11 = M1 + M4 - M5 + M7. Yes it does.
            w4 = [0, 0, 1, 1] # C10 (W[2]) and C11 (W[3])
            
            # M5: (A00+A01)B11 -> C01, C11?
            # C01 = M3 + M5.
            # C11 = ... - M5 ...
            # A00+A01: indices 0, 1.
            # B11: index 3.
            u5 = [1, 1, 0, 0]
            v5 = [0, 0, 0, 1]
            w5 = [0, 1, 0, -1] # C01 (W[1]) and C11 (W[3])
            
            # M6: (A10-A00)(B00+B01) -> C11
            # A10-A00: indices 2, 0.
            # B00+B01: indices 0, 1.
            u6 = [1, 0, -1, 0] # A00 - A10? Or A10 - A00?
            # Standard: M6 = (A21 - A11)(B11 + B12) in 1-based.
            # A21 is A10 (0-based). A11 is A00.
            # So A10 - A00.
            u6 = [-1, 0, 1, 0]
            v6 = [1, 1, 0, 0] # B00 + B01
            w6 = [0, 0, 0, 1] # C11 (W[3])
            
            # M7: (A01-A11)(B10+B11) -> C00
            # A01-A11: indices 1, 3.
            # B10+B11: indices 2, 3.
            u7 = [0, 1, 0, -1]
            v7 = [0, 0, 1, 1]
            w7 = [1, 0, 0, 0] # C00 (W[0])
            
            U = np.array([u1, u2, u3, u4, u5, u6, u7]).T # Shape (4, 7)
            V = np.array([v1, v2, v3, v4, v5, v6, v7]).T # Shape (4, 7)
            W = np.array([w1, w2, w3, w4, w5, w6, w7]).T # Shape (4, 7)
            return U, V, W

        # Recursive step
        # Split p into p1, p2
        p1 = p // 2
        p2 = p - p1
        
        # B is m x p. Split into B1 (m x p1) and B2 (m x p2).
        # C is n x p. Split into C1 (n x p1) and C2 (n x p2).
        # C1 = A B1. C2 = A B2.
        
        U1, V1, W1 = get_decomposition(n, m, p1)
        U2, V2, W2 = get_decomposition(n, m, p2)
        
        # Combine U
        # U is n x m. Same for both.
        U = np.hstack([U1, U2])
        
        # Combine V
        # V is m x p.
        # V1 is m x p1. V2 is m x p2.
        # We need to map V1 to the first p1 columns of B, and V2 to the last p2 columns.
        # V1 indices are j*p1 + k1.
        # V2 indices are j*p2 + k2.
        # Global V index is j*p + k.
        # For V1: k = k1. Global index j*p + k1.
        # For V2: k = p1 + k2. Global index j*p + p1 + k2.
        
        V = np.zeros((m*p, U.shape[1]))
        for j in range(m):
            for k1 in range(p1):
                V[j*p + k1, :U1.shape[1]] = V1[j*p1 + k1, :]
            for k2 in range(p2):
                V[j*p + p1 + k2, U1.shape[1]:] = V2[j*p2 + k2, :]
        
        # Combine W
        # W is p x n.
        # W1 is p1 x n. W2 is p2 x n.
        # Global W index is k*n + i.
        # For W1: k = k1. Global index k1*n + i.
        # For W2: k = p1 + k2. Global index (p1+k2)*n + i.
        
        W = np.zeros((p*n, U.shape[1]))
        for k1 in range(p1):
            for i in range(n):
                W[k1*n + i, :U1.shape[1]] = W1[k1*n1 + i, :] # Wait, W1 is for n x m x p1.
                # W1 shape is (p1*n, rank1).
                # Index in W1 is k1*n + i.
                W[k1*n + i, :U1.shape[1]] = W1[k1*n + i, :]
        for k2 in range(p2):
            for i in range(n):
                W[(p1+k2)*n + i, U1.shape[1]:] = W2[k2*n + i, :]
                
        return U, V, W

    U, V, W = get_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
