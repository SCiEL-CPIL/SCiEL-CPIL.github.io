import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 140

    def schoolbook():
        """Schoolbook decomposition: rank = n*m*p = 80"""
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
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    def verify(U, V, W):
        """Verify that the decomposition exactly reproduces matrix multiplication."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # Build the multiplication tensor from the decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Compare with the actual multiplication tensor: T[i,j,k] = 1 for all (i,j,k)
        
        # For efficiency, check a sample of random matrices
        np.random.seed(42)
        for _ in range(20):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute C from decomposition
            C_decomp = np.zeros((p, n))
            for r in range(rank):
                u = U[:, r]
                v = V[:, r]
                w = W[:, r]
                A_u = np.zeros((n, m))
                for i in range(n):
                    for j in range(m):
                        A_u[i, j] = u[i * m + j]
                B_v = np.zeros((m, p))
                for j in range(m):
                    for k in range(p):
                        B_v[j, k] = v[j * p + k]
                C_w = np.zeros((p, n))
                for k in range(p):
                    for i in range(n):
                        C_w[k, i] = w[k * n + i]
                C_decomp += (A_u @ A) * (B @ B_v).T * 0  # This is wrong
                # Correct: C_decomp[k,i] += sum over r of U[i*m+j,r]*A[i,j] * V[j*p+k,r]*B[j,k] * W[k*n+i,r]
                # Let me redo this properly
            
            # Better approach: directly compute
            C_decomp = np.zeros((p, n))
            for r in range(rank):
                u_r = U[:, r]
                v_r = V[:, r]
                w_r = W[:, r]
                # A_u = matrix where A_u[i,j] = u_r[i*m+j]
                # B_v = matrix where B_v[j,k] = v_r[j*p+k]
                # C_w = matrix where C_w[k,i] = w_r[k*n+i]
                # C_decomp[k,i] += (A^T @ A_u)[i, :] ... no
                # C_decomp[k,i] += sum_{j} A[i,j] * u_r[i*m+j] * v_r[j*p+k] * B[j,k] * w_r[k*n+i]
                # = w_r[k*n+i] * sum_j A[i,j] * u_r[i*m+j] * v_r[j*p+k] * B[j,k]
                
                # Let's compute: for each i, j, k:
                # C_decomp[k,i] += A[i,j] * B[j,k] * u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
                
                # Vectorized:
                Au = u_r.reshape(n, m)
                Bv = v_r.reshape(m, p)
                Cw = w_r.reshape(p, n)
                
                # C_decomp[k,i] += sum_j A[i,j] * Au[i,j] * Bv[j,k] * B[j,k] * Cw[k,i]
                # = Cw[k,i] * sum_j (A[i,j]*Au[i,j]) * (B[j,k]*Bv[j,k])
                
                A_masked = A * Au
                B_masked = B * Bv
                C_decomp += (A_masked @ B_masked) * Cw
            
            if not np.allclose(C_decomp, C_expected, atol=1e-10):
                return False
        return True

    # Try to construct a rank-65 decomposition using Strassen for 4x4x4 and schoolbook for 4x4x1
    # The 4x4x4 part uses double Strassen (rank 49), the 4x4x1 part uses schoolbook (rank 16)
    
    def build_strassen_4x4x4_terms():
        """Build 49 rank-1 terms for 4x4 x 4x4 matrix multiplication using double Strassen.
        
        Returns list of (u, v, w) where u, v, w are length-16 vectors.
        """
        # Strassen 2x2 factors: each is (U_2x2, V_2x2, W_2x2)
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 - M4 - M6 + M7
        # C01 = M3 + M4
        # C10 = M2 + M5
        # C11 = M1 - M2 + M5 - M3
        
        s2 = [
            # (U, V, W) where U, V, W are 2x2 matrices
            # U defines left operand: X = U @ A_blocks
            # V defines right operand: Y = V @ B_blocks  
            # W defines contribution to C: C += W * M (where M is the scalar product)
            (np.array([[1,0],[0,1]], dtype=float), np.array([[1,0],[0,1]], dtype=float), np.array([[1,0],[0,1]], dtype=float)),   # M1
            (np.array([[0,0],[1,1]], dtype=float), np.array([[1,0],[0,0]], dtype=float), np.array([[0,0],[1,0]], dtype=float)),   # M2
            (np.array([[1,0],[0,0]], dtype=float), np.array([[0,0],[1,-1]], dtype=float), np.array([[0,1],[0,0]], dtype=float)),   # M3
            (np.array([[0,0],[0,1]], dtype=float), np.array([[0,1],[0,0]], dtype=float), np.array([[-1,1],[0,0]], dtype=float)),  # M4
            (np.array([[1,1],[0,0]], dtype=float), np.array([[0,0],[0,1]], dtype=float), np.array([[0,0],[1,1]], dtype=float)),   # M5
            (np.array([[-1,0],[1,0]], dtype=float), np.array([[1,1],[0,0]], dtype=float), np.array([[-1,0],[0,0]], dtype=float)), # M6
            (np.array([[0,1],[0,-1]], dtype=float), np.array([[0,0],[1,1]], dtype=float), np.array([[1,0],[0,0]], dtype=float)),  # M7
        ]
        
        terms = []
        
        for Uo, Vo, Wo in s2:
            for Ui, Vi, Wi in s2:
                # Effective coefficients
                U_eff = Ui @ Uo  # 2x2
                V_eff = Vi @ Vo  # 2x2
                
                # The left operand is a 2x2 matrix of 2x2 blocks:
                # L[r,c] = sum_l U_eff[r,l] * A_block[l,c]
                # where A_block[l,c] = A[2*l:2*l+2, 2*c:2*c+2]
                
                # The right operand is a 2x2 matrix of 2x2 blocks:
                # R[r,c] = sum_l V_eff[r,l] * B_block[l,c]
                
                # The product P = L * R is a 2x2 matrix of 2x2 blocks:
                # P[r,c] = sum_k L[r,k] * R[k,c]
                # P[r,c][i,j] = sum_k sum_a L[r,k][i,a] * R[k,c][a,j]
                
                # Each entry P[r,c][i,j] is a sum of products of linear combinations.
                # For the tensor decomposition, we need to express this as rank-1 terms.
                
                # The contribution to C from this (Uo, Vo, Ui, Vi) combination is:
                # C_block[r,c] += Wi[r,c] * P[r,c]  (after applying W_outer)
                # Actually, the full contribution is:
                # C += Wo @ (Wi @ P)  where P = L*R
                
                # Let me compute the effective W:
                W_eff = Wi @ Wo  # 2x2
                
                # Now, for each entry (r, c) of the 2x2 block result, and each entry (i, j) within the block:
                # C[2*c+i, 2*r+j] += W_eff[r,c] * P[r,c][i,j]
                # P[r,c][i,j] = sum_k sum_a L[r,k][i,a] * R[k,c][a,j]
                # L[r,k][i,a] = sum_l U_eff[r,l] * A[2*l+i, 2*k+a]
                # R[k,c][a,j] = sum_m V_eff[k,m] * B[2*m+a, 2*c+j]
                
                # So: P[r,c][i,j] = sum_k sum_a sum_l sum_m U_eff[r,l] * V_eff[k,m] * A[2*l+i, 2*k+a] * B[2*m+a, 2*c+j]
                
                # This is a sum of 16 products (4 values of k, 4 values of a... wait, k and a each range over 2)
                # Actually k ranges over 2 (block index), a ranges over 2 (within block), l ranges over 2, m ranges over 2
                # So 2*2*2*2 = 16 terms per (r,c,i,j)
                
                # But we want rank-1 terms, not sums. The issue is that P[r,c][i,j] is itself a sum of products.
                
                # The correct approach: each (Uo, Vo, Ui, Vi) combination gives ONE rank-1 term
                # where the u vector has the coefficients of the linear combination of A,
                # the v vector has the coefficients of the linear combination of B,
                # and the w vector has the coefficients for the contribution to C.
                
                # Let me construct the u, v, w vectors:
                u = np.zeros(16)
                v = np.zeros(16)
                w = np.zeros(16)
                
                # u[i,j] = coefficient of A[i,j] in the left linear combination
                # The left operand L is a 2x2 matrix of 2x2 blocks:
                # L[r,k][i,a] = sum_l U_eff[r,l] * A[2*l+i, 2*k+a]
                # So the coefficient of A[2*l+i, 2*k+a] in L[r,k][i,a] is U_eff[r,l]
                
                # But we need a single linear combination, not a matrix of linear combinations.
                # The product P[r,c] = sum_k L[r,k] * R[k,c] involves all entries of L and R.
                
                # I think the issue is that the double Strassen doesn't directly give rank-1 terms
                # in the way I'm thinking. Let me try a different approach.
                
                # Actually, I recall that the rank-49 decomposition of 4x4 matrix multiply
                # can be constructed by taking the tensor product of two rank-7 decompositions
                # of 2x2 matrix multiply, but with appropriate identifications.
                
                # Let me just use schoolbook for now and accept rank 80.
                pass
        
        return None  # Not implemented correctly

    # For now, use schoolbook decomposition: rank 80
    result = schoolbook()
    return result
# EVOLVE-BLOCK-END
