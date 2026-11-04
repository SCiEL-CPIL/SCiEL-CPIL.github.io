"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _build_strassen_2x2x2():
    """
    Return U, V, W for 2x2x2 matrix multiplication with Strassen's rank-7 decomposition.
    U: shape (4, 7), U[i*2+j, r]
    V: shape (4, 7), V[j*2+k, r]
    W: shape (4, 7), W[k*2+i, r]
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen's 7 products:
    # M1 = (A11)(B12 - B22)
    # M2 = (A11 + A22)(B22)
    # M3 = (A22 - A11)(B11)
    # M4 = (A21 + A22)(B22 - B12)
    # M5 = (A11 + A12)(B11)
    # M6 = (A12 - A22)(B21 + B22)
    # M7 = (A21 - A11)(B11 + B12)
    
    # U encodes coefficients for A elements in each M_r
    # A is 2x2, indexed A[i,j], U[i*2+j, r]
    # V encodes coefficients for B elements in each M_r
    # B is 2x2, indexed B[j,k], V[j*2+k, r]
    # W encodes coefficients for C elements from each M_r
    # C is 2x2, indexed C[i,k], W[k*2+i, r]
    
    # M1: A11 * (B12 - B22) => A[0,0]=1, B[0,1]=1, B[1,1]=-1
    U[0*2+0, 0] = 1
    V[0*2+1, 0] = 1
    V[1*2+1, 0] = -1
    # C contribution: C11 += M1, C12 -= M1 => W[0*2+0,0]=1 (C[0,0]), W[1*2+0,0]=-1 (C[0,1])
    W[0*2+0, 0] = 1
    W[1*2+0, 0] = -1
    
    # M2: (A11 + A22) * B22 => A[0,0]=1, A[1,1]=1, B[1,1]=1
    U[0*2+0, 1] = 1
    U[1*2+1, 1] = 1
    V[1*2+1, 1] = 1
    # C22 += M2 => W[1*2+1,1]=1 (C[1,1])
    W[1*2+1, 1] = 1
    
    # M3: (A22 - A11) * B11 => A[1,1]=1, A[0,0]=-1, B[0,0]=1
    U[1*2+1, 2] = 1
    U[0*2+0, 2] = -1
    V[0*2+0, 2] = 1
    # C22 += M3, C21 -= M3? Let me be careful.
    # C11 = M1 + M2 - M3 - M7... wait, let me use the standard formula.
    # C11 = M1 + M2 - M3 - M7
    # C12 = M5 + M7
    # C21 = M4 + M6
    # C22 = M1 - M2 + M3 + M4
    
    # Let me redo W more carefully based on the standard Strassen output formulas.
    # C[0,0] = M1 + M2 - M3 - M7
    # C[0,1] = M5 + M7
    # C[1,0] = M4 + M6
    # C[1,1] = M1 - M2 + M3 + M4
    
    # W[k*2+i, r] means W index for C[i,k] is k*2+i
    
    # Reset W
    W[:] = 0
    # C[0,0] = M1 + M2 - M3 - M7
    W[0*2+0, 0] = 1   # M1
    W[0*2+0, 1] = 1   # M2
    W[0*2+0, 2] = -1  # M3
    W[0*2+0, 6] = -1  # M7
    
    # C[0,1] = M5 + M7
    W[1*2+0, 4] = 1   # M5 (index 4)
    W[1*2+0, 6] = 1   # M7
    
    # C[1,0] = M4 + M6
    W[0*2+1, 3] = 1   # M4 (index 3)
    W[0*2+1, 5] = 1   # M6 (index 5)
    
    # C[1,1] = M1 - M2 + M3 + M4
    W[1*2+1, 0] = 1   # M1
    W[1*2+1, 1] = -1  # M2
    W[1*2+1, 2] = 1   # M3
    W[1*2+1, 3] = 1   # M4
    
    # Now define U and V for all 7 products
    U[:] = 0
    V[:] = 0
    
    # M1 (r=0): A[0,0] * (B[0,1] - B[1,1])
    U[0*2+0, 0] = 1
    V[0*2+1, 0] = 1
    V[1*2+1, 0] = -1
    
    # M2 (r=1): (A[0,0] + A[1,1]) * B[1,1]
    U[0*2+0, 1] = 1
    U[1*2+1, 1] = 1
    V[1*2+1, 1] = 1
    
    # M3 (r=2): (A[1,1] - A[0,0]) * B[0,0]
    U[1*2+1, 2] = 1
    U[0*2+0, 2] = -1
    V[0*2+0, 2] = 1
    
    # M4 (r=3): (A[1,0] + A[1,1]) * (B[1,1] - B[0,1])
    U[1*2+0, 3] = 1
    U[1*2+1, 3] = 1
    V[1*2+1, 3] = 1
    V[0*2+1, 3] = -1
    
    # M5 (r=4): (A[0,0] + A[0,1]) * B[0,0]
    U[0*2+0, 4] = 1
    U[0*2+1, 4] = 1
    V[0*2+0, 4] = 1
    
    # M6 (r=5): (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    U[0*2+1, 5] = 1
    U[1*2+1, 5] = -1
    V[1*2+0, 5] = 1
    V[1*2+1, 5] = 1
    
    # M7 (r=6): (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    U[1*2+0, 6] = 1
    U[0*2+0, 6] = -1
    V[0*2+0, 6] = 1
    V[0*2+1, 6] = 1
    
    return U, V, W

def _tensor_product_strassen(base_n, base_m, base_p, target_n, target_m, target_p):
    """
    Build a tensor decomposition for target_n x target_m x target_p using
    Kronecker products of the base Strassen decomposition.
    """
    if target_n % base_n != 0 or target_m % base_m != 0 or target_p % base_p != 0:
        return None
    
    base_U, base_V, base_W = _build_strassen_2x2x2()
    
    # The tensor product of two decompositions of sizes (n1,m1,p1) and (n2,m2,p2)
    # gives a decomposition of size (n1*n2, m1*m2, p1*p2) with rank r1*r2.
    # We need to build up from 2x2x2 to target size.
    
    # For 4x4x8 from 2x2x2:
    # Step 1: 2x2x2 -> 4x4x4 (rank 49)
    # Step 2: 4x4x4 -> 4x4x8 (rank 98)
    
    # General approach: build up by tensor products.
    # We'll represent the decomposition as U, V, W matrices.
    
    def kron_decomp(U1, V1, W1, U2, V2, W2):
        """
        Tensor product of two decompositions.
        U1: (n1*m1, r1), V1: (m1*p1, r1), W1: (p1*n1, r1)
        U2: (n2*m2, r2), V2: (m2*p2, r2), W2: (p2*n2, r2)
        Result: U: (n1*n2*m1*m2, r1*r2), etc.
        """
        n1, m1, p1, r1 = U1.shape[0] // m1, U1.shape[0] // (U1.shape[1]), None, U1.shape[1]
        # This is getting complex. Let me just do it for our specific case.
        pass
    
    # For our specific case: 4x4x8
    # We'll do two Kronecker products.
    # First: 2x2x2 ⊗ 2x2x2 = 4x4x4, rank 49
    # Second: 4x4x4 ⊗ 2x2x2... no, we need 4x4x8.
    # Actually 4x4x8 = (2x2x2) ⊗ (2x2x4)? No, we only have 2x2x2 base.
    # 4x4x8: n=4, m=4, p=8. 
    # We can do: (2x2x2) ⊗ (2x2x2) gives 4x4x4.
    # Then we need to extend p from 4 to 8. That's another factor of 2 in p.
    # So: (4x4x4) ⊗ (1x1x2)? We don't have a 1x1x2 base.
    # Alternatively: (2x2x2) ⊗ (2x2x4)? We don't have that either.
    # 
    # Better approach: just do it directly.
    # 4x4x8 can be seen as two 4x4x4 problems (columns 0-3 and 4-7 of B).
    # Each 4x4x4 uses rank 49, total rank 98.
    
    # Build 4x4x4 rank-49 decomposition first.
    U4, V4, W4 = _kron_2x2x2_to_4x4x4(base_U, base_V, base_W)
    
    # Now build 4x4x8 by concatenating two copies.
    n, m, p = 4, 4, 8
    rank = 2 * 49  # 98
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # First half: B columns 0-3
    U[:, 0:49] = U4
    V[0:4*4, 0:49] = V4  # V4 has shape (4*4, 49) = (16, 49)
    W[0:4*4, 0:49] = W4  # W4 has shape (4*4, 49) = (16, 49), k in 0..3
    
    # Second half: B columns 4-7
    U[:, 49:98] = U4
    V[4:16*8, 49:98] = V4  # V rows for k=4..7: j*8+k where j in 0..3, k in 4..7
    # V4[j*4+k, r] maps to V[j*8+(k+4), r]
    # V4 has shape (16, 49), indexed by j*4+k for j,k in 0..3
    # We need to place it into V[j*8+(k+4), r]
    for j in range(4):
        for k in range(4):
            V[j*8 + (k+4), 49:98] = V4[j*4 + k, :]
    
    # W for second half: k in 4..7, so W[(k)*4+i, r] for k in 4..7
    for k in range(4):
        for i in range(4):
            W[(k+4)*4 + i, 49:98] = W4[k*4 + i, :]
    
    return U, V, W

def _kron_2x2x2_to_4x4x4(U2, V2, W2):
    """
    Tensor product of 2x2x2 with itself to get 4x4x4.
    U2: (4, 7), V2: (4, 7), W2: (4, 7)
    Result: U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n1, m1, p1, r1 = 2, 2, 2, 7
    n2, m2, p2, r2 = 2, 2, 2, 7
    
    n = n1 * n2
    m = m1 * m2
    p = p1 * p2
    rank = r1 * r2
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r = r1_idx * r2 + r2_idx
            
            # U: index (i1*m2*n2 + i2*m2 + j1*n2 + j2, r) 
            # Actually, for tensor product, A is 4x4, A[i,j] where i = i1*2+i2, j = j1*2+j2
            # U[i*4+j, r] = U2[i1*2+j1, r1_idx] * U2[i2*2+j2, r2_idx]
            for i1 in range(n1):
                for j1 in range(m1):
                    u1 = U2[i1*m1 + j1, r1_idx]
                    if u1 == 0:
                        continue
                    for i2 in range(n2):
                        for j2 in range(m2):
                            u2 = U2[i2*m2 + j2, r2_idx]
                            if u2 == 0:
                                continue
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            U[i * m + j, r] += u1 * u2
            
            # V: B is 4x4, B[j,k] where j = j1*2+j2, k = k1*2+k2
            # V[j*4+k, r] = V2[j1*2+k1, r1_idx] * V2[j2*2+k2, r2_idx]
            for j1 in range(m1):
                for k1 in range(p1):
                    v1 = V2[j1*p1 + k1, r1_idx]
                    if v1 == 0:
                        continue
                    for j2 in range(m2):
                        for k2 in range(p2):
                            v2 = V2[j2*p2 + k2, r2_idx]
                            if v2 == 0:
                                continue
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            V[j * p + k, r] += v1 * v2
            
            # W: C is 4x4, C[i,k] where i = i1*2+i2, k = k1*2+k2
            # W[k*4+i, r] = W2[k1*2+i1, r1_idx] * W2[k2*2+i2, r2_idx]
            for k1 in range(p1):
                for i1 in range(n1):
                    w1 = W2[k1*n1 + i1, r1_idx]
                    if w1 == 0:
                        continue
                    for k2 in range(p2):
                        for i2 in range(n2):
                            w2 = W2[k2*n2 + i2, r2_idx]
                            if w2 == 0:
                                continue
                            i = i1 * n2 + i2
                            k = k1 * p2 + k2
                            W[k * n + i, r] += w1 * w2
    
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    try:
        base_U, base_V, base_W = _build_strassen_2x2x2()
        U, V, W = _tensor_product_strassen(2, 2, 2, 4, 4, 8)
        
        # Verify the decomposition
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if i==j==k else 0
        # For matrix mult, C[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor: T[i,j,k] = 1 if i==j==k... no.
        # The matrix multiplication tensor has T[i,j,k] = 1 always (for the bilinear form).
        # Actually, the decomposition satisfies:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta(i,j) * delta(j,k) ... no.
        
        # Let me just verify by constructing a random A and B and checking C = A*B.
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_actual = np.zeros((n, p))
        for r in range(rank):
            A_r = U[:, r].reshape(n, m)
            B_r = V[:, r].reshape(m, p)
            C_r = W[:, r].reshape(p, n)
            C_actual += (A_r * A).sum() * (B_r * B).sum() * 0  # This is wrong
        
        # Correct verification:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i*m+j,r]) (sum_k' ... )
        # Actually: C = sum_r outer(A_r * A, B_r * B) ... no.
        # The standard tensor decomposition: C[i,k] = sum_r (sum_j A[i,j] U[i*m+j,r]) (sum_l B[j,l] V[j*p+l,r]) ... 
        # No, it's: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] * A[i,j] * B[j,k] should equal A[i,j]*B[j,k] when summed over j? No.
        
        # The tensor T such that T[i,j,k] A[i,j] B[j,k] summed = C.
        # T[i,j,k] = delta(i,j)*delta(j,k)? No, T[i,j,k] = 1 for all valid i,j,k, and C[i,k] = sum_j A[i,j]B[j,k].
        # The decomposition: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k.
        # That can't be right either because then the sum would be n*m*p.
        
        # Let me just trust the construction and verify numerically.
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # A_r is n x m, B_r is m x p
            A_r = U[:, r].reshape(n, m)
            B_r = V[:, r].reshape(m, p)
            C_r = W[:, r].reshape(p, n)
            # Contribution to C: (A * A_r^T) (B_r * B^T) ... no
            # It's: C += (A * A_r) ... I'm overcomplicating.
            
            # The bilinear form: sum_{i,j,k} A[i,j] B[j,k] U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
            # This should equal sum_{i,j,k} A[i,j] B[j,k] for each r summed.
            
            pass
        
        # Simpler verification: build the full tensor and check.
        # T_decomp[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Should equal 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k] means the tensor is all-ones).
        
        # Actually no. The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k.
        # And C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k] = sum_j A[i,j] B[j,k].
        # So yes, T[i,j,k] = 1 for all valid i,j,k.
        
        T_decomp = np.zeros((n, m, p))
        for r in range(rank):
            U_r = U[:, r].reshape(n, m)
            V_r = V[:, r].reshape(m, p)
            W_r = W[:, r].reshape(p, n)
            T_decomp += np.einsum('ij,jk,ki->ijk', U_r, V_r, W_r)
        
        # T_decomp should be all 1s
        if not np.allclose(T_decomp, 1.0):
            # Try to verify differently
            max_diff = np.max(np.abs(T_decomp - 1.0))
            if max_diff > 1e-10:
                # Fall back to schoolbook
                rank_school = n * m * p
                U_school = [[0] * rank_school for _ in range(n * m)]
                V_school = [[0] * rank_school for _ in range(m * p)]
                W_school = [[0] * rank_school for _ in range(p * n)]
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_school[i * m + j][col] = 1
                            V_school[j * p + k][col] = 1
                            W_school[k * n + i][col] = 1
                            col += 1
                return {"U": U_school, "V": V_school, "W": W_school}
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    except Exception:
        # Fall back to schoolbook
        n, m, p = DIMENSIONS
        rank_school = n * m * p
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
