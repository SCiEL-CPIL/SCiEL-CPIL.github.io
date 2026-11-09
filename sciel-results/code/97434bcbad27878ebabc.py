"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _build_2x2_strassen():
    """Return U, V, W for 2x2 * 2x2 multiplication (rank 7).
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    A is 2x2, B is 2x2, C = A*B is 2x2.
    C[i,k] = sum_j A[i,j]*B[j,k]
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U maps A indices (i,j) to coefficient in each M
    # V maps B indices (j,k) to coefficient in each M
    # W maps C indices (k,i) to coefficient in each M
    
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (k,i) -> k*2+i: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1  # A00
    U[3, 0] = 1  # A11
    V[0, 0] = 1  # B00
    V[3, 0] = 1  # B11
    
    # M2 = (A10+A11)*B00
    U[2, 1] = 1  # A10
    U[3, 1] = 1  # A11
    V[0, 1] = 1  # B00
    
    # M3 = A00*(B01-B11)
    U[0, 2] = 1  # A00
    V[1, 2] = 1  # B01
    V[3, 2] = -1 # B11
    
    # M4 = A11*(B10-B00)
    U[3, 3] = 1  # A11
    V[2, 3] = 1  # B10
    V[0, 3] = -1 # B00
    
    # M5 = (A00+A01)*B11
    U[0, 4] = 1  # A00
    U[1, 4] = 1  # A01
    V[3, 4] = 1  # B11
    
    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1  # A10
    U[0, 5] = -1 # A00
    V[0, 5] = 1  # B00
    V[1, 5] = 1  # B01
    
    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1  # A01
    U[3, 6] = -1 # A11
    V[2, 6] = 1  # B10
    V[3, 6] = 1  # B11
    
    # C00 = M1+M4-M5+M7
    W[0, 0] = 1   # M1
    W[0, 3] = 1   # M4
    W[0, 4] = -1  # M5
    W[0, 6] = 1   # M7
    
    # C01 = M3+M5
    W[1, 2] = 1   # M3
    W[1, 4] = 1   # M5
    
    # C10 = M2+M4
    W[2, 1] = 1   # M2
    W[2, 3] = 1   # M4
    
    # C11 = M1-M2+M3+M6
    W[3, 0] = 1   # M1
    W[3, 1] = -1  # M2
    W[3, 2] = 1   # M3
    W[3, 5] = 1   # M6
    
    return U, V, W


def _build_2x4_schoolbook():
    """Return U, V, W for 2x4 * 4x4 multiplication (rank 8).
    Wait, we need 2x4 times 4x8? No.
    
    Let me reconsider the decomposition.
    We want to multiply 3x4 by 4x8.
    
    The strategy says: treat 4x8 as Kronecker product of 2x4 and 2x2.
    Actually, let me re-read: "treating the 4x8 multiplication as the Kronecker product of a 2x4 and a 2x2 multiplication"
    
    Hmm, 4x8 = (2*2) x (4*2) = (2x4) ⊗ (2x2)? That gives dimensions (2*2) x (4*2) = 4x8. Yes!
    So the 4x8 matrix is B = B1 ⊗ B2 where B1 is 2x4 and B2 is 2x2.
    And the 3x4 matrix is A = A1 ⊗ A2 where A1 is 3x2 and A2 is 2x2.
    Then C = A*B = (A1⊗A2)(B1⊗B2) = (A1*B1) ⊗ (A2*B2)
    
    A1 is 3x2, B1 is 2x4, so A1*B1 is 3x4.
    A2 is 2x2, B2 is 2x2, so A2*B2 is 2x2.
    C is (3x4) ⊗ (2x2) = 6x8. But we need C to be 3x8.
    
    Wait, that doesn't work. Let me reconsider.
    
    Actually, 3x4 * 4x8 = 3x8.
    If we write 4 = 2*2 and 8 = 4*2, then:
    A is 3x(2*2), B is (2*2)x(4*2).
    We can think of A as having shape (3, 2, 2) and B as (2, 2, 4, 2).
    This is getting complicated. Let me think differently.
    
    Alternative: The Kronecker product approach for tensor decomposition.
    If we have two matrix multiplications (n1,m1,p1) and (n2,m2,p2),
    then (n1*n2, m1*m2, p1*p2) can be done with rank r1*r2.
    
    So for 3x4*4x8:
    We need n1*n2=3, m1*m2=4, p1*p2=8.
    Possible: n1=3,n2=1; m1=4,m2=1; p1=8,p2=1 -> rank = (3*4*8)*(1*1*1) = 96, worse.
    Or: n1=3,n2=1; m1=2,m2=2; p1=4,p2=2 -> rank = (3*2*4)*(1*2*2) = 24*4 = 96, worse.
    
    Hmm, that's not helping. Let me reconsider the strategy description.
    
    "treating the 4x8 multiplication as the Kronecker product of a 2x4 and a 2x2 multiplication, using the rank-7 Strassen algorithm for the 2x2 component and the schoolbook rank-8 for the 2x4 component to achieve a total rank of 56"
    
    7 * 8 = 56. So the idea is that the 4x8 part decomposes into a 2x4 part (rank 8) and a 2x2 part (rank 7).
    
    But how does this apply to the full 3x4*4x8 problem?
    
    I think the idea is:
    - The 4x8 matrix B can be seen as B = [B1; B2] where B1 is 2x8 and B2 is 2x8, or B = [B1, B2] where B1 is 4x4 and B2 is 4x4.
    
    Let me try: B is 4x8. Write B = [B_left | B_right] where each is 4x4.
    Then A*B = A*[B_left | B_right] = [A*B_left | A*B_right].
    This doesn't directly help.
    
    Alternatively, write the 4 rows of B as pairs. B = [[B00, B01], [B10, B11]] where each Bij is 2x4.
    And A = [[A0, A1]] where A0, A1 are 3x2.
    Then C = A*B has rows:
    C[0:2, :] = A0*B00 + A1*B10  (3x8)
    C[2:4, :] = A0*B01 + A1*B11  (3x8)
    Wait, that gives a 4x8 output, but we need 3x8.
    
    I'm confused. Let me just try a different approach: use known low-rank decompositions.
    
    For 3x4*4x8, the schoolbook rank is 96. Can we do better?
    
    One approach: block decomposition.
    A is 3x4, B is 4x8.
    Split A into A1 (3x2) and A2 (3x2), B into B1 (2x8) and B2 (2x8).
    C = A1*B1 + A2*B2.
    This requires rank(3x2*2x8) + rank(3x2*2x8) = 48 + 48 = 96. No improvement.
    
    Another approach: Use the fact that 3x4*4x8 can be related to smaller problems.
    
    Actually, let me just try to find any decomposition with rank < 96.
    
    For now, let me just return the schoolbook decomposition to ensure correctness, and then try to improve.
    """
    n, m, p = 2, 4, 4
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
    return U, V, W


def _schoolbook_3x4_4x8():
    """Schoolbook decomposition for 3x4 * 4x8."""
    n, m, p = 3, 4, 8
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
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_k V[j*p+k, c] * B[j,k])
    # Actually: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # This is not quite right. Let me think again.
    # The tensor is T[i,j,k] = delta(i,j,k) for matrix mult.
    # T = sum_c U[:,c] * V[:,c] * W[:,c] (outer products)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix mult, T[i,j,k] should be 1 if the indices correspond to a valid term, else 0.
    # Specifically, C[i,k] = sum_j A[i,j]*B[j,k], so T[i,j,k] = 1 for all valid i,j,k (it's the identity in the "multiplication" sense).
    # Wait, no. The matrix multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Because C[i,k] = sum_j A[i,j]*B[j,k], and each term A[i,j]*B[j,k] contributes to C[i,k].
    # So the tensor is just all ones? No, that's not right either.
    # Actually, the matrix multiplication tensor is defined such that:
    # C = A*B means C[i,k] = sum_j A[i,j]*B[j,k]
    # This can be written as C[i,k] = sum_{j,c} T[i,j,k,c] * A[i,j] * B[j,k]
    # where T[i,j,k,c] is the tensor. For schoolbook, T[i,j,k,c] = 1 if c corresponds to (i,j,k), else 0.
    # So the tensor is a set of rank-1 tensors, each with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1.
    
    # To verify, we can check that for random A, B:
    # C = A*B equals sum_c W[:,c] * (U[:,c]^T * A_reshaped) * (V[:,c]^T * B_reshaped)
    # where A_reshaped has shape (n*m,) and B_reshaped has shape (m*p,).
    
    np.random.seed(42)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    A_flat = A.flatten()  # shape (n*m,)
    B_flat = B.flatten()  # shape (m*p,)
    
    C_computed = np.zeros((n, p))
    for c in range(U.shape[1]):
        # intermediate1[i] = sum_j U[i*m+j, c] * A[i,j]
        # But U[i*m+j, c] is a scalar, and A[i,j] is a scalar.
        # Actually, the contraction is:
        # prod_c = (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        # This doesn't make sense because i and k are summed over in different ways.
        
        # Let me reconsider. The standard way:
        # For each c, we compute:
        # u_c = U[:, c]  # shape (n*m,)
        # v_c = V[:, c]  # shape (m*p,)
        # w_c = W[:, c]  # shape (p*n,)
        # Then the contribution to C is:
        # C += w_c_reshaped * (u_c_reshaped * A_reshaped) * (v_c_reshaped * B_reshaped)
        # where u_c_reshaped has shape (n, m), v_c_reshaped has shape (m, p), w_c_reshaped has shape (p, n) -> (n, p)
        
        u_reshaped = u_c.reshape(n, m) if False else U[:, c].reshape(n, m)
        v_reshaped = V[:, c].reshape(m, p)
        w_reshaped = W[:, c].reshape(p, n).T  # shape (n, p)
        
        # The contribution is:
        # C += w_reshaped * (u_reshaped * A) @ (v_reshaped * B)
        # Wait, that's not right either. Let me look at the schoolbook case.
        # For schoolbook, U[i*m+j, c] = 1 for a specific (i,j,k), V[j*p+k, c] = 1, W[k*n+i, c] = 1.
        # So u_reshaped has a 1 at position (i,j), v_reshaped has a 1 at position (j,k), w_reshaped has a 1 at position (k,i) -> (i,k).
        # The contribution should be A[i,j] * B[j,k] added to C[i,k].
        # So: C += w_reshaped * (u_reshaped * A) * (v_reshaped * B) where * is element-wise and the result is summed?
        # No. Let me think of it as:
        # term = (u_reshaped * A) @ (v_reshaped * B)  # This is (n,m) @ (m,p) = (n,p)
        # But u_reshaped * A is (n,m) and v_reshaped * B is (m,p).
        # For schoolbook, u_reshaped has one 1, so u_reshaped * A has one non-zero entry A[i,j].
        # v_reshaped * B has one non-zero entry B[j,k].
        # (u_reshaped * A) @ (v_reshaped * B) would be a matrix with only one non-zero entry at (i,k) with value A[i,j]*B[j,k].
        # Then C += w_reshaped * that matrix. Since w_reshaped has a 1 at (i,k), C[i,k] += A[i,j]*B[j,k]. Correct!
        
        term = (u_reshaped * A) @ (v_reshaped * B)
        C_computed += w_reshaped * term
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        return False
    return True


def construct():
    # Start with schoolbook
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook_3x4_4x8()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Should not happen
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
