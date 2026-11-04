"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook: rank = n*m*p = 4*4*5 = 80
    # U[i*m+j, t] = 1 for term (i,j,k)
    # V[j*p+k, t] = 1 for term (i,j,k)
    # W[k*n+i, t] = 1 for term (i,j,k)
    
    # Strategy: Try to find a lower-rank decomposition using Strassen-style
    # sub-block decompositions. We'll use Strassen for 2x2x2 blocks where possible.
    
    # For 4x4 * 4x5:
    # We can partition A into 2x2 blocks: A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12, B13], [B21, B22, B23]] where B11,B21 are 2x2, B12,B22 are 2x2, B13,B23 are 2x1
    # Wait, B is 4x5, so if we split rows into 2+2, columns into 2+2+1:
    # B = [[B11(2x2), B12(2x2), B13(2x1)], [B21(2x2), B22(2x2), B23(2x1)]]
    
    # C = A*B is 4x5: C = [[C11(2x2), C12(2x2), C13(2x1)], [C21(2x2), C22(2x2), C23(2x1)]]
    
    # C11 = A11*B11 + A12*B21  (2x2 * 2x2 + 2x2 * 2x2)
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    
    # For each 2x2 * 2x2 product, we can use Strassen (rank 7 instead of 8).
    # There are 6 such 2x2*2x2 products: C11 has 2, C12 has 2, C21 has 2, C22 has 2.
    # Wait: C11 = A11*B11 + A12*B21: that's 2 separate 2x2*2x2 multiplications.
    # C12 = A11*B12 + A12*B22: 2 more.
    # C21 = A21*B11 + A22*B21: 2 more.
    # C22 = A21*B12 + A22*B22: 2 more.
    # Total: 8 2x2*2x2 multiplications.
    # Each uses rank 7 instead of 8, saving 1 each: total save 8.
    
    # For 2x2 * 2x1 products:
    # C13 = A11*B13 + A12*B23: A11 is 2x2, B13 is 2x1 -> 2x1 result
    # This is just 4 scalar multiplications (2*2*1=4) for each, total 4*2=8 terms.
    # Similarly C23: 8 terms.
    # No savings here.
    
    # Total schoolbook: 80
    # With Strassen on 8 blocks: 80 - 8 = 72
    # But wait, the Strassen decomposition shares intermediate products.
    # Actually, for separate 2x2*2x2 multiplications, each independently uses 7 mults.
    # So total rank = 8*7 + 8*4 = 56 + 32 = 88? No, that's worse.
    
    # Let me recount. Schoolbook for 4x4*4x5:
    # Each output C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    # Total scalar multiplications: 4*5*4 = 80.
    
    # If I use Strassen for each 2x2*2x2 sub-product:
    # The 8 2x2*2x2 products would normally take 8*8=64 scalar mults.
    # With Strassen: 8*7=56 scalar mults.
    # The remaining 2x2*2x1 and 2x1*2x1 products: 
    # C13 = A11*B13 + A12*B23: A11(2x2)*B13(2x1) = 4 mults, A12(2x2)*B23(2x1) = 4 mults. Total 8.
    # C23 = A21*B13 + A22*B23: similarly 8 mults.
    # Total: 56 + 8 + 8 = 72.
    
    # So rank 72 is achievable. Let me construct it.
    
    # Actually, let me just implement the schoolbook for now since it's guaranteed correct,
    # and the task says "score is the rank" so lower is better.
    # But 72 < 80, so let me try to implement the Strassen version.
    
    # Let me build the factor matrices for the Strassen-based decomposition.
    
    # Strassen 2x2*2x2 rank-7 decomposition:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)*(e+h)
    # M2 = (c+d)*e
    # M3 = a*(g-e)
    # M4 = d*(h-g)
    # M5 = (a+b)*(h+f)
    # M6 = (c+a)*(e+g)
    # M7 = (b+d)*(g-f)
    # C = [[M1+M4-M2-M7, M3+M7], [M2+M6, M1-M3+M5-M6]]
    
    # For our problem, we need to map these to the global indices.
    
    # Let me define the rank-72 decomposition explicitly.
    # I'll use a list of (U_entry, V_entry, W_entry) tuples.
    
    # Each rank-1 term has:
    # U[i*m+j] = coefficient for A[i,j]
    # V[j*p+k] = coefficient for B[j,k]
    # W[k*n+i] = coefficient for C[i,k]
    
    # Let me build this systematically.
    
    # First, handle the 8 2x2*2x2 Strassen blocks.
    # Block (bi, bk) where bi in {0,1} (row block of A), bk in {0,1} (col block of B for 2x2 part)
    # Actually, let me index the 2x2 blocks:
    # A blocks: A[0:2,0:2], A[0:2,2:4], A[2:4,0:2], A[2:4,2:4]
    # B 2x2 blocks: B[0:2,0:2], B[0:2,2:4], B[2:4,0:2], B[2:4,2:4]
    
    # The 8 2x2*2x2 products are:
    # 1. A[0:2,0:2] * B[0:2,0:2] -> contributes to C[0:2,0:2]
    # 2. A[0:2,2:4] * B[2:4,0:2] -> contributes to C[0:2,0:2]
    # 3. A[0:2,0:2] * B[0:2,2:4] -> contributes to C[0:2,2:4]
    # 4. A[0:2,2:4] * B[2:4,2:4] -> contributes to C[0:2,2:4]
    # 5. A[2:4,0:2] * B[0:2,0:2] -> contributes to C[2:4,0:2]
    # 6. A[2:4,2:4] * B[2:4,0:2] -> contributes to C[2:4,0:2]
    # 7. A[2:4,0:2] * B[0:2,2:4] -> contributes to C[2:4,2:4]
    # 8. A[2:4,2:4] * B[2:4,2:4] -> contributes to C[2:4,2:4]
    
    # And the 2x2*2x1 products:
    # 9. A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5] (2x1)
    # 10. A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5] (2x1)
    # 11. A[2:4,0:2] * B[0:2,4:5] -> C[2:4,4:5] (2x1)
    # 12. A[2:4,2:4] * B[2:4,4:5] -> C[2:4,4:5] (2x1)
    
    # Total: 8*7 + 4*4 = 56 + 16 = 72. 
    
    # Let me construct the factor matrices.
    # I'll use numpy for efficiency.
    
    rank = 72
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    col = 0
    
    # Strassen helper: adds 7 rank-1 terms for a 2x2*2x2 product
    # A_block at rows ai:ai+2, cols aj:aj+2
    # B_block at rows bj:bj+2, cols bk:bk+2
    # C_block at rows ci:ci+2, cols ck:ck+2
    def add_strassen_2x2x2(U, V, W, n, m, p, ai, aj, bj, bk, ci, ck, col):
        # A indices: (ai,aj), (ai,aj+1), (ai+1,aj), (ai+1,aj+1)
        # B indices: (bj,bk), (bj,bk+1), (bj+1,bk), (bj+1,bk+1)
        # C indices: (ci,ck), (ci,ck+1), (ci+1,ck), (ci+1,ck+1)
        
        a00 = ai*m + aj
        a01 = ai*m + aj + 1
        a10 = (ai+1)*m + aj
        a11 = (ai+1)*m + aj + 1
        
        b00 = bj*p + bk
        b01 = bj*p + bk + 1
        b10 = (bj+1)*p + bk
        b11 = (bj+1)*p + bk + 1
        
        c00 = ck*n + ci
        c01 = ck*n + ci + 1
        c10 = (ck+1)*n + ci
        c11 = (ck+1)*n + ci + 1
        
        # M1 = (a00+a11)*(b00+b11)
        U[a00, col] = 1
        U[a11, col] = 1
        V[b00, col] = 1
        V[b11, col] = 1
        W[c00, col] = 1
        W[c01, col] = 0  # M1 contributes to C00 and C11 with signs
        W[c10, col] = 0
        W[c11, col] = 1
        # Actually, M1 contributes +1 to C00 and +1 to C11
        # But we need to track the full contribution. Let me use a different approach.
        # Each Strassen intermediate M_t contributes to multiple C entries with coefficients.
        # So for each M_t, I need to add its contribution to all relevant C entries.
        
        # Let me redo this. For each of the 7 Strassen products, I create one rank-1 term
        # but the W matrix will have multiple non-zero entries.
        
        # Actually, in the tensor decomposition, each rank-1 term is a single outer product
        # u * v * w where u, v, w are vectors. So each rank-1 term has exactly one non-zero
        # in U, one in V, and one in W per row/column. But the W vector can have multiple
        # non-zero entries.
        
        # So for M1 = (a00+a11)*(b00+b11):
        # U has +1 at a00 and +1 at a11
        # V has +1 at b00 and +1 at b11
        # W has +1 at c00 and +1 at c11
        # This is ONE rank-1 term in the tensor decomposition.
        
        # Let me clear what I just wrote and redo properly.
        U[a00, col] = 1
        U[a11, col] = 1
        V[b00, col] = 1
        V[b11, col] = 1
        W[c00, col] = 1
        W[c11, col] = 1
        col += 1
        
        # M2 = (a10+a11)*b00
        U[a10, col] = 1
        U[a11, col] = 1
        V[b00, col] = 1
        W[c10, col] = 1
        W[c11, col] = -1  # M2 contributes +1 to C10, -1 to C11
        col += 1
        
        # M3 = a00*(b10-b00)
        U[a00, col] = 1
        V[b10, col] = 1
        V[b00, col] = -1
        W[c01, col] = 1
        W[c11, col] = -1  # M3 contributes +1 to C01, -1 to C11
        col += 1
        
        # M4 = a11*(b01-b11)
        U[a11, col] = 1
        V[b01, col] = 1
        V[b11, col] = -1
        W[c00, col] = 1
        W[c11, col] = 1  # M4 contributes +1 to C00, +1 to C11
        col += 1
        
        # M5 = (a00+a01)*(b11+b01)
        U[a00, col] = 1
        U[a01, col] = 1
        V[b11, col] = 1
        V[b01, col] = 1
        W[c11, col] = 1
        col += 1
        
        # M6 = (a10+a00)*(b00+b10)
        U[a10, col] = 1
        U[a00, col] = 1
        V[b00, col] = 1
        V[b10, col] = 1
        W[c10, col] = 1
        W[c11, col] = -1  # M6 contributes +1 to C10, -1 to C11
        col += 1
        
        # M7 = (a01+a11)*(b10-b01)
        U[a01, col] = 1
        U[a11, col] = 1
        V[b10, col] = 1
        V[b01, col] = -1
        W[c01, col] = 1
        W[c11, col] = -1  # M7 contributes +1 to C01, -1 to C11
        col += 1
        
        return col
    
    # Add the 8 Strassen blocks
    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    col = add_strassen_2x2x2(U, V, W, n, m, p, 0, 0, 0, 0, 0, 0, col)
    
    # Block 2: A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
    col = add_strassen_2x2x2(U, V, W, n, m, p, 0, 2, 2, 0, 0, 0, col)
    
    # Block 3: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
    col = add_strassen_2x2x2(U, V, W, n, m, p, 0, 0, 0, 2, 0, 2, col)
    
    # Block 4: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    col = add_strassen_2x2x2(U, V, W, n, m, p, 0, 2, 2, 2, 0, 2, col)
    
    # Block 5: A[2:4,0:2] * B[0:2,0:2] -> C[2:4,0:2]
    col = add_strassen_2x2x2(U, V, W, n, m, p, 2, 0, 0, 0, 2, 0, col)
    
    # Block 6: A[2:4,2:4] * B[2:4,0:2] -> C[2:4,0:2]
    col = add_strassen_2x2x2(U, V, W, n, m, p, 2, 2, 2, 0, 2, 0, col)
    
    # Block 7: A[2:4,0:2] * B[0:2,2:4] -> C[2:4,2:4]
    col = add_strassen_2x2x2(U, V, W, n, m, p, 2, 0, 0, 2, 2, 2, col)
    
    # Block 8: A[2:4,2:4] * B[2:4,2:4] -> C[2:4,2:4]
    col = add_strassen_2x2x2(U, V, W, n, m, p, 2, 2, 2, 2, 2, 2, col)
    
    # Now handle the 2x2*2x1 products (4 of them, each with 4 scalar mults = 16 total)
    # These are for the last column of C (column 4).
    
    # Block 9: A[0:2,0:2] * B[0:2,4:5] -> C[0:2,4:5]
    # C[0,4] += A[0,0]*B[0,4] + A[0,1]*B[1,4]
    # C[1,4] += A[1,0]*B[0,4] + A[1,1]*B[1,4]
    for i in range(2):
        for j in range(2):
            U[i*m+j, col] = 1
            V[j*p+4, col] = 1
            W[4*n+i, col] = 1
            col += 1
    
    # Block 10: A[0:2,2:4] * B[2:4,4:5] -> C[0:2,4:5]
    for i in range(2):
        for j in range(2):
            U[(i)*m+(2+j), col] = 1
            V[(2+j)*p+4, col] = 1
            W[4*n+i, col] = 1
            col += 1
    
    # Block 11: A[2:4,0:2] * B[0:2,4:5] -> C[2:4,4:5]
    for i in range(2):
        for j in range(2):
            U[(2+i)*m+j, col] = 1
            V[j*p+4, col] = 1
            W[4*n+(2+i), col] = 1
            col += 1
    
    # Block 12: A[2:4,2:4] * B[2:4,4:5] -> C[2:4,4:5]
    for i in range(2):
        for j in range(2):
            U[(2+i)*m+(2+j), col] = 1
            V[(2+j)*p+4, col] = 1
            W[4*n+(2+i), col] = 1
            col += 1
    
    # Total should be 56 + 16 = 72
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify correctness with a random test
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    # Compute C from decomposition
    C_computed = np.zeros((p, n))
    for t in range(rank):
        u_vec = U[:, t]
        v_vec = V[:, t]
        w_vec = W[:, t]
        # u_vec is of length n*m, representing A flattened
        # v_vec is of length m*p, representing B flattened
        # w_vec is of length p*n, representing C flattened (with W[k*n+i, t])
        
        # The contribution to C is: for each (i,j) with U[i*m+j,t]!=0 and (j,k) with V[j*p+k,t]!=0,
        # add U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] to C[i,k]
        # But since this is a rank-1 term, it's simpler:
        # C += outer product of (A * u_vec reshaped) and (v_vec reshaped * B) weighted by w_vec
        
        # Actually, the tensor decomposition says:
        # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # where T[i,j,k] = 1 if this is the (i,j,k) entry of the multiplication tensor.
        # The multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the identity-like tensor).
        # Wait, no. The matrix multiplication tensor is defined as:
        # t(A,B) = sum_{i,j,k} C[i,k] * e_i ⊗ e_j ⊗ e_k where C[i,k] = sum_j A[i,j]*B[j,k]
        # So the tensor is T = sum_{i,j,k} A[i,j] * B[j,k] * C[i,k]
        # And we want T = sum_t (u_t · A) * (v_t · B) * (w_t · C)
        # where u_t, v_t, w_t are the t-th columns of U, V, W.
        
        # So to verify: for random A, B, compute C = A@B, then check that
        # sum_t (u_t · A_flat) * (v_t · B_flat) * (w_t · C_flat) == sum_{i,k} C[i,k]^2
        # Actually, that's not right either. Let me think more carefully.
        
        # The bilinear form: for each t, the term is (sum_{i,j} U[i*m+j,t]*A[i,j]) * (sum_{j,k} V[j*p+k,t]*B[j,k]) * (sum_{i,k} W[k*n+i,t]*C[i,k])
        # And we want sum_t of these to equal sum_{i,j,k} A[i,j]*B[j,k]*C[i,k] for all A,B,C where C=A@B.
        
        # So let me just compute the full tensor contraction.
        
        a_flat = A_test.flatten()  # length n*m
        b_flat = B_test.flatten()  # length m*p
        c_flat = C_expected.flatten()  # length p*n (C[i,k] at index k*n+i)
        
        # Wait, C_expected is (p, n) so C_expected[k, i] is at index k*n+i in row-major.
        # But we defined W[k*n+i, t], so W is indexed by (k, i) with k first.
        # C_expected[k, i] corresponds to W index k*n+i. Good.
        
        # For each t, compute:
        # prod_t = (u_t · a_flat) * (v_t · b_flat) * (w_t · c_flat)
        # Sum over t should equal sum_{i,j,k} A[i,j]*B[j,k]*C[k,i]
        
    # Let me do the verification differently: reconstruct the multiplication.
    # The decomposition gives: C[i,k] = sum_t (sum_j U[i*m+j,t]*A[i,j]) * (sum_j V[j*p+k,t]*B[j,k]) * W[k*n+i,t]
    # Wait, that's not quite right. The correct formula is:
    # C[i,k] = sum_t W[k*n+i,t] * (sum_j U[i*m+j,t] * A[i,j]) * (sum_j V[j*p+k,t] * B[j,k])
    # No, that's not right either. Let me look at this more carefully.
    
    # The standard tensor decomposition for matrix multiplication:
    # We have tensors U, V, W of shape (n*m, r), (m*p, r), (p*n, r).
    # The multiplication is: C = sum_t (U_t^T A) * (B V_t) * W_t
    # where U_t, V_t, W_t are the t-th columns.
    # More precisely: C[i,k] = sum_t W_t[k*n+i] * (sum_j U_t[i*m+j] * A[i,j]) * (sum_j V_t[j*p+k] * B[j,k])
    
    # Let me verify this formula.
    
    C_recon = np.zeros((p, n))
    for t in range(rank):
        u_t = U[:, t]  # length n*m
        v_t = V[:, t]  # length m*p
        w_t = W[:, t]  # length p*n
        
        # Compute (U_t · A): for each (i,j), this is U_t[i*m+j] * A[i,j]
        # But the formula says: sum_j U_t[i*m+j] * A[i,j] for each i
        # This gives a vector of length n (one per row i)
        
        # And (B · V_t): for each (j,k), this is B[j,k] * V_t[j*p+k]
        # sum_j B[j,k] * V_t[j*p+k] for each k
        # This gives a vector of length p (one per col k)
        
        a_proj = np.zeros(n)
        for i in range(n):
            for j in range(m):
                a_proj[i] += U[i*m+j, t] * A_test[i, j]
        
        b_proj = np.zeros(p)
        for k in range(p):
            for j in range(m):
                b_proj[k] += B_test[j, k] * V[j*p+k, t]
        
        # Now C[i,k] += W[k*n+i, t] * a_proj[i] * b_proj[k]
        for i in range(n):
            for k in range(p):
                C_recon[k, i] += W[k*n+i, t] * a_proj[i] * b_proj[k]
    
    # Check if C_recon matches C_expected
    if not np.allclose(C_recon, C_expected, atol=1e-9):
        # Verification failed, fall back to schoolbook
        rank_school = n * m * p
        U_school = np.zeros((n*m, rank_school), dtype=np.int64)
        V_school = np.zeros((m*p, rank_school), dtype=np.int64)
        W_school = np.zeros((p*n, rank_school), dtype=np.int64)
        col_s = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j, col_s] = 1
                    V_school[j*p+k, col_s] = 1
                    W_school[k*n+i, col_s] = 1
                    col_s += 1
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Convert to lists for the return interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
