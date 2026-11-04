"""Program for matrix multiplication 3x4 x 4x8 with row-block structured decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

def _build_schoolbook(n, m, p):
    """Build schoolbook tensor decomposition for n x m * m x p matrix multiplication."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _build_rowblock(n, m, p):
    """Build row-block decomposition: 1-row block (rank p) + 2-row block (rank 7*p/2).
    
    For n=3, m=4, p=8: rank = 8 + 14 = 22.
    Strategy:
      - Row 0: 1x4 * 4x8, rank = 8 (each output C[0,k] = sum_j A[0,j]*B[j,k])
      - Rows 1-2: 2x4 * 4x8, split m=4 into two 2-dim blocks, use Strassen for each
        2x2x2 block, total rank = 7 + 7 = 14
    """
    if n != 3 or m != 4 or p != 8:
        return None
    
    rank = 22
    U = [[0.0] * rank for _ in range(n * m)]  # 12 x 22
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 22
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 22
    
    # --- Block 1: Row 0 (1x4 * 4x8), rank 8 ---
    # For each k in [0,8): C[0,k] = sum_j A[0,j] * B[j,k]
    # u[0,t] = A[0,t] (t = j index 0..3)
    # v[k,t] = B[t,k]
    # W[k*n+0, t] = 1 for t in 0..7
    for k in range(8):
        for j in range(4):
            t = k  # No, we need t to index j, not k
            pass
    # Correct: for each k, we need 4 terms (one per j), but we can share the U/V structure
    # Actually: u[i,t] = sum_j U[i*m+j, t] * A[i,j]
    # For i=0: u[0,t] = sum_j U[0*m+j, t] * A[0,j] = sum_j U[j, t] * A[0,j]
    # We want u[0,t] * v[k,t] * W[k*n+0, t] to sum to C[0,k] = sum_j A[0,j]*B[j,k]
    # 
    # Simple approach: use 8 terms, one per (j,k) pair? No, that's rank 32 for this block.
    # Better: for each k, use 4 terms where u[0,t] = A[0,j_t], v[k,t] = B[j_t, k]
    # But t must be shared across all k... 
    # 
    # Actually the standard way: for 1x4*4x8, the bilinear complexity is 4*8=32? No!
    # The rank of the 1x4x8 tensor T[j,k] = 1 (for all j,k) is... 
    # T is a 4x8 matrix of ones. Its rank as a matrix is 1, but as a bilinear map...
    # The bilinear map is (a, B) -> a * B (row vector times matrix).
    # This has rank = 4*8 = 32? No, it's rank = 8.
    # 
    # Let me think again. C[k] = sum_j a[j] * B[j,k]
    # = (a, B) -> sum_j a[j] * B[j,k]
    # For fixed k, this is a linear functional of a and B.
    # The tensor is T[j,k] = 1 for all j in [0,4), k in [0,8).
    # This is a 4x8 matrix of all ones. As a 3-tensor (with the output index being k),
    # it's T_{j,k} where j is the "A" index and k is the "B/C" index.
    # The bilinear complexity is the rank of this tensor.
    # 
    # T_{j,k} = 1 = sum_{t=0}^{7} delta_{j, t%4} * delta_{k, t//4}
    # So rank = 8. Each term: u[j] = delta_{j, t%4}, v[k] = delta_{k, t//4}, W[k] = 1
    # 
    # In our formulation:
    # u[0,t] = sum_j U[j, t] * A[0,j] = A[0, t%4]  (since U[j,t] = delta_{j, t%4})
    # v[k,t] = sum_j V[j*p+k, t] * B[j,k] = B[t%4, k] * delta_{k, t//4}
    #   Wait, V[j*p+k, t] should give v[k,t] = B[t%4, k] only when k = t//4
    #   So V[j*p+k, t] = delta_{j, t%4} * delta_{k, t//4}
    # W[k*n+0, t] = 1 (for all k, t)
    # 
    # Then C[0,k] = sum_t W[k*n+0, t] * u[0,t] * v[k,t]
    #   = sum_t 1 * A[0, t%4] * B[t%4, k] * delta_{k, t//4}
    #   = sum_{t: t//4=k} A[0, t%4] * B[t%4, k]
    #   = sum_{j=0}^{3} A[0, j] * B[j, k]  (since t = k*4 + j for j in 0..3)
    # 
    # Wait, t ranges 0..7, t//4 gives 0 or 1, not 0..7. That's wrong.
    # 
    # Let me redo: we need 8 terms. For term t (0..7):
    #   Let j_t = t (0..3 for t<4, 4..7 for t>=4)? No, j must be in 0..3.
    #   Actually: for k in 0..7, we need C[0,k] = sum_{j=0}^{3} A[0,j]*B[j,k]
    #   This is 8 separate linear combinations. The tensor T_{j,k} = 1 (4x8) has rank 8
    #   because it can be written as sum_{k=0}^{7} e_k (output) * (sum_j e_j (A-index)) * e_k (B-index)
    #   Hmm, that's not quite right either.
    #
    # Let me use the standard approach: the 1xm*mxp bilinear map has rank m*p.
    # For each output k and each input j, we need a term. So rank = m*p = 32? 
    # No! The rank of the 1xm*mxp multiplication tensor is m*p.
    # 
    # Actually, I recall that the bilinear complexity of 1xm*mxp is m*p.
    # Proof: C[k] = sum_j a_j * B[j,k]. The tensor is T_{j,k} = 1 for all j,k.
    # As a matrix (j x k), this is a 4x8 matrix of ones, which has matrix rank 1.
    # But the bilinear rank is different. The bilinear rank of T is the minimum r such that
    # T_{j,k} = sum_{t=1}^{r} u_t[j] * v_t[k] * w_t
    # where w_t are scalars. This is just the matrix rank of T! So rank = 1?
    # No, that's for the case where w_t can be absorbed. Actually:
    # T_{j,k} = sum_t u_t[j] * v_t[k]
    # This is the matrix rank of the 4x8 matrix of ones, which is 1.
    # 
    # But wait, in our formulation, W[k*n+0, t] multiplies u[0,t]*v[k,t].
    # So T_{j,k} = sum_t U[j,t] * V[j*p+k,t] * W[k*n+0,t]
    # For the 1-row case, this becomes:
    # 1 = sum_t U[j,t] * V[j*p+k,t] * W[k,t]  (for all j,k)
    # 
    # If we set W[k,t] = 1 for all k,t, then we need:
    # 1 = sum_t U[j,t] * V[j*p+k,t]
    # This is a 4x8 matrix of ones = U (4xr) * V_reshaped (r x 8) where V_reshaped[k,t] = V[j*p+k,t]... 
    # No, V depends on j too. This is more complex.
    #
    # Let me just use the straightforward rank-32 decomposition for the 1-row block:
    # For each (j,k) pair, one term. That's 4*8=32 terms.
    # U[j, t] = 1 where t = j*8+k
    # V[j*p+k, t] = 1 where t = j*8+k  
    # W[k*n+0, t] = 1 where t = j*8+k
    # But that's rank 32 for just the first row, total 32+14=46. Still better than 96.
    #
    # Actually, I think the 1x4*4x8 has rank 8, not 32. Let me verify:
    # C[k] = sum_j a_j * B[j,k]
    # Term t (t=0..7): let t = 4*k_t + j_t where k_t = t//4, j_t = t%4
    # u[0,t] = a_{j_t}  => U[j, t] = delta_{j, j_t} = delta_{j, t%4}
    # v[k,t] = B[j_t, k] * delta_{k, k_t} => V[j*p+k, t] = delta_{j, j_t} * delta_{k, k_t}
    #   = delta_{j, t%4} * delta_{k, t//4}
    # W[k*n+0, t] = 1
    # 
    # Check: sum_t W[k*n+0,t] * u[0,t] * v[k,t]
    #   = sum_t 1 * a_{t%4} * B[t%4, k] * delta_{k, t//4}
    #   = sum_{t: t//4=k} a_{t%4} * B[t%4, k]
    #   = sum_{j: 4k+j < 8 and t=4k+j} ... 
    #   For k=0: t in {0,1,2,3}, t//4=0=k, t%4 = t = 0,1,2,3
    #     sum = a_0*B[0,0] + a_1*B[1,0] + a_2*B[2,0] + a_3*B[3,0] = C[0,0] ✓
    #   For k=1: t in {4,5,6,7}, t//4=1=k, t%4 = 0,1,2,3
    #     sum = a_0*B[0,1] + a_1*B[1,1] + a_2*B[2,1] + a_3*B[3,1] = C[0,1] ✓
    #   ...
    #   For k=7: t in {28,29,30,31}? But t only goes to 7!
    #
    # Ah, I see the issue. t ranges 0..7, so t//4 is only 0 or 1. This only works for k=0,1.
    # I need t to range 0..31 for all 8 values of k. So rank = 32 for the 1-row block.
    #
    # Wait no. Let me reconsider. For k=2, I need t such that t//4=2, i.e., t in {8,9,10,11}.
    # So t ranges 0..31, giving rank 32. That's the same as schoolbook for this block.
    #
    # Hmm, but I thought the bilinear complexity of 1xm*mxp was m*p. Let me check:
    # For 1x2*2x2: C[0] = a_0*B[0,0]+a_1*B[1,0], C[1] = a_0*B[0,1]+a_1*B[1,1]
    # Rank = 4 (each output needs 2 terms, 2 outputs). m*p = 2*2 = 4. ✓
    # For 1x2*2x3: C[0] = a_0*B[0,0]+a_1*B[1,0], C[1] = a_0*B[0,1]+a_1*B[1,1], C[2] = a_0*B[0,2]+a_1*B[1,2]
    # Rank = 6 = m*p. ✓
    # So yes, 1x4*4x8 has rank 32. My earlier claim of rank 8 was wrong.
    #
    # So the row-block approach gives: 32 + 14 = 46. Still better than 96!
    #
    # Let me implement rank 46.
    
    rank = 46
    U = [[0.0] * rank for _ in range(n * m)]  # 12 x 46
    V = [[0.0] * rank for _ in range(m * p)]  # 32 x 46
    W = [[0.0] * rank for _ in range(p * n)]  # 24 x 46
    
    # --- Block 1: Row 0 (1x4 * 4x8), rank 32 ---
    # For t = 0..31: j_t = t % 4, k_t = t // 4
    for t in range(32):
        j_t = t % 4
        k_t = t // 4
        U[j_t][t] = 1.0  # U[i*m+j, t] with i=0, so U[0*4+j_t, t] = U[j_t, t]
        V[j_t * p + k_t][t] = 1.0  # V[j*p+k, t]
        W[k_t * n + 0][t] = 1.0  # W[k*n+i, t] with i=0
    
    # --- Block 2: Rows 1-2 (2x4 * 4x8), rank 14 ---
    # Split m=4 into two 2-dim blocks: j in {0,1} and j in {2,3}
    # For each block, use Strassen 2x2x2 with p=4 (each output k in 0..3)
    # Wait, Strassen is for 2x2*2x2. Here we have 2x2*2x4 for each block.
    # 
    # For the first block (j=0,1): C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=1,2, k=0..3
    # This is 2x2*2x4, rank = 2*2*4 = 16? Or can we do better?
    # The tensor is T_{i,j,k} = 1 for i in {0,1}, j in {0,1}, k in {0,3}.
    # As a 2x2x4 tensor, its rank is... For fixed i, it's a 2x4 matrix of ones (rank 1).
    # The full tensor rank: T_{i,j,k} = sum_t u_t[i,j] * v_t[j,k] * w_t[i,k]
    # This is the bilinear complexity of 2x2*2x4, which is 2*2*4 = 16.
    #
    # Hmm, so each 2x2*2x4 block has rank 16, total 32 for both blocks.
    # Total: 32 + 16 + 16 = 64. Still better than 96.
    #
    # But the directive says to use Strassen for the 2x4*4x8 block, giving rank 14.
    # Let me re-read: "the 2x4 by 4x8 product via a rank-14 decomposition (applying Strassen to the 2x2 block structure of the inner dimension)"
    #
    # I think the idea is: 2x4*4x8 = 2x(2+2)*(2+2)x8 = 2x2x2x2 + 2x2x2x2 + 2x2x2x2 + 2x2x2x2
    # No, that's 4 blocks of 2x2*2x2, each with Strassen rank 7, total 28.
    #
    # Or: 2x4*4x8 = (2x2*2x8) + (2x2*2x8) = two 2x2*2x8 blocks.
    # Each 2x2*2x8 has rank 2*2*8 = 32? Or can Strassen help?
    #
    # I think the intended decomposition is:
    # 2x4*4x8: split the 4-dim middle as (2,2) and the 8-dim output as (2,2,2,2)
    # Then we have 2*4 = 8 blocks of 2x2*2x2, each with Strassen rank 7.
    # Total: 8*7 = 56. That's worse.
    #
    # Or: split only the middle: 2x4*4x8 = 2x(2,2)*(2,2)x8
    # = (2x2*2x8) + (2x2*2x8)
    # Each 2x2*2x8: rank = 2*2*8 = 32 (schoolbook) or can we do better?
    # The tensor T_{i,j,k} for 2x2*2x8 is 2x2x8 all ones.
    # Rank = 2*2*8 = 32? Let me check: for 2x2*2x2, rank is 7 (Strassen).
    # For 2x2*2x8, I believe rank = 7*4 = 28 (apply Strassen to each 2x2*2x2 subblock).
    # So total for 2x4*4x8 = 28 + 28 = 56.
    #
    # I'm confused by the "rank 14" claim. Let me just implement the straightforward
    # decomposition and see what rank we get.
    #
    # Let me try a different approach: just build the full decomposition directly.
    # Total rank for 3x4*4x8 with the block structure:
    # - Row 0: 1x4*4x8, rank 32
    # - Rows 1-2: 2x4*4x8, rank 64 (schoolbook) or 56 (Strassen per 2x2x2 block)
    # Total: 32+64=96 (same as schoolbook) or 32+56=88.
    #
    # This isn't helping much. Let me just implement a clean decomposition with rank 46
    # (32 for row 0 + 14 for rows 1-2 using a specific Strassen-like structure).
    #
    # Actually, let me just implement the schoolbook (rank 96) as fallback and try to
    # find a better one. For now, let me implement rank 46 with the structure described.
    
    # Reset and use rank 46
    rank = 46
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Block 1: Row 0, rank 32 (terms 0-31)
    for t in range(32):
        j_t = t % 4
        k_t = t // 4
        U[0 * m + j_t][t] = 1.0
        V[j_t * p + k_t][t] = 1.0
        W[k_t * n + 0][t] = 1.0
    
    # Block 2: Rows 1-2, rank 14 (terms 32-45)
    # 2x4*4x8: split m=4 into (j0,j1) and (j2,j3)
    # For the first 2-dim block (j=0,1): 2x2*2x8
    # Use 7 Strassen terms for each of 4 groups of 2 outputs
    # Actually, let me use a simpler structure: 2x4*4x8 with rank 14
    # 
    # I'll use the following: for 2x2*2x2, Strassen has 7 terms.
    # 2x4*4x8 = 4 blocks of 2x2*2x2 (splitting both m and p into 2s)
    # But that gives 4*7=28.
    #
    # For rank 14, I need a different structure. Let me use:
    # 2x4*4x8: treat as (2x4) * (4x8)
    # Split A into [A1; A2] where A1, A2 are 2x2
    # Split B into [B1, B2] where B1, B2 are 2x4
    # C = A1*B1 + A2*B2
    # Each 2x2*2x4 has rank 2*2*4 = 16 (schoolbook)
    # Total: 32. Not 14.
    #
    # I think rank 14 for 2x4*4x8 is not achievable with simple Strassen.
    # Let me just use rank 64 for the 2x4*4x8 block (schoolbook).
    # Total: 32 + 64 = 96. Same as schoolbook.
    #
    # Let me try yet another approach: use the full tensor structure.
    # The 3x4x8 multiplication tensor T_{i,j,k} = 1 for all valid i,j,k.
    # This is a 3x4x8 tensor. Its rank is at most 96 (schoolbook).
    # 
    # For now, let me just return the schoolbook decomposition.
    return None

def _verify_decomposition(U, V, W, n, m, p, num_tests=20):
    """Verify that the decomposition correctly computes C = A @ B."""
    random.seed(42)
    for _ in range(num_tests):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        true_C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        rank = len(U[0]) if U else 0
        u = [[0] * rank for _ in range(n)]
        for i in range(n):
            for t in range(rank):
                s = 0
                for j in range(m):
                    s += U[i * m + j][t] * A[i][j]
                u[i][t] = s
        
        v = [[0] * rank for _ in range(p)]
        for k in range(p):
            for t in range(rank):
                s = 0
                for j in range(m):
                    s += V[j * p + k][t] * B[j][k]
                v[k][t] = s
        
        for i in range(n):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += W[k * n + i][t] * u[i][t] * v[k][t]
                if s != true_C[i][k]:
                    return False
    return True


def _verify_numpy(U, V, W, n, m, p, num_tests=50):
    """Faster numpy-based verification."""
    np.random.seed(42)
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    for _ in range(num_tests):
        A = np.random.randint(-10, 10, size=(n, m)).astype(float)
        B = np.random.randint(-10, 10, size=(m, p)).astype(float)
        
        U_reshaped = U_np.reshape(n, m, rank)
        u = np.einsum('imt,ij->it', U_reshaped, A)
        
        V_reshaped = V_np.reshape(m, p, rank)
        v = np.einsum('jkt,jk->kt', V_reshaped, B)
        
        W_reshaped = W_np.reshape(p, n, rank)
        C = np.einsum('kit,it,kt->ik', W_reshaped, u, v)
        
        true_C = A @ B
        if not np.allclose(C, true_C, atol=1e-10):
            return False
    return True



def _build_kronecker(n, m, p):
    """Build decomposition using Kronecker product structure.
    
    For n=3, m=4, p=8, we can write:
    - Split m=4 into two blocks of 2
    - Split p=8 into four blocks of 2
    - Use 2x2 Strassen-like structure
    
    This gives rank = n * (m/2) * (p/2) * 7 = 3 * 2 * 4 * 7 = 168 (worse)
    Or simpler: n * m * p = 96 (same as schoolbook)
    
    Actually, let's use a different approach: treat the 4-dim middle as 
    two independent 2-dim spaces and use rank-7 for each 2x2x2 block.
    """
    # For 3x4*4x8: split m=4 into m1=2, m2=2; p=8 into p1=2,p2=2,p3=2,p4=2
    # Each 3x2 * 2x2 subproblem has rank 3*2*2=12 schoolbook, or we can use structure
    # Total rank with independent blocks: 3 * (2*2) * 4 = 48? No...
    
    # Let me think differently. The tensor is T_{i,j,k} where i in [0,3), j in [0,4), k in [0,8)
    # Schoolbook: T = sum_{j} e_{i,j} ⊗ e_{j,k} ⊗ e_{i,k} with rank 3*4*8=96
    
    # Try: decompose j into (j1, j2) where j1 in [0,2), j2 in [0,2)
    # Then T_{i,(j1,j2),k} = sum_{j1,j2} ...
    # This doesn't directly help unless we use low-rank structure in j2
    
    # Alternative: use the fact that for fixed i, the map A_i (row i of A, length 4) 
    # times B (4x8) gives row i of C.
    # rank_3 = min rank of {A_i * B : A_i in R^4, B in R^{4x8}}
    # This is the rank of the 4x4x8 tensor as a 4x(4*8) matrix... 
    # Actually it's the bilinear complexity of 1x4*4x8 which is 32.
    # Total = 3 * 32 = 96. Same as schoolbook.
    
    # So Kronecker doesn't help here. Return schoolbook.
    return _build_schoolbook(n, m, p)


def _build_strassen_2x2x2():
    """Return the Strassen 2x2x2 decomposition as (U, V, W) lists.
    
    U: shape (2*2, 7) = (4, 7)
    V: shape (2*2, 7) = (4, 7)  
    W: shape (2*2, 7) = (4, 7)
    
    Indexing: U[i*2+j, t], V[i*2+j, t], W[i*2+j, t]
    where i,j in {0,1} for the 2x2 matrices.
    """
    rank = 7
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(4)]
    W = [[0.0] * rank for _ in range(4)]
    
    # Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # U[i*2+j, t] = coefficient of A[i,j] in M_t
    # V[i*2+j, t] = coefficient of B[i,j] in M_t
    # W[i*2+j, t] = coefficient of M_t in C[i,j]
    
    # M1: A00+A11 times B00+B11
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[3][0] = 1  # B00, B11
    W[0][0] = 1; W[1][0] = 1; W[2][0] = 0; W[3][0] = 1  # C00+=M1, C01+=0, C10+=0, C11+=M1
    # Wait, W is indexed by [i*2+j] for C[i,j]. Let me redo.
    # W[i*2+j, t] = coefficient of M_t in C[i,j]
    
    # Let me restart with clear indexing.
    # U row index: i*2+j where i,j in {0,1} for A[i,j]
    # V row index: i*2+j where i,j in {0,1} for B[i,j]  
    # W row index: i*2+j where i,j in {0,1} for C[i,j]
    
    # M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1 => U[0]=1, U[3]=1
    # V: B00=1, B11=1 => V[0]=1, V[3]=1
    # C00 += M1, C11 += M1 => W[0]+=1, W[3]+=1
    
    # M2 = (A10+A11)B00
    # U: A10=1, A11=1 => U[2]=1, U[3]=1
    # V: B00=1 => V[0]=1
    # C10 += M2 => W[2]+=1
    
    # M3 = A00(B10-B11)
    # U: A00=1 => U[0]=1
    # V: B10=1, B11=-1 => V[2]=1, V[3]=-1
    # C01 += M3, C11 += M3 => W[1]+=1, W[3]+=1
    
    # M4 = A11(B01-B10)
    # U: A11=1 => U[3]=1
    # V: B01=1, B10=-1 => V[1]=1, V[2]=-1
    # C00 += M4, C10 += M4 => W[0]+=1, W[2]+=1
    
    # M5 = (A00+A01)B11
    # U: A00=1, A01=1 => U[0]=1, U[1]=1
    # V: B11=1 => V[3]=1
    # C00 -= M5, C01 += M5 => W[0]-=1, W[1]+=1
    
    # M6 = (A10-A00)(B00+B01)
    # U: A10=1, A00=-1 => U[2]=1, U[0]=-1
    # V: B00=1, B01=1 => V[0]=1, V[1]=1
    # C11 += M6 => W[3]+=1
    
    # M7 = (A01-A11)(B10+B11)
    # U: A01=1, A11=-1 => U[1]=1, U[3]=-1
    # V: B10=1, B11=1 => V[2]=1, V[3]=1
    # C00 += M7 => W[0]+=1
    
    # Now accumulate:
    # W[0] (C00): M1+1, M4+1, M5-1, M7+1 => [1,0,0,1,-1,0,1]
    # W[1] (C01): M3+1, M5+1 => [0,0,1,0,1,0,0]
    # W[2] (C10): M2+1, M4+1 => [0,1,0,1,0,0,0]
    # W[3] (C11): M1+1, M3+1, M6+1 => [1,0,1,0,0,1,0]
    
    U = [[0.0]*7 for _ in range(4)]
    V = [[0.0]*7 for _ in range(4)]
    W = [[0.0]*7 for _ in range(4)]
    
    # M1
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    # M2
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1
    # M3
    U[0][2] += 1
    V[2][2] += 1; V[3][2] += -1
    W[1][2] += 1; W[3][2] += 1
    # M4
    U[3][3] += 1
    V[1][3] += 1; V[2][3] += -1
    W[0][3] += 1; W[2][3] += 1
    # M5
    U[0][4] += 1; U[1][4] += 1
    V[3][4] += 1
    W[0][4] += -1; W[1][4] += 1
    # M6
    U[2][5] += 1; U[0][5] += -1
    V[0][5] += 1; V[1][5] += 1
    W[3][5] += 1
    # M7
    U[1][6] += 1; U[3][6] += -1
    V[2][6] += 1; V[3][6] += 1
    W[0][6] += 1
    
    return U, V, W

def _random_search(n, m, p, target_rank, time_budget, seed=42):
    """Search for a decomposition of given rank using random initialization + refinement.
    
    Uses the tensor network structure:
    C[i,k] = sum_t W[k*n+i, t] * (sum_j U[i*m+j, t] * A[i,j]) * (sum_j V[j*p+k, t] * B[j,k])
    
    We can view this as: for each t, we have a rank-1 "slice" in the bilinear map.
    """
    rng = np.random.RandomState(seed)
    best_result = None
    best_error = float('inf')
    
    start = time.time()
    
    for attempt in range(50):
        if time.time() - start > time_budget:
            break
        
        # Initialize randomly
        U = rng.randn(n * m, target_rank)
        V = rng.randn(m * p, target_rank)
        W = rng.randn(p * n, target_rank)
        
        # Scale to reasonable magnitudes
        U /= np.sqrt(target_rank)
        V /= np.sqrt(target_rank)
        W /= np.sqrt(target_rank)
        
        # Iterate: fix two, solve for third
        for it in range(200):
            if time.time() - start > time_budget:
                break
            
            # Solve for W given U, V
            A_rand = rng.randn(n, m)
            B_rand = rng.randn(m, p)
            C_true = A_rand @ B_rand
            
            U_reshaped = U.reshape(n, m, target_rank)
            u = np.einsum('imt,ij->it', U_reshaped, A_rand)
            
            V_reshaped = V.reshape(m, p, target_rank)
            v = np.einsum('jkt,jk->kt', V_reshaped, B_rand)
            
            rhs = np.zeros((n * p, target_rank))
            for i in range(n):
                for k in range(p):
                    rhs[k * n + i, :] = u[i, :] * v[k, :]
            
            C_flat = C_true.flatten()
            
            try:
                W_flat, residuals, rank_ls, sv = np.linalg.lstsq(rhs, C_flat, rcond=None)
                W = W_flat.reshape(p, n, target_rank).transpose(1, 0, 2).reshape(p * n, target_rank)
            except:
                continue
            
            # Check error
            err = 0
            for _ in range(5):
                A_test = rng.randn(n, m)
                B_test = rng.randn(m, p)
                U_r = U.reshape(n, m, target_rank)
                u_test = np.einsum('imt,ij->it', U_r, A_test)
                V_r = V.reshape(m, p, target_rank)
                v_test = np.einsum('jkt,jk->kt', V_r, B_test)
                W_r = W.reshape(p, n, target_rank)
                C_test = np.einsum('kit,it,kt->ik', W_r, u_test, v_test)
                err = max(err, np.abs(C_test - A_test @ B_test).max())
            
            if err < best_error:
                best_error = err
                if err < 1e-10:
                    best_result = (U.tolist(), V.tolist(), W.tolist())
                    return best_result
        
        if best_result is not None:
            break
    
    return best_result

def construct():
    """Construct a rank-minimal (or near-minimal) exact tensor decomposition for 3x4 * 4x8.
    
    Uses a row-block decomposition:
    - Row 0: 1x4 * 4x8, rank 32
    - Rows 1-2: 2x4 * 4x8, rank 64 (schoolbook) or lower with Strassen
    Total: 96 (same as schoolbook) or potentially lower.
    
    For now, returns the schoolbook decomposition as a verified baseline.
    """
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=8
    
    # Build schoolbook decomposition (rank = 96) as fallback
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    
    # Try to find lower rank decomposition
    # Try ranks from low to high
    for target_rank in range(20, 96):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time() - 5
        if remaining < 1:
            break
        
        result = _random_search(n, m, p, target_rank, remaining, seed=target_rank)
        
        if result is not None:
            U, V, W = result
            # Verify with numpy
            if _verify_numpy(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    if not _verify_decomposition(U_school, V_school, W_school, n, m, p):
        raise RuntimeError("Schoolbook decomposition failed verification")
    
    return {"U": U_school, "V": V_school, "W": W_school}
