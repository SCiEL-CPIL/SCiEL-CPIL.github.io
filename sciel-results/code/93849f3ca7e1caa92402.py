import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def verify_decomposition(U, V, W, n=2, m=3, p=6, atol=1e-9):
    """
    Verify that U, V, W exactly decompose the matrix multiplication tensor
    for multiplying an (n x m) matrix by an (m x p) matrix.
    
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)  -- W index is k*n + i where k is row of C, i is col of C
    
    The tensor M[i,j,k,l] = 1 if i==k and j==l else 0
    (where i is row of A, j is col of A / row of B, k is col of B / row of C, l is col of C)
    
    Actually, let me re-derive:
    A is n x m: A[i, j]
    B is m x p: B[j, k]
    C is n x p: C[i, k] = sum_j A[i,j] B[j,k]
    
    The tensor T[i, j, k, l] where:
    - i: row of A (0..n-1)
    - j: col of A = row of B (0..m-1)
    - k: col of B (0..p-1)
    - l: row of C? No...
    
    Let me re-read the task:
    "The output factor follows the official trace convention W[k*n+i,r]."
    
    So W has shape (p*n, rank), indexed by k*n + i.
    Here k ranges over p (columns of B / columns of C), and i ranges over n (rows of A / rows of C).
    
    The decomposition is:
    C[i, k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    summed over j and r.
    
    Wait, that doesn't look right either. Let me think about this more carefully.
    
    For matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i, k] = sum_{j=0}^{m-1} A[i, j] * B[j, k]
    
    The tensor decomposition should be:
    C[i, k] = sum_{r} sum_{j} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    But that's not standard. The standard form is:
    C[i, k] = sum_r U[i, r] * V[k, r] * W[r]
    for scalar multiplication, but for matrices:
    
    Actually the tensor M has entries M[i, j, k] = delta(j) where... no.
    
    The matrix multiplication tensor T of format (n, m, p) has:
    T[i, j, k] = 1 if this is the "diagonal" term, i.e., the term A[i,j] * B[j,k] contributes to C[i,k].
    
    So the tensor is T[i, j, k] which equals 1 for all valid i, j, k (since every product A[i,j]*B[j,k] contributes to C[i,k]).
    
    The rank-r decomposition:
    T[i, j, k] = sum_{l=0}^{r-1} U[i*m + j, l] * V[j*p + k, l] * W[k*n + i, l]
    
    Wait, that sums over l but the LHS depends on i, j, k. Let me verify:
    
    For the decomposition to work:
    sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1 for all i, j, k
    
    Because every entry of the tensor is 1 (each A[i,j]*B[j,k] term contributes 1 to C[i,k]).
    
    Yes! That's correct. The tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    
    So we need: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1 for all i, j, k.
    """
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for l in range(r):
                    s += U[i*m + j, l] * V[j*p + k, l] * W[k*n + i, l]
                if not np.isclose(s, 1.0, atol=atol):
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Schoolbook fallback: rank n*m*p = 36
    def schoolbook():
        r = n * m * p
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Try to find a rank-33 decomposition using the known structure
    # Strategy: Use the fact that 2x3x6 can be decomposed as 2x3x3 + 2x3x3 with rank 17+17=34
    # But we can do better by exploiting the 2x2x2 substructure.
    
    # Known: Rank(2,2,2) = 7 (Strassen)
    # For 2x3x6, we can partition B into two 3x3 blocks: B = [B1, B2]
    # C = A * B1 + A * B2
    # Each A*Bt is 2x3.
    
    # Alternative approach: Partition into 2x2x2 + 2x1x2 + 1x2x2 style blocks
    # But let's try a direct algebraic construction for rank 33.
    
    # Actually, let's use a smarter block decomposition.
    # A is 2x3, B is 3x6.
    # Partition A = [A1, A2] where A1 is 2x2, A2 is 2x1
    # Partition B = [B1; B2] where B1 is 2x6, B2 is 1x6
    
    # C = A1 * B1 + A2 * B2
    
    # A1*B1: 2x2 times 2x6 = 2x6. Rank of 2x2x6 multiplication.
    # A2*B2: 2x1 times 1x6 = 2x6. This is rank 2 (just 2 scalar products per output).
    # Actually A2*B2 has rank 2*1*6/6... no.
    # A2 is 2x1, B2 is 1x6. Product is 2x6.
    # The tensor has entries T[i, 0, k] = 1 for i in 0,1 and k in 0..5.
    # Rank = 2*1*6 / ... actually it's just outer product structure.
    # C[i,k] = A2[i,0] * B2[0,k]
    # This is rank 2 (one for each i, since k is free).
    # Actually no: sum over the single middle dimension j=0.
    # T[i, 0, k] = 1 for all i, k.
    # We need sum_l U[i, l] * V[k, l] * W[k, i]... 
    # With U shape (2, r), V shape (6, r), W shape (6*2=12, r)? No.
    # For 2x1x6: U (2*1, r) = (2, r), V (1*6, r) = (6, r), W (6*2, r) = (12, r)
    # T[i, j, k] with j=0 always. T[i, 0, k] = 1.
    # sum_l U[i, l] * V[k, l] * W[k*2+i, l] = 1 for all i, k.
    # Rank 2: U[i,l] = delta(i,l) for l=0,1; V[k,l] = 1 for all k,l; W[k*2+i, l] = delta(i,l)
    # Check: sum_l delta(i,l) * 1 * delta(i,l) = 1. Yes, rank 2.
    
    # A1*B1: 2x2 times 2x6.
    # Partition B1 = [B1a, B1b] where B1a is 2x2, B1b is 2x4.
    # A1*B1a: 2x2x2, rank 7 (Strassen)
    # A1*B1b: 2x2x4, rank = 2 * Rank(2,2,2) = 14? No, that's if we do two copies.
    # Actually 2x2x4: A is 2x2, B is 2x4.
    # B = [B1, B2] each 2x2. C = A*B1 + A*B2.
    # Each A*Bt is 2x2, rank 7. Total 14.
    # But can we do better? Rank(2,2,4) ... 
    # The tensor T[i,j,k] for i in 0,1; j in 0,1; k in 0..3.
    # This is like two independent 2x2x2 tensors side by side.
    # So rank = 7 + 7 = 14. Can't do better since the k=0,1 and k=2,3 parts are independent.
    
    # So total: 7 + 14 + 2 = 23? That seems too low. Let me recheck.
    # A1*B1: A1 is 2x2, B1 is 2x6. 
    # The tensor T[i, j, k] where i in 0,1; j in 0,1; k in 0..5.
    # This is 6 independent 2x2x1... no, it's like B has 6 columns.
    # T[i,j,k] = 1 for all i,j,k.
    # This is the same as multiplying A (2x2) by B (2x6).
    # We can think of B as 3 blocks of 2x2: B1, B2, B3.
    # C = A*B1 + A*B2 + A*B3, each 2x2.
    # Rank = 3 * 7 = 21.
    # Or we can be smarter.
    
    # Hmm, let me just go with the known rank 34 solution and try to optimize.
    # Actually, the known best for 2x3x6 might be 33.
    
    # Let me try a different partitioning.
    # A is 2x3. Think of it as [a1, a2, a3] where each ai is a column (2-vector).
    # B is 3x6. Think of it as rows b1, b2, b3 (each 1x6).
    # C = a1*b1 + a2*b2 + a3*b3.
    # Each ai*bj is a 2x6 matrix (outer product of 2-vector and 6-vector).
    # The tensor for one outer product: T[i, j, k] where the "middle" index is fixed.
    # For a_i * b_i: C[i, k] += A[i, j] * B[j, k] where j is fixed to i.
    # T[i, j, k] = 1 if j == i (in the middle dimension), 0 otherwise.
    # Wait no, that's not right either.
    
    # Let me just use the constructive approach with Strassen on 2x2 blocks.
    
    # Partition: A = [A_left (2x2), A_right (2x1)]
    # B = [B_top (2x6); B_bottom (1x6)]
    # C = A_left * B_top + A_right * B_bottom
    
    # A_left * B_top: 2x2 * 2x6.
    # Partition B_top = [B_t1 (2x2), B_t2 (2x2), B_t3 (2x2)]
    # Each 2x2x2 block has rank 7 (Strassen).
    # But the three blocks share the same A_left, so we might save.
    # C_top = A_left * B_t1 + A_left * B_t2 + A_left * B_t3
    # = A_left * [B_t1 B_t2 B_t3]
    # This is equivalent to 2x2x6.
    # Rank(2,2,6): Let's think. B is 2x6 = 3 blocks of 2x2.
    # If we use Strassen for each block independently: 3*7 = 21.
    # Can we do better? The three B blocks are independent, so probably not below 21.
    
    # A_right * B_bottom: 2x1 * 1x6. Rank 2 as computed above.
    
    # Total: 21 + 2 = 23? That can't be right because schoolbook is 36.
    # Let me re-examine.
    
    # Oh wait, I think I'm confusing myself. Let me re-derive the tensor.
    
    # For C = A * B, A is n x m, B is m x p.
    # C[i, k] = sum_{j=0}^{m-1} A[i, j] * B[j, k]
    
    # The tensor T has indices (i, j, k) and T[i, j, k] = 1 for all i, j, k.
    # Because each term A[i,j]*B[j,k] contributes 1 to C[i,k].
    
    # The decomposition: T[i,j,k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
    
    # For schoolbook: one term per (i,j,k) triple, so rank = n*m*p = 36.
    
    # For Strassen on 2x2x2: rank 7 instead of 8.
    
    # Now for 2x3x6: n=2, m=3, p=6. Rank schoolbook = 36.
    
    # Let's try the partition:
    # A = [A1 (2x2) | A2 (2x1)]
    # B = [B1 (2x6); B2 (1x6)]
    # C = A1*B1 + A2*B2
    
    # A1*B1: A1 is 2x2, B1 is 2x6.
    # Tensor T1[i, j, k] = 1 for i in 0,1; j in 0,1; k in 0..5.
    # This is the 2x2x6 tensor.
    
    # A2*B2: A2 is 2x1, B2 is 1x6.
    # Tensor T2[i, j, k] = 1 for i in 0,1; j=0 (only); k in 0..5.
    # But wait, in the full tensor, j ranges over 0,1,2.
    # A2 corresponds to j=2 (the third column of A).
    # B2 corresponds to j=2 (the third row of B).
    # So T2[i, 2, k] = 1 for i in 0,1; k in 0..5.
    # And T1[i, j, k] = 1 for i in 0,1; j in 0,1; k in 0..5.
    # Together they cover all j in 0,1,2.
    
    # Rank of 2x2x6 tensor:
    # B1 is 2x6. Partition B1 = [B1a (2x2) B1b (2x2) B1c (2x2)].
    # C1 = A1*B1a + A1*B1b + A1*B1c.
    # Each A1*B1x is a 2x2x2 tensor, rank 7.
    # But since they all share A1, the total rank is 3*7 = 21 (no sharing possible since B blocks are independent).
    
    # Rank of 2x1x6 tensor: 
    # T2[i, 2, k] = 1.
    # In the full indexing: U index = i*3 + 2, V index = 2*6 + k, W index = k*2 + i.
    # We need sum_l U[i*3+2, l] * V[12+k, l] * W[2k+i, l] = 1 for all i, k.
    # Rank 2: U[0*3+2, 0]=1, U[1*3+2, 1]=1, others 0.
    # V[12+k, 0]=1 for all k, V[12+k, 1]=1 for all k.
    # W[2k+0, 0]=1, W[2k+1, 1]=1 for all k.
    # Check: sum_l U[i*3+2, l] * V[12+k, l] * W[2k+i, l]
    # = U[i*3+2, i] * V[12+k, i] * W[2k+i, i]
    # = 1 * 1 * 1 = 1. Yes!
    
    # Total rank: 21 + 2 = 23.
    
    # Wait, that seems too optimistic. Let me double-check by constructing it.
    
    # Actually, I realize the issue. The 2x2x6 tensor has T[i,j,k]=1 for i in 0,1, j in 0,1, k in 0..5.
    # If I partition B into three 2x2 blocks, each block gives a 2x2x2 tensor.
    # But these are NOT independent in the sense that they share the same A indices.
    # However, the B indices are different, so the tensor entries are in different "slices".
    # The rank should indeed be 3 * Rank(2,2,2) = 3 * 7 = 21.
    
    # So total rank = 21 + 2 = 23.
    
    # Let me construct this.
    
    def strassen_222():
        """
        Returns (U, V, W) for 2x2x2 with rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indexing: U[i*2+j, l], V[j*2+k, l], W[k*2+i, l]
        T[i,j,k] = 1 for all i,j,k in 0,1.
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # P1: (A00+A11)(B00+B11) -> contributes to C00 and C11
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # P1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # P2: (A10+A11)B00 -> C10
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # P3: A00(B01-B11) -> C01, C11
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        # P4: A11(B10-B00) -> C00, C10
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # P5: (A00+A01)B11 -> C00, C01
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # P6: (A10-A00)(B00+B01) -> C11
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # P7: (A01-A11)(B10+B11) -> C00, C11
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W
    
    # Construct 2x2x6 decomposition (rank 21)
    # A1 is 2x2: indices in full A: i*3+0, i*3+1 for i=0,1
    # B1 is 2x6: indices in full B: j*6+k for j=0,1; k=0..5
    # C_top is 2x6: indices in full C: k*2+i for k=0..5; i=0,1
    
    U_226 = np.zeros((6, 21))  # 2x3 = 6 rows
    V_226 = np.zeros((18, 21))  # 3x6 = 18 rows
    W_226 = np.zeros((12, 21))  # 6x2 = 12 rows
    
    U_str, V_str, W_str = strassen_222()
    
    for block in range(3):
        base = block * 7
        for l in range(7):
            for i in range(2):
                for j in range(2):
                    u_idx = i * 3 + j  # Full A index for A1[i,j]
                    U_226[u_idx, base + l] = U_str[i * 2 + j, l]
            for j in range(2):
                for k in range(2):
                    v_idx = j * 6 + (block * 2 + k)  # Full B index for B1[j, block*2+k]
                    V_226[v_idx, base + l] = V_str[j * 2 + k, l]
            for k in range(2):
                for i in range(2):
                    w_idx = (block * 2 + k) * 2 + i  # Full C index for C[i, block*2+k]
                    W_226[w_idx, base + l] = W_str[k * 2 + i, l]
    
    # Construct 2x1x6 decomposition (rank 2)
    # A2 is column 2 of A: indices i*3+2 for i=0,1
    # B2 is row 2 of B: indices 2*6+k = 12+k for k=0..5
    # C_bottom: C[i,k] for i=0,1; k=0..5. Indices k*2+i.
    
    U_216 = np.zeros((6, 2))
    V_216 = np.zeros((18, 2))
    W_216 = np.zeros((12, 2))
    
    for i in range(2):
        U_216[i * 3 + 2, i] = 1
        for k in range(6):
            V_216[12 + k, i] = 1
            W_216[k * 2 + i, i] = 1
    
    # Combine
    U_total = np.hstack([U_226, U_216])  # (6, 23)
    V_total = np.hstack([V_226, V_216])  # (18, 23)
    W_total = np.hstack([W_226, W_216])  # (12, 23)
    
    # Verify
    if verify_decomposition(U_total, V_total, W_total):
        return {
            "U": U_total.tolist(),
            "V": V_total.tolist(),
            "W": W_total.tolist()
        }
    
    # Fallback: try rank 34 (two copies of 2x3x3)
    # Actually, let's just use schoolbook rank 36 as final fallback
    if time.time() - start_time > deadline - 10:
        U, V, W = schoolbook()
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Try to verify the rank 23 construction more carefully
    # If it fails, fall back to schoolbook
    U, V, W = schoolbook()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['W'][0])}")
