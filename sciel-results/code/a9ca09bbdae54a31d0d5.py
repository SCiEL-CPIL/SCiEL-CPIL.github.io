import numpy as np
import time

def _verify_fast(U, V, W, n=2, m=4, p=8, trials=50, tol=1e-9):
    """
    Fast vectorized verification.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = U.shape[1]
    rng = np.random.RandomState(42)
    
    U_reshaped = U.reshape(n, m, rank)  # U[i, j, c]
    V_reshaped = V.reshape(m, p, rank)  # V[l, k, c]
    W_reshaped = W.reshape(p, n, rank)  # W[k, i, c]
    
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        
        factor_A = np.einsum('ij,ijc->ic', A, U_reshaped)
        factor_B = np.einsum('lk,lkc->kc', B, V_reshaped)
        
        C_computed = np.einsum('kic,ic,kc->ik', W_reshaped, factor_A, factor_B)
        
        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication with rank 7.
    Dimensions: n=2, m=2, p=2.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices:
    U row: i*m + j (0..3) -> i in 0..1, j in 0..1
    V row: l*p + k (0..3) -> l in 0..1, k in 0..1
    W row: k*n + i (0..3) -> k in 0..1, i in 0..1
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: A11+A22, B11+B22 -> C11 += 1, C22 += 1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: A21+A22, B11 -> C21 += 1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3: A11, B21-B22 -> C12 += 1, C22 += 1
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4: A22, B12-B11 -> C11 += 1, C21 += 1
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5: A11+A12, B22 -> C11 -= 1, C12 += 1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6: A21-A11, B11+B12 -> C22 += 1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7: A12-A22, B21+B22 -> C11 += 1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 2, 4, 8
    
    # Strategy: Recursive Strassen on the 4x8 part of the multiplication.
    # We view the multiplication A (2x4) * B (4x8) as:
    # Split B into 4 blocks of 4x2: B0, B1, B2, B3
    # Split the middle dimension (4) into two 2s: B_top (2x8), B_bot (2x8)
    # C = A_top * B_top + A_bot * B_bot
    # where A_top = A[:, 0:2] (2x2), A_bot = A[:, 2:4] (2x2)
    # B_top = B[0:2, :] (2x8), B_bot = B[2:4, :] (2x8)
    
    # This is sum of two 2x2x8 multiplications.
    # R(2x2x8) can be done recursively.
    # Split B (2x8) into two 2x4 blocks: B_top0, B_top1
    # R(2x2x4) recursively.
    # Split B (2x4) into two 2x2 blocks: B_top00, B_top01
    # R(2x2x2) is rank 7.
    
    # So R(2x2x8) = 2 * R(2x2x4) = 4 * R(2x2x2) = 4 * 7 = 28.
    # Total Rank = R(A_top * B_top) + R(A_bot * B_bot) = 28 + 28 = 56.
    
    # Wait, the previous program also achieved 56.
    # Can we do better?
    # The previous program did:
    # 4 sub-problems of 2x4x2.
    # R(2x4x2) = R(2x2x2) + R(2x2x2) = 14.
    # Total = 4 * 14 = 56.
    
    # Is there a way to mix them?
    # The tensor is T_{i, j, k, l} = delta_{j,l}.
    # This is a direct sum of 4 tensors T^{(l)}_{i, j, k} = delta_{j,l} for l=0..3? No.
    # The indices are i (0..1), j (0..3), k (0..7), l (0..3).
    # T_{i, j, k, l} is non-zero only if j=l.
    # So T = sum_{l=0}^3 T_l where T_l has support on j=l, and depends on i, k.
    # For a fixed l, T_l is a 2 x 1 x 8 x 1 tensor? No.
    # T_l(i, j, k, l') = delta_{j,l} delta_{l',l} * 1.
    # This is a rank 1 tensor? No, it's a product of deltas.
    # Actually, for fixed l, the tensor is T_l(i, j, k) = delta_{j,l}.
    # This is a 2 x 4 x 8 tensor with only 16 non-zero entries (2 i's, 8 k's, 1 j=l).
    # Wait, if j is fixed to l, then for each i, k, the value is 1.
    # So T_l is a matrix of ones (2x8) repeated?
    # T_l(i, j, k) = 1 if j=l else 0.
    # This tensor T_l can be written as u_l v_l^T w_l^T?
    # Let u_l be a vector in R^{2*4} with 1 at position i*4+l? No.
    # The tensor is T_{i, j, k}.
    # We can write T_l(i, j, k) = (e_i \otimes e_l) \cdot (e_l \otimes e_k)? No.
    # Let's look at the structure.
    # T_l is non-zero only when j=l.
    # The entries are T_l(i, l, k) = 1 for all i, k.
    # This is a rank 1 tensor in the (i, k) plane for each fixed l?
    # No, it's a matrix of ones of size 2x8.
    # The rank of a matrix of ones 2x8 is 1.
    # So T_l can be decomposed into 1 term?
    # Let's check.
    # Term: U_l(i, j) V_l(j, k) W_l(k, i)?
    # We need sum_c U(i,j,c) V(j,k,c) W(k,i,c) = T(i,j,k).
    # If we pick c=l, and set U(i,l,l)=1, V(l,k,l)=1, W(k,i,l)=1?
    # Then sum is 1*1*1 = 1 for j=l.
    # And 0 for j!=l.
    # So T_l has rank 1?
    # If so, total rank is 4 (one for each l).
    # Let's verify this hypothesis.
    
    # Hypothesis: Rank is 4.
    # Construct:
    # For each l in 0..3:
    #   U[:, l] is a vector of size n*m = 8.
    #   U[i*m + j, l] = 1 if j==l else 0.
    #   V[:, l] is a vector of size m*p = 32.
    #   V[l*p + k, l] = 1 for all k.
    #   W[:, l] is a vector of size p*n = 16.
    #   W[k*n + i, l] = 1 for all k, i.
    
    # Let's check the sum:
    # C_{i, k} = sum_c U_{i, j, c} V_{j, k, c} W_{k, i, c} ?
    # Wait, the indices in the problem are:
    # U: (n*m, rank) -> U[i, j, c]
    # V: (m*p, rank) -> V[j, k, c]  (Note: V row index is j*p + k)
    # W: (p*n, rank) -> W[k, i, c]  (Note: W row index is k*n + i)
    
    # My proposed decomposition:
    # For c=l:
    # U[i, j, l] = 1 if j==l else 0.
    # V[j, k, l] = 1 if j==l else 0?
    # Wait, V is indexed by (j, k).
    # If V[j, k, l] = 1 for all k when j==l, and 0 otherwise.
    # W[k, i, l] = 1 for all k, i.
    
    # Then C_{i, k} = sum_l U[i, l, l] V[l, k, l] W[k, i, l]
    # U[i, l, l] = 1 (since j=l).
    # V[l, k, l] = 1.
    # W[k, i, l] = 1.
    # So C_{i, k} = sum_l 1 = 4?
    # But we want C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # This decomposition computes a constant 4 for all i, k?
    # No, it ignores A and B.
    # The tensor decomposition is for the *multiplication map*.
    # The tensor T_{i, j, k, l} represents the coefficient of A_{i, j} B_{j, k} in C_{i, k}.
    # Wait, the standard tensor for matrix multiplication C = A B is:
    # T_{i, j, k} = delta_{j, l}? No.
    # The tensor is T_{i, j, k, l} where C_{i, k} = sum_{j, l} T_{i, j, k, l} A_{i, j} B_{l, k}?
    # No, standard is C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # So the tensor is T_{i, j, k, j} = 1?
    # The tensor is T_{i, j, k, l} = delta_{j, l}.
    # And the decomposition is C_{i, k} = sum_c (sum_{i, j} U_{i, j, c} A_{i, j}) (sum_{l, k} V_{l, k, c} B_{l, k}) W_{k, i, c}?
    # No, the factors are:
    # f_c(A) = sum_{i, j} U_{i, j, c} A_{i, j}
    # g_c(B) = sum_{l, k} V_{l, k, c} B_{l, k}
    # h_c(C) = sum_{k, i} W_{k, i, c} C_{k, i} ? No.
    
    # The decomposition is:
    # T = sum_c U_c \otimes V_c \otimes W_c
    # where U_c is in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}.
    # And T_{i, j, l, k} = sum_c U_c[i, j] V_c[l, k] W_c[k, i].
    # Note the indices of W are (k, i) corresponding to C_{i, k}.
    # So W_c is a matrix W_c[k, i].
    
    # We want T_{i, j, l, k} = delta_{j, l}.
    # So we need sum_c U_c[i, j] V_c[l, k] W_c[k, i] = delta_{j, l}.
    
    # Let's try the rank 4 hypothesis again with correct indices.
    # For c = l (0..3):
    # U_c[i, j] = 1 if j == c else 0.
    # V_c[l, k] = 1 if l == c else 0.
    # W_c[k, i] = 1 for all k, i.
    
    # Sum over c:
    # Term c: U_c[i, j] V_c[l, k] W_c[k, i] = (1 if j=c) * (1 if l=c) * 1.
    # This is 1 if j=c and l=c, i.e., j=l=c.
    # Summing over c, we get 1 if j=l (since c must be j=l), 0 otherwise.
    # This matches delta_{j, l}!
    
    # So the rank is 4!
    
    # Let's construct it.
    rank = 4
    U_total = np.zeros((n*m, rank))
    V_total = np.zeros((m*p, rank))
    W_total = np.zeros((p*n, rank))
    
    for c in range(rank):
        # c corresponds to the middle index j/l
        # U_c[i, j] = 1 if j == c else 0
        for i in range(n):
            for j in range(m):
                if j == c:
                    U_total[i*m + j, c] = 1
        
        # V_c[l, k] = 1 if l == c else 0
        for l in range(m):
            for k in range(p):
                if l == c:
                    V_total[l*p + k, c] = 1
        
        # W_c[k, i] = 1 for all k, i
        for k in range(p):
            for i in range(n):
                W_total[k*n + i, c] = 1
    
    if not _verify_fast(U_total, V_total, W_total, n, m, p, trials=50):
        # Fallback to schoolbook if verification fails (should not)
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

