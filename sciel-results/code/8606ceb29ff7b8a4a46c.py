"""Matrix multiplication 4x4 by 4x7 rank-minimal tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 7]

def _build_tensor():
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, T):
    """Verify that sum_d U[:,d]*V[:,d]*W[:,d] == T"""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    T_rec = np.zeros_like(T)
    for d in range(rank):
        T_rec += np.einsum('a,b,c->abc', U[:,d], V[:,d], W[:,d])
    return np.allclose(T_rec, T, atol=1e-9)

def _strassen_4x4x4():
    """Return Strassen decomposition factors for 4x4 * 4x4 -> 4x4 multiplication.
    
    Returns (U, V, W) each of shape (16, 7) such that
    sum_d U[a,d]*V[b,d]*W[c,d] = T[a,b,c] for the 4x4x4 multiplication tensor.
    
    U indexes (i,j) of A: a = i*4 + j
    V indexes (j,k) of B: b = j*4 + k  
    W indexes (k,i) of C: c = k*4 + i
    """
    # Strassen's 7 products for 2x2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4x4, we partition into 2x2 blocks of size 2x2.
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # The 4x4x4 multiplication tensor can be decomposed using Strassen recursively.
    # Each 2x2x2 Strassen term involves products of 2x2 matrices.
    # A 2x2x2 multiplication has rank 7 (Strassen).
    # So 4x4x4 has rank 7*7 = 49? No, that's not right either.
    
    # Actually, the standard approach: the 4x4x4 tensor can be viewed as
    # a 2x2 block matrix where each entry is a 2x2 matrix multiplication.
    # Using Strassen at the block level: 7 block multiplications, each of rank 7.
    # Total rank = 7 * 7 = 49.
    
    # But we can do better. The 4x4x4 multiplication tensor has known rank.
    # Let me just construct it directly.
    
    # Alternative: use the fact that 4x4x4 = kron(2x2x2, 2x2x2) in some sense.
    # The tensor T_4x4x4[i*4+j, j*4+k, k*4+i] = 1
    # This equals T_2x2x2[i*2+i2, j*2+j2, k*2+k2] * T_2x2x2[i2, j2, k2]
    # where i = 2*i2 + i3, j = 2*j2 + j3, k = 2*k2 + k3
    # Wait, that's not quite right. Let me think again.
    
    # T_4[i,j,k] = 1 for all valid (i,j,k) in [0,4)^3 with the constraint that
    # the indices are linked: a=i*4+j, b=j*4+k, c=k*4+i
    # 
    # We can write i = 2*i_hi + i_lo, j = 2*j_hi + j_lo, k = 2*k_hi + k_lo
    # a = (2*i_hi+i_lo)*4 + (2*j_hi+j_lo) = 8*i_hi + 4*i_lo + 2*j_hi + j_lo
    # b = (2*j_hi+j_lo)*4 + (2*k_hi+k_lo) = 8*j_hi + 4*j_lo + 2*k_hi + k_lo
    # c = (2*k_hi+k_lo)*4 + (2*i_hi+i_lo) = 8*k_hi + 4*k_lo + 2*i_hi + i_lo
    
    # The tensor is T[a,b,c] = 1 iff the indices are consistent.
    # This is exactly the Kronecker product of two 2x2x2 tensors!
    # T_4 = T_2 ⊗ T_2 where the indices decompose appropriately.
    
    # For the Kronecker product of two rank-r tensors, the rank is at most r^2.
    # T_2 has rank 7 (Strassen), so T_4 has rank at most 49.
    
    # Let's construct this explicitly.
    # T_2[i*2+j, j*2+k, k*2+i] = 1 for i,j,k in {0,1}
    # Strassen decomposition of T_2:
    
    # For 2x2x2: a=i*2+j, b=j*2+k, c=k*2+i
    # The 7 Strassen terms:
    # Term 0: M1=(A11+A22)(B11+B22) -> U: a=0(A00),a=3(A11) get +1; V: b=0(B00),b=3(B11) get +1; W: c=0(C00),c=3(C11) get +1
    # Wait, let me be more careful.
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Now I need to express each M_d as a rank-1 tensor contribution.
    # M_d = (linear combo of A entries) * (linear combo of B entries)
    # This contributes to C entries via the W factors.
    
    # For each Strassen term d, we have:
    # U[a,d] = coefficient of A[a] in the first factor
    # V[b,d] = coefficient of B[b] in the second factor  
    # W[c,d] = coefficient for C[c] in the output
    
    # M1 = (A00+A11)(B00+B11)
    # U: a=0(A00):+1, a=3(A11):+1
    # V: b=0(B00):+1, b=3(B11):+1
    # W: contributes to C00:+1, C11:+1
    # So W[0]:+1, W[3]:+1
    
    # M2 = (A10+A11)B00
    # U: a=2(A10):+1, a=3(A11):+1
    # V: b=0(B00):+1
    # W: contributes to C10:+1
    # W[2]:+1
    
    # M3 = A00(B10-B11)
    # U: a=0(A00):+1
    # V: b=2(B10):+1, b=3(B11):-1
    # W: contributes to C01:+1, C11:+1
    # W[1]:+1, W[3]:+1
    
    # M4 = A11(B01-B00)
    # U: a=3(A11):+1
    # V: b=1(B01):+1, b=0(B00):-1
    # W: contributes to C00:+1, C10:+1
    # W[0]:+1, W[2]:+1
    
    # M5 = (A00+A01)B11
    # U: a=0(A00):+1, a=1(A01):+1
    # V: b=3(B11):+1
    # W: contributes to C00:-1, C01:+1
    # W[0]:-1, W[1]:+1
    
    # M6 = (A10-A00)(B00+B01)
    # U: a=2(A10):+1, a=0(A00):-1
    # V: b=0(B00):+1, b=1(B01):+1
    # W: contributes to C11:+1
    # W[3]:+1
    
    # M7 = (A01-A11)(B10+B11)
    # U: a=1(A01):+1, a=3(A11):-1
    # V: b=2(B10):+1, b=3(B11):+1
    # W: contributes to C00:+1, C11:+1
    # W[0]:+1, W[3]:+1
    
    U2 = np.zeros((4, 7), dtype=np.float64)
    V2 = np.zeros((4, 7), dtype=np.float64)
    W2 = np.zeros((4, 7), dtype=np.float64)
    
    # Term 0: M1
    U2[0,0] = 1; U2[3,0] = 1
    V2[0,0] = 1; V2[3,0] = 1
    W2[0,0] = 1; W2[3,0] = 1
    
    # Term 1: M2
    U2[2,1] = 1; U2[3,1] = 1
    V2[0,1] = 1
    W2[2,1] = 1
    
    # Term 2: M3
    U2[0,2] = 1
    V2[2,2] = 1; V2[3,2] = -1
    W2[1,2] = 1; W2[3,2] = 1
    
    # Term 3: M4
    U2[3,3] = 1
    V2[1,3] = 1; V2[0,3] = -1
    W2[0,3] = 1; W2[2,3] = 1
    
    # Term 4: M5
    U2[0,4] = 1; U2[1,4] = 1
    V2[3,4] = 1
    W2[0,4] = -1; W2[1,4] = 1
    
    # Term 5: M6
    U2[2,5] = 1; U2[0,5] = -1
    V2[0,5] = 1; V2[1,5] = 1
    W2[3,5] = 1
    
    # Term 6: M7
    U2[1,6] = 1; U2[3,6] = -1
    V2[2,6] = 1; V2[3,6] = 1
    W2[0,6] = 1; W2[3,6] = 1
    
    # Verify 2x2x2
    T2 = np.zeros((4,4,4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T2[i*2+j, j*2+k, k*2+i] = 1.0
    T2_rec = np.zeros((4,4,4))
    for d in range(7):
        T2_rec += np.einsum('a,b,c->abc', U2[:,d], V2[:,d], W2[:,d])
    assert np.allclose(T2_rec, T2, atol=1e-9), "Strassen 2x2x2 verification failed"
    
    # Now construct 4x4x4 as Kronecker product of two 2x2x2 tensors.
    # The 4x4x4 tensor T4[i*4+j, j*4+k, k*4+i] = 1
    # Decompose i=2*i2+i3, j=2*j2+j3, k=2*k2+k3
    # Then a = i*4+j = (2*i2+i3)*4+(2*j2+j3) = 8*i2+4*i3+2*j2+j3
    # b = j*4+k = 8*j2+4*j3+2*k2+k3
    # c = k*4+i = 8*k2+4*k3+2*i2+i3
    
    # T4[a,b,c] = T2[8*i2+2*j2, 8*j2+2*k2, 8*k2+2*i2] * T2[4*i3+j3, 4*j3+k3, 4*k3+i3]
    # Wait, that's not quite right. Let me reconsider.
    
    # Actually T4[i*4+j, j*4+k, k*4+i] = 1 for all i,j,k in [0,4).
    # This is just the all-ones pattern on the valid index set.
    # The Kronecker product interpretation: 
    # T4 = T2 ⊗ T2 in the appropriate tensor product sense.
    
    # For the Kronecker product of two tensors X (size a1×b1×c1) and Y (size a2×b2×c2):
    # (X⊗Y)[a1*a2+a2', b1*b2+b2', c1*c2+c2'] = X[a1,b1,c1]*Y[a2,b2,c2]
    
    # For our case: T2 has size 4×4×4. T2⊗T2 has size 16×16×16.
    # (T2⊗T2)[a*4+a', b*4+b', c*4+c'] = T2[a,b,c]*T2[a',b',c']
    
    # Let's check: a = i*4+j = (2*i2+i3)*4+(2*j2+j3)
    # If we write a = a_hi*4 + a_lo where a_hi = 2*i2+i3 and a_lo = 2*j2+j3
    # Hmm, this doesn't decompose cleanly into a 4×4 grid.
    
    # Let me try a different indexing. Write:
    # i = 2*i_hi + i_lo, j = 2*j_hi + j_lo, k = 2*k_hi + k_lo
    # a = i*4+j = 8*i_hi + 4*i_lo + 2*j_hi + j_lo
    # b = j*4+k = 8*j_hi + 4*j_lo + 2*k_hi + k_lo
    # c = k*4+i = 8*k_hi + 4*k_lo + 2*i_hi + i_lo
    
    # Define: a_hi = 2*i_hi + j_hi, a_lo = 2*i_lo + j_lo
    # b_hi = 2*j_hi + k_hi, b_lo = 2*j_lo + k_lo
    # c_hi = 2*k_hi + i_hi, c_lo = 2*k_lo + i_lo
    
    # Then a = 4*a_hi + a_lo, b = 4*b_hi + b_lo, c = 4*c_hi + c_lo
    # And T2[a_hi, b_hi, c_hi] = 1 (since i_hi,j_hi,k_hi are valid 2x2x2 indices)
    # And T2[a_lo, b_lo, c_lo] = 1 (since i_lo,j_lo,k_lo are valid 2x2x2 indices)
    
    # So T4[4*a_hi+a_lo, 4*b_hi+b_lo, 4*c_hi+c_lo] = T2[a_hi,b_hi,c_hi] * T2[a_lo,b_lo,c_lo]
    
    # This means T4 = T2 ⊗ T2 (Kronecker product)!
    
    # For the Kronecker product of two rank-r tensors:
    # If X = sum_{d=1}^{r} u_d ⊗ v_d ⊗ w_d and Y = sum_{e=1}^{r} u'_e ⊗ v'_e ⊗ w'_e
    # Then X⊗Y = sum_{d,e} (u_d⊗u'_e) ⊗ (v_d⊗v'_e) ⊗ (w_d⊗w'_e)
    # So rank(X⊗Y) ≤ rank(X) * rank(Y)
    
    # rank(T2) = 7, so rank(T4) ≤ 49.
    
    U4 = np.zeros((16, 49), dtype=np.float64)
    V4 = np.zeros((16, 49), dtype=np.float64)
    W4 = np.zeros((16, 49), dtype=np.float64)
    
    for d in range(7):
        for e in range(7):
            col = d*7 + e
            # u = U2[:,d] ⊗ U2[:,e], shape (4,4) -> flatten to (16,)
            # (u⊗u')[a*4+a'] = u[a]*u'[a']
            U4[:, col] = np.kron(U2[:,d], U2[:,e])
            V4[:, col] = np.kron(V2[:,d], V2[:,e])
            W4[:, col] = np.kron(W2[:,d], W2[:,e])
    
    # Verify 4x4x4
    T4 = np.zeros((16,16,16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T4[i*4+j, j*4+k, k*4+i] = 1.0
    T4_rec = np.zeros((16,16,16))
    for d in range(49):
        T4_rec += np.einsum('a,b,c->abc', U4[:,d], V4[:,d], W4[:,d])
    assert np.allclose(T4_rec, T4, atol=1e-9), "Strassen 4x4x4 verification failed"
    
    return U4, V4, W4

def _build_4x4x7_decomposition():
    """Build the 4x4x7 decomposition using Strassen(4x4x4) + schoolbook(4x4x3).
    
    Split B into B = [B_left | B_right] where B_left is 4x4 and B_right is 4x3.
    C = A * B = A * B_left + A * B_right
    
    For A * B_left (4x4 * 4x4 -> 4x4): use Strassen, rank 49.
    For A * B_right (4x4 * 4x3 -> 4x3): use schoolbook, rank 4*4*3 = 48.
    
    Total rank = 49 + 48 = 97.
    
    Wait, but the output C is 4x7, not 4x4 + 4x3 separately.
    The tensor T has dimensions (n*m, m*p, p*n) = (16, 28, 28).
    
    Let me reconsider. We need to decompose T[i*4+j, j*7+k, k*4+i] = 1.
    
    Split k into k = 4*k_hi + k_lo where k_hi in {0,1} and k_lo in {0,1,2,3}.
    - When k_hi = 0 (k in {0,1,2,3}): this is the 4x4x4 part
    - When k_hi = 1 (k in {4,5,6}): this is the 4x4x3 part
    
    For the 4x4x4 part (k in {0,1,2,3}):
    T[i*4+j, j*7+k, k*4+i] = 1
    The middle index is j*7+k. For the Strassen part, we need to map this to the 4x4x4 convention.
    
    Actually, let me think about this differently. The full tensor is:
    T[a,b,c] where a=i*4+j (i,j in [0,4)), b=j*7+k (j in [0,4), k in [0,7)), c=k*4+i (k in [0,7), i in [0,4))
    
    T[a,b,c] = 1 iff the indices are consistent (j matches between a and b, i matches between a and c, k matches between b and c).
    
    We can split this into two parts:
    Part 1: k in {0,1,2,3} (4x4x4 block)
    Part 2: k in {4,5,6} (4x4x3 block)
    
    For Part 1:
    a = i*4+j, b = j*7+k (k in {0,1,2,3}), c = k*4+i
    We can use the Strassen 4x4x4 decomposition but with modified V and W indices.
    
    For the Strassen 4x4x4, the standard indexing is:
    a' = i*4+j, b' = j*4+k, c' = k*4+i
    
    But in our problem, b = j*7+k instead of j*4+k.
    So we need to map b' -> b. The Strassen V factor has entries at positions j*4+k for k in {0,1,2,3}.
    In our problem, these correspond to positions j*7+k.
    
    Similarly, c = k*4+i is the same in both cases for k in {0,1,2,3}.
    
    So for Part 1, we take the Strassen U4, V4, W4 and:
    - U stays the same (size 16)
    - V: remap from j*4+k to j*7+k (size 16 -> 28)
    - W: stays the same for k in {0,1,2,3} (size 16 -> 28, only first 16 entries used)
    
    For Part 2 (k in {4,5,6}):
    a = i*4+j, b = j*7+k (k in {4,5,6}), c = k*4+i
    Schoolbook: for each (i,j,k) with k in {4,5,6}:
    U[i*4+j] = 1, V[j*7+k] = 1, W[k*4+i] = 1
    
    Number of terms: 4*4*3 = 48
    
    Total rank: 49 + 48 = 97.
    
    But wait - can we do better? The 4x4x3 part might also have a lower-rank decomposition.
    4x4x3 schoolbook is 48. Can we do better?
    
    Actually, let me reconsider the whole approach. Maybe I should split differently.
    
    Alternative: split A's columns instead.
    A = [A_left | A_right] where A_left is 4x4 and A_right is 4x0? No, A is 4x4.
    
    Let me try: split the summation index j.
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    This doesn't help directly since j only goes to 3.
    
    Let me try splitting k (the output column index):
    C[:, 0:4] = A * B[:, 0:4]  (4x4 * 4x4 -> 4x4)
    C[:, 4:7] = A * B[:, 4:7]  (4x4 * 4x3 -> 4x3)
    
    This is what I had before. Rank = 49 + 48 = 97.
    
    Hmm, but 97 > 112? No, 97 < 112. Good.
    
    Actually wait, let me double-check. The schoolbook for 4x4x7 is 4*4*7 = 112.
    My decomposition gives 49 + 48 = 97. That's an improvement!
    
    Let me verify this is correct by constructing it explicitly.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Get Strassen 4x4x4 factors
    U4, V4, W4 = _strassen_4x4x4()
    rank_strassen = 49
    
    # Part 1: k in {0,1,2,3} - use Strassen
    # U: same as U4 (16 x 49)
    # V: remap V4 from j*4+k to j*7+k (28 x 49)
    # W: remap W4 from k*4+i to k*4+i (28 x 49), only k in {0,1,2,3}
    
    U1 = np.zeros((16, rank_strassen), dtype=np.float64)
    V1 = np.zeros((28, rank_strassen), dtype=np.float64)
    W1 = np.zeros((28, rank_strassen), dtype=np.float64)
    
    U1 = U4.copy()
    for j in range(4):
        for k in range(4):
            V1[j*7+k] = V4[j*4+k]
    for k in range(4):
        for i in range(4):
            W1[k*4+i] = W4[k*4+i]
    
    # Part 2: k in {4,5,6} - schoolbook
    rank_school = 4 * 4 * 3  # 48
    U2 = np.zeros((16, rank_school), dtype=np.float64)
    V2 = np.zeros((28, rank_school), dtype=np.float64)
    W2 = np.zeros((28, rank_school), dtype=np.float64)
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U2[i*4+j, col] = 1
                V2[j*7+k, col] = 1
                W2[k*4+i, col] = 1
                col += 1
    
    # Combine
    total_rank = rank_strassen + rank_school
    U = np.hstack([U1, U2])
    V = np.hstack([V1, V2])
    W = np.hstack([W1, W2])
    
    # Verify
    T = _build_tensor()
    T_rec = np.zeros_like(T)
    for d in range(total_rank):
        T_rec += np.einsum('a,b,c->abc', U[:,d], V[:,d], W[:,d])
    assert np.allclose(T_rec, T, atol=1e-9), "4x4x7 decomposition verification failed"
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return U_list, V_list, W_list

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS
    
    # Build the Strassen + schoolbook decomposition
    U, V, W = _build_4x4x7_decomposition()
    
    rank = len(U[0])
    
    # Verify one more time
    T = _build_tensor()
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    T_rec = np.zeros_like(T)
    for d in range(rank):
        T_rec += np.einsum('a,b,c->abc', U_np[:,d], V_np[:,d], W_np[:,d])
    assert np.allclose(T_rec, T, atol=1e-9), "Final verification failed"
    
    return {"U": U, "V": V, "W": W}
