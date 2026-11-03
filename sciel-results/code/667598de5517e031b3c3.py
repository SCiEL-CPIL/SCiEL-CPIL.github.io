import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    # We'll build a decomposition using block structure.
    # A is 4x5, B is 5x5, C is 4x5.
    # We can split B into B = [B_left (5x4), B_right (5x1)]? No, B is 5x5.
    # Let's split indices of B (which is m x p = 5x5).
    # Let's split A (n x m = 4x5) and B (m x p = 5x5).
    # Actually, the tensor is T_{i,j,k} where i in 0..3, j in 0..4, k in 0..4.
    # T[i, j, k] = 1 if ... no, it's the multiplication tensor.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is T_{(i,j), (j,k), (k,i)}? No.
    # The standard tensor for matrix multiplication M_{n,m,p} has indices (i,j,k)
    # such that T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
    # In the flattened form:
    # U has rows indexed by (i,j) -> i*m + j
    # V has rows indexed by (j,k) -> j*p + k
    # W has rows indexed by (k,i) -> k*n + i
    # T_{i*m+j, j*p+k, k*n+i} = 1.
    
    # We want to find a low rank decomposition.
    # Rank 100 is schoolbook.
    # Let's try to use Strassen for the 4x4x4 part and handle the 5th column/row separately.
    # Split j into j1 in 0..3 and j2=4.
    # Split k into k1 in 0..3 and k2=4.
    # C_{i,k} = sum_{j=0}^3 A_{i,j} B_{j,k} + A_{i,4} B_{4,k}
    # = sum_{j=0}^3 A_{i,j} B_{j,k1} + sum_{j=0}^3 A_{i,j} B_{j,k2} + A_{i,4} B_{4,k1} + A_{i,4} B_{4,k2}
    
    # Let's define sub-tensors:
    # 1. C^{(1)}_{i,k1} = sum_{j=0}^3 A_{i,j} B_{j,k1}  (4x4x4 multiplication)
    # 2. C^{(2)}_{i,k2} = sum_{j=0}^3 A_{i,j} B_{j,k2}  (4x4x1 multiplication, but B part is 4x1)
    #    Actually, B_{j,k2} for j=0..3, k2=4 is a 4x1 matrix.
    #    So this is M_{4,4,1}. Rank is 4.
    # 3. C^{(3)}_{i,k1} = A_{i,4} B_{4,k1}  (1x1x4? No, A part is 4x1, B part is 1x4)
    #    This is M_{4,1,4}. Rank is 4.
    # 4. C^{(4)}_{i,k2} = A_{i,4} B_{4,k2}  (4x1x1). Rank is 4? No, it's just a scalar product for each i?
    #    C_{i,4} += A_{i,4} B_{4,4}. This is a rank 4 tensor?
    #    T_{i,4,4} = A_{i,4} B_{4,4}.
    #    This can be done with 4 rank-1 terms?
    #    U_{i,4} = A_{i,4}, V_{4,4} = B_{4,4}, W_{4,i} = C_{i,4}.
    #    Wait, the index structure is (i,j), (j,k), (k,i).
    #    For term 4: i in 0..3, j=4, k=4.
    #    U_{i,4} is non-zero. V_{4,4} is non-zero. W_{4,i} is non-zero.
    #    This is a tensor of shape (4,1,4) in (i,j,k) subspace?
    #    Actually, it's just a set of 4 independent scalar multiplications.
    #    So rank 4.
    
    # Total rank = Rank(M_{4,4,4}) + Rank(M_{4,4,1}) + Rank(M_{4,1,4}) + Rank(M_{4,1,1})
    # Rank(M_{4,4,4}) using Strassen: 49?
    # Strassen for 2x2x2 has rank 7.
    # 4x4x4 can be viewed as 4 blocks of 2x2x2.
    # Rank is 7 * 4 = 28? No, that's not right.
    # The rank of M_{2,2,2} is 7.
    # The rank of M_{4,4,4} is at most 7 * (4/2)^3? No.
    # Kronecker product of two M_{2,2,2} tensors gives M_{4,4,4}.
    # Rank of Kronecker product is product of ranks.
    # So Rank(M_{4,4,4}) <= 7 * 7 = 49.
    # Is there a better one?
    # Lower bound for M_{4,4,4} is 49? No, lower bound is 4*4=16?
    # Actually, the rank of M_{4,4,4} is known to be 49?
    # Let's assume 49 for now.
    # Rank(M_{4,4,1}) = 4.
    # Rank(M_{4,1,4}) = 4.
    # Rank(M_{4,1,1}) = 4.
    # Total rank = 49 + 4 + 4 + 4 = 61.
    
    # Can we do better?
    # Maybe use a different split.
    # What if we use Strassen for the whole thing?
    # No, 5 is not a power of 2.
    
    # Let's implement the block decomposition.
    
    # Helper to create rank-1 tensors
    def add_term(U, V, W, i, j, k, coeff=1.0):
        # i: row index of U (i*m + j)
        # j: row index of V (j*p + k)
        # k: row index of W (k*n + i)
        # Wait, the indices in the tensor T are (i, j, k) where:
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        # This corresponds to the term A_{i,j} B_{j,k} contributing to C_{i,k}.
        # So for a specific term in the decomposition, we need to specify
        # which (i,j) in U, which (j,k) in V, and which (k,i) in W.
        
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        U[u_idx] += coeff
        V[v_idx] += coeff
        W[w_idx] += coeff
    
    # We need to build U, V, W matrices of shape (n*m, R), (m*p, R), (p*n, R).
    # Let's determine R first.
    # R = 49 + 4 + 4 + 4 = 61.
    
    # But wait, we can optimize the 4x4x4 part.
    # Is there a rank 49 decomposition for 4x4x4?
    # Yes, the Kronecker product of two Strassen decompositions.
    
    # Let's define the Strassen decomposition for 2x2x2.
    # Indices: i in 0,1; j in 0,1; k in 0,1.
    # Terms:
    # M1 = A00(B00+B11) -> U(0,0), V(0,0), V(1,1), W(0,0)
    #    Wait, this is not a single rank-1 term.
    #    A rank-1 term is u_i v_j w_k.
    #    Strassen's algorithm computes 7 products:
    #    M1 = A00(B00+B11)
    #    M2 = A10(B00+B01)
    #    M3 = (A00+A10)B10
    #    M4 = (A01+A11)B11
    #    M5 = (A00+A01)B01
    #    M6 = (A10-A00)B00
    #    M7 = (A11-A01)B11
    #    Then C is computed from these.
    #    This is a decomposition of the *algorithm*, but is it a rank-7 tensor decomposition?
    #    Yes, the tensor M_{2,2,2} has rank 7.
    #    The 7 rank-1 tensors are:
    #    1. u=(1,0,1,0), v=(1,0,1,0), w=(1,0,0,0) ? No.
    #    Let's look up the explicit rank-7 decomposition of M_{2,2,2}.
    #    It is given by:
    #    r1: u=(1,0,0,0), v=(1,1,0,0), w=(1,0,0,0)  -> A00(B00+B01)C00? No.
    
    #    Let's derive it.
    #    We want T = sum r_t (u_t ⊗ v_t ⊗ w_t).
    #    T_{i,j,k} = 1 if i,j,k are such that A_{i,j} B_{j,k} -> C_{i,k}.
    #    Indices in flattened form:
    #    U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    #    V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    #    W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)  (Note: W index is k*n+i. For n=2, k*2+i)
    #       k=0,i=0 -> 0
    #       k=0,i=1 -> 1
    #       k=1,i=0 -> 2
    #       k=1,i=1 -> 3
    
    #    The 7 terms for M_{2,2,2}:
    #    1. A00(B00+B11):
    #       This corresponds to u=(1,0,0,0), v=(1,0,0,1), w=(1,0,0,0)?
    #       Let's check: sum u_i v_j w_k.
    #       If u=(1,0,0,0), v=(1,0,0,1), w=(1,0,0,0):
    #       Non-zero at (0,0,0) and (0,3,0).
    #       T_{0,0,0} = 1 (A00 B00 -> C00). Correct.
    #       T_{0,3,0} = 1 (A00 B11 -> C00)? No, B11 is V index 3.
    #       V index 3 is (1,1). So j=1, k=1.
    #       W index 0 is (0,0). So k=0, i=0.
    #       But j=1 and k=1 implies k=1, but W says k=0. Contradiction.
    #       The indices must be consistent: j from U must equal j from V. k from V must equal k from W. i from W must equal i from U.
    #       So a rank-1 tensor is defined by a triple (i,j,k) and has 1 at:
    #       U[i*m+j], V[j*p+k], W[k*n+i].
    #       And it is 0 elsewhere.
    #       Wait, a rank-1 tensor u ⊗ v ⊗ w has entries u_a v_b w_c.
    #       For it to match T, we need u_a v_b w_c to be non-zero only when (a,b,c) is a valid triple.
    #       This implies u, v, w must be such that their supports are consistent.
    #       Actually, the rank-1 decomposition of the multiplication tensor is NOT just a sum of "basis" tensors e_i ⊗ e_j ⊗ e_k.
    #       Those are the schoolbook terms.
    #       Strassen's terms involve sums of basis elements.
    #       For example, term 1: A00(B00+B11).
    #       This is not a single rank-1 tensor in the standard basis.
    #       It is a linear combination.
    #       The rank-7 decomposition means we can write T as sum of 7 rank-1 tensors, where each rank-1 tensor is a product of 3 vectors.
    #       The vectors are not necessarily basis vectors.
    
    #    Let's find the 7 vectors for M_{2,2,2}.
    #    Reference: 
    #    u1 = (1, 0, 0, 0)
    #    v1 = (1, 0, 0, 1)
    #    w1 = (1, 0, 0, 0)
    #    This gives T(0,0,0) and T(0,3,0).
    #    But T(0,3,0) is A00 B11 C00.
    #    Is this a valid term?
    #    A00 is U0. B11 is V3. C00 is W0.
    #    Yes.
    #    So term 1 is u1 ⊗ v1 ⊗ w1.
    #    Let's check if this is part of the standard Strassen decomposition.
    #    Actually, the standard Strassen decomposition is:
    #    M1 = A00(B00+B11)
    #    This corresponds to u=(1,0,0,0), v=(1,0,0,1), w=(1,0,0,0).
    #    Yes.
    #    
    #    M2 = A10(B00+B01)
    #    u=(0,0,1,0), v=(1,1,0,0), w=(0,1,0,0)
    #    A10 is U2. B00 is V0, B01 is V1. C10 is W1 (k=0, i=1 -> 0*2+1=1).
    #    So w=(0,1,0,0).
    #    
    #    M3 = (A00+A10)B10
    #    u=(1,0,1,0), v=(0,0,1,0), w=(0,0,1,0)
    #    B10 is V2. C01 is W2 (k=1, i=0 -> 1*2+0=2).
    #    
    #    M4 = (A01+A11)B11
    #    u=(0,1,0,1), v=(0,0,0,1), w=(0,0,0,1)
    #    B11 is V3. C01 is W2? No, C11 is W3 (k=1, i=1 -> 3).
    #    Wait, M4 contributes to C01 and C11?
    #    (A01+A11)B11 = A01 B11 + A11 B11.
    #    A01 B11 -> C01. A11 B11 -> C11.
    #    So w should have non-zeros at W2 and W3.
    #    w=(0,0,1,1).
    #    
    #    Let's verify the full set of 7 terms for M_{2,2,2}.
    #    1. u=(1,0,0,0), v=(1,0,0,1), w=(1,0,0,0)
    #    2. u=(0,0,1,0), v=(1,1,0,0), w=(0,1,0,0)
    #    3. u=(1,0,1,0), v=(0,0,1,0), w=(0,0,1,0)
    #    4. u=(0,1,0,1), v=(0,0,0,1), w=(0,0,1,1)
    #    5. u=(1,1,0,0), v=(0,1,0,0), w=(1,0,0,0)
    #    6. u=(0,0,1,-1), v=(1,0,0,0), w=(0,1,0,0)
    #    7. u=(0,1,0,-1), v=(0,0,0,1), w=(0,0,1,1)
    
    #    Let's sum these and see if we get T.
    #    This is tedious but doable.
    #    Instead of hardcoding, I can just use the known fact that Rank(M_{4,4,4}) = 49.
    #    I will construct the 4x4x4 decomposition by Kronecker product of the 2x2x2 decomposition.
    
    #    Let U2, V2, W2 be the 4x7 matrices for M_{2,2,2}.
    #    Then U4 = U2 ⊗ U2 (4*4=16 rows, 7*7=49 cols)
    #    V4 = V2 ⊗ V2
    #    W4 = W2 ⊗ W2
    #    This gives a rank 49 decomposition for M_{4,4,4}.
    
    #    Now I need to map this to the global indices.
    #    The 4x4x4 block corresponds to i in 0..3, j in 0..3, k in 0..3.
    #    In the global tensor:
    #    U index: i*5 + j
    #    V index: j*5 + k
    #    W index: k*4 + i
    
    #    For the 4x4x4 part:
    #    i in 0..3, j in 0..3, k in 0..3.
    #    U_local index: i*4 + j
    #    V_local index: j*4 + k
    #    W_local index: k*4 + i
    
    #    We need to embed U4 (16x49) into U_global (20x61).
    #    Row i*4+j of U4 goes to row i*5+j of U_global.
    #    Similarly for V and W.
    
    #    Let's define the 2x2x2 decomposition matrices.
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1
    U2[0, 0] = 1; V2[0, 0] = 1; V2[3, 0] = 1; W2[0, 0] = 1
    # Term 2
    U2[2, 1] = 1; V2[0, 1] = 1; V2[1, 1] = 1; W2[1, 1] = 1
    # Term 3
    U2[0, 2] = 1; U2[2, 2] = 1; V2[2, 2] = 1; W2[2, 2] = 1
    # Term 4
    U2[1, 3] = 1; U2[3, 3] = 1; V2[3, 3] = 1; W2[2, 3] = 1; W2[3, 3] = 1
    # Term 5
    U2[0, 4] = 1; U2[1, 4] = 1; V2[1, 4] = 1; W2[0, 4] = 1
    # Term 6
    U2[2, 5] = 1; U2[0, 5] = -1; V2[0, 5] = 1; W2[1, 5] = 1
    # Term 7
    U2[1, 6] = 1; U2[3, 6] = -1; V2[3, 6] = 1; W2[2, 6] = 1; W2[3, 6] = 1
    
    # Verify M_{2,2,2}
    T2 = np.einsum('ar,br,cr->abc', U2, V2, W2)
    T2_target = np.zeros((4,4,4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T2_target[i*2+j, j*2+k, k*2+i] = 1
    # Check
    if not np.allclose(T2, T2_target):
        # Fallback to schoolbook if Strassen fails
        pass

    # Create 4x4x4 decomposition
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # Now handle the remaining terms.
    # Total rank R = 49 + 4 + 4 + 4 = 61.
    R = 61
    U = np.zeros((20, R))
    V = np.zeros((25, R))
    W = np.zeros((20, R))
    
    # Embed 4x4x4 part
    # U4 rows: i*4+j for i,j in 0..3
    # U rows: i*5+j
    for i in range(4):
        for j in range(4):
            U[i*5+j, :49] = U4[i*4+j, :]
    
    # V4 rows: j*4+k for j,k in 0..3
    # V rows: j*5+k
    for j in range(4):
        for k in range(4):
            V[j*5+k, :49] = V4[j*4+k, :]
    
    # W4 rows: k*4+i for k,i in 0..3
    # W rows: k*4+i (since n=4, W index is k*n+i = k*4+i)
    # So W4 rows map directly to W rows for k,i in 0..3
    for k in range(4):
        for i in range(4):
            W[k*4+i, :49] = W4[k*4+i, :]
    
    # Now add the other terms.
    # Term 2: C^{(2)}_{i,k2} = sum_{j=0}^3 A_{i,j} B_{j,k2}
    # k2=4. j in 0..3. i in 0..3.
    # This is M_{4,4,1}.
    # Rank 4.
    # Terms: for each j in 0..3, A_{i,j} B_{j,4} -> C_{i,4}.
    # This is 4 independent dot products?
    # No, for fixed j, it's A_{:,j} * B_{j,4} -> C_{:,4}.
    # This is a rank-4 update?
    # Actually, it's 4 rank-1 terms?
    # For fixed j, the term is u_i v_j w_k.
    # u_i = A_{i,j}, v_j = B_{j,4}, w_k = C_{i,4}.
    # But w depends on i.
    # So for each j, we have a term that is a sum over i?
    # No, the tensor term is T_{i,j,k} = 1 if k=4 and j is fixed?
    # No, T_{i,j,k} = 1 if A_{i,j} B_{j,k} contributes to C_{i,k}.
    # For k=4, T_{i,j,4} = 1 for all i, j.
    # So we need to decompose the sub-tensor T^{(2)}_{i,j,4} = 1 for i,j in 0..3.
    # This is a 4x4x1 tensor.
    # Rank is 4.
    # Decomposition: sum_{j=0}^3 e_i ⊗ e_j ⊗ e_4?
    # No, that would be 16 terms.
    # We can do: sum_{j=0}^3 (sum_i e_i) ⊗ e_j ⊗ e_4?
    # Let u = (1,1,1,1), v = e_j, w = e_4.
    # Then u ⊗ v ⊗ w has 1s at (i,j,4) for all i.
    # So 4 such terms cover all i,j.
    # So:
    for j in range(4):
        col = 49 + j
        for i in range(4):
            U[i*5+j, col] = 1
        V[j*5+4, col] = 1
        for i in range(4):
            W[4*4+i, col] = 1  # k=4, i in 0..3 -> 16+i
    
    # Term 3: C^{(3)}_{i,k1} = A_{i,4} B_{4,k1}
    # j=4. i in 0..3. k in 0..3.
    # This is M_{4,1,4}.
    # Rank 4.
    # T_{i,4,k} = 1 for i in 0..3, k in 0..3.
    # Decomposition: for each k, sum_i e_i ⊗ e_4 ⊗ e_k.
    for k in range(4):
        col = 53 + k
        for i in range(4):
            U[i*5+4, col] = 1
        V[4*5+k, col] = 1
        for i in range(4):
            W[k*4+i, col] = 1
    
    # Term 4: C^{(4)}_{i,k2} = A_{i,4} B_{4,k2}
    # j=4, k=4. i in 0..3.
    # T_{i,4,4} = 1.
    # Rank 4.
    # Decomposition: for each i, e_i ⊗ e_4 ⊗ e_4?
    # No, w is e_{k,i}. For k=4, w is e_{16+i}.
    # So for each i, u=e_i, v=e_4, w=e_{16+i}.
    for i in range(4):
        col = 57 + i
        U[i*5+4, col] = 1
        V[4*5+4, col] = 1
        W[4*4+i, col] = 1
    
    # Verify the full decomposition
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = np.zeros((20, 25, 20))
    for i in range(4):
        for j in range(5):
            for k in range(5):
                T_target[i*5+j, j*5+k, k*4+i] = 1
    
    # Check error
    err = np.max(np.abs(T_approx - T_target))
    
    if err > 1e-6:
        # Fallback to schoolbook
        rank_sb = 100
        U = np.zeros((20, rank_sb))
        V = np.zeros((25, rank_sb))
        W = np.zeros((20, rank_sb))
        col = 0
        for i in range(4):
            for j in range(5):
                for k in range(5):
                    U[i*5+j, col] = 1
                    V[j*5+k, col] = 1
                    W[k*4+i, col] = 1
                    col += 1
    else:
        # Round to integers if possible
        U = np.round(U)
        V = np.round(V)
        W = np.round(W)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
