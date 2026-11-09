import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using Fractions."""
    rank = len(U[0])
    
    # Convert to Fractions
    U_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in U]
    V_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in V]
    W_frac = [[Fraction(x).limit_denominator(1000) for x in row] for row in W]
    
    # Check all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                s = Fraction(0)
                for r in range(rank):
                    s += U_frac[u][r] * V_frac[v][r] * W_frac[w][r]
                if s != 1:
                    return False
                
                # Check zero entries: for all (u,v,w) not of the form (i*m+j, j*p+k, k*n+i)
                # This is expensive (2500 entries). But we can just check the non-1 entries.
                # Actually, we can just check that the reconstruction matches the tensor.
                # Let's reconstruct the whole tensor? No, that's too slow with Fractions.
                # Instead, we check that for any u,v,w, the sum is 1 if (u,v,w) is a valid index, else 0.
                # To do this efficiently, we can precompute the set of valid indices.
    
    # Let's do a full check to be safe, but optimized.
    valid_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_indices.add((i*m+j, j*p+k, k*n+i))
    
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                s = Fraction(0)
                for r in range(rank):
                    s += U_frac[u][r] * V_frac[v][r] * W_frac[w][r]
                expected = 1 if (u, v, w) in valid_indices else 0
                if s != expected:
                    return False
    return True

def _strassen_2x2():
    """
    Return the rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of lists.
    U: 4x7, V: 4x7, W: 4x7
    Indices: u = i*2+j, v = j*2+k, w = k*2+i
    """
    # Strassen's 7 terms
    # Let A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)b00
    # m3 = a00(b01 - b11)
    # m4 = a11(b10 - b00)
    # m5 = (a00 + a01)b11
    # m6 = (a10 - a00)(b00 + b01)
    # m7 = (a01 - a11)(b10 + b11)
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # We need to map this to the tensor decomposition.
    # The terms are linear combinations of products of entries of A and B.
    # Each term m_r corresponds to a rank-1 tensor.
    # m_r = (sum_i alpha_{i,r} A_{i,j}) * (sum_j beta_{j,r} B_{j,k}) * (sum_k gamma_{k,r} C_{k,l})? No.
    # The decomposition is T = sum_r U_r V_r W_r^T (outer product).
    # For each r, U_r is a vector in R^4 (for A), V_r in R^4 (for B), W_r in R^4 (for C).
    # The term is U_r[u] V_r[v] W_r[w].
    # This must equal the contribution of m_r to T.
    
    # Let's construct U, V, W for the 7 terms.
    # Term 1: (a00+a11)(b00+b11)
    # This contributes to c00, c11 (and others? No, it's part of the sum).
    # Actually, the standard way is to express the C entries as linear combinations of the m's.
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    
    # The tensor T for 2x2 is defined by T[u,v,w]=1 if w = (k*n+i) where u=i*2+j, v=j*2+k.
    # Wait, the decomposition is for the multiplication itself.
    # The rank-1 tensors are formed by the "variables" of the m's.
    # m_r = (sum_i a_i v_r[i]) * (sum_j b_j w_r[j])? No.
    # m_r = (sum_{i,j} A_{i,j} * U_r[i,j]) * (sum_{j,k} B_{j,k} * V_r[j,k])
    # And C_{k,l} = sum_r m_r * W_r[k,l]
    # So the decomposition is sum_r (sum_{i,j} U_r[i,j] A_{i,j}) (sum_{j,k} V_r[j,k] B_{j,k}) (sum_{k,l} W_r[k,l] C_{k,l})?
    # No, the tensor is T_{u,v,w}.
    # T = sum_r U_r \otimes V_r \otimes W_r.
    # U_r is indexed by u (index of A), V_r by v (index of B), W_r by w (index of C).
    # The condition is that sum_r U_r[u] V_r[v] W_r[w] = 1 if (u,v,w) is a valid product term, else 0.
    
    # Let's use the known Strassen decomposition.
    # Reference:
    # m1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1] (contributes to c00, c11? No)
    # Actually, let's look up the exact U,V,W matrices for Strassen.
    
    # It's easier to derive or find a table.
    # Let's try to construct it.
    # m1 = (A00+A11)(B00+B11)
    # This term appears in C00 (coeff 1) and C11 (coeff 1).
    # So W_1 has 1s at indices for C00 and C11.
    # C indices: c00 -> w=0*2+0=0? No.
    # w = k*n + i.
    # C is k x i? No, C is result of A (2x2) * B (2x2) -> C (2x2).
    # Indices for C: c_{k,i} ? No, standard is C = A B.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # In our tensor: T[u, v, w] where u=i*m+j, v=j*p+k, w=k*n+i.
    # For 2x2x2: u=i*2+j, v=j*2+k, w=k*2+i.
    # C index w corresponds to (k, i).
    # So C_{i,k} is at w = k*2 + i.
    
    # m1 contributes to C_{0,0} (w=0) and C_{1,1} (w=3).
    # So W_1 = [1, 0, 0, 1] (indices 0 and 3).
    # U_1 corresponds to A_{0,0} and A_{1,1}.
    # u for A_{0,0} is 0*2+0=0.
    # u for A_{1,1} is 1*2+1=3.
    # So U_1 = [1, 0, 0, 1].
    # V_1 corresponds to B_{0,0} and B_{1,1}.
    # v for B_{0,0} is 0*2+0=0.
    # v for B_{1,1} is 1*2+1=3.
    # So V_1 = [1, 0, 0, 1].
    
    # m2 = (A10+A11)B00
    # Contributes to C10 (w=2*0+1=1? No. C_{1,0} -> i=1, k=0. w=0*2+1=1) and C11 (w=3).
    # Wait. C10 is i=1, k=0. w = 0*2 + 1 = 1.
    # C11 is i=1, k=1. w = 1*2 + 1 = 3.
    # So W_2 has 1s at 1 and 3.
    # W_2 = [0, 1, 0, 1].
    # U_2: A10 (u=2), A11 (u=3).
    # U_2 = [0, 0, 1, 1].
    # V_2: B00 (v=0).
    # V_2 = [1, 0, 0, 0].
    
    # m3 = A00(B01-B11)
    # Contributes to C01 (i=0, k=1. w=1*2+0=2) and C11 (i=1, k=1. w=3).
    # W_3 = [0, 0, 1, 1].
    # U_3: A00 (u=0).
    # U_3 = [1, 0, 0, 0].
    # V_3: B01 (v=1), B11 (v=3). Coeffs 1, -1.
    # V_3 = [0, 1, 0, -1].
    
    # m4 = A11(B10-B00)
    # Contributes to C00 (w=0) and C10 (w=1).
    # W_4 = [1, 1, 0, 0].
    # U_4: A11 (u=3).
    # U_4 = [0, 0, 0, 1].
    # V_4: B10 (v=2), B00 (v=0). Coeffs 1, -1.
    # V_4 = [-1, 0, 1, 0].
    
    # m5 = (A00+A01)B11
    # Contributes to C00 (w=0, coeff -1) and C01 (w=2, coeff 1).
    # W_5 = [-1, 0, 1, 0].
    # U_5: A00 (u=0), A01 (u=1).
    # U_5 = [1, 1, 0, 0].
    # V_5: B11 (v=3).
    # V_5 = [0, 0, 0, 1].
    
    # m6 = (A10-A00)(B00+B01)
    # Contributes to C11 (w=3, coeff 1).
    # W_6 = [0, 0, 0, 1].
    # U_6: A10 (u=2), A00 (u=0). Coeffs 1, -1.
    # U_6 = [-1, 0, 1, 0].
    # V_6: B00 (v=0), B01 (v=1).
    # V_6 = [1, 1, 0, 0].
    
    # m7 = (A01-A11)(B10+B11)
    # Contributes to C00 (w=0, coeff 1).
    # W_7 = [1, 0, 0, 0].
    # U_7: A01 (u=1), A11 (u=3). Coeffs 1, -1.
    # U_7 = [0, 1, 0, -1].
    # V_7: B10 (v=2), B11 (v=3).
    # V_7 = [0, 0, 1, 1].
    
    U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    # Note: U in decomposition is usually (rows=indices, cols=rank).
    # So U should be 4x7.
    U = np.array(U).T
    V = np.array([
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [-1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1]
    ]).T
    # Wait, let's re-verify V.
    # m1: B00, B11 -> [1, 0, 0, 1]
    # m2: B00 -> [1, 0, 0, 0]
    # m3: B01, B11 -> [0, 1, 0, -1]
    # m4: B10, B00 -> [-1, 0, 1, 0]
    # m5: B11 -> [0, 0, 0, 1]
    # m6: B00, B01 -> [1, 1, 0, 0]
    # m7: B10, B11 -> [0, 0, 1, 1]
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]).T
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 1, 0]
    ]).T
    
    # Let's verify this.
    # T = U @ V @ W? No, T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r].
    # Let's check one entry. T[0,0,0] should be 1.
    # u=0, v=0, w=0.
    # U[0,:] = [1, 0, 1, 0, 1, -1, 0]
    # V[0,:] = [1, 1, 0, -1, 0, 1, 0]
    # W[0,:] = [1, 0, 0, 1, -1, 0, 1]
    # Dot products:
    # r=0: 1*1*1 = 1
    # r=1: 0*1*0 = 0
    # r=2: 1*0*0 = 0
    # r=3: 0*(-1)*1 = 0
    # r=4: 1*0*(-1) = 0
    # r=5: -1*1*0 = 0
    # r=6: 0*0*1 = 0
    # Sum = 1. Correct.
    
    # Check T[0,0,1] (should be 0).
    # u=0, v=0, w=1.
    # W[1,:] = [0, 1, 0, 1, 0, 0, 0]
    # r=0: 1*1*0 = 0
    # r=1: 0*1*1 = 0
    # r=2: 1*0*0 = 0
    # r=3: 0*(-1)*1 = 0
    # r=4: 1*0*0 = 0
    # r=5: -1*1*0 = 0
    # r=6: 0*0*0 = 0
    # Sum = 0. Correct.
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V_school = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W_school = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
                
    best_U, best_V, best_W = U_school.tolist(), V_school.tolist(), W_school.tolist()
    best_rank = rank_schoolbook
    
    # Try to construct a rank-46 solution using Strassen for four 2x2 blocks.
    # The 5x5 matrix can be viewed as:
    # A (2x5) = [A_top (2x2), A_mid (2x1), A_bot (2x2)]? No.
    # A is 2x5. B is 5x5. C is 2x5.
    # We can split the 5 dimensions of A's columns and B's rows into:
    # indices 0,1 (Strassen 1), 2 (Scalar 1), 3,4 (Strassen 2)?
    # Let's try splitting into 2x2 and 2x2 and 1x1.
    # A columns: 0,1 | 2 | 3,4
    # B rows: 0,1 | 2 | 3,4
    # This gives 3 blocks.
    # Block 1: A[:, 0:2] (2x2), B[0:2, :] (2x5).
    # Block 2: A[:, 2:3] (2x1), B[2:3, :] (1x5).
    # Block 3: A[:, 3:5] (2x2), B[3:5, :] (2x5).
    
    # For Block 1:
    # A1 is 2x2, B1 is 2x5.
    # This is multiplication of 2x2 and 2x5.
    # We can split B1 into two 2x2 blocks? No, B1 is 2x5.
    # Let's split the output C (2x5) into C1 (2x2), C2 (2x1), C3 (2x2).
    # Then C1 = A1 * B1[:, 0:2] (2x2 * 2x2). Use Strassen. Rank 7.
    # C2 = A1 * B1[:, 2:3] (2x2 * 2x1). Schoolbook. Rank 4.
    # C3 = A1 * B1[:, 3:5] (2x2 * 2x2). Use Strassen?
    # Wait, Strassen is for 2x2 * 2x2.
    # So for Block 1, we have 3 sub-products.
    # Sub 1.1: A1 * B1_left (2x2). Rank 7.
    # Sub 1.2: A1 * B1_mid (2x1). Rank 4.
    # Sub 1.3: A1 * B1_right (2x2). Rank 7.
    # Total for Block 1: 7+4+7 = 18.
    
    # Block 2: A2 (2x1) * B2 (1x5).
    # Sub 2.1: A2 * B2_left (2x1 * 1x2). Rank 2.
    # Sub 2.2: A2 * B2_mid (2x1 * 1x1). Rank 1.
    # Sub 2.3: A2 * B2_right (2x1 * 1x2). Rank 2.
    # Total for Block 2: 2+1+2 = 5.
    
    # Block 3: A3 (2x2) * B3 (2x5).
    # Sub 3.1: A3 * B3_left (2x2 * 2x2). Rank 7.
    # Sub 3.2: A3 * B3_mid (2x2 * 2x1). Rank 4.
    # Sub 3.3: A3 * B3_right (2x2 * 2x2). Rank 7.
    # Total for Block 3: 7+4+7 = 18.
    
    # Total Rank = 18 + 5 + 18 = 41.
    # This is better than 46.
    
    # Let's verify the logic.
    # A = [A1, A2, A3] where A1 is 2x2, A2 is 2x1, A3 is 2x2.
    # B = [B1; B2; B3] where B1 is 2x5, B2 is 1x5, B3 is 2x5.
    # C = A1 B1 + A2 B2 + A3 B3.
    
    # We need to construct the U, V, W matrices for the whole thing.
    # The indices are global.
    # U has 10 rows (2x5).
    # V has 25 rows (5x5).
    # W has 10 rows (2x5).
    
    # We will build U, V, W column by column (each column is a rank-1 term).
    # We'll use a helper to add a Strassen block or a schoolbook block.
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        # This is not how it works. We need to add a full rank-1 tensor.
        # For Strassen, we add 7 terms.
        # For Schoolbook, we add terms for each (i,j,k).
        
        pass
    
    # Let's define the blocks precisely.
    # A indices: i in {0,1}, j in {0..4}.
    # B indices: j in {0..4}, k in {0..4}.
    # C indices: k in {0..4}, i in {0,1}.
    
    # Block 1: A[:, 0:2], B[0:2, :]
    # A1: i in {0,1}, j in {0,1}.
    # B1: j in {0,1}, k in {0..4}.
    
    # Sub 1.1: A1, B1[:, 0:2]. Output C[:, 0:2].
    # This is 2x2 * 2x2 -> 2x2.
    # Use Strassen.
    U_s, V_s, W_s = _strassen_2x2()
    # Map indices.
    # U local indices 0..3 correspond to A1[i, j_local] where j_local in {0,1}.
    # Global A index: i*5 + j_local.
    # V local indices 0..3 correspond to B1[j_local, k_local] where k_local in {0,1}.
    # Global B index: j_local*5 + k_local.
    # W local indices 0..3 correspond to C[k_local, i_local] where i_local in {0,1}.
    # Global C index: k_local*2 + i_local.
    
    for r in range(7):
        u_vec = np.zeros(10)
        v_vec = np.zeros(25)
        w_vec = np.zeros(10)
        
        for lu in range(4):
            i, jl = divmod(lu, 2)
            u_vec[i*5 + jl] = U_s[lu, r]
            
        for lv in range(4):
            jl, lk = divmod(lv, 2)
            v_vec[jl*5 + lk] = V_s[lv, r]
            
        for lw in range(4):
            lk, li = divmod(lw, 2)
            w_vec[lk*2 + li] = W_s[lw, r]
            
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    # Sub 1.2: A1, B1[:, 2:3]. Output C[:, 2:3].
    # 2x2 * 2x1 -> 2x1.
    # Schoolbook. 4 terms.
    for i in range(2):
        for jl in range(2):
            for lk in range(1): # k_local=0
                u_vec = np.zeros(10)
                v_vec = np.zeros(25)
                w_vec = np.zeros(10)
                
                u_idx = i*5 + jl
                v_idx = jl*5 + (2 + lk)
                w_idx = (2 + lk)*2 + i
                
                u_vec[u_idx] = 1
                v_vec[v_idx] = 1
                w_vec[w_idx] = 1
                
                U_list.append(u_vec)
                V_list.append(v_vec)
                W_list.append(w_vec)
                
    # Sub 1.3: A1, B1[:, 3:5]. Output C[:, 3:5].
    # 2x2 * 2x2 -> 2x2.
    # Use Strassen.
    for r in range(7):
        u_vec = np.zeros(10)
        v_vec = np.zeros(25)
        w_vec = np.zeros(10)
        
        for lu in range(4):
            i, jl = divmod(lu, 2)
            u_vec[i*5 + jl] = U_s[lu, r]
            
        for lv in range(4):
            jl, lk = divmod(lv, 2)
            v_vec[jl*5 + (3 + lk)] = V_s[lv, r]
            
        for lw in range(4):
            lk, li = divmod(lw, 2)
            w_vec[(3 + lk)*2 + li] = W_s[lw, r]
            
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    # Block 2: A[:, 2:3], B[2:3, :]
    # A2: i in {0,1}, j in {2}.
    # B2: j in {2}, k in {0..4}.
    
    # Sub 2.1: A2, B2[:, 0:2]. Output C[:, 0:2].
    # 2x1 * 1x2 -> 2x2.
    # Schoolbook. 2 terms?
    # C[i, k] = sum_j A[i, j] B[j, k].
    # Here j=2 only.
    # C[i, k] = A[i, 2] B[2, k].
    # So for each (i, k), it's one term.
    # i in {0,1}, k in {0,1}. 4 terms.
    for i in range(2):
        for k in range(2):
            u_vec = np.zeros(10)
            v_vec = np.zeros(25)
            w_vec = np.zeros(10)
            
            u_idx = i*5 + 2
            v_idx = 2*5 + k
            w_idx = k*2 + i
            
            u_vec[u_idx] = 1
            v_vec[v_idx] = 1
            w_vec[w_idx] = 1
            
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
            
    # Sub 2.2: A2, B2[:, 2:3]. Output C[:, 2:3].
    # 2x1 * 1x1 -> 2x1.
    # 2 terms.
    for i in range(2):
        u_vec = np.zeros(10)
        v_vec = np.zeros(25)
        w_vec = np.zeros(10)
        
        u_idx = i*5 + 2
        v_idx = 2*5 + 2
        w_idx = 2*2 + i
        
        u_vec[u_idx] = 1
        v_vec[v_idx] = 1
        w_vec[w_idx] = 1
        
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    # Sub 2.3: A2, B2[:, 3:5]. Output C[:, 3:5].
    # 2x1 * 1x2 -> 2x2.
    # 4 terms.
    for i in range(2):
        for k in range(2, 5):
            u_vec = np.zeros(10)
            v_vec = np.zeros(25)
            w_vec = np.zeros(10)
            
            u_idx = i*5 + 2
            v_idx = 2*5 + k
            w_idx = k*2 + i
            
            u_vec[u_idx] = 1
            v_vec[v_idx] = 1
            w_vec[w_idx] = 1
            
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
            
    # Block 3: A[:, 3:5], B[3:5, :]
    # A3: i in {0,1}, j in {3,4}.
    # B3: j in {3,4}, k in {0..4}.
    
    # Sub 3.1: A3, B3[:, 0:2]. Output C[:, 0:2].
    # 2x2 * 2x2 -> 2x2.
    # Use Strassen.
    for r in range(7):
        u_vec = np.zeros(10)
        v_vec = np.zeros(25)
        w_vec = np.zeros(10)
        
        for lu in range(4):
            i, jl = divmod(lu, 2)
            u_vec[i*5 + (3 + jl)] = U_s[lu, r]
            
        for lv in range(4):
            jl, lk = divmod(lv, 2)
            v_vec[(3 + jl)*5 + lk] = V_s[lv, r]
            
        for lw in range(4):
            lk, li = divmod(lw, 2)
            w_vec[lk*2 + li] = W_s[lw, r]
            
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    # Sub 3.2: A3, B3[:, 2:3]. Output C[:, 2:3].
    # 2x2 * 2x1 -> 2x1.
    # 4 terms.
    for i in range(2):
        for jl in range(2):
            u_vec = np.zeros(10)
            v_vec = np.zeros(25)
            w_vec = np.zeros(10)
            
            u_idx = i*5 + (3 + jl)
            v_idx = (3 + jl)*5 + 2
            w_idx = 2*2 + i
            
            u_vec[u_idx] = 1
            v_vec[v_idx] = 1
            w_vec[w_idx] = 1
            
            U_list.append(u_vec)
            V_list.append(v_vec)
            W_list.append(w_vec)
            
    # Sub 3.3: A3, B3[:, 3:5]. Output C[:, 3:5].
    # 2x2 * 2x2 -> 2x2.
    # Use Strassen.
    for r in range(7):
        u_vec = np.zeros(10)
        v_vec = np.zeros(25)
        w_vec = np.zeros(10)
        
        for lu in range(4):
            i, jl = divmod(lu, 2)
            u_vec[i*5 + (3 + jl)] = U_s[lu, r]
            
        for lv in range(4):
            jl, lk = divmod(lv, 2)
            v_vec[(3 + jl)*5 + (3 + lk)] = V_s[lv, r]
            
        for lw in range(4):
            lk, li = divmod(lw, 2)
            w_vec[(3 + lk)*2 + li] = W_s[lw, r]
            
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    # Total rank
    new_rank = len(U_list)
    U_new = np.array(U_list).T
    V_new = np.array(V_list).T
    W_new = np.array(W_list).T
    
    # Verify
    if _verify_exact(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
        best_U, best_V, best_W = U_new.tolist(), V_new.tolist(), W_new.tolist()
        best_rank = new_rank
    
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
