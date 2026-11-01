import numpy as np

def construct():
    n, m, p = 2, 2, 6
    rank = 21
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen definitions for 2x2x2
    # P1: (A00+A11)(B00+B11) -> M00 += 1, M11 += 1
    # P2: (A10+A11)B00 -> M00 -= 1
    # P3: A00(B01-B11) -> M01 += 1, M11 -= 1
    # P4: A11(B10-B00) -> M00 += 1
    # P5: (A00+A01)B11 -> M10 += 1, M11 += 1
    # P6: (A10-A00)(B00+B01) -> M10 += 1, M11 -= 1
    # P7: (A01-A11)(B10+B11) -> M00 -= 1, M01 += 1
    
    # We need to map these to the 2x2x6 tensor.
    # The tensor is T[i, j, k] = delta[i, j] ? No, T is the multiplication tensor.
    # T[i, j, k] = 1 if ...? No.
    # The decomposition is T = sum c_r (u_r \otimes v_r \otimes w_r).
    # T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r].
    # For matrix mult, T[i, j, k] is 1 if i is row of A, j is col of A/row of B, k is col of B?
    # No, standard tensor for matrix mult C = AB:
    # C_{i, k} = sum_j A_{i, j} B_{j, k}.
    # The tensor T has indices i, j, k.
    # T_{i, j, k} = 1 if the term A_{i, j} B_{j, k} contributes to C_{i, k}.
    # Actually, the tensor is defined such that <T, A \otimes B> = C.
    # So T_{i, j, k} = 1.
    
    # Let's verify the Strassen decomposition of the 2x2x2 tensor.
    # The 2x2x2 tensor has 8 ones.
    # Sum of 7 rank 1 tensors must reconstruct these 8 ones.
    
    # Let's define the 7 terms for one block.
    # Block b corresponds to k indices 2*b, 2*b+1.
    
    terms = []
    
    # Helper to add a term
    # u: vector of length 4 (A indices)
    # v: vector of length 12 (B indices)
    # w: vector of length 12 (C indices)
    
    def add_term(u, v, w):
        idx = len(terms)
        U[:, idx] = u
        V[:, idx] = v
        W[:, idx] = w
        terms.append((u, v, w))

    # Block 0: k=0, 1
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: j=0,1; k=0,1 -> 0, 1, 6, 7
    # C indices: k=0,1; i=0,1 -> 0, 1, 2, 3
    
    # P1: (A00+A11)(B00+B11)
    u = np.array([1, 0, 0, 1])
    v = np.zeros(12); v[0] = 1; v[6] = 1
    w = np.zeros(12); w[0] = 1; w[3] = 1
    add_term(u, v, w)
    
    # P2: (A10+A11)B00
    u = np.array([0, 0, 1, 1])
    v = np.zeros(12); v[0] = 1
    w = np.zeros(12); w[0] = -1
    add_term(u, v, w)
    
    # P3: A00(B01-B11)
    u = np.array([1, 0, 0, 0])
    v = np.zeros(12); v[1] = 1; v[7] = -1
    w = np.zeros(12); w[1] = 1; w[3] = -1
    add_term(u, v, w)
    
    # P4: A11(B10-B00)
    u = np.array([0, 0, 0, 1])
    v = np.zeros(12); v[6] = 1; v[0] = -1
    w = np.zeros(12); w[0] = 1
    add_term(u, v, w)
    
    # P5: (A00+A01)B11
    u = np.array([1, 1, 0, 0])
    v = np.zeros(12); v[7] = 1
    w = np.zeros(12); w[2] = 1; w[3] = 1
    add_term(u, v, w)
    
    # P6: (A10-A00)(B00+B01)
    u = np.array([-1, 0, 1, 0])
    v = np.zeros(12); v[0] = 1; v[1] = 1
    w = np.zeros(12); w[2] = 1; w[3] = -1
    add_term(u, v, w)
    
    # P7: (A01-A11)(B10+B11)
    u = np.array([0, 1, 0, -1])
    v = np.zeros(12); v[6] = 1; v[7] = 1
    w = np.zeros(12); w[0] = -1; w[1] = 1
    add_term(u, v, w)
    
    # Block 1: k=2, 3
    # B indices: j=0,1; k=2,3 -> 2, 3, 8, 9
    # C indices: k=2,3; i=0,1 -> 4, 5, 6, 7
    
    # P1
    u = np.array([1, 0, 0, 1])
    v = np.zeros(12); v[2] = 1; v[8] = 1
    w = np.zeros(12); w[4] = 1; w[7] = 1
    add_term(u, v, w)
    
    # P2
    u = np.array([0, 0, 1, 1])
    v = np.zeros(12); v[2] = 1
    w = np.zeros(12); w[4] = -1
    add_term(u, v, w)
    
    # P3
    u = np.array([1, 0, 0, 0])
    v = np.zeros(12); v[3] = 1; v[9] = -1
    w = np.zeros(12); w[5] = 1; w[7] = -1
    add_term(u, v, w)
    
    # P4
    u = np.array([0, 0, 0, 1])
    v = np.zeros(12); v[8] = 1; v[2] = -1
    w = np.zeros(12); w[4] = 1
    add_term(u, v, w)
    
    # P5
    u = np.array([1, 1, 0, 0])
    v = np.zeros(12); v[9] = 1
    w = np.zeros(12); w[6] = 1; w[7] = 1
    add_term(u, v, w)
    
    # P6
    u = np.array([-1, 0, 1, 0])
    v = np.zeros(12); v[2] = 1; v[3] = 1
    w = np.zeros(12); w[6] = 1; w[7] = -1
    add_term(u, v, w)
    
    # P7
    u = np.array([0, 1, 0, -1])
    v = np.zeros(12); v[8] = 1; v[9] = 1
    w = np.zeros(12); w[4] = -1; w[5] = 1
    add_term(u, v, w)
    
    # Block 2: k=4, 5
    # B indices: j=0,1; k=4,5 -> 4, 5, 10, 11
    # C indices: k=4,5; i=0,1 -> 8, 9, 10, 11
    
    # P1
    u = np.array([1, 0, 0, 1])
    v = np.zeros(12); v[4] = 1; v[10] = 1
    w = np.zeros(12); w[8] = 1; w[11] = 1
    add_term(u, v, w)
    
    # P2
    u = np.array([0, 0, 1, 1])
    v = np.zeros(12); v[4] = 1
    w = np.zeros(12); w[8] = -1
    add_term(u, v, w)
    
    # P3
    u = np.array([1, 0, 0, 0])
    v = np.zeros(12); v[5] = 1; v[11] = -1
    w = np.zeros(12); w[9] = 1; w[11] = -1
    add_term(u, v, w)
    
    # P4
    u = np.array([0, 0, 0, 1])
    v = np.zeros(12); v[10] = 1; v[4] = -1
    w = np.zeros(12); w[8] = 1
    add_term(u, v, w)
    
    # P5
    u = np.array([1, 1, 0, 0])
    v = np.zeros(12); v[11] = 1
    w = np.zeros(12); w[10] = 1; w[11] = 1
    add_term(u, v, w)
    
    # P6
    u = np.array([-1, 0, 1, 0])
    v = np.zeros(12); v[4] = 1; v[5] = 1
    w = np.zeros(12); w[10] = 1; w[11] = -1
    add_term(u, v, w)
    
    # P7
    u = np.array([0, 1, 0, -1])
    v = np.zeros(12); v[10] = 1; v[11] = 1
    w = np.zeros(12); w[8] = -1; w[9] = 1
    add_term(u, v, w)
    
    # Convert to lists of integers/half-integers if needed, but they are integers.
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
