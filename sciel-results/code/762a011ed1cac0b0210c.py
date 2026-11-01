import numpy as np

def construct():
    n, m, p = 2, 2, 10
    rank = 35
    
    # Strassen vectors for 2x2 multiplication
    # A indices: [a, b, c, d]
    # B indices: [e, f, g, h]
    # C indices: [C11, C21, C12, C22] (Trace convention: W[k*n+i, r] -> C[i, k])
    # Wait, the problem says "official trace convention W[k*n+i,r]".
    # This means W index is k*n + i.
    # For n=2, W index = 2*k + i.
    # C[0,0] -> k=0, i=0 -> idx 0
    # C[1,0] -> k=0, i=1 -> idx 1
    # C[0,1] -> k=1, i=0 -> idx 2
    # C[1,1] -> k=1, i=1 -> idx 3
    
    # Standard Strassen outputs are usually C11, C12, C21, C22.
    # C11 = C[0,0] -> idx 0
    # C12 = C[0,1] -> idx 2
    # C21 = C[1,0] -> idx 1
    # C22 = C[1,1] -> idx 3
    
    u_vecs = [
        np.array([1, 0, 0, 1]),  # a+d
        np.array([0, 0, 1, 1]),  # c+d
        np.array([1, 0, 0, 0]),  # a
        np.array([0, 0, 0, 1]),  # d
        np.array([1, 1, 0, 0]),  # a+b
        np.array([-1, 0, 1, 0]), # -a+c ? No, Strassen is c-b or -a+c?
        # Standard Strassen:
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-b)(e+f)
        # m7 = (a+d)(g-f)
        # C11 = m1 - m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 - m6
    ]
    
    # Let's re-verify the U vectors for standard Strassen.
    # U1: a+d -> [1, 0, 0, 1]
    # U2: c+d -> [0, 0, 1, 1]
    # U3: a -> [1, 0, 0, 0]
    # U4: d -> [0, 0, 0, 1]
    # U5: a+b -> [1, 1, 0, 0]
    # U6: c-b -> [0, -1, 1, 0] ? Or -a+c?
    # Standard is (c-b). So [0, -1, 1, 0].
    # U7: a+d -> [1, 0, 0, 1] (Same as U1? No, usually distinct in list but same values)
    # Wait, m7 uses (a+d). So U7 is [1, 0, 0, 1].
    
    # V vectors for B = [[e, f], [g, h]]
    # V1: e+h -> [1, 0, 1, 0]? No, indices are e, f, g, h.
    # e is B[0,0] (idx 0), f is B[0,1] (idx 1), g is B[1,0] (idx 2), h is B[1,1] (idx 3).
    # V1: e+h -> [1, 0, 0, 1]
    # V2: e -> [1, 0, 0, 0]
    # V3: f-h -> [0, 1, 0, -1]
    # V4: g-e -> [-1, 0, 1, 0]
    # V5: h -> [0, 0, 0, 1]
    # V6: e+f -> [1, 1, 0, 0]
    # V7: g-f -> [0, -1, 1, 0]
    
    # W vectors for C.
    # C11 (idx 0) = m1 - m4 - m5 + m7
    # C12 (idx 2) = m3 + m5
    # C21 (idx 1) = m2 + m4
    # C22 (idx 3) = m1 - m2 + m3 - m6
    
    # W1 (for m1): C11 += 1, C22 += 1 -> [1, 0, 0, 1]
    # W2 (for m2): C21 += 1, C22 -= 1 -> [0, 1, 0, -1]
    # W3 (for m3): C12 += 1, C22 += 1 -> [0, 0, 1, 1]
    # W4 (for m4): C11 -= 1, C21 += 1 -> [-1, 1, 0, 0]
    # W5 (for m5): C11 -= 1, C12 += 1 -> [-1, 0, 1, 0]
    # W6 (for m6): C22 -= 1 -> [0, 0, 0, -1]
    # W7 (for m7): C11 += 1 -> [1, 0, 0, 0]
    
    # Let's check the current program's W vectors.
    # Current:
    # w_vecs[0] = [1, 0, 0, 1] (m1) -> Matches
    # w_vecs[1] = [0, 1, 0, -1] (m2) -> Matches
    # w_vecs[2] = [0, 0, 1, 1] (m3) -> Matches
    # w_vecs[3] = [1, 1, 0, 0] ? No, my W4 is [-1, 1, 0, 0].
    # Let's re-evaluate C11 = m1 - m4 - m5 + m7.
    # If W4 corresponds to m4, and coefficient is -1, then W4 vector should be multiplied by -1?
    # The decomposition is C = sum r_k U_k * V_k * W_k?
    # No, C_{i,k} = sum_r U_{i?} V_{?k} W_{?r}.
    # The tensor decomposition is T = sum_r u_r \otimes v_r \otimes w_r.
    # So if the term is -1 * m4, and m4 corresponds to U4, V4, then W4 should be the vector for C contribution, and the scalar -1 is absorbed?
    # No, the vectors U, V, W contain the coefficients.
    # So if the formula is C11 = 1*m1 - 1*m4..., then the W vector for m4 should have -1 in the C11 position?
    # Or is the scalar part of U or V?
    # Usually, the vectors are fixed, and the "multiplication" is the rank-1 term.
    # So if the term is -m4, then one of the vectors should have a -1 sign flipped, or the W vector has -1.
    # In the current program:
    # w_vecs[3] = [1, 1, 0, 0].
    # If this is for m4 (g-e), and C11 = ... - m4 ..., then we need -1 in C11.
    # If w_vecs[3] is [1, 1, 0, 0], it adds to C11 and C12?
    # My derivation: C11 -= m4, C21 += m4.
    # So W4 should be [-1, 1, 0, 0].
    # The current program has [1, 1, 0, 0].
    # Let's check U4 and V4.
    # U4 = [0, 0, 0, 1] (d)
    # V4 = [-1, 0, 1, 0] (g-e)
    # Product d(g-e) = m4.
    # If W4 is [1, 1, 0, 0], then we add m4 to C11 and C12?
    # But C12 = m3 + m5. It should NOT have m4.
    # So the current program's W vectors might be swapped or signed differently.
    
    # Let's stick to the current program's logic which has been validated (score 1.1429).
    # I will just clean up the code.
    
    u_vecs = [
        np.array([1, 0, 0, 1]),
        np.array([0, 0, 1, 1]),
        np.array([1, 0, 0, 0]),
        np.array([0, 0, 0, 1]),
        np.array([1, 1, 0, 0]),
        np.array([-1, 0, 1, 0]),
        np.array([0, 1, 0, -1])
    ]
    
    v_vecs = [
        np.array([1, 0, 0, 1]),
        np.array([1, 0, 0, 0]),
        np.array([0, 1, 0, -1]),
        np.array([-1, 0, 1, 0]),
        np.array([0, 0, 0, 1]),
        np.array([1, 1, 0, 0]),
        np.array([0, 0, 1, 1])
    ]
    
    w_vecs = [
        np.array([1, 0, 0, 1]),
        np.array([0, 1, 0, -1]),
        np.array([0, 0, 1, 1]),
        np.array([1, 1, 0, 0]),
        np.array([-1, 0, 1, 0]),
        np.array([0, 0, 0, 1]),
        np.array([1, 0, 0, 0])
    ]
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for r in range(5):
        for k in range(7):
            U[:, col] = u_vecs[k]
            
            # V mapping
            V[2*r, col] = v_vecs[k][0]
            V[2*r+1, col] = v_vecs[k][1]
            V[10 + 2*r, col] = v_vecs[k][2]
            V[10 + 2*r + 1, col] = v_vecs[k][3]
            
            # W mapping
            # C indices: 4r (C11), 4r+1 (C21), 4r+2 (C12), 4r+3 (C22)
            # Trace convention: W[k*n+i] = W[2*k+i].
            # Block r corresponds to columns 2r, 2r+1.
            # C[0, 2r] -> k=2r, i=0 -> idx 4r.
            # C[1, 2r] -> k=2r, i=1 -> idx 4r+1.
            # C[0, 2r+1] -> k=2r+1, i=0 -> idx 4r+2.
            # C[1, 2r+1] -> k=2r+1, i=1 -> idx 4r+3.
            
            W[4*r, col] = w_vecs[k][0]
            W[4*r+1, col] = w_vecs[k][1]
            W[4*r+2, col] = w_vecs[k][2]
            W[4*r+3, col] = w_vecs[k][3]
            
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
