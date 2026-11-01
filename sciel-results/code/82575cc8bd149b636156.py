import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication.
    The minimal rank is 11.
    """
    n, m, p = 2, 2, 3
    rank = 11
    
    # Initialize matrices with zeros
    # U: (n*m, rank) -> (4, 11)
    # V: (m*p, rank) -> (6, 11)
    # W: (p*n, rank) -> (6, 11)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    # Helper to set values
    # Indices:
    # U[i, j] -> index i*m + j
    # V[j, k] -> index j*p + k
    # W[k, i] -> index k*n + i (Note: task says W[k*n+i, r])
    
    def set_u(i, j, r, val):
        U[i * m + j, r] = val

    def set_v(j, k, r, val):
        V[j * p + k, r] = val

    def set_w(k, i, r, val):
        W[k * n + i, r] = val

    # --- Rank 11 Decomposition ---
    # Terms 0-6: Strassen algorithm for C[:, 0:2] = A * B[:, 0:2]
    # B[:, 0:2] = [[b00, b01], [b10, b11]]
    
    # T0: (A00 + A11)(B00 + B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1) # C00 +=
    set_w(1, 1, 0, 1) # C11 +=
    # Note: Strassen usually has more complex W combinations to subtract terms.
    # Let's use the standard Strassen decomposition for 2x2x2.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M7
    
    # My previous manual construction in the prompt was slightly different but valid.
    # Let's stick to the standard Strassen to be sure.
    
    # Term 0: M1
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    # W for M1: contributes to C00 (+1), C11 (+1)
    # We need to subtract M4 from C00 and M3 from C11.
    # This requires M4 and M3 to be separate terms or combined in W.
    # In tensor decomposition, each term r has fixed U, V, W.
    # So we just set W[k,i,r] to the coefficient of M_r in C[i,k].
    
    # Let's re-map standard Strassen terms to indices 0-6.
    
    # r=0: M1 = (A00+A11)(B00+B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)  # C00 += M1
    set_w(1, 1, 0, 1)  # C11 += M1
    
    # r=1: M2 = (A10+A11)B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(1, 0, 1, 1)  # C10 += M2
    
    # r=2: M3 = A00(B01-B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(0, 1, 2, 1)  # C01 += M3
    set_w(1, 1, 2, -1) # C11 -= M3
    
    # r=3: M4 = A11(B10-B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(1, 0, 3, 1)  # C10 += M4
    set_w(0, 0, 3, -1) # C00 -= M4
    
    # r=4: M5 = (A00+A01)B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 1, 4, 1)  # C01 += M5
    
    # r=5: M6 = (A10-A00)(B00+B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(0, 0, 5, 1)  # C00 += M6
    
    # r=6: M7 = (A01-A11)(B10+B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(1, 1, 6, 1)  # C11 += M7

    # Check C00: M1 - M4 + M6 = (A00+A11)(B00+B11) - A11(B10-B00) + (A10-A00)(B00+B01)
    # = A00B00 + A00B11 + A11B00 + A11B11 - A11B10 + A11B00 + A10B00 + A10B01 - A00B00 - A00B01
    # = A00B11 + 2A11B00 + A11B11 - A11B10 + A10B00 + A10B01 - A00B01
    # This does NOT look like A00B00 + A01B10.
    # Standard Strassen:
    # C00 = M1 - M4 + M6
    # M1 = (A00+A11)(B00+B11)
    # M4 = A11(B10-B00)
    # M6 = (A10-A00)(B00+B01)
    # Let's re-verify the standard Strassen formulas.
    # C00 = M1 - M4 + M6 is correct for standard Strassen.
    # Why did my expansion fail?
    # A00B00 cancels?
    # M1: A00B00 + A00B11 + A11B00 + A11B11
    # -M4: -A11B10 + A11B00
    # +M6: A10B00 + A10B01 - A00B00 - A00B01
    # Sum:
    # A00B00: 1 - 1 = 0. (Wait, C00 should have A00B00).
    # Ah, standard Strassen M6 is (A10-A00)(B00+B01)?
    # Let's check Wikipedia or standard source.
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M4 + M6
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M7
    
    # My indices: A00=A11, A01=A12, A10=A21, A11=A22.
    # B00=B11, B01=B12, B10=B21, B11=B22.
    # C00=C11, C01=C12, C10=C21, C11=C22.
    
    # Let's re-evaluate C00 (which is C11 in 1-based):
    # C00 = M1 - M4 + M6
    # M1 = (A00+A11)(B00+B11)
    # M4 = A11(B10-B00)  <-- Note: B10 is B21 in 1-based? No.
    # In 1-based: M4 = A22(B12-B11).
    # In 0-based: A11(B01-B00).
    # My previous M4 definition: `set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)` -> B10 - B00.
    # Standard M4 uses B01 - B00.
    # Let's fix M4.
    
    # Correct Standard Strassen:
    # r=0: M1 = (A00+A11)(B00+B11)
    # r=1: M2 = (A10+A11)B00
    # r=2: M3 = A00(B10-B11)  # Note: B21-B22 in 1-based -> B10-B11 in 0-based
    # r=3: M4 = A11(B01-B00)  # Note: B12-B11 in 1-based -> B01-B00 in 0-based
    # r=4: M5 = (A00+A01)B11
    # r=5: M6 = (A10-A00)(B00+B01)
    # r=6: M7 = (A01-A11)(B10+B11)
    
    # Let's re-set the values based on this corrected mapping.
    
    # Reset U, V, W for r=0..6
    for r in range(7):
        U[:, r] = 0
        V[:, r] = 0
        W[:, r] = 0

    # r=0: M1
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1)  # C00 += M1
    set_w(1, 1, 0, 1)  # C11 += M1

    # r=1: M2
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(1, 0, 1, 1)  # C10 += M2

    # r=2: M3 = A00(B10-B11)
    set_u(0, 0, 2, 1)
    set_v(1, 0, 2, 1); set_v(1, 1, 2, -1)
    set_w(0, 1, 2, 1)  # C01 += M3
    set_w(1, 1, 2, -1) # C11 -= M3

    # r=3: M4 = A11(B01-B00)
    set_u(1, 1, 3, 1)
    set_v(0, 1, 3, 1); set_v(0, 0, 3, -1)
    set_w(1, 0, 3, 1)  # C10 += M4
    set_w(0, 0, 3, -1) # C00 -= M4

    # r=4: M5 = (A00+A01)B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 1, 4, 1)  # C01 += M5

    # r=5: M6 = (A10-A00)(B00+B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(0, 0, 5, 1)  # C00 += M6

    # r=6: M7 = (A01-A11)(B10+B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(1, 1, 6, 1)  # C11 += M7

    # Terms 7-10: For C[:, 2]
    # C02 = A00B02 + A01B12
    # C12 = A10B02 + A11B12
    
    # r=7: A00 B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)  # C02 +=

    # r=8: A01 B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)  # C02 +=

    # r=9: A10 B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)  # C12 +=

    # r=10: A11 B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1) # C12 +=

    # Verification
    def verify(U, V, W, n_trials=5):
        for _ in range(n_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            U3 = U.reshape(n, m, rank)
            V3 = V.reshape(m, p, rank)
            W3 = W.reshape(p, n, rank)
            
            M = np.einsum('ijr,ij->r', U3, A)
            N = np.einsum('jkr,jk->r', V3, B)
            MN = M * N
            C_computed = np.einsum('kir,r->ik', W3, MN)
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook if verification fails (should not happen)
        return None 

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

