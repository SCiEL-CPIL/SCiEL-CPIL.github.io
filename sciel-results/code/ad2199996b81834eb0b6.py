import time
import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 50
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_222(i0, j0, k0, r_start):
        # i0, j0, k0 are top-left indices
        # r_start is the column index in U, V, W
        
        # M1 = (a00 + a11)(b00 + b11)
        # C00 += M1, C11 += M1
        U[i0*m + j0, r_start] += 1
        U[(i0+1)*m + (j0+1), r_start] += 1
        V[j0*p + k0, r_start] += 1
        V[(j0+1)*p + (k0+1), r_start] += 1
        W[k0*n + i0, r_start] += 1
        W[(k0+1)*n + (i0+1), r_start] += 1
        
        # M2 = (a10 + a11)b00
        # C10 += M2, C11 -= M2
        U[(i0+1)*m + j0, r_start+1] += 1
        U[(i0+1)*m + (j0+1), r_start+1] += 1
        V[j0*p + k0, r_start+1] += 1
        W[k0*n + (i0+1), r_start+1] += 1
        W[(k0+1)*n + (i0+1), r_start+1] += -1
        
        # M3 = a00(b01 - b11)
        # C01 += M3, C11 += M3
        U[i0*m + j0, r_start+2] += 1
        V[j0*p + (k0+1), r_start+2] += 1
        V[(j0+1)*p + (k0+1), r_start+2] += -1
        W[k0*n + (i0+1), r_start+2] += 1  # C01 is i=0, k=1? No.
        
        # Correction on indices for C:
        # C[i, k].
        # C01 means i=0, k=1. Index in W: k*n + i = 1*n + 0 = n.
        # C11 means i=1, k=1. Index in W: k*n + i = 1*n + 1 = n+1.
        # C10 means i=1, k=0. Index in W: k*n + i = 0*n + 1 = 1.
        # C00 means i=0, k=0. Index in W: k*n + i = 0*n + 0 = 0.
        
        # Let's re-evaluate W indices for M3.
        # M3 contributes to C01 and C11.
        # C01: i=0, k=1. W index: 1*n + 0 = n.
        # C11: i=1, k=1. W index: 1*n + 1 = n+1.
        
        # My previous M1 W indices:
        # C00: i=0, k=0. W index 0.
        # C11: i=1, k=1. W index n+1.
        
        # My previous M2 W indices:
        # C10: i=1, k=0. W index 1.
        # C11: i=1, k=1. W index n+1.
        
        # Let's fix the W indices in the helper.
        
    # Re-implementing with correct W indices
    
    def add_strassen_222_correct(i0, j0, k0, r_start):
        n_local = 2
        # W index mapping: w_idx = k_local * n + i_local
        # C00: k=0, i=0 -> 0
        # C01: k=1, i=0 -> n
        # C10: k=0, i=1 -> 1
        # C11: k=1, i=1 -> n+1
        
        # M1: C00 (+1), C11 (+1)
        U[i0*m + j0, r_start] += 1
        U[(i0+1)*m + (j0+1), r_start] += 1
        V[j0*p + k0, r_start] += 1
        V[(j0+1)*p + (k0+1), r_start] += 1
        W[k0*n + i0, r_start] += 1
        W[(k0+1)*n + (i0+1), r_start] += 1
        
        # M2: C10 (+1), C11 (-1)
        U[(i0+1)*m + j0, r_start+1] += 1
        U[(i0+1)*m + (j0+1), r_start+1] += 1
        V[j0*p + k0, r_start+1] += 1
        W[k0*n + (i0+1), r_start+1] += 1
        W[(k0+1)*n + (i0+1), r_start+1] += -1
        
        # M3: C01 (+1), C11 (+1)
        U[i0*m + j0, r_start+2] += 1
        V[j0*p + (k0+1), r_start+2] += 1
        V[(j0+1)*p + (k0+1), r_start+2] += -1
        W[(k0+1)*n + i0, r_start+2] += 1
        W[(k0+1)*n + (i0+1), r_start+2] += 1
        
        # M4: C00 (+1), C10 (+1)
        U[(i0+1)*m + (j0+1), r_start+3] += 1
        V[(j0+1)*p + k0, r_start+3] += 1
        V[j0*p + k0, r_start+3] += -1
        W[k0*n + i0, r_start+3] += 1
        W[k0*n + (i0+1), r_start+3] += 1
        
        # M5: C00 (-1), C01 (+1)
        U[i0*m + j0, r_start+4] += 1
        U[i0*m + (j0+1), r_start+4] += 1
        V[(j0+1)*p + (k0+1), r_start+4] += 1
        W[k0*n + i0, r_start+4] += -1
        W[(k0+1)*n + i0, r_start+4] += 1
        
        # M6: C11 (+1)
        # Wait, M6 also contributes to others which cancel?
        # Standard Strassen:
        # M6 = (a10 - a00)(b00 + b01)
        # Terms: a10b00 (C10), a10b01 (C11), a00b00 (C00), a00b01 (C01)
        # Formula: C11 = M1 - M2 + M3 + M6.
        # So M6 must provide the net contribution to C11 after cancellations.
        # Actually, it's safer to just use the linear forms and let the sum work out.
        # But we need specific W values.
        # Let's check M6 contribution to C11.
        # a10b01 is in C11. Coeff +1.
        # a10b00 is in C10. Coeff +1.
        # a00b00 is in C00. Coeff -1.
        # a00b01 is in C01. Coeff -1.
        
        # So M6 should have:
        # W[C11] += 1
        # W[C10] += 1
        # W[C00] -= 1
        # W[C01] -= 1
        
        # Let's verify if this is consistent with other factors.
        # C10 = M2 + M4.
        # M2 has +1 at C10.
        # M4 has +1 at C10.
        # If M6 has +1 at C10, total C10 = M2 + M4 + M6?
        # But formula is C10 = M2 + M4.
        # So M6 must NOT contribute to C10 in the final sum, or M2/M4 coefficients are different.
        
        # Let's re-read Strassen.
        # M6 = (a10 - a00)(b00 + b01)
        # Expansion: a10b00 + a10b01 - a00b00 - a00b01
        # C10 = sum a1j b j0.
        # a10b00 is in C10.
        # a11b10 is in C10.
        # M2 = (a10+a11)b00 = a10b00 + a11b00.
        # a11b00 is NOT in C10. It is in C10? No, b00 is col 0. a11 is row 1.
        # C10 = a10b00 + a11b10.
        # So a11b00 is 0 in C10.
        
        # This means M2 contains a term a11b00 which is 0 in C10.
        # So M2's contribution to C10 is effectively a10b00.
        # But M2 is a single rank term.
        # The decomposition is exact.
        # The sum of products $\sum U V W$ must equal 1 for the existing terms and 0 for non-existing.
        
        # I will rely on the standard Strassen U,V,W definitions.
        
        U[(i0+1)*m + j0, r_start+5] += 1
        U[i0*m + j0, r_start+5] += -1
        V[j0*p + k0, r_start+5] += 1
        V[j0*p + (k0+1), r_start+5] += 1
        W[(k0+1)*n + (i0+1), r_start+5] += 1
        W[k0*n + (i0+1), r_start+5] += 1
        W[k0*n + i0, r_start+5] += -1
        W[(k0+1)*n + i0, r_start+5] += -1
        
        # M7: C00 (+1)
        # M7 = (a01 - a11)(b10 + b11)
        # a01b10 (C01), a01b11 (C00), a11b10 (C10), a11b11 (C11)
        # Formula: C00 = M1 + M4 - M5 + M7.
        # So M7 contributes +1 to C00.
        # Other terms cancel?
        
        U[i0*m + (j0+1), r_start+6] += 1
        U[(i0+1)*m + (j0+1), r_start+6] += -1
        V[(j0+1)*p + k0, r_start+6] += 1
        V[(j0+1)*p + (k0+1), r_start+6] += 1
        W[k0*n + i0, r_start+6] += 1
        W[(k0+1)*n + i0, r_start+6] += 1
        W[k0*n + (i0+1), r_start+6] += -1
        W[(k0+1)*n + (i0+1), r_start+6] += -1

    def add_trivial_221(i0, j0, k0, r_start):
        # C[i, k] = A[i, 0]B[0, k] + A[i, 1]B[1, k]
        # k is fixed (size 1).
        # C00: A00B00 + A01B10
        # C10: A10B00 + A11B10
        
        # 1. A00B00 -> C00
        U[i0*m + j0, r_start] += 1
        V[j0*p + k0, r_start] += 1
        W[k0*n + i0, r_start] += 1
        
        # 2. A01B10 -> C00
        U[i0*m + (j0+1), r_start+1] += 1
        V[(j0+1)*p + k0, r_start+1] += 1
        W[k0*n + i0, r_start+1] += 1
        
        # 3. A10B00 -> C10
        U[(i0+1)*m + j0, r_start+2] += 1
        V[j0*p + k0, r_start+2] += 1
        W[k0*n + (i0+1), r_start+2] += 1
        
        # 4. A11B10 -> C10
        U[(i0+1)*m + (j0+1), r_start+3] += 1
        V[(j0+1)*p + k0, r_start+3] += 1
        W[k0*n + (i0+1), r_start+3] += 1

    r = 0
    for j0 in [0, 2]:
        # k=0, size 2
        add_strassen_222_correct(0, j0, 0, r)
        r += 7
        # k=2, size 2
        add_strassen_222_correct(0, j0, 2, r)
        r += 7
        # k=4, size 2
        add_strassen_222_correct(0, j0, 4, r)
        r += 7
        # k=6, size 1
        add_trivial_221(0, j0, 6, r)
        r += 4
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    # Verification
    n, m, p = DIMENSIONS
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Reconstruct tensor
    T = np.zeros((n*m, m*p, p*n))
    for r in range(U.shape[1]):
        T += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    # Check against expected
    Expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i*m + j
                b_idx = j*p + k
                c_idx = k*n + i
                Expected[a_idx, b_idx, c_idx] = 1
                
    if np.allclose(T, Expected):
        print("Valid decomposition")
    else:
        print("Invalid decomposition")
        print("Diff:", np.sum(T - Expected))
