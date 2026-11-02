import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 135.0  # Safety margin for 150s kill limit

    n, m, p = 2, 2, 10
    nm, mp, pn = n * m, m * p, p * n  # 4, 20, 20

    def build_strassen_rank35():
        rank = 35
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        
        # Strassen 7-term decomposition for 2x2x2
        # The indices for U, V, W in a single 2x2 block are:
        # U: 4 entries (i, j) -> i*2+j
        # V: 4 entries (j, k) -> j*2+k
        # W: 4 entries (k, i) -> k*2+i (Trace convention: W[k*n+i])
        
        # Standard Strassen matrices for A (2x2) and B (2x2)
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B10 - B11)
        # M4 = A11 * (B01 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # C00 = M1 - M4 + M5 + M7
        # C01 = M3 + M4
        # C10 = M1 + M2 - M5 - M7
        # C11 = M2 + M3 - M6 + M7
        
        # We map these to the tensor decomposition vectors.
        # Each term r corresponds to one of the 7 Strassen products.
        # We repeat this for each of the 5 blocks.
        
        strassen_terms = [
            # (U_vec, V_vec, W_vec)
            # M1: (A00+A11)(B00+B11) -> C00, C10
            # U: [1, 0, 0, 1] (A00, A01, A10, A11)
            # V: [1, 0, 0, 1] (B00, B01, B10, B11)
            # W: [1, 0, 0, 1] (C00, C01, C10, C11) ? 
            # Wait, C = [[C00, C01], [C10, C11]]
            # W index is k*n + i.
            # W[0] = C00 (k=0, i=0)
            # W[1] = C10 (k=0, i=1)
            # W[2] = C01 (k=1, i=0)
            # W[3] = C11 (k=1, i=1)
            
            # Let's re-verify the W mapping.
            # W[k*n + i]
            # k=0, i=0 -> W[0] = C[0,0]
            # k=0, i=1 -> W[1] = C[1,0]
            # k=1, i=0 -> W[2] = C[0,1]
            # k=1, i=1 -> W[3] = C[1,1]
            
            # So W vector is [C00, C10, C01, C11]
            
            # M1 contributes to C00 and C10.
            # C00 += M1
            # C10 += M1
            # So W contribution is [1, 1, 0, 0]
            
            # M2: (A10+A11)B00 -> C10, C11
            # U: [0, 0, 1, 1]
            # V: [1, 0, 0, 0]
            # C10 += M2, C11 += M2
            # W: [0, 1, 0, 1]
            
            # M3: A00(B10-B11) -> C01, C11
            # U: [1, 0, 0, 0]
            # V: [0, -1, 1, 0] (B10 - B11) -> V indices: B10 is idx 2, B11 is idx 3.
            # V: [0, 0, 1, -1]
            # C01 += M3, C11 += M3
            # W: [0, 0, 1, 1]
            
            # M4: A11(B01-B00) -> C00, C01
            # U: [0, 0, 0, 1]
            # V: [-1, 1, 0, 0] (B01 - B00) -> V indices: B00 idx 0, B01 idx 1.
            # V: [-1, 1, 0, 0]
            # C00 -= M4, C01 += M4
            # W: [-1, 0, 1, 0]
            
            # M5: (A00+A01)B11 -> C00, C10
            # U: [1, 1, 0, 0]
            # V: [0, 0, 0, 1]
            # C00 += M5, C10 -= M5
            # W: [1, -1, 0, 0]
            
            # M6: (A10-A00)(B00+B01) -> C11
            # U: [-1, 0, 1, 0]
            # V: [1, 1, 0, 0]
            # C11 -= M6
            # W: [0, 0, 0, -1]
            
            # M7: (A01-A11)(B10+B11) -> C00, C10
            # U: [0, 1, 0, -1]
            # V: [0, 0, 1, 1]
            # C00 += M7, C10 -= M7
            # W: [1, -1, 0, 0]
            
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 0, 0]),   # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 1]),   # M2
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]),  # M3
            ([0, 0, 0, 1], [-1, 1, 0, 0], [-1, 0, 1, 0]), # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [1, -1, 0, 0]),  # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, -1]), # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, -1, 0, 0]), # M7
        ]
        
        col = 0
        for block in range(5):
            for a, b, c in strassen_terms:
                # U: A indices [i*m+j]
                for i in range(nm):
                    U[i, col] = a[i]
                
                # V: B indices [j*p+k]
                # Block covers columns 2*block and 2*block+1
                # V index: j*p + k
                # j=0, k=2*block -> 0*p + 2*block
                # j=0, k=2*block+1 -> 0*p + 2*block+1
                # j=1, k=2*block -> 1*p + 2*block
                # j=1, k=2*block+1 -> 1*p + 2*block+1
                
                V[0*p + 2*block, col] = b[0]    # b[0] is B00
                V[0*p + 2*block+1, col] = b[1]  # b[1] is B01
                V[1*p + 2*block, col] = b[2]    # b[2] is B10
                V[1*p + 2*block+1, col] = b[3]  # b[3] is B11
                
                # W: C indices [k*n+i]
                # W index: k*n + i
                # k=2*block, i=0 -> (2*block)*2 + 0
                # k=2*block, i=1 -> (2*block)*2 + 1
                # k=2*block+1, i=0 -> (2*block+1)*2 + 0
                # k=2*block+1, i=1 -> (2*block+1)*2 + 1
                
                W[(2*block)*2 + 0, col] = c[0]  # c[0] is C00
                W[(2*block)*2 + 1, col] = c[1]  # c[1] is C10
                W[(2*block+1)*2 + 0, col] = c[2] # c[2] is C01
                W[(2*block+1)*2 + 1, col] = c[3] # c[3] is C11
                
                col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W, tol=1e-9):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rank = Ua.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    total = 0.0
                    for r in range(rank):
                        total += Ua[idx_u, r] * Va[idx_v, r] * Wa[idx_w, r]
                    expected = 1.0 if j == k else 0.0
                    if abs(total - expected) > tol:
                        return False
        return True

    best = build_strassen_rank35()
    
    # Verify
    if not verify(best["U"], best["V"], best["W"]):
        # Fallback to schoolbook if Strassen construction is flawed (should not happen)
        rank = 40
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if j == k:
                        U[i*m+j, col] = 1.0
                        V[j*p+k, col] = 1.0
                        W[k*n+i, col] = 1.0
                        col += 1
        best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return best
