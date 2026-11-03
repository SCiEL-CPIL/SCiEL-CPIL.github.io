import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Try rank 16 first, then rank 17, then fallback to schoolbook rank 18
    for rank in [16, 17]:
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        # We'll use a systematic approach:
        # For 2x3x3, we can use Strassen-like decomposition for the 2x2x2 core
        # and handle the extra column/row with additional terms.
        
        # Let's build a rank-17 decomposition using Strassen for 2x2x2 + 10 schoolbook terms
        # U indices: i*m+j where i in {0,1}, j in {0,1,2}
        # V indices: j*p+k where j in {0,1,2}, k in {0,1,2}
        # W indices: k*n+i where k in {0,1,2}, i in {0,1}
        
        if rank == 17:
            # Strassen terms for the 2x2x2 block (i in 0,1; j in 0,1; k in 0,1)
            # Term 0: M1 = (A00+A11)(B00+B11)
            U[0, 0] = 1; U[4, 0] = 1
            V[0, 0] = 1; V[4, 0] = 1
            W[0, 0] = 1; W[3, 0] = 1
            
            # Term 1: M2 = (A10+A11)B00
            U[3, 1] = 1; U[4, 1] = 1
            V[0, 1] = 1
            W[1, 1] = 1; W[3, 1] = -1
            
            # Term 2: M3 = A00(B10-B11)
            U[0, 2] = 1
            V[3, 2] = 1; V[4, 2] = -1
            W[2, 2] = 1; W[3, 2] = 1
            
            # Term 3: M4 = A11(B01-B00)
            U[4, 3] = 1
            V[1, 3] = 1; V[0, 3] = -1
            W[2, 3] = 1; W[0, 3] = -1
            
            # Term 4: M5 = (A00+A01)B11
            U[0, 4] = 1; U[1, 4] = 1
            V[4, 4] = 1
            W[0, 4] = 1
            
            # Term 5: M6 = (A10-A00)(B00+B10)
            U[3, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[3, 5] = 1
            W[3, 5] = 1
            
            # Term 6: M7 = (A01-A11)(B10+B11)
            U[1, 6] = 1; U[4, 6] = -1
            V[3, 6] = 1; V[4, 6] = 1
            W[1, 6] = 1
            
            # Schoolbook terms for remaining products (7 terms)
            # We need to cover: A02*B00, A02*B01, A02*B02, A02*B10, A02*B11, A02*B12,
            #                   A12*B00, A12*B01, A12*B02, A12*B10, A12*B11, A12*B12
            # But some may be covered by Strassen terms. Let me enumerate all 18 products:
            # C[i,k] = sum_j A[i,j]*B[j,k]
            # i in {0,1}, j in {0,1,2}, k in {0,1,2}
            
            # Products not covered by Strassen (which covers j in {0,1}, k in {0,1}):
            # A02*B00 -> C00, A02*B01 -> C01, A02*B02 -> C02
            # A02*B10 -> C00, A02*B11 -> C01, A02*B12 -> C02
            # A12*B00 -> C10, A12*B01 -> C11, A12*B02 -> C12
            # A12*B10 -> C10, A12*B11 -> C11, A12*B12 -> C12
            
            # That's 12 terms, but Strassen uses 7 terms to cover 4 products (with cancellation)
            # So we need 12 more terms? That would be 19 total. Let me reconsider.
            
            # Actually, Strassen covers the 2x2x2 sub-tensor (i in 0,1; j in 0,1; k in 0,1)
            # The remaining terms are those with j=2 or k=2.
            # j=2: A02*B00, A02*B01, A02*B02, A02*B10, A02*B11, A02*B12 (6 terms)
            # k=2: A00*B02, A00*B12, A01*B02, A01*B12, A10*B02, A10*B12, A11*B02, A11*B12 (8 terms)
            # But A02*B02 is counted once, so total = 6 + 8 = 14? No, let me list all 18:
            # (i,j,k) pairs:
            # i=0: (0,0,0),(0,0,1),(0,0,2),(0,1,0),(0,1,1),(0,1,2),(0,2,0),(0,2,1),(0,2,2)
            # i=1: (1,0,0),(1,0,1),(1,0,2),(1,1,0),(1,1,1),(1,1,2),(1,2,0),(1,2,1),(1,2,2)
            
            # Strassen covers: (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)
            # Remaining: (0,0,2),(0,1,2),(0,2,0),(0,2,1),(0,2,2),(1,0,2),(1,1,2),(1,2,0),(1,2,1),(1,2,2)
            # That's 10 terms. So 7 + 10 = 17. Good.
            
            # Term 7: A00*B02 -> C02
            U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
            # Term 8: A01*B02 -> C02
            U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
            # Term 9: A02*B00 -> C00
            U[2, 9] = 1; V[0, 9] = 1; W[0, 9] = 1
            # Term 10: A02*B01 -> C01
            U[2, 10] = 1; V[1, 10] = 1; W[2, 10] = 1
            # Term 11: A02*B02 -> C02
            U[2, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
            # Term 12: A02*B10 -> C00
            U[2, 12] = 1; V[3, 12] = 1; W[0, 12] = 1
            # Term 13: A02*B11 -> C01
            U[2, 13] = 1; V[4, 13] = 1; W[2, 13] = 1
            # Term 14: A02*B12 -> C02
            U[2, 14] = 1; V[5, 14] = 1; W[4, 14] = 1
            # Term 15: A10*B02 -> C12
            U[3, 15] = 1; V[2, 15] = 1; W[5, 15] = 1
            # Term 16: A11*B02 -> C12
            U[4, 16] = 1; V[5, 16] = 1; W[5, 16] = 1
            # Wait, I'm missing A12 terms. Let me recount.
            
            # Remaining after Strassen:
            # (0,0,2): A00*B02 -> C02
            # (0,1,2): A01*B02 -> C02
            # (0,2,0): A02*B00 -> C00
            # (0,2,1): A02*B01 -> C01
            # (0,2,2): A02*B02 -> C02
            # (1,0,2): A10*B02 -> C12
            # (1,1,2): A11*B02 -> C12
            # (1,2,0): A12*B00 -> C10
            # (1,2,1): A12*B01 -> C11
            # (1,2,2): A12*B02 -> C12
            
            # That's 10 terms. I had 10 terms (7-16) but I missed A12 terms. Let me fix:
            # I'll redo terms 7-16:
            
            # Reset terms 7-16
            U[0:6, 7:17] = 0
            V[0:9, 7:17] = 0
            W[0:6, 7:17] = 0
            
            # Term 7: A00*B02 -> C02 (W index: k=2, i=0 -> 2*2+0=4)
            U[0, 7] = 1; V[2, 7] = 1; W[4, 7] = 1
            # Term 8: A01*B02 -> C02
            U[1, 8] = 1; V[5, 8] = 1; W[4, 8] = 1
            # Term 9: A02*B00 -> C00 (W: k=0, i=0 -> 0)
            U[2, 9] = 1; V[0, 9] = 1; W[0, 9] = 1
            # Term 10: A02*B01 -> C01 (W: k=1, i=0 -> 2)
            U[2, 10] = 1; V[1, 10] = 1; W[2, 10] = 1
            # Term 11: A02*B02 -> C02 (W: k=2, i=0 -> 4)
            U[2, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
            # Term 12: A02*B10 -> C00
            U[2, 12] = 1; V[3, 12] = 1; W[0, 12] = 1
            # Term 13: A02*B11 -> C01
            U[2, 13] = 1; V[4, 13] = 1; W[2, 13] = 1
            # Term 14: A02*B12 -> C02
            U[2, 14] = 1; V[5, 14] = 1; W[4, 14] = 1
            # Term 15: A10*B02 -> C12 (W: k=2, i=1 -> 5)
            U[3, 15] = 1; V[2, 15] = 1; W[5, 15] = 1
            # Term 16: A11*B02 -> C12
            U[4, 16] = 1; V[5, 16] = 1; W[5, 16] = 1
            
            # Wait, I'm still missing A12 terms. I have 10 slots (7-16) but need to cover:
            # A00*B02, A01*B02, A02*B00, A02*B01, A02*B02, A02*B10, A02*B11, A02*B12,
            # A10*B02, A11*B02, A12*B00, A12*B01, A12*B02
            # That's 13 terms! I can't fit them in 10 slots.
            
            # Let me reconsider. Maybe I can combine some terms or use a different decomposition.
            # For now, let me just verify if the current 17-term setup works.
            
            def verify(U, V, W):
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                            error += (val - 1.0) ** 2
                return error
            
            if verify(U, V, W) < 1e-9:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if time.time() > deadline - 5:
            break
    
    # Fallback to schoolbook rank 18
    U_sb = np.zeros((n * m, n * m * p), dtype=int)
    V_sb = np.zeros((m * p, n * m * p), dtype=int)
    W_sb = np.zeros((p * n, n * m * p), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_sb[u_idx, idx] = 1
                V_sb[v_idx, idx] = 1
                W_sb[w_idx, idx] = 1
                idx += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
