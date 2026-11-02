import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    t0 = time.time()
    deadline = t0 + 105
    
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # A: 2x2, B: 2x10, C: 2x10
    # U: (4, R), V: (20, R), W: (20, R)
    # C[k*2 + i] = sum_j sum_l A[i,j] B[j,l] C_out[l,k]? 
    # Wait, standard MM: C[i,k] = sum_j A[i,j] B[j,k]
    # Here C is 2x10. i in 0..1, k in 0..9.
    # Convention: W[k*n+i, r]. So W index is k*2+i.
    
    def build_strassen_35():
        U = np.zeros((4, 35))
        V = np.zeros((20, 35))
        W = np.zeros((20, 35))
        
        # Strassen terms for 2x2 * 2x2
        # A indices: 00, 01, 10, 11
        # B indices: 00, 01, 10, 11
        # C indices: 00, 01, 10, 11
        # Terms: (U_vec, V_vec, W_vec)
        terms = [
            # M1 = (A00+A11)(B00+B11)
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            # M2 = (A10+A11)B00
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
            # M3 = A00(B10-B11)
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 0, 1]),
            # M4 = A11(B00-B01)
            ([0, 0, 0, 1], [1, -1, 0, 0], [0, 1, 0, 0]),
            # M5 = (A00+A01)B11
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]),
            # M6 = (A10-A00)(B00+B01)
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0]),
            # M7 = (A01-A11)(B10+B11)
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])
        ]
        
        col = 0
        for q in range(5):
            # q corresponds to pair of columns 2q, 2q+1 in B and C
            for a, b, c in terms:
                # U
                U[0, col] = a[0]
                U[1, col] = a[1]
                U[2, col] = a[2]
                U[3, col] = a[3]
                
                # V
                # B indices: j*10 + k. j in 0,1. k in 2q, 2q+1.
                # b = [B00, B01, B10, B11] local
                # Local 00 -> global 0, 2q
                # Local 01 -> global 0, 2q+1
                # Local 10 -> global 1, 2q
                # Local 11 -> global 1, 2q+1
                V[0, col] = 0 # Just to be safe, though we overwrite
                V[10, col] = 0
                V[0*10 + 2*q, col] = b[0]
                V[0*10 + 2*q+1, col] = b[1]
                V[1*10 + 2*q, col] = b[2]
                V[1*10 + 2*q+1, col] = b[3]
                
                # W
                # W indices: k*2 + i. i in 0,1. k in 2q, 2q+1.
                # c = [C00, C01, C10, C11] local? 
                # Let's check standard Strassen output mapping.
                # C00 = M1 - M4 + M6 - M7
                # C01 = M5 + M7
                # C10 = M2 + M4
                # C11 = M1 - M3 + M2 + M5
                # My terms above:
                # T1: c=[1,0,0,1] -> C00+=1, C11+=1
                # T2: c=[0,0,1,1] -> C10+=1, C11+=1
                # T3: c=[0,0,0,1] -> C11+=1
                # T4: c=[0,1,0,0] -> C01+=1
                # T5: c=[0,0,0,1] -> C11+=1
                # T6: c=[0,1,0,0] -> C01+=1
                # T7: c=[1,0,0,0] -> C00+=1
                
                # Summing up:
                # C00: T1(1) + T7(1) = 2? 
                # Wait, standard Strassen:
                # M1 = (A00+A11)(B00+B11)
                # M2 = (A10+A11)B00
                # M3 = A00(B10-B11)
                # M4 = A11(B00-B01)
                # M5 = (A00+A01)B11
                # M6 = (A10-A00)(B00+B01)
                # M7 = (A01-A11)(B10+B11)
                # C00 = M1 - M4 + M6 - M7
                # C01 = M5 + M7
                # C10 = M2 + M4
                # C11 = M1 - M3 + M2 + M5
                
                # Let's check my vectors against this.
                # T1: M1. c should be [1, 0, 0, 1] (C00, C01, C10, C11). Correct.
                # T2: M2. c should be [0, 0, 1, 1]. Correct.
                # T3: M3. c should be [0, 0, 0, -1]. My vector was [0, 0, 0, 1] with A[0,0]=1, B[10]=1, B[11]=-1.
                #    M3 = A00(B10-B11). Term: A00 * (B10 - B11).
                #    If I use A_vec=[1,0,0,0], B_vec=[0,0,1,-1], product is A00(B10-B11).
                #    This contributes to C11 with a - sign in the formula C11 = ... - M3 ...
                #    So the coefficient in W for C11 should be -1.
                #    My current T3 c_vec is [0,0,0,1]. This would add +M3 to C11.
                #    So I need to flip the sign of B_vec or c_vec.
                #    Let's adjust T3: A=[1,0,0,0], B=[0,0,-1,1], c=[0,0,0,1].
                #    Then product is A00(B11-B10) = -M3. Adding this to C11 gives -M3. Correct.
                #    Or keep B=[0,0,1,-1] and set c=[0,0,0,-1].
                
                # T4: M4. A11(B00-B01).
                #    Formula C10 = M2 + M4. C01 = ... - M4 ... (in C00 = M1 - M4 ...).
                #    Wait, C00 = M1 - M4 + M6 - M7.
                #    So M4 contributes to C10 (+1) and C00 (-1).
                #    My T4 c_vec was [0,1,0,0] (C01). This is wrong. It should affect C00 and C10.
                #    Let's re-verify standard Strassen assignments.
                #    C00 = M1 - M4 + M6 - M7
                #    C01 = M5 + M7
                #    C10 = M2 + M4
                #    C11 = M1 - M3 + M2 + M5
                
                #    So:
                #    M1: C00=1, C01=0, C10=0, C11=1. Vec: [1, 0, 0, 1]
                #    M2: C00=0, C01=0, C10=1, C11=1. Vec: [0, 0, 1, 1]
                #    M3: C00=0, C01=0, C10=0, C11=-1. Vec: [0, 0, 0, -1]
                #    M4: C00=-1, C01=0, C10=1, C11=0. Vec: [-1, 0, 1, 0]
                #    M5: C00=0, C01=1, C10=0, C11=1. Vec: [0, 1, 0, 1]
                #    M6: C00=1, C01=0, C10=0, C11=0. Vec: [1, 0, 0, 0]
                #    M7: C00=-1, C01=1, C10=0, C11=0. Vec: [-1, 1, 0, 0]
                
                #    Let's rebuild the terms with these correct W vectors.
                #    And ensure A and B vectors match the definitions.
                #    M1 = (A00+A11)(B00+B11). A=[1,0,0,1], B=[1,0,0,1]
                #    M2 = (A10+A11)B00. A=[0,0,1,1], B=[1,0,0,0]
                #    M3 = A00(B10-B11). A=[1,0,0,0], B=[0,0,1,-1]
                #    M4 = A11(B00-B01). A=[0,0,0,1], B=[1,-1,0,0]
                #    M5 = (A00+A01)B11. A=[1,1,0,0], B=[0,0,0,1]
                #    M6 = (A10-A00)(B00+B01). A=[-1,0,1,0], B=[1,1,0,0]
                #    M7 = (A01-A11)(B10+B11). A=[0,1,0,-1], B=[0,0,1,1]
                
                pass # Re-initialize inside the loop to be safe, or just define outside.
        
        # Let's define the corrected terms outside the loop for clarity
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 1]),
            ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 0, -1]),
            ([0, 0, 0, 1], [1, -1, 0, 0], [-1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 1, 0, 0])
        ]
        
        col = 0
        for q in range(5):
            for a, b, c in terms:
                U[0, col] = a[0]
                U[1, col] = a[1]
                U[2, col] = a[2]
                U[3, col] = a[3]
                
                V[0*10 + 2*q, col] = b[0]
                V[0*10 + 2*q+1, col] = b[1]
                V[1*10 + 2*q, col] = b[2]
                V[1*10 + 2*q+1, col] = b[3]
                
                # W indices: k*2 + i
                # c = [C00, C01, C10, C11] corresponding to local k=0, i=0; k=0, i=1; k=1, i=0; k=1, i=1
                # Global k = 2q + local_k. Global i = local_i.
                # W index = (2q + lk)*2 + li = 4q + 2*lk + li
                # c[0] (lk=0, li=0) -> 4q
                # c[1] (lk=0, li=1) -> 4q + 1
                # c[2] (lk=1, li=0) -> 4q + 2
                # c[3] (lk=1, li=1) -> 4q + 3
                
                W[4*q + 0, col] = c[0]
                W[4*q + 1, col] = c[1]
                W[4*q + 2, col] = c[2]
                W[4*q + 3, col] = c[3]
                
                col += 1
                
        return U, V, W

    U, V, W = build_strassen_35()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
