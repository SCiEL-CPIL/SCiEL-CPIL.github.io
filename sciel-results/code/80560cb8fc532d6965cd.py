import numpy as np
import time

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Target tensor T where T[i*m+j, j*p+k, k*n+i] = 1
    # i: 0..n-1 (2), j: 0..m-1 (2), k: 0..p-1 (3)
    # U index: i*m + j (0..3)
    # V index: j*p + k (0..5)
    # W index: k*n + i (0..5)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # Fallback: Schoolbook Rank 12
    # We will construct a Rank 11 solution using Strassen for the 2x2 block (cols 0,1) 
    # and Schoolbook for the 3rd column (col 2).
    
    # Strassen decomposition for 2x2 matrix multiplication:
    # M = A * B
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # 
    # Strassen uses 7 products:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # M11 = m1 + m4 - m5 + m7
    # M12 = m3 + m5
    # M21 = m2 + m4
    # M22 = m1 - m2 + m3 + m6
    
    # Our problem: A is 2x2, B is 2x3.
    # Let A = [[a, b], [c, d]] (indices i=0,1; j=0,1)
    # Let B = [[e, f, g], [h, i, j]] (indices j=0,1; k=0,1,2)
    # Note: The task uses indices i,j,k for A(i,j), B(j,k), C(i,k).
    # A is 2x2: indices (0,0), (0,1), (1,0), (1,1)
    # B is 2x3: indices (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    # C is 2x3: indices (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    
    # We split B into B1 (2x2, cols 0,1) and B2 (2x1, col 2).
    # B1 = [[B00, B01], [B10, B11]] corresponding to B[0,0], B[0,1], B[1,0], B[1,1]
    # B2 = [[B02], [B12]] corresponding to B[0,2], B[1,2]
    
    # C1 = A * B1 (2x2) -> Use Strassen (Rank 7)
    # C2 = A * B2 (2x1) -> Use Schoolbook (Rank 4)
    # Total Rank = 7 + 4 = 11.
    
    # Let's map the variables to the tensor indices.
    # U corresponds to A (2x2). Rows: i*m+j.
    # V corresponds to B (2x3). Rows: j*p+k.
    # W corresponds to C (2x3). Rows: k*n+i.
    
    # Strassen part (for C1, k=0,1):
    # We need to express C1[i,k] as linear combinations of the 7 products.
    # The 7 products are of the form u_r[i,j] * v_r[j,k].
    # We need to define U_strassen (4x7), V_strassen (6x7) such that
    # sum_r U_strassen[i,j] * V_strassen[j,k] = C1[i,k] for k=0,1.
    # And W_strassen (6x7) such that sum_r W_strassen[k*n+i] * 1 = 1? 
    # No, the decomposition is T = sum_r U_r \otimes V_r \otimes W_r.
    # For the Strassen part, the "output" W vector for a specific term r is 
    # the coefficient vector for the corresponding C1 entry.
    # Since C1 is 2x2, each Strassen term contributes to specific (i,k) pairs.
    
    # Let's define the 7 Strassen terms explicitly.
    # A = [[a, b], [c, d]]
    # B1 = [[e, f], [h, i]] (using e,f for B00,B01 and h,i for B10,B11 to avoid confusion with index i)
    # Let's rename B1 entries: B100=B[0,0], B101=B[0,1], B110=B[1,0], B111=B[1,1].
    
    # m1 = (a+d)(B100+B111)
    # m2 = (c+d)B100
    # m3 = a(B101-B111)
    # m4 = d(B110-B100)
    # m5 = (a+b)B111
    # m6 = (c-a)(B100+B101)
    # m7 = (b-d)(B110+B111)
    
    # C100 = m1 + m4 - m5 + m7
    # C101 = m3 + m5
    # C110 = m2 + m4
    # C111 = m1 - m2 + m3 + m6
    
    # We need to construct U, V, W matrices of width 11.
    # Cols 0-6: Strassen terms.
    # Cols 7-10: Schoolbook terms for C2.
    
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    # --- Strassen Part (Cols 0-6) ---
    
    # Term 0: m1 = (a+d)(B100+B111)
    # U: a=1, d=1 -> U[0,0]=1, U[2,0]=1 (indices: 0=0*2+0, 2=1*2+0? No. U index is i*m+j. 
    # i=0,j=0 -> 0. i=0,j=1 -> 1. i=1,j=0 -> 2. i=1,j=1 -> 3.
    # a=A[0,0] -> idx 0. b=A[0,1] -> idx 1. c=A[1,0] -> idx 2. d=A[1,1] -> idx 3.
    U[0, 0] = 1; U[3, 0] = 1
    # V: B100=B[0,0] idx 0. B111=B[1,1] idx 1*3+1=4.
    V[0, 0] = 1; V[4, 0] = 1
    # W: Coefficients for C100, C101, C110, C111.
    # C100 coeff 1 -> W[k*n+i]. k=0,i=0 -> 0*2+0=0.
    # C101 coeff 0.
    # C110 coeff 0.
    # C111 coeff 1 -> k=1,i=1 -> 1*2+1=3.
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 1: m2 = (c+d)B100
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C110 coeff 1 -> k=1,i=0 -> 1*2+0=2.
    W[2, 1] = 1
    
    # Term 2: m3 = a(B101-B111)
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    # C101 coeff 1 -> k=0,i=1 -> 0*2+1=1.
    # C111 coeff 1 -> k=1,i=1 -> 3.
    W[1, 2] = 1; W[3, 2] = 1
    
    # Term 3: m4 = d(B110-B100)
    U[3, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1  # B110 is B[1,0] idx 3. B100 is B[0,0] idx 0.
    # C100 coeff 1 -> W[0,3]=1.
    # C110 coeff 1 -> W[2,3]=1.
    W[0, 3] = 1; W[2, 3] = 1
    
    # Term 4: m5 = (a+b)B111
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    # C100 coeff -1 -> W[0,4]=-1.
    # C101 coeff 1 -> W[1,4]=1.
    W[0, 4] = -1; W[1, 4] = 1
    
    # Term 5: m6 = (c-a)(B100+B101)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C111 coeff 1 -> W[3,5]=1.
    W[3, 5] = 1
    
    # Term 6: m7 = (b-d)(B110+B111)
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    # C100 coeff 1 -> W[0,6]=1.
    W[0, 6] = 1
    
    # --- Schoolbook Part for C2 (Cols 7-10) ---
    # C2 = A * B2
    # B2 = [[B02], [B12]] -> B[0,2] idx 2, B[1,2] idx 5.
    # C20 = a*B02 + b*B12
    # C21 = c*B02 + d*B12
    
    # Term 7: a * B02
    U[0, 7] = 1
    V[2, 7] = 1
    # C20 -> k=2, i=0 -> 2*2+0=4.
    W[4, 7] = 1
    
    # Term 8: b * B12
    U[1, 8] = 1
    V[5, 8] = 1
    # C20 -> k=2, i=0 -> 4.
    W[4, 8] = 1
    
    # Term 9: c * B02
    U[2, 9] = 1
    V[2, 9] = 1
    # C21 -> k=2, i=1 -> 2*2+1=5.
    W[5, 9] = 1
    
    # Term 10: d * B12
    U[3, 10] = 1
    V[5, 10] = 1
    # C21 -> k=2, i=1 -> 5.
    W[5, 10] = 1
    
    # Verify the solution
    # T_calc = sum_r U[:,r] \otimes V[:,r] \otimes W[:,r]
    T_calc = np.zeros((4, 6, 6))
    for r in range(11):
        T_calc += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    if not np.allclose(T_calc, T_target, atol=1e-9):
        # Fallback to Schoolbook Rank 12 if construction failed (should not happen)
        U = np.zeros((4, 12))
        V = np.zeros((6, 12))
        W = np.zeros((6, 12))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    else:
        # It is Rank 11
        pass
        
    # Time check (should be instant)
    if time.time() > deadline:
        # Return whatever we have, but this won't happen
        pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
