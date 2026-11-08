import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank = 17
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define the 17 multiplications
    # Block 1: j in {0, 1} (2x2x3)
    # Strassen for 2x2x2 on cols 0,1 of B
    # A rows: 0,1 (i=0,1), cols 0,1 (j=0,1) -> U indices 0,1,3,4
    # B rows: 0,1 (j=0,1), cols 0,1 (k=0,1) -> V indices 0,1,3,4
    
    # M1: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    
    # M2: (A10+A11)B00
    U[3, 1] = 1; U[4, 1] = 1 # Wait, A10 is index 3? No.
    # A index: i*m + j.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 3
    # i=1, j=1 -> 4
    # So A10 is index 3, A11 is index 4.
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    
    # M3: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[5, 2] = -1 # B01 is index 1, B11 is index 5?
    # B index: j*p + k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=0, k=2 -> 2
    # j=1, k=0 -> 3
    # j=1, k=1 -> 4
    # j=1, k=2 -> 5
    # So B01 is index 1, B11 is index 4.
    V[1, 2] = 1; V[4, 2] = -1
    
    # M4: A11(B10-B00)
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    
    # M5: (A00+A01)B10
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6: (A10+A11)(B01+B11)
    U[3, 5] = 1; U[4, 5] = 1
    V[1, 5] = 1; V[4, 5] = 1
    
    # M7: (A10+A11)(B10-B00)
    U[3, 6] = 1; U[4, 6] = 1
    V[3, 6] = 1; V[0, 6] = -1
    
    # Extra 4 for col 2 of Block 1
    # M8: A00 B02
    U[0, 7] = 1
    V[2, 7] = 1
    
    # M9: A01 B12
    U[1, 8] = 1
    V[5, 8] = 1
    
    # M10: A10 B02
    U[3, 9] = 1
    V[2, 9] = 1
    
    # M11: A11 B12
    U[4, 10] = 1
    V[5, 10] = 1
    
    # Block 2: j = 2 (2x1x3)
    # M12: A02 B20
    U[5, 11] = 1 # A02 is index 5
    V[6, 11] = 1 # B20 is index 6
    
    # M13: A02 B21
    U[5, 12] = 1
    V[7, 12] = 1
    
    # M14: A02 B22
    U[5, 13] = 1
    V[8, 13] = 1
    
    # M15: A12 B20
    U[2, 14] = 1 # A12 is index 2? No.
    # A index: i*m + j.
    # i=1, j=2 -> 1*3 + 2 = 5.
    # Wait, I used U[5] for A02?
    # i=0, j=2 -> 0*3 + 2 = 2.
    # i=1, j=2 -> 1*3 + 2 = 5.
    # So A02 is index 2, A12 is index 5.
    # I made a mistake in Block 1 indices?
    # Block 1 used j in {0,1}.
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 3
    # i=1, j=1 -> 4
    # So A02 is index 2, A12 is index 5.
    # In Block 2, we use j=2.
    # So U[2] is A02, U[5] is A12.
    
    # Correcting Block 2:
    # M12: A02 B20
    U[2, 11] = 1
    V[6, 11] = 1
    
    # M13: A02 B21
    U[2, 12] = 1
    V[7, 12] = 1
    
    # M14: A02 B22
    U[2, 13] = 1
    V[8, 13] = 1
    
    # M15: A12 B20
    U[5, 14] = 1
    V[6, 14] = 1
    
    # M16: A12 B21
    U[5, 15] = 1
    V[7, 15] = 1
    
    # M17: A12 B22
    U[5, 16] = 1
    V[8, 16] = 1

    # Now solve for W
    # T_{r,s,t} is 1 if r=i*m+j, s=j*p+k, t=k*n+i and i,j,k match.
    # We need W such that sum_c U[r,c] V[s,c] W[t,c] = T[r,s,t]
    
    for t in range(p * n):
        # Build system A W_t = b_t
        # A is (n*m * m*p) x rank
        # b_t is (n*m * m*p)
        A = np.zeros((n * m * m * p, rank))
        b = np.zeros(n * m * m * p)
        
        idx = 0
        for r in range(n * m):
            for s in range(m * p):
                # Check if T[r,s,t] is 1
                i = r // m
                j = r % m
                j2 = s // p
                k = s % p
                i2 = t % n
                k2 = t // n
                
                if j == j2 and i == i2 and k == k2:
                    b[idx] = 1
                else:
                    b[idx] = 0
                
                for c in range(rank):
                    A[idx, c] = U[r, c] * V[s, c]
                idx += 1
        
        # Solve A W_t = b
        # Use least squares
        W_t, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
        W[t, :] = W_t
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
