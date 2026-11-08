import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s kill

    n, m, p = DIMENSIONS  # A: n x m, B: m x p, C: n x p

    def _elapsed_ok():
        return (time.time() - start) < deadline

    # Schoolbook for (n,m,p): rank = n*m*p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        t = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][t] = 1
                    V[j * p + k][t] = 1
                    W[k * n + i][t] = 1
                    t += 1
        return U, V, W

    # Strassen for 2x2 * 2x2: rank 7
    def strassen_222():
        # A indices: 0=A11, 1=A12, 2=A21, 3=A22
        # B indices: 0=B11, 1=B12, 2=B21, 3=B22
        # C indices: 0=C11, 1=C12, 2=C21, 3=C22
        # Standard Strassen terms:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to represent these as sums of rank-1 terms.
        # Each "product" in Strassen is a product of linear forms.
        # To decompose exactly into rank-1 terms with integer coefficients,
        # we can expand each product if it involves sums, or keep it if it's simple.
        # However, the standard Strassen algorithm defines 7 scalar multiplications.
        # The tensor rank of 2x2x2 multiplication is 7.
        # We need to map these 7 multiplications to the U, V, W matrices.
        
        # Let's define the 7 terms explicitly.
        # Term 0: M1 = (A11+A22)(B11+B22)
        #   U: A11=1, A22=1 -> U[0][0]=1, U[3][0]=1
        #   V: B11=1, B22=1 -> V[0][0]=1, V[3][0]=1
        #   W: C11=1, C22=1 -> W[0][0]=1, W[3][0]=1
        #   Wait, this is a product of sums. It contributes to C11 and C22.
        #   C11 += M1, C22 += M1.
        #   So W[0][0] += 1, W[3][0] += 1.
        
        # Term 1: M2 = (A21+A22)B11
        #   U: A21=1, A22=1 -> U[2][1]=1, U[3][1]=1
        #   V: B11=1 -> V[0][1]=1
        #   W: C21=1, C22=-1 -> W[2][1]=1, W[3][1]=-1
        
        # Term 2: M3 = A11(B21-B22)
        #   U: A11=1 -> U[0][2]=1
        #   V: B21=1, B22=-1 -> V[2][2]=1, V[3][2]=-1
        #   W: C12=1, C22=1 -> W[1][2]=1, W[3][2]=1
        
        # Term 3: M4 = A22(B12-B11)
        #   U: A22=1 -> U[3][3]=1
        #   V: B12=1, B11=-1 -> V[1][3]=1, V[0][3]=-1
        #   W: C11=1, C21=1 -> W[0][3]=1, W[2][3]=1
        
        # Term 4: M5 = (A11+A12)B22
        #   U: A11=1, A12=1 -> U[0][4]=1, U[1][4]=1
        #   V: B22=1 -> V[3][4]=1
        #   W: C11=-1, C12=1 -> W[0][4]=-1, W[1][4]=1
        
        # Term 5: M6 = (A21-A11)(B11+B12)
        #   U: A21=1, A11=-1 -> U[2][5]=1, U[0][5]=-1
        #   V: B11=1, B12=1 -> V[0][5]=1, V[1][5]=1
        #   W: C22=1 -> W[3][5]=1
        
        # Term 6: M7 = (A12-A22)(B21+B22)
        #   U: A12=1, A22=-1 -> U[1][6]=1, U[3][6]=-1
        #   V: B21=1, B22=1 -> V[2][6]=1, V[3][6]=1
        #   W: C11=1 -> W[0][6]=1
        
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        
        # T0
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # T1
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # T2
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # T3
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # T4
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # T5
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # T6
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W

    def make_221():
        # 2x2 * 2x1. Rank 4 (schoolbook) or less?
        # C1 = A11 B11 + A12 B21
        # C2 = A21 B11 + A22 B21
        # Can we do it in 3?
        # Strassen-like for 2x2x1?
        # Actually, rank of 2x2x1 is 4. (It's the identity tensor I_2 x I_2 x I_1? No, it's just two separate dot products).
        # Wait, C1 depends on A row 1, C2 on A row 2. They are independent.
        # So rank is sum of ranks? No, they share B.
        # C1 = <A1, B1>, C2 = <A2, B1>.
        # This is equivalent to computing A @ B where A is 2x2, B is 2x1.
        # The tensor is T[i, j, k] = delta_{i,k} for i in 0..1, j in 0..1, k in 0..1? No.
        # Indices: U (2x2), V (2x1), W (1x2).
        # T[i, j, k] = 1 if i==k?
        # C_k = sum_j A_{k,j} B_{j,k}? No, B is 2x1, so B_{j,0}.
        # C_k = sum_j A_{k,j} B_{j,0}.
        # So T[k, j, 0] = A_{k,j} B_{j,0}.
        # The tensor has non-zero entries at (k, j, 0) for all k,j.
        # This is the identity matrix I_2 tensor I_2? No.
        # It's the tensor that maps A, B to C.
        # It is known that rank of 2x2x1 is 4.
        return schoolbook(2, 2, 1)

    def make_223():
        # 2x2 * 2x3. Rank?
        # Schoolbook: 2*2*3 = 12.
        # Can we do better?
        # Split B into B1 (2x1) and B2 (2x2).
        # C1 = A B1 (2x1), C2 = A B2 (2x2).
        # Rank(2x2x1) = 4. Rank(2x2x2) = 7.
        # Total rank = 4 + 7 = 11.
        # Is it possible to do better than 11?
        # Known lower bounds for 2x2x3 is 11?
        # Actually, rank of 2x2xn is known.
        # For 2x2x3, rank is 11.
        # Let's construct it.
        U1, V1, W1 = make_221() # Rank 4
        U2, V2, W2 = strassen_222() # Rank 7
        
        # Concatenate
        U = []
        for i in range(4):
            U.append(U1[i] + U2[i])
        V = []
        for i in range(2):
            V.append(V1[i] + [0]*7)
        for i in range(4):
            V.append([0]*4 + V2[i])
        W = []
        for i in range(2):
            W.append(W1[i] + [0]*7)
        for i in range(4):
            W.append([0]*4 + W2[i])
        return U, V, W

    def make_224():
        # 2x2 * 2x4. Rank?
        # Split B into two 2x2 blocks.
        # Rank = 7 + 7 = 14.
        U1, V1, W1 = strassen_222()
        U2, V2, W2 = strassen_222()
        
        U = []
        for i in range(4):
            U.append(U1[i] + U2[i])
        V = []
        for i in range(4):
            V.append(V1[i] + [0]*7)
        for i in range(4):
            V.append([0]*7 + V2[i])
        W = []
        for i in range(4):
            W.append(W1[i] + [0]*7)
        for i in range(4):
            W.append([0]*7 + W2[i])
        return U, V, W

    def make_227():
        # 2x2 * 2x7. Rank?
        # Split B into 2x2, 2x2, 2x2, 2x1? No, 7 = 2+2+2+1.
        # Or 2x3 + 2x4?
        # Rank(2x2x3) = 11. Rank(2x2x4) = 14. Total 25.
        # Or 2x2 + 2x2 + 2x2 + 2x1 = 7+7+7+4 = 25.
        # Let's use 2x3 + 2x4.
        U1, V1, W1 = make_224() # Rank 14
        U2, V2, W2 = make_223() # Rank 11
        
        U = []
        for i in range(4):
            U.append(U1[i] + U2[i])
        V = []
        for i in range(8):
            V.append(V1[i] + [0]*11)
        for i in range(6):
            V.append([0]*14 + V2[i])
        W = []
        for i in range(8):
            W.append(W1[i] + [0]*11)
        for i in range(6):
            W.append([0]*14 + W2[i])
        return U, V, W

    def make_247():
        # 2x4 * 4x7.
        # Split A into two 2x2 blocks (rows 0-1, rows 2-3? No, A is 2x4).
        # A is 2x4. Split into A1 (2x2) and A2 (2x2).
        # B is 4x7. Split into B1 (2x7) and B2 (2x7).
        # C = A1 B1 + A2 B2.
        # Rank(A1 B1) = Rank(2x2 * 2x7) = 25.
        # Rank(A2 B2) = Rank(2x2 * 2x7) = 25.
        # Total Rank = 50.
        
        U1, V1, W1 = make_227()
        U2, V2, W2 = make_227()
        
        U = []
        for i in range(4):
            U.append(U1[i] + [0]*25)
        for i in range(4):
            U.append([0]*25 + U2[i])
        V = []
        for i in range(14):
            V.append(V1[i] + [0]*25)
        for i in range(14):
            V.append([0]*25 + V2[i])
        W = []
        for i in range(14):
            W.append(W1[i] + [0]*25)
        for i in range(14):
            W.append([0]*25 + W2[i])
        return U, V, W

    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    try:
        if not _elapsed_ok():
            raise TimeoutError()

        # Construct the rank 50 solution
        U, V, W = make_247()
        rank = len(U[0])

        # Verify
        if verify(U, V, W, rank):
            return {"U": U, "V": V, "W": W}
        else:
            # Fallback to schoolbook
            U, V, W = schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}

    except Exception as e:
        # Fall back to schoolbook
        U, V, W = schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
