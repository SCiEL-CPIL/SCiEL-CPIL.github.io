import time
import random
import sys

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _construct_rank_26():
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Indices for U (A): i*m+j
    # A00=0, A01=1, A02=2, A10=3, A11=4, A12=5, A20=6, A21=7, A22=8
    # Indices for V (B): j*p+k
    # B00=0, B01=1, B02=2, B10=3, B11=4, B12=5, B20=6, B21=7, B22=8
    # Indices for W (C): k*n+i
    # C00=0, C10=1, C20=2, C01=3, C11=4, C21=5, C02=6, C12=7, C22=8
    # Note: W[k*n+i] means row k, col i in the output matrix C?
    # Task says: "output factor follows the official trace convention W[k*n+i,r]"
    # And verify uses W[k*n+i].
    # In standard matrix multiplication C[i][k] = sum A[i][j] B[j][k].
    # The tensor is T[i,j,k] = delta[j,?]. No, T is the multiplication tensor.
    # T[i,j,k] = 1 if the term A[i,j]B[j,k] contributes to C[i,k].
    # So T[i,j,k] is 1 for all i,j,k? No.
    # The tensor for matrix multiplication is defined such that
    # C[i,k] = sum_{j} A[i,j] B[j,k].
    # So the tensor T has indices i (row of A), j (col of A / row of B), k (col of B).
    # T[i,j,k] = 1.
    # The decomposition is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T[i,j,k].
    # Wait, the verify function checks:
    # s = sum_c U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
    # if abs(s - 1.0) > 1e-9: return False
    # This implies T[i,j,k] = 1 for all i,j,k.
    # This is correct for the standard matrix multiplication tensor.
    
    # Let's re-verify W index mapping.
    # W[k*n+i] corresponds to output C[i,k]?
    # If C[i,k] = sum ... W[k*n+i] ...
    # Then W index is k*n + i.
    # So W[0] is C[0,0] (k=0, i=0).
    # W[1] is C[0,1]? No, k=0, i=1 -> W[1].
    # W[3] is C[1,0] (k=1, i=0).
    # W[4] is C[1,1] (k=1, i=1).
    # W[6] is C[2,0] (k=2, i=0).
    # W[7] is C[2,1] (k=2, i=1).
    # W[8] is C[2,2] (k=2, i=2).
    
    # My previous mapping for W was:
    # C00=0, C01=1, C10=3, C11=4, C20=6, C21=7, C22=8?
    # Let's check k*n+i.
    # C[0,0]: i=0, k=0 -> 0*3+0=0.
    # C[0,1]: i=0, k=1 -> 1*3+0=3.
    # C[0,2]: i=0, k=2 -> 2*3+0=6.
    # C[1,0]: i=1, k=0 -> 0*3+1=1.
    # C[1,1]: i=1, k=1 -> 1*3+1=4.
    # C[1,2]: i=1, k=2 -> 2*3+1=7.
    # C[2,0]: i=2, k=0 -> 0*3+2=2.
    # C[2,1]: i=2, k=1 -> 1*3+2=5.
    # C[2,2]: i=2, k=2 -> 2*3+2=8.
    
    # So W indices:
    # 0: C00
    # 1: C10
    # 2: C20
    # 3: C01
    # 4: C11
    # 5: C21
    # 6: C02
    # 7: C12
    # 8: C22

    # Strassen Terms (1-7)
    
    # Term 1: M1 = (A00+A11)(B00+B11) -> C00, C11
    # U: A00(0)=1, A11(4)=1
    # V: B00(0)=1, B11(4)=1
    # W: C00(0)=1, C11(4)=1
    U[0][0] = 1; U[4][0] = 1
    V[0][0] = 1; V[4][0] = 1
    W[0][0] = 1; W[4][0] = 1

    # Term 2: M2 = (A10+A11)B00 -> C10, C11(-1)
    # U: A10(3)=1, A11(4)=1
    # V: B00(0)=1
    # W: C10(1)=1, C11(4)=-1
    U[3][1] = 1; U[4][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[4][1] = -1

    # Term 3: M3 = A00(B10-B11) -> C01, C11
    # U: A00(0)=1
    # V: B10(3)=1, B11(4)=-1
    # W: C01(3)=1, C11(4)=1
    U[0][2] = 1
    V[3][2] = 1; V[4][2] = -1
    W[3][2] = 1; W[4][2] = 1

    # Term 4: M4 = A11(B01-B00) -> C00, C10
    # U: A11(4)=1
    # V: B01(1)=1, B00(0)=-1
    # W: C00(0)=1, C10(1)=1
    U[4][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1

    # Term 5: M5 = (A00+A01)B11 -> C00(-1), C01
    # U: A00(0)=1, A01(1)=1
    # V: B11(4)=1
    # W: C00(0)=-1, C01(3)=1
    U[0][4] = 1; U[1][4] = 1
    V[4][4] = 1
    W[0][4] = -1; W[3][4] = 1

    # Term 6: M6 = (A10-A00)(B00+B01) -> C11
    # U: A10(3)=1, A00(0)=-1
    # V: B00(0)=1, B01(1)=1
    # W: C11(4)=1
    U[3][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[4][5] = 1

    # Term 7: M7 = (A01-A11)(B10+B11) -> C00
    # U: A01(1)=1, A11(4)=-1
    # V: B10(3)=1, B11(4)=1
    # W: C00(0)=1
    U[1][6] = 1; U[4][6] = -1
    V[3][6] = 1; V[4][6] = 1
    W[0][6] = 1

    # Standard Terms (8-26)
    
    # 8. A02 B20 -> C00
    U[2][7] = 1; V[6][7] = 1; W[0][7] = 1
    
    # 9. A02 B21 -> C01
    U[2][8] = 1; V[7][8] = 1; W[3][8] = 1
    
    # 10. A12 B20 -> C10
    U[5][9] = 1; V[6][9] = 1; W[1][9] = 1
    
    # 11. A12 B21 -> C11
    U[5][10] = 1; V[7][10] = 1; W[4][10] = 1
    
    # 12. A00 B02 -> C02
    U[0][11] = 1; V[2][11] = 1; W[6][11] = 1
    
    # 13. A01 B12 -> C02
    U[1][12] = 1; V[5][12] = 1; W[6][12] = 1
    
    # 14. A02 B22 -> C02
    U[2][13] = 1; V[8][13] = 1; W[6][13] = 1
    
    # 15. A10 B02 -> C12
    U[3][14] = 1; V[2][14] = 1; W[7][14] = 1
    
    # 16. A11 B12 -> C12
    U[4][15] = 1; V[5][15] = 1; W[7][15] = 1
    
    # 17. A12 B22 -> C12
    U[5][16] = 1; V[8][16] = 1; W[7][16] = 1
    
    # 18. A20 B00 -> C20
    U[6][17] = 1; V[0][17] = 1; W[2][17] = 1
    
    # 19. A21 B10 -> C20
    U[7][18] = 1; V[3][18] = 1; W[2][18] = 1
    
    # 20. A22 B20 -> C20
    U[8][19] = 1; V[6][19] = 1; W[2][19] = 1
    
    # 21. A20 B01 -> C21
    U[6][20] = 1; V[1][20] = 1; W[5][20] = 1
    
    # 22. A21 B11 -> C21
    U[7][21] = 1; V[4][21] = 1; W[5][21] = 1
    
    # 23. A22 B21 -> C21
    U[8][22] = 1; V[7][22] = 1; W[5][22] = 1
    
    # 24. A20 B02 -> C22
    U[6][23] = 1; V[2][23] = 1; W[8][23] = 1
    
    # 25. A21 B12 -> C22
    U[7][24] = 1; V[5][24] = 1; W[8][24] = 1
    
    # 26. A22 B22 -> C22
    U[8][25] = 1; V[8][25] = 1; W[8][25] = 1

    return {"U": U, "V": V, "W": W}

def construct():
    return _construct_rank_26()
