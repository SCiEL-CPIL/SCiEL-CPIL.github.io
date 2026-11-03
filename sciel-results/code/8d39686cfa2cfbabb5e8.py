import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    
    The known minimal rank for 2x3x3 is 17, achievable by using Strassen's algorithm
    for the 2x2 core and schoolbook for the remaining terms.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=3
    
    # Index conventions:
    # U[i*m+j, c] for A[i,j] where i in [0,n), j in [0,m)
    # V[j*p+k, c] for B[j,k] where j in [0,m), k in [0,p)
    # W[k*n+i, c] for C[i,k] where k in [0,p), i in [0,n)
    
    # We'll build rank-17 decomposition.
    # Strassen for 2x2 core (columns 0,1 of A, rows 0,1 of B, output C[0:2, 0:2])
    # Then schoolbook for remaining terms involving column 2 of A or row 2 of B.
    
    # Total rank = 17
    rank = 17
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, col, u_val=1, v_val=1, w_val=1):
        U[u_idx, col] = Fraction(u_val)
        V[v_idx, col] = Fraction(v_val)
        W[w_idx, col] = Fraction(w_val)
    
    # Strassen's 7 terms for the 2x2 core (A[0:2,0:2] * B[0:2,0:2])
    # A = [[a11, a12], [a21, a22]] = A[0:2, 0:2]
    # B = [[b11, b12], [b21, b22]] = B[0:2, 0:2]
    # C = [[c11, c12], [c21, c22]]
    #
    # M1 = (a11 + a22)(b11 + b22)
    # M2 = (a21 + a22)b11
    # M3 = a11(b12 - b22)
    # M4 = a22(b21 - b11)
    # M5 = (a11 + a12)b22
    # M6 = (a21 - a11)(b11 + b12)
    # M7 = (a12 - a22)(b21 + b22)
    #
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    #
    # Indices:
    # A[i,j] -> U[i*m+j] = U[i*3+j]
    # B[j,k] -> V[j*p+k] = V[j*3+k]
    # C[i,k] -> W[k*n+i] = W[k*2+i]
    
    # Term 1: M1 = (a11 + a22)(b11 + b22)
    # a11 = A[0,0] -> U[0], a22 = A[1,1] -> U[1*3+1]=U[4]
    # b11 = B[0,0] -> V[0], b22 = B[1,1] -> V[1*3+1]=V[4]
    # This is NOT a single rank-1 term; it's a product of sums.
    # We need to expand each Strassen term into rank-1 components.
    #
    # Actually, each Mi is a product of a linear form in A and a linear form in B,
    # so each Mi IS a rank-1 term in the tensor decomposition.
    # The U, V, W factors for each Mi capture the linear forms.
    
    # Let me re-think. For each Mi:
    # U has the coefficients for the linear form in A
    # V has the coefficients for the linear form in B  
    # W has the coefficients for how Mi contributes to each C[i,k]
    
    # Term 1: M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # U: A[0,0] coeff 1, A[1,1] coeff 1 -> U[0]=1, U[4]=1
    # V: B[0,0] coeff 1, B[1,1] coeff 1 -> V[0]=1, V[4]=1
    # W: M1 contributes to c11 (+1), c22 (+1) -> W[0]=1 (C[0,0], k=0,i=0), W[3]=1 (C[1,1], k=1,i=1)
    col = 0
    U[0, col] = 1; U[4, col] = 1
    V[0, col] = 1; V[4, col] = 1
    W[0, col] = 1; W[3, col] = 1
    
    # Term 2: M2 = (A[1,0] + A[1,1]) * B[0,0]
    # U: A[1,0] coeff 1, A[1,1] coeff 1 -> U[3]=1, U[4]=1
    # V: B[0,0] coeff 1 -> V[0]=1
    # W: M2 contributes to c21 (+1), c22 (-1) -> W[1]=1 (C[1,0]), W[3]=-1 (C[1,1])
    col = 1
    U[3, col] = 1; U[4, col] = 1
    V[0, col] = 1
    W[1, col] = 1; W[3, col] = -1
    
    # Term 3: M3 = A[0,0] * (B[0,1] - B[1,1])
    # U: A[0,0] coeff 1 -> U[0]=1
    # V: B[0,1] coeff 1, B[1,1] coeff -1 -> V[1]=1, V[4]=-1
    # W: M3 contributes to c12 (+1), c22 (+1) -> W[2]=1 (C[0,1]), W[3]=1 (C[1,1])
    col = 2
    U[0, col] = 1
    V[1, col] = 1; V[4, col] = -1
    W[2, col] = 1; W[3, col] = 1
    
    # Term 4: M4 = A[1,1] * (B[1,0] - B[0,0])
    # U: A[1,1] coeff 1 -> U[4]=1
    # V: B[1,0] coeff 1, B[0,0] coeff -1 -> V[3]=1, V[0]=-1
    # W: M4 contributes to c11 (+1), c21 (+1) -> W[0]=1 (C[0,0]), W[1]=1 (C[1,0])
    col = 3
    U[4, col] = 1
    V[3, col] = 1; V[0, col] = -1
    W[0, col] = 1; W[1, col] = 1
    
    # Term 5: M5 = (A[0,0] + A[0,1]) * B[1,1]
    # U: A[0,0] coeff 1, A[0,1] coeff 1 -> U[0]=1, U[1]=1
    # V: B[1,1] coeff 1 -> V[4]=1
    # W: M5 contributes to c11 (-1), c12 (+1) -> W[0]=-1 (C[0,0]), W[2]=1 (C[0,1])
    col = 4
    U[0, col] = 1; U[1, col] = 1
    V[4, col] = 1
    W[0, col] = -1; W[2, col] = 1
    
    # Term 6: M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # U: A[1,0] coeff 1, A[0,0] coeff -1 -> U[3]=1, U[0]=-1
    # V: B[0,0] coeff 1, B[0,1] coeff 1 -> V[0]=1, V[1]=1
    # W: M6 contributes to c22 (+1) -> W[3]=1 (C[1,1])
    col = 5
    U[3, col] = 1; U[0, col] = -1
    V[0, col] = 1; V[1, col] = 1
    W[3, col] = 1
    
    # Term 7: M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    # U: A[0,1] coeff 1, A[1,1] coeff -1 -> U[1]=1, U[4]=-1
    # V: B[1,0] coeff 1, B[1,1] coeff 1 -> V[3]=1, V[4]=1
    # W: M7 contributes to c11 (+1) -> W[0]=1 (C[0,0])
    col = 6
    U[1, col] = 1; U[4, col] = -1
    V[3, col] = 1; V[4, col] = 1
    W[0, col] = 1
    
    # Now schoolbook terms for the remaining products:
    # We need all C[i,k] = sum_j A[i,j]*B[j,k]
    # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0] + A[i,2]*B[2,0]
    # C[i,1] = A[i,0]*B[0,1] + A[i,1]*B[1,1] + A[i,2]*B[2,1]
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2] + A[i,2]*B[2,2]
    #
    # Strassen handles the 2x2 core (j=0,1 for A; k=0,1 for B; i=0,1 for C).
    # Missing terms involve j=2 (A[i,2]) or k=2 (B[j,2]).
    #
    # Let's list all 18 schoolbook terms and see which are already covered:
    # Terms with j=2: A[0,2]*B[2,0], A[0,2]*B[2,1], A[0,2]*B[2,2], A[1,2]*B[2,0], A[1,2]*B[2,1], A[1,2]*B[2,2]
    # Terms with k=2: A[0,0]*B[0,2], A[0,1]*B[1,2], A[1,0]*B[0,2], A[1,1]*B[1,2]
    # Total: 6 + 4 = 10 additional terms, but some may overlap.
    #
    # Actually, all 10 are distinct from the Strassen-covered terms.
    # 7 + 10 = 17. 
    
    # Schoolbook terms involving A[i,2] (j=2):
    # A[0,2]*B[2,0] -> C[0,0]
    col = 7
    U[0*3+2, col] = 1; V[2*3+0, col] = 1; W[0*2+0, col] = 1
    
    # A[0,2]*B[2,1] -> C[0,1]
    col = 8
    U[0*3+2, col] = 1; V[2*3+1, col] = 1; W[0*2+1, col] = 1
    
    # A[0,2]*B[2,2] -> C[0,2]
    col = 9
    U[0*3+2, col] = 1; V[2*3+2, col] = 1; W[2*2+0, col] = 1
    
    # A[1,2]*B[2,0] -> C[1,0]
    col = 10
    U[1*3+2, col] = 1; V[2*3+0, col] = 1; W[0*2+1, col] = 1
    
    # A[1,2]*B[2,1] -> C[1,1]
    col = 11
    U[1*3+2, col] = 1; V[2*3+1, col] = 1; W[1*2+1, col] = 1
    
    # A[1,2]*B[2,2] -> C[1,2]
    col = 12
    U[1*3+2, col] = 1; V[2*3+2, col] = 1; W[2*2+1, col] = 1
    
    # Schoolbook terms involving B[j,2] (k=2), j=0,1:
    # A[0,0]*B[0,2] -> C[0,2]
    col = 13
    U[0*3+0, col] = 1; V[0*3+2, col] = 1; W[2*2+0, col] = 1
    
    # A[0,1]*B[1,2] -> C[0,2]
    col = 14
    U[0*3+1, col] = 1; V[1*3+2, col] = 1; W[2*2+0, col] = 1
    
    # A[1,0]*B[0,2] -> C[1,2]
    col = 15
    U[1*3+0, col] = 1; V[0*3+2, col] = 1; W[2*2+1, col] = 1
    
    # A[1,1]*B[1,2] -> C[1,2]
    col = 16
    U[1*3+1, col] = 1; V[1*3+2, col] = 1; W[2*2+1, col] = 1
    
    # Convert to lists for the return interface
    U_list = [[int(x) if x == int(x) else float(x) for x in row] for row in U]
    V_list = [[int(x) if x == int(x) else float(x) for x in row] for row in V]
    W_list = [[int(x) if x == int(x) else float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
