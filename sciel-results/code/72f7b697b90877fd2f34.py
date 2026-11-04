"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time
import itertools

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = int(n), int(m), int(p)
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    u_val = U[i*m + j][c]
                    v_val = V[j*p + k][c]
                    w_val = W[k*n + i][c]
                    val += u_val * v_val * w_val
                if val != 1:
                    return False
    return True

def _strassen_2x2x2():
    """Return Strassen decomposition for 2x2 matrix multiplication (rank 7)."""
    # Strassen's 7 products for 2x2 matrices
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # Result: [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    # U: 4x7, V: 4x7, W: 4x7
    # U[i,j,c] where i,j in {0,1}, c in {0..6}
    # V[j,k,c] where j,k in {0,1}, c in {0..6}
    # W[k,i,c] where k,i in {0,1}, c in {0..6}
    
    # m1 = (a+d)(e+h): U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[1,1]=1
    # m2 = (c+d)e: U[1,0]=1, U[1,1]=1, V[0,0]=1, W[0,1]=-1
    # m3 = a(f-h): U[0,0]=1, V[0,1]=1, V[1,1]=-1, W[1,0]=1
    # m4 = d(g-e): U[1,1]=1, V[1,0]=1, V[0,0]=-1, W[0,1]=1
    # m5 = (a+b)h: U[0,0]=1, U[0,1]=1, V[1,1]=1, W[1,0]=-1
    # m6 = (c-a)(e+f): U[1,0]=1, U[0,0]=-1, V[0,0]=1, V[0,1]=1, W[1,1]=1
    # m7 = (b-d)(g+h): U[0,1]=1, U[1,1]=-1, V[1,0]=1, V[1,1]=1, W[0,0]=1
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # m1: (a+d)(e+h)
    U[0][0] = 1; U[1][0] = 1  # a and d
    V[0][0] = 1; V[1][0] = 1  # e and h
    W[0][0] = 1; W[1][0] = 1  # contributes to [0,0] and [1,1]
    
    # m2: (c+d)e
    U[2][1] = 1; U[3][1] = 1  # c and d
    V[0][1] = 1  # e
    W[1][1] = -1  # contributes to [1,1] with -1
    
    # m3: a(f-h)
    U[0][2] = 1  # a
    V[0][2] = 1; V[1][2] = -1  # f - h
    W[0][2] = 1  # contributes to [0,1]
    
    # m4: d(g-e)
    U[3][3] = 1  # d
    V[1][3] = 1; V[0][3] = -1  # g - e
    W[1][3] = 1  # contributes to [1,1]
    
    # m5: (a+b)h
    U[0][4] = 1; U[1][4] = 1  # a + b
    V[1][4] = 1  # h
    W[0][4] = -1  # contributes to [0,1] with -1
    
    # m6: (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1  # c - a
    V[0][5] = 1; V[1][5] = 1  # e + f
    W[1][5] = 1  # contributes to [1,1]
    
    # m7: (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1  # b - d
    V[1][6] = 1; V[0][6] = 1  # g + h
    W[0][6] = 1  # contributes to [0,0]
    
    # Let me redo this more carefully using the standard formulation
    # Actually, let me just return the known Strassen matrices directly
    
    # For 2x2x2, the tensor T[i,j,k] = 1 for all i,j,k in {0,1}
    # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # where m=p=n=2, so U[i*2+j,c], V[j*2+k,c], W[k*2+i,c]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Standard Strassen:
    # m1 = (a+d)(e+h) -> U: [1,0,0,1] for indices 0,1,2,3 (i*2+j)
    #   V: [1,0,0,1] for (e+h)
    #   W: [1,1,0,0] for (output[0,0] + output[1,1])
    # Actually W[k*2+i]: output[0,0] is k=0,i=0 -> index 0; output[1,1] is k=1,i=1 -> index 3
    # So W[0]=1, W[3]=1
    
    # Let me use the explicit formulas:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C = AB where C[i,k] = sum_j A[i,j]*B[j,k]
    
    # m1 = (A00+A11)(B00+B11)
    U[0][0]=1; U[3][0]=1  # A00 and A11
    V[0][0]=1; V[1][0]=1  # B00 and B11
    W[0][0]=1; W[3][0]=1  # C00 and C11
    
    # m2 = (A10+A11)B00
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[1][1]=-1  # C10 gets -m2? No...
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # Let me look up the exact Strassen formulas again.
    
    # Standard Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e  
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    # Mapping to our indices:
    # A[i,j]: i=row, j=col. A[0,0]=a, A[0,1]=b, A[1,0]=c, A[1,1]=d
    # B[j,k]: B[0,0]=e, B[0,1]=f, B[1,0]=g, B[1,1]=h
    # C[i,k]: C[0,0], C[0,1], C[1,0], C[1,1]
    
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # C[i,k] = sum_j A[i,j]B[j,k] = sum_c U[i*2+j,c]V[j*2+k,c]W[k*2+i,c] summed over j,c
    
    # p1 = (a+d)(e+h):
    # A terms: a=A[0,0], d=A[1,1] -> U[0,0]=1, U[3,0]=1
    # B terms: e=B[0,0], h=B[1,1] -> V[0,0]=1, V[1,0]=1
    # C terms: C[0,0] and C[1,1] -> W[0,0]=1, W[3,0]=1
    
    U[0][0]=1; U[3][0]=1
    V[0][0]=1; V[1][0]=1
    W[0][0]=1; W[3][0]=1
    
    # p2 = (c+d)e:
    # A terms: c=A[1,0], d=A[1,1] -> U[2,1]=1, U[3,1]=1
    # B terms: e=B[0,0] -> V[0,1]=1
    # C terms: C[1,0] -> W[1,1]=1 (but with sign from formula)
    # C[1,0] = p2 + p4, so p2 contributes +1 to C[1,0]
    # W[k*2+i]: C[1,0] is i=1,k=0 -> index 0*2+1=1
    U[2][1]=1; U[3][1]=1
    V[0][1]=1
    W[1][1]=1
    
    # p3 = a(f-h):
    # A terms: a=A[0,0] -> U[0,2]=1
    # B terms: f=B[0,1], h=B[1,1] -> V[1,2]=1, V[3,2]=-1
    # C terms: C[0,1] -> W[2,2]=1 (i=0,k=1 -> index 1*2+0=2)
    U[0][2]=1
    V[1][2]=1; V[3][2]=-1
    W[2][2]=1
    
    # p4 = d(g-e):
    # A terms: d=A[1,1] -> U[3,3]=1
    # B terms: g=B[1,0], e=B[0,0] -> V[2,3]=1, V[0,3]=-1
    # C terms: C[1,1] -> W[3,3]=1 (i=1,k=1 -> index 1*2+1=3)
    U[3][3]=1
    V[2][3]=1; V[0][3]=-1
    W[3][3]=1
    
    # p5 = (a+b)h:
    # A terms: a=A[0,0], b=A[0,1] -> U[0,4]=1, U[1,4]=1
    # B terms: h=B[1,1] -> V[3,4]=1
    # C terms: C[0,1] -> W[2,4]=-1 (from C[0,1]=p3+p5, so p5 contributes +1)
    # Wait, C[0,1]=p3+p5, so p5 should be +1
    U[0][4]=1; U[1][4]=1
    V[3][4]=1
    W[2][4]=1
    
    # p6 = (c-a)(e+f):
    # A terms: c=A[1,0], a=A[0,0] -> U[2,5]=1, U[0,5]=-1
    # B terms: e=B[0,0], f=B[0,1] -> V[0,5]=1, V[1,5]=1
    # C terms: C[1,1] -> W[3,5]=1 (from C[1,1]=p1-p2+p3+p6)
    U[2][5]=1; U[0][5]=-1
    V[0][5]=1; V[1][5]=1
    W[3][5]=1
    
    # p7 = (b-d)(g+h):
    # A terms: b=A[0,1], d=A[1,1] -> U[1,6]=1, U[3,6]=-1
    # B terms: g=B[1,0], h=B[1,1] -> V[2,6]=1, V[3,6]=1
    # C terms: C[0,0] -> W[0,6]=1 (from C[0,0]=p1+p4-p5+p7... wait)
    # C[0,0] = p1 + p4 - p5 + p7? Let me check.
    # Actually C[0,0] = a*e + b*g
    # p1 = (a+d)(e+h) = ae+ah+de+dh
    # p4 = d(g-e) = dg-de
    # p5 = (a+b)h = ah+bh
    # p7 = (b-d)(g+h) = bg+bh-dg-dh
    # p1+p4-p5+p7 = ae+ah+de+dh + dg-de - ah-bh + bg+bh-dg-dh = ae+bg. Yes!
    U[1][6]=1; U[3][6]=-1
    V[2][6]=1; V[3][6]=1
    W[0][6]=1
    
    return U, V, W

def _try_rank_linear(rank, n, m, p, time_limit=10):
    """Try to find a rank decomposition using linear algebra approach."""
    start_time = time.time()
    
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    # For each rank c, we have U[:,c], V[:,c], W[:,c]
    # The constraint is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is a system of polynomial equations. For small rank, we can try to solve it.
    # One approach: fix W and solve for U and V linearly, or use iterative methods.
    
    # Try a structured approach: assume some entries are 0, ±1, ±1/2
    # and use random search with more iterations
    
    np.random.seed(42)
    
    for attempt in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Random sparse initialization
        U = np.zeros((U_rows, rank))
        V = np.zeros((V_rows, rank))
        W = np.zeros((W_rows, rank))
        
        for c in range(rank):
            # Each column of U, V, W has a few non-zero entries
            num_nonzero_U = np.random.randint(1, 4)
            num_nonzero_V = np.random.randint(1, 4)
            num_nonzero_W = np.random.randint(1, 4)
            
            U_indices = np.random.choice(U_rows, size=min(num_nonzero_U, U_rows), replace=False)
            V_indices = np.random.choice(V_rows, size=min(num_nonzero_V, V_rows), replace=False)
            W_indices = np.random.choice(W_rows, size=min(num_nonzero_W, W_rows), replace=False)
            
            for idx in U_indices:
                U[idx, c] = np.random.choice([-2, -1, -0.5, 0.5, 1, 2])
            for idx in V_indices:
                V[idx, c] = np.random.choice([-2, -1, -0.5, 0.5, 1, 2])
            for idx in W_indices:
                W[idx, c] = np.random.choice([-2, -1, -0.5, 0.5, 1, 2])
        
        # Check if this works
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # First try to construct using Strassen-based block decomposition
    # For 2x3x4, we can try to use a combination of smaller decompositions
    
    # Approach: Use the fact that 2x3 = 2x2 + 2x1, and 3x4 = 3x2 + 3x2
    # Or try direct random search for low ranks
    
    # Try ranks from 19 to 23
    for rank in range(19, 24):
        result = _try_rank_linear(rank, n, m, p, time_limit=8)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition of rank 24
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
