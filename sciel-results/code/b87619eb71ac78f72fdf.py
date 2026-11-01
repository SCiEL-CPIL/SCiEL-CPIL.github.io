import numpy as np
import time

def construct():
    n, m, p = 2, 3, 5
    deadline = time.time() + 130
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # 28-rank construction:
    # Split m=3 into m1=2, m2=1
    # Split p=5 into p1=4, p2=1
    # 
    # A is (n x m) = (2 x 3), B is (m x p) = (3 x 5)
    # A = [A1 (2x2), A2 (2x1)]
    # B = [B1 (2x5); B2 (1x5)]
    # 
    # C = A1*B1 + A2*B2
    # 
    # A1*B1: (2x2)*(2x5). Split p=5 into 4+1:
    #   B1 = [B1a (2x4), B1b (2x1)]
    #   A1*B1a: (2x2)*(2x4) -> split into two (2x2)*(2x2), each Strassen rank 7, total 14
    #   A1*B1b: (2x2)*(2x1) -> schoolbook rank 4
    # A2*B2: (2x1)*(1x5) -> schoolbook rank 10
    # Total: 14 + 4 + 10 = 28
    
    def strassen_2x2x2():
        """Return U,V,W for 2x2 x 2x2 multiplication, rank 7.
        Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        i,k in {0,1}, j in {0,1}
        """
        # Strassen formulas for C = A*B where A,B are 2x2
        # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)*e
        # M3 = a*(f-h)
        # M4 = d*(g-e)
        # M5 = (a+b)*h
        # M6 = (c-a)*(e+f)
        # M7 = (b-d)*(g+h)
        # C = [[M1+M4-M5+M7, M3+M5],
        #      [M2+M4, M1-M2+M3+M6]]
        
        # A indices: a=00, b=01, c=10, d=11 (row*2+col)
        # B indices: e=00, f=01, g=10, h=11
        # C indices: 00, 01, 10, 11
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (a+d)(e+h)
        # U: a=0, d=3 -> U[0,0]=1, U[3,0]=1
        # V: e=0, h=3 -> V[0,0]=1, V[3,0]=1
        # W: contributes to C[0,0] and C[1,1] with +1, C[1,0] with -1, C[0,1] with +1
        # C[0,0]: M1+M4-M5+M7 -> +1
        # C[0,1]: M3+M5 -> +1
        # C[1,0]: M2+M4 -> -1
        # C[1,1]: M1-M2+M3+M6 -> +1
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[1,0] = 1; W[2,0] = -1; W[3,0] = 1
        
        # M2 = (c+d)*e
        # U: c=2, d=3
        # V: e=0
        # W: C[1,0]: +1, C[1,1]: -1
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1; W[3,1] = -1
        
        # M3 = a*(f-h)
        # U: a=0
        # V: f=1, h=3 -> V[1,2]=1, V[3,2]=-1
        # W: C[0,1]: +1, C[1,1]: +1
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1; W[3,2] = 1
        
        # M4 = d*(g-e)
        # U: d=3
        # V: g=2, e=0 -> V[2,3]=1, V[0,3]=-1
        # W: C[0,0]: +1, C[1,0]: +1
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1
        
        # M5 = (a+b)*h
        # U: a=0, b=1
        # V: h=3
        # W: C[0,0]: -1, C[0,1]: +1
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1
        
        # M6 = (c-a)*(e+f)
        # U: c=2, a=0 -> U[2,5]=1, U[0,5]=-1
        # V: e=0, f=1
        # W: C[1,1]: +1
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        
        # M7 = (b-d)*(g+h)
        # U: b=1, d=3 -> U[1,6]=1, U[3,6]=-1
        # V: g=2, h=3
        # W: C[0,0]: +1
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1
        
        return U, V, W
    
    # Build the 28-rank decomposition
    U_full = np.zeros((6, 28))  # n*m = 2*3 = 6
    V_full = np.zeros((15, 28)) # m*p = 3*5 = 15
    W_full = np.zeros((10, 28)) # p*n = 5*2 = 10
    
    col = 0
    
    # Helper: add a Strassen block for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[k0:k0+2, i0:i0+2]
    def add_strassen(i0, j0, k0):
        nonlocal col
        U_s, V_s, W_s = strassen_2x2x2()
        for r in range(7):
            for a_local in range(4):
                i_l, j_l = divmod(a_local, 2)
                U_full[(i0+i_l)*3 + (j0+j_l), col] = U_s[a_local, r]
            for b_local in range(4):
                j_l, k_l = divmod(b_local, 2)
                V_full[(j0+j_l)*5 + (k0+k_l), col] = V_s[b_local, r]
            for c_local in range(4):
                k_l, i_l = divmod(c_local, 2)
                W_full[(k0+k_l)*2 + (i0+i_l), col] = W_s[c_local, r]
            col += 1
    
    # Helper: add schoolbook block for A[i0:i0+ni, j0:j0+mj] * B[j0:j0+mj, k0:k0+pk] -> C[k0:k0+pk, i0:i0+ni]
    def add_schoolbook(i0, j0, k0, ni, mj, pk):
        nonlocal col
        for i_l in range(ni):
            for j_l in range(mj):
                for k_l in range(pk):
                    U_full[(i0+i_l)*3 + (j0+j_l), col] = 1
                    V_full[(j0+j_l)*5 + (k0+k_l), col] = 1
                    W_full[(k0+k_l)*2 + (i0+i_l), col] = 1
                    col += 1
    
    # Block 1a: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen(0, 0, 0)
    
    # Block 1b: A[0:2, 0:2] * B[0:2, 2:4] -> C[2:4, 0:2]
    add_strassen(0, 0, 2)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 4:5] -> C[4:5, 0:2]
    # This is (2x2)*(2x1), schoolbook rank 4
    add_schoolbook(0, 0, 4, 2, 2, 1)
    
    # Block 3: A[0:2, 2:3] * B[2:3, 0:5] -> C[0:5, 0:2]
    # This is (2x1)*(1x5), schoolbook rank 10
    add_schoolbook(0, 2, 0, 2, 1, 5)
    
    # Total rank used: 7 + 7 + 4 + 10 = 28
    assert col == 28, f"Expected 28, got {col}"
    
    # Verify
    S = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
    if not np.allclose(S, T, atol=1e-10):
        # Fallback to schoolbook
        rank = 30
        U = np.zeros((6, rank))
        V = np.zeros((15, rank))
        W = np.zeros((10, rank))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(5):
                    U[i*3+j, col] = 1
                    V[j*5+k, col] = 1
                    W[k*2+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank: check if any two columns can be merged
    # For rank reduction, we need U[:,r1]*V[:,r1]*W[:,r1] + U[:,r2]*V[:,r2]*W[:,r2]
    # to be expressible as a single rank-1 term (or fewer)
    # A necessary condition: the "flattened" vectors must be collinear in some sense
    # Try: for each pair (r1,r2), check if we can find alpha,beta such that
    # alpha*U[:,r1] + beta*U[:,r2] = U_new, etc.
    # This is generally not possible, but let's check for the schoolbook terms
    # The last 10 terms are all schoolbook with U,V,W entries being 1
    # Terms 14-23 are schoolbook for A[0:2,0:2]*B[0:2,4:5]
    # Terms 24-33... wait, we only have 28 terms
    # Terms 14-17: schoolbook (2x2x1)
    # Terms 18-27: schoolbook (2x1x5)
    
    # Check if any of the 10 schoolbook terms in the last block are linearly dependent
    # with Strassen terms. The Strassen terms use indices in U: rows 0-3, V: rows 0-7, W: rows 0-3
    # The schoolbook (2x1x5) terms use U: rows 4-5, V: rows 10-14, W: rows 0-9
    # These are disjoint from Strassen terms in U and V, so no direct merging possible
    
    # The schoolbook (2x2x1) terms use U: rows 0-3, V: rows 8-9, W: rows 8-9
    # Strassen terms use V: rows 0-7. So V rows 8-9 are only in schoolbook block 2.
    # W rows 8-9 are only in schoolbook block 2.
    # So no overlap there either.
    
    # Conclusion: the 28-rank decomposition is block-independent and cannot be reduced
    # by simple column merging. Return it.
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
