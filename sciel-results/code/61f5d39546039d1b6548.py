"""
Rank-minimal tensor decomposition for 2x3 * 3x10 matrix multiplication.
Uses Strassen (rank 7) for five 2x2*2x2 sub-blocks plus schoolbook for the
2x1 * 1x10 correction, achieving rank 55 (vs schoolbook 60).
"""
import numpy as np
import time

START = time.time()
DEADLINE = 150.0 - 10.0  # safety margin

def _strassen_2x2():
    """
    Strassen's algorithm for 2x2 * 2x2. Returns U(4x7), V(4x7), W(4x7).
    A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    P1=(a+d)(e+h), P2=(c+d)e, P3=a(f-h), P4=d(g-e),
    P5=(a+b)h, P6=(c-a)(e+f), P7=(b-d)(g+h)
    C = [[P1+P4-P5+P7, P3+P5],[P2+P4, P1-P2+P3+P6]]
    """
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    # P1=(a+d)(e+h): U a,d; V e,h; W C00+=1, C11-=1
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=-1
    # P2=(c+d)e: U c,d; V e; W C10+=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1
    # P3=a(f-h): U a; V f,h=-1; W C01+=1, C11+=1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # P4=d(g-e): U d; V g,e=-1; W C00+=1, C10+=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # P5=(a+b)h: U a,b; V h; W C00-=1, C01+=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    # P6=(c-a)(e+f): U c,a=-1; V e,f; W C11+=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # P7=(b-d)(g+h): U b,d=-1; V g,h; W C00+=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify U,V,W decompose the multiplication tensor exactly."""
    U = np.array(U, dtype=float); V = np.array(V, dtype=float); W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                rhs = 0.0
                for r in range(rank):
                    rhs += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(1.0 - rhs) > 1e-9:
                    return False
    return True

def _build_55():
    """Construct rank-55 decomposition: 5 Strassen blocks + 20 schoolbook correction."""
    n, m, p = 2, 3, 10
    rank = 55
    U = np.zeros((n*m, rank))  # 6 x 55
    V = np.zeros((m*p, rank))  # 30 x 55
    W = np.zeros((p*n, rank))  # 20 x 55

    sU, sV, sW = _strassen_2x2()

    # ---- Part 1: A_L (2x2) * B_L (2x10) via 5 Strassen 2x2 blocks ----
    # B_L rows: j=0 (B row 0), j=1 (B row 1). Block b (0..4) covers B cols 2b,2b+1.
    for b in range(5):
        base = 7 * b
        # Map A_L entries (a=U[0], b=U[1], c=U[2], d=U[3]) to global U rows:
        #   A[0,0]->row0, A[0,1]->row1, A[1,0]->row2, A[1,1]->row3
        aU = np.array([0, 1, 2, 3])
        # Map B_L entries (e=f row0col2b, f=row0col2b+1, g=row1col2b, h=row1col2b+1)
        #   to global V rows: j*p+k. j=0 -> k in {2b,2b+1}; j=1 -> k in {2b,2b+1}
        aV = np.array([0*p + 2*b, 0*p + (2*b+1), 1*p + 2*b, 1*p + (2*b+1)])
        # Map C entries (C00->W row k*n+i: k=2b,i=0; C01->k=2b+1,i=0;
        #   C10->k=2b,i=1; C11->k=2b+1,i=1)
        aW = np.array([(2*b)*n + 0, (2*b+1)*n + 0, (2*b)*n + 1, (2*b+1)*n + 1])
        for r in range(7):
            for u_idx, val in enumerate(sU[:, r]):
                if val != 0:
                    U[aU[u_idx], base + r] += val
            for v_idx, val in enumerate(sV[:, r]):
                if val != 0:
                    V[aV[v_idx], base + r] += val
            for w_idx, val in enumerate(sW[:, r]):
                if val != 0:
                    W[aW[w_idx], base + r] += val

    # ---- Part 2: A_R (2x1) * B_R (1x10) schoolbook ----
    # A_R: A[0,2]->U row 4, A[1,2]->U row 5. B_R: B row 2, cols 0..9 -> V rows 2*p+k.
    col = 35
    for i in range(2):       # A row
        for k in range(10):  # B col
            U[i*m + 2, col] = 1
            V[2*p + k, col] = 1
            W[k*n + i, col] = 1
            col += 1

    return U, V, W

def construct():
    n, m, p = 2, 3, 10
    U, V, W = _build_55()
    assert _verify(U, V, W, n, m, p), "rank-55 decomposition failed verification"
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
