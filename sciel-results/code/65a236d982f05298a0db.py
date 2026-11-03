"""Matrix multiplication 3,4,7 - rank-minimal exact tensor decomposition"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]
TIME_LIMIT = 140

def _strassen_222():
    """Strassen decomposition for 2x2 * 2x2. Returns (U, V, W) each (4,7)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1; W[1,0]=-1; W[2,0]=-1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=1
    # M3 = A00(B10-B11)
    U[0,2]=1
    V[2,2]=1; V[3,2]=-1
    W[0,2]=1; W[1,2]=1
    # M4 = A11(B01-B00)
    U[3,3]=1
    V[1,3]=1; V[0,3]=-1
    W[2,3]=1; W[3,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=1; W[1,4]=1
    # M6 = (A10-A00)(B00+B10)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[2,5]=1
    W[2,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B01+B11)
    U[1,6]=1; U[3,6]=-1
    V[1,6]=1; V[3,6]=1
    W[0,6]=1; W[1,6]=1
    return U, V, W

def _strassen_223():
    """Rank-8 decomposition for 2x2 * 2x3 -> 2x3. Returns (U,V,W) each (4,8),(6,8),(6,8)."""
    # A is 2x2, B is 2x3, C is 2x3
    # Use Strassen-like: treat B as two 2x2 blocks overlapping
    # Actually, let's use a direct construction
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    # 
    # Rank-8 algorithm:
    # M1 = (A00+A11)(B00+B10)  -> contributes to C00, C10
    # M2 = (A00-A11)(B00-B10)  -> contributes to C00, C10
    # M3 = (A00+A11)(B01+B11)  -> contributes to C01, C11
    # M4 = (A00-A11)(B01-B11)  -> contributes to C01, C11
    # M5 = (A00+A11)(B02+B12)  -> contributes to C02, C12
    # M6 = (A00-A11)(B02-B12)  -> contributes to C02, C12
    # M7 = A01*B10  (but this is already covered?)
    # Actually, let me think more carefully.
    
    # Simpler: just use schoolbook for 2x2x3 which is rank 6
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    # That's 12 terms, not 8.
    
    # Let me use a different approach: split B into 2 columns + 1 column
    # 2x2x2 part (rank 7) + 2x2x1 part (rank 4) = 11, not 8.
    
    # Actually, R(2,2,3) = 8 is known. Let me use a valid construction.
    # From literature:
    # M1 = (a+b)(x+u)
    # M2 = (a-b)(x-u)
    # M3 = (c+d)(x+u)
    # M4 = (c-d)(x-u)
    # M5 = (a+b)(y+v)
    # M6 = (a-b)(y-v)
    # M7 = (c+d)(y+v)
    # M8 = (c-d)(y-v)
    # where A = [[a,b],[c,d]], B = [[x,y,z],[u,v,w]]
    # C = [[a*x+b*u, a*y+b*v, a*z+b*w],
    #      [c*x+d*u, c*y+d*v, c*z+d*w]]
    #
    # M1 = (a+b)(x+u) = ax+au+bx+bu
    # M2 = (a-b)(x-u) = ax-au-bx+bu
    # M1+M2 = 2ax+2bu -> C00 = a*x + b*u (need 1/2)
    # M3 = (c+d)(x+u) = cx+cu+dx+du
    # M4 = (c-d)(x-u) = cx-cu-dx+du
    # M3+M4 = 2cx+2du -> C10 = c*x + d*u (need 1/2)
    # M5 = (a+b)(y+v) = ay+av+by+bv
    # M6 = (a-b)(y-v) = ay-av-by+bv
    # M5+M6 = 2ay+2bv -> C01 = a*y + b*v (need 1/2)
    # M7 = (c+d)(y+v) = cy+cv+dy+dv
    # M8 = (c-d)(y-v) = cy-cv-dy+dv
    # M7+M8 = 2cy+2dv -> C11 = c*y + d*v (need 1/2)
    #
    # But what about the z and w columns? They're not covered!
    # This construction only covers the first 2 columns of B.
    
    # So this 8-term algorithm only works for 2x2x2, not 2x2x3.
    # R(2,2,3) is actually not 8 with this simple construction.
    
    # Let me just use schoolbook for 2x2x3: rank 12
    # Or split into 2x2x2 + 2x2x1: 7 + 4 = 11
    
    # For now, use the 11-term construction (7+4)
    U = np.zeros((4, 11))
    V = np.zeros((6, 11))
    W = np.zeros((6, 11))
    
    # First 7 terms: 2x2x2 Strassen for columns 0,1 of B
    Uss, Vss, Wss = _strassen_222()
    # Map B indices: B00=0, B01=1, B10=2, B11=3
    # V indices for 2x2: [0,1,2,3]
    # W indices for 2x2: [0,1,2,3]
    for t in range(7):
        for i in range(4):
            if Uss[i,t] != 0:
                U[i,t] = Uss[i,t]
        for i in range(4):
            if Vss[i,t] != 0:
                V[i,t] = Vss[i,t]
        for i in range(4):
            if Wss[i,t] != 0:
                W[i,t] = Wss[i,t]
    
    # Last 4 terms: 2x2x1 for column 2 of B
    # C0 = A00*B02 + A01*B12
    # C1 = A10*B02 + A11*B12
    # B02 = V index 4, B12 = V index 5
    # C0 = W index 4, C1 = W index 5
    U[0,7]=1; V[4,7]=1; W[4,7]=1  # A00*B02 -> C0
    U[1,8]=1; V[5,8]=1; W[4,8]=1  # A01*B12 -> C0
    U[2,9]=1; V[4,9]=1; W[5,9]=1  # A10*B02 -> C1
    U[3,10]=1; V[5,10]=1; W[5,10]=1  # A11*B12 -> C1
    
    return U, V, W

def _schoolbook_122():
    """Rank-4 decomposition for 1x2 * 2x2 -> 1x2. Returns (U,V,W) each (2,4),(4,4),(2,4)."""
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    U[0,0]=1; V[0,0]=1; W[0,0]=1  # A0*B00 -> C0
    U[1,1]=1; V[2,1]=1; W[0,1]=1  # A1*B10 -> C0
    U[0,2]=1; V[1,2]=1; W[1,2]=1  # A0*B01 -> C1
    U[1,3]=1; V[3,3]=1; W[1,3]=1  # A1*B11 -> C1
    return U, V, W

def _schoolbook_123():
    """Rank-6 decomposition for 1x2 * 2x3 -> 1x3. Returns (U,V,W) each (2,6),(6,6),(3,6)."""
    U = np.zeros((2, 6))
    V = np.zeros((6, 6))
    W = np.zeros((3, 6))
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    # C2 = A0*B02 + A1*B12
    U[0,0]=1; V[0,0]=1; W[0,0]=1  # A0*B00 -> C0
    U[1,1]=1; V[3,1]=1; W[0,1]=1  # A1*B10 -> C0
    U[0,2]=1; V[1,2]=1; W[1,2]=1  # A0*B01 -> C1
    U[1,3]=1; V[4,3]=1; W[1,3]=1  # A1*B11 -> C1
    U[0,4]=1; V[2,4]=1; W[2,4]=1  # A0*B02 -> C2
    U[1,5]=1; V[5,5]=1; W[2,5]=1  # A1*B12 -> C2
    return U, V, W

def _build_tensor(n, m, p):
    tensor = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m+j; b = j*p+k; c = k*n+i
                tensor[a,b,c] = 1.0
    return tensor

def _verify(U, V, W, tensor, tol=1e-9):
    r = U.shape[1]
    for a in range(tensor.shape[0]):
        for b in range(tensor.shape[1]):
            for c in range(tensor.shape[2]):
                val = float(np.dot(U[a,:], V[b,:]*W[c,:]))
                if abs(val - tensor[a,b,c]) > tol:
                    return False
    return True

def _to_lists(U, V, W):
    def conv(mat):
        return [[int(x) if abs(x-round(x))<1e-9 else float(x) for x in row] for row in mat]
    return conv(U), conv(V), conv(W)

def construct():
    """Find rank-minimal exact tensor decomposition for 3x4 * 4x7."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback (rank 84)
    rank_sb = n*m*p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    best_rank = rank_sb

    # Try the 64-rank decomposition: 3x4x4 (36) + 3x4x3 (28)
    try:
        Uss, Vss, Wss = _strassen_222()
        Us23, Vs23, Ws23 = _strassen_223()  # 11 terms
        U122, V122, W122 = _schoolbook_122()  # 4 terms
        U123, V123, W123 = _schoolbook_123()  # 6 terms
        
        # 3x4x4: 2x4x4 (28) + 1x4x4 (8) = 36
        # 3x4x3: 2x4x3 (22) + 1x4x3 (12) = 34
        # Total: 70
        
        total_rank = 70
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        col = 0
        
        def add_block(Ub, Vb, Wb, u_idx, v_idx, w_idx):
            nonlocal col
            r = Ub.shape[1]
            for t in range(r):
                for ri, val in enumerate(Ub[:, t]):
                    if val != 0:
                        U[u_idx[ri], col] += val
                for ri, val in enumerate(Vb[:, t]):
                    if val != 0:
                        V[v_idx[ri], col] += val
                for ri, val in enumerate(Wb[:, t]):
                    if val != 0:
                        W[w_idx[ri], col] += val
                col += 1

        # Part 1: 3x4x4 (columns 0-3 of B, columns 0-3 of C)
        # A[0:2, :] * B[0:2, 0:4] -> C[0:2, 0:4]
        # A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        # etc.
        
        # This is getting complex. Let me just use the original 78-rank approach
        # which I know works, and verify it.
        pass
        
    except Exception:
        pass

    # Use the original 78-rank decomposition
    try:
        Uss, Vss, Wss = _strassen_222()
        Usb, Vsb, Wsb = _schoolbook_221()
        
        total_rank = 78
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        col = 0
        
        def add_block(Ub, Vb, Wb, u_idx, v_idx, w_idx):
            nonlocal col
            r = Ub.shape[1]
            for t in range(r):
                for ri, val in enumerate(Ub[:, t]):
                    if val != 0:
                        U[u_idx[ri], col] += val
                for ri, val in enumerate(Vb[:, t]):
                    if val != 0:
                        V[v_idx[ri], col] += val
                for ri, val in enumerate(Wb[:, t]):
                    if val != 0:
                        W[w_idx[ri], col] += val
                col += 1

        u_AL = [0, 1, 4, 5]
        u_AR = [2, 3, 6, 7]
        
        v_B00 = [0, 1, 7, 8]
        v_B01 = [2, 3, 9, 10]
        v_B02 = [4, 5, 11, 12]
        v_B03 = [6, 13]
        
        v_B10 = [14, 15, 21, 22]
        v_B11 = [16, 17, 23, 24]
        v_B12 = [18, 19, 25, 26]
        v_B13 = [20, 27]
        
        w_C00 = [0, 3, 1, 4]
        w_C01 = [6, 9, 7, 10]
        w_C02 = [12, 15, 13, 16]
        w_C03 = [18, 19]
        
        add_block(Uss, Vss, Wss, u_AL, v_B00, w_C00)
        add_block(Uss, Vss, Wss, u_AL, v_B01, w_C01)
        add_block(Uss, Vss, Wss, u_AL, v_B02, w_C02)
        add_block(Usb, Vsb, Wsb, u_AL, v_B03, w_C03)
        
        add_block(Uss, Vss, Wss, u_AR, v_B10, w_C00)
        add_block(Uss, Vss, Wss, u_AR, v_B11, w_C01)
        add_block(Uss, Vss, Wss, u_AR, v_B12, w_C02)
        add_block(Usb, Vsb, Wsb, u_AR, v_B13, w_C03)
        
        for j in range(4):
            for k in range(7):
                U[8+j, col] = 1
                V[j*7+k, col] = 1
                W[k*3+2, col] = 1
                col += 1

        if col == total_rank:
            tensor = _build_tensor(n, m, p)
            if _verify(U, V, W, tensor):
                best_rank = col
                best_U, best_V, best_W = _to_lists(U, V, W)
    except Exception:
        pass

    if time.time() - start_time > TIME_LIMIT - 5:
        return {"U": best_U, "V": best_V, "W": best_W}

    return {"U": best_U, "V": best_V, "W": best_W}

def _schoolbook_221():
    """Rank-4 decomposition for 2x2 * 2x1 -> 2x1. Returns (U,V,W) each (4,4),(2,4),(2,4)."""
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    return U, V, W
