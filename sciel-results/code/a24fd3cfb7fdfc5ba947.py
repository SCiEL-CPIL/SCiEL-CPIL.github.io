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

def _schoolbook_221():
    """Rank-4 decomposition for 2x2 * 2x1 -> 2x1. Returns (U,V,W) each (4,4),(2,4),(2,4)."""
    U = np.zeros((4, 4))
    V = np.zeros((2, 4))
    W = np.zeros((2, 4))
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # Term 1: A00 B0 -> C0
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    # Term 2: A01 B1 -> C0
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    # Term 3: A10 B0 -> C1
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    # Term 4: A11 B1 -> C1
    U[3,3]=1; V[1,3]=1; W[1,3]=1
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

    # Rank 78 decomposition
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

        # Helper functions for index mapping
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i

        # Part A: A[0:2, :] * B -> C[0:2, :]
        # A_L: rows 0-1, cols 0-1. U indices: 0, 1, 4, 5
        u_AL = [0, 1, 4, 5]
        # A_R: rows 0-1, cols 2-3. U indices: 2, 3, 6, 7
        u_AR = [2, 3, 6, 7]
        
        # B blocks for A_L (rows 0-1 of B)
        # B[0:2, 0:2] -> V indices: 0, 1, 7, 8
        v_B00 = [0, 1, 7, 8]
        # B[0:2, 2:4] -> V indices: 2, 3, 9, 10
        v_B01 = [2, 3, 9, 10]
        # B[0:2, 4:6] -> V indices: 4, 5, 11, 12
        v_B02 = [4, 5, 11, 12]
        # B[0:2, 6:7] -> V indices: 6, 13
        v_B03 = [6, 13]
        
        # B blocks for A_R (rows 2-3 of B)
        # B[2:4, 0:2] -> V indices: 14, 15, 21, 22
        v_B10 = [14, 15, 21, 22]
        # B[2:4, 2:4] -> V indices: 16, 17, 23, 24
        v_B11 = [16, 17, 23, 24]
        # B[2:4, 4:6] -> V indices: 18, 19, 25, 26
        v_B12 = [18, 19, 25, 26]
        # B[2:4, 6:7] -> V indices: 20, 27
        v_B13 = [20, 27]
        
        # C blocks (W indices)
        # C[0:2, 0:2] -> W indices: 0, 3, 1, 4
        w_C00 = [0, 3, 1, 4]
        # C[0:2, 2:4] -> W indices: 6, 9, 7, 10
        w_C01 = [6, 9, 7, 10]
        # C[0:2, 4:6] -> W indices: 12, 15, 13, 16
        w_C02 = [12, 15, 13, 16]
        # C[0:2, 6:7] -> W indices: 18, 19
        w_C03 = [18, 19]
        
        # Add A_L * B[0:2, :]
        add_block(Uss, Vss, Wss, u_AL, v_B00, w_C00)
        add_block(Uss, Vss, Wss, u_AL, v_B01, w_C01)
        add_block(Uss, Vss, Wss, u_AL, v_B02, w_C02)
        add_block(Usb, Vsb, Wsb, u_AL, v_B03, w_C03)
        
        # Add A_R * B[2:4, :]
        add_block(Uss, Vss, Wss, u_AR, v_B10, w_C00)
        add_block(Uss, Vss, Wss, u_AR, v_B11, w_C01)
        add_block(Uss, Vss, Wss, u_AR, v_B12, w_C02)
        add_block(Usb, Vsb, Wsb, u_AR, v_B13, w_C03)
        
        # Part B: A[2, :] * B -> C[2, :]
        # A[2, j] U index: 8+j
        # B[j, k] V index: j*7+k
        # C[2, k] W index: k*3+2
        
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
