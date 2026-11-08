import numpy as np
import time

DIMENSIONS = [3, 4, 5]
n, m, p = DIMENSIONS

def _build_strassen_222():
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_221():
    U = np.zeros((4,4)); V = np.zeros((2,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    return U, V, W

def _build_122():
    U = np.zeros((2,4)); V = np.zeros((4,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[2,1]=1; W[0,1]=1
    U[0,2]=1; V[1,2]=1; W[1,2]=1
    U[1,3]=1; V[3,3]=1; W[1,3]=1
    return U, V, W

def _build_121():
    U = np.zeros((2,2)); V = np.zeros((2,2)); W = np.zeros((1,2))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    return U, V, W

def _construct_block_decomposition():
    U_parts, V_parts, W_parts = [], [], []
    def add(u_idx, v_idx, w_idx, Ublk, Vblk, Wblk):
        for r in range(Ublk.shape[1]):
            u = np.zeros(n*m); v = np.zeros(m*p); w = np.zeros(p*n)
            for i, a in enumerate(u_idx): u[a] = Ublk[i, r]
            for i, a in enumerate(v_idx): v[a] = Vblk[i, r]
            for i, a in enumerate(w_idx): w[a] = Wblk[i, r]
            U_parts.append(u); V_parts.append(v); W_parts.append(w)

    # Part A: rows 0,1 of A
    Uu, Vv, Ww = _build_strassen_222()
    add([0,1,4,5],[0,1,5,6],[0,3,1,4], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_strassen_222()
    add([0,1,4,5],[2,3,7,8],[6,9,7,10], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_221()
    add([0,1,4,5],[4,9],[12,13], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_strassen_222()
    add([2,3,6,7],[10,11,15,16],[0,3,1,4], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_strassen_222()
    add([2,3,6,7],[12,13,17,18],[6,9,7,10], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_221()
    add([2,3,6,7],[14,19],[12,13], Uu, Vv, Ww)

    # Part B: row 2 of A
    Uu, Vv, Ww = _build_122()
    add([8,9],[0,1,5,6],[2,5], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_122()
    add([8,9],[2,3,7,8],[8,11], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_121()
    add([8,9],[4,9],[14], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_122()
    add([10,11],[10,11,15,16],[2,5], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_122()
    add([10,11],[12,13,17,18],[8,11], Uu, Vv, Ww)
    Uu, Vv, Ww = _build_121()
    add([10,11],[14,19],[14], Uu, Vv, Ww)

    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W

def _als_decompose(T, R, U_init=None, V_init=None, W_init=None, max_iter=100, tol=1e-6):
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    if U_init is None:
        U = np.random.randn(dim_u, R)
    else:
        U = U_init.copy()
    if V_init is None:
        V = np.random.randn(dim_v, R)
    else:
        V = V_init.copy()
    if W_init is None:
        W = np.random.randn(dim_w, R)
    else:
        W = W_init.copy()

    T_flat = T.flatten()
    
    for it in range(max_iter):
        # Solve for W
        M = np.zeros((dim_u * dim_v, R))
        for r in range(R):
            M[:, r] = (U[:, r] * V[:, r]).flatten()
        T_mat = T.reshape(dim_u * dim_v, dim_w).T
        W_new, _, _, _ = np.linalg.lstsq(M, T_mat.T, rcond=None)
        W = W_new.T
        
        # Solve for U
        T_mat_u = T.reshape(dim_u, dim_v * dim_w)
        N = np.zeros((dim_v * dim_w, R))
        for r in range(R):
            N[:, r] = (V[:, r] * W[:, r]).flatten()
        U_new, _, _, _ = np.linalg.lstsq(N, T_mat_u.T, rcond=None)
        U = U_new.T
        
        # Solve for V
        T_mat_v = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
        P = np.zeros((dim_u * dim_w, R))
        for r in range(R):
            P[:, r] = (U[:, r] * W[:, r]).flatten()
        V_new, _, _, _ = np.linalg.lstsq(P, T_mat_v.T, rcond=None)
        V = V_new.T
        
        if it % 20 == 0:
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T - T_rec)
            if err < tol:
                break
                
    return U, V, W

def construct():
    t0 = time.time()
    deadline = 120.0
    
    U56, V56, W56 = _construct_block_decomposition()
    T = np.ones((n, m, p))
    
    # Try rank 55
    if time.time() - t0 < deadline - 30:
        try:
            U55, V55, W55 = _als_decompose(T, 55, U56, V56, W56, max_iter=200)
            T_rec = np.einsum('ar,br,cr->abc', U55, V55, W55)
            if np.allclose(T_rec, T, atol=1e-5):
                return {
                    "U": U55.tolist(),
                    "V": V55.tolist(),
                    "W": W55.tolist()
                }
        except Exception:
            pass

    return {
        "U": U56.tolist(),
        "V": V56.tolist(),
        "W": W56.tolist()
    }
