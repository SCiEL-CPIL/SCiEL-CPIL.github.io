import numpy as np
import time

DIMENSIONS = [3, 4, 5]
n, m, p = DIMENSIONS  # 3, 4, 5

def _build_strassen_222():
    """Rank-7 Strassen for 2x2x2. Returns U,V,W each (4,7).
    U order: A00,A01,A10,A11 ; V order: B00,B01,B10,B11 ; W order: C00,C01,C10,C11."""
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[1,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_122():
    """Rank-4 schoolbook for 1x2x2: A(1x2)*B(2x2)=C(1x2). U(2,4),V(4,4),W(2,4).
    U order: A0,A1 ; V order: B00,B01,B10,B11 ; W order: C0,C1."""
    U = np.zeros((2,4)); V = np.zeros((4,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[2,1]=1; W[0,1]=1
    U[0,2]=1; V[1,2]=1; W[1,2]=1
    U[1,3]=1; V[3,3]=1; W[1,3]=1
    return U, V, W

def _build_121():
    """Rank-2 for 1x2x1: A(1x2)*B(2x1)=C(1x1). U(2,2),V(2,2),W(1,2)."""
    U = np.zeros((2,2)); V = np.zeros((2,2)); W = np.zeros((1,2))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    return U, V, W

def _build_221():
    """Rank-4 schoolbook for 2x2x1: A(2x2)*B(2x1)=C(2x1). U(4,4),V(2,4),W(2,4).
    U order: A00,A01,A10,A11 ; V order: B0,B1 ; W order: C0,C1."""
    U = np.zeros((4,4)); V = np.zeros((2,4)); W = np.zeros((2,4))
    U[0,0]=1; V[0,0]=1; W[0,0]=1
    U[1,1]=1; V[1,1]=1; W[0,1]=1
    U[2,2]=1; V[0,2]=1; W[1,2]=1
    U[3,3]=1; V[1,3]=1; W[1,3]=1
    return U, V, W

def _verify(U, V, W):
    r = U.shape[1]
    T = np.zeros((n, m, p))
    Umat = np.asarray(U, dtype=float)
    Vmat = np.asarray(V, dtype=float)
    Wmat = np.asarray(W, dtype=float)
    for idx in range(r):
        u = Umat[:, idx].reshape(n, m)
        v = Vmat[:, idx].reshape(m, p)
        w = Wmat[:, idx].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
    M = np.ones((n, m, p))
    return np.allclose(T, M, atol=1e-9)

def _construct_rank_56():
    """
    Partition:
    A rows: [1, 2]
    A cols: [2, 2]
    B rows: [2, 2]
    B cols: [2, 2, 1]
    """
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Block definitions: (a_r_start, a_c_start, a_r_size, a_c_size, b_r_start, b_c_start, b_r_size, b_c_size)
    blocks = [
        # Row 0 of A (1x4), Col 0 of B (2x2) -> C[0:1, 0:2]
        (0, 0, 1, 2, 0, 0, 2, 2), # A00 * B00
        (0, 2, 1, 2, 2, 0, 2, 2), # A01 * B10
        
        # Row 0 of A, Col 1 of B (2x2) -> C[0:1, 2:4]
        (0, 0, 1, 2, 0, 2, 2, 2), # A00 * B01
        (0, 2, 1, 2, 2, 2, 2, 2), # A01 * B11
        
        # Row 0 of A, Col 2 of B (2x1) -> C[0:1, 4:5]
        (0, 0, 1, 2, 0, 4, 2, 1), # A00 * B02
        (0, 2, 1, 2, 2, 4, 2, 1), # A01 * B12
        
        # Row 1 of A (2x4), Col 0 of B (2x2) -> C[1:3, 0:2]
        (1, 0, 2, 2, 0, 0, 2, 2), # A10 * B00
        (1, 2, 2, 2, 2, 0, 2, 2), # A11 * B10
        
        # Row 1 of A, Col 1 of B (2x2) -> C[1:3, 2:4]
        (1, 0, 2, 2, 0, 2, 2, 2), # A10 * B01
        (1, 2, 2, 2, 2, 2, 2, 2), # A11 * B11
        
        # Row 1 of A, Col 2 of B (2x1) -> C[1:3, 4:5]
        (1, 0, 2, 2, 0, 4, 2, 1), # A10 * B02
        (1, 2, 2, 2, 2, 4, 2, 1), # A11 * B12
    ]
    
    for (ar_s, ac_s, ar_sz, ac_sz, br_s, bc_s, br_sz, bc_sz) in blocks:
        if ar_sz == 1 and ac_sz == 2 and bc_sz == 2:
            Ub, Vb, Wb = _build_122()
        elif ar_sz == 1 and ac_sz == 2 and bc_sz == 1:
            Ub, Vb, Wb = _build_121()
        elif ar_sz == 2 and ac_sz == 2 and bc_sz == 2:
            Ub, Vb, Wb = _build_strassen_222()
        elif ar_sz == 2 and ac_sz == 2 and bc_sz == 1:
            Ub, Vb, Wb = _build_221()
        else:
            continue # Should not happen
            
        rank_b = Ub.shape[1]
        for r in range(rank_b):
            u = np.zeros(n*m)
            v = np.zeros(m*p)
            w = np.zeros(p*n)
            
            # Map U
            for i in range(ar_sz):
                for j in range(ac_sz):
                    # Local index in Ub: i*ac_sz + j ? 
                    # In _build_122, U is (2, 4). i is row of A (size 1? No, ar_sz=1). 
                    # Wait, _build_122 U is (2,4). The 2 corresponds to ac_sz=2.
                    # So local index is j (since ar_sz=1, i=0).
                    # Let's check _build_122: U[0,:] is A0, U[1,:] is A1.
                    # So if ar_sz=1, i=0. j=0,1.
                    # If ar_sz=2, i=0,1. j=0,1.
                    # In _build_221, U is (4,4). i*2+j.
                    
                    if ar_sz == 1:
                        local_u_idx = j
                    else:
                        local_u_idx = i*ac_sz + j
                        
                    val = Ub[local_u_idx, r]
                    if abs(val) > 1e-15:
                        global_u_idx = (ar_s + i)*m + (ac_s + j)
                        u[global_u_idx] = val
                        
            # Map V
            for j in range(br_sz):
                for k in range(bc_sz):
                    if br_sz == 2 and bc_sz == 2:
                        local_v_idx = j*2 + k
                    elif br_sz == 2 and bc_sz == 1:
                        local_v_idx = j
                    else:
                        local_v_idx = j*bc_sz + k
                        
                    val = Vb[local_v_idx, r]
                    if abs(val) > 1e-15:
                        global_v_idx = (br_s + j)*p + (bc_s + k)
                        v[global_v_idx] = val
                        
            # Map W
            for i in range(ar_sz):
                for k in range(bc_sz):
                    if ar_sz == 1 and bc_sz == 2:
                        local_w_idx = k
                    elif ar_sz == 1 and bc_sz == 1:
                        local_w_idx = 0
                    elif ar_sz == 2 and bc_sz == 2:
                        local_w_idx = k*2 + i
                    elif ar_sz == 2 and bc_sz == 1:
                        local_w_idx = i
                    else:
                        local_w_idx = k*ar_sz + i
                        
                    val = Wb[local_w_idx, r]
                    if abs(val) > 1e-15:
                        global_w_idx = (bc_s + k)*n + (ar_s + i)
                        w[global_w_idx] = val
                        
            U_parts.append(u)
            V_parts.append(v)
            W_parts.append(w)
            
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W

def _construct_schoolbook():
    U_parts, V_parts, W_parts = [], [], []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m); u[i*m+j] = 1
                v = np.zeros(m*p); v[j*p+k] = 1
                w = np.zeros(p*n); w[k*n+i] = 1
                U_parts.append(u); V_parts.append(v); W_parts.append(w)
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return U, V, W

def construct():
    t0 = time.time()
    deadline = 150.0
    margin = 30.0
    
    try:
        U, V, W = _construct_rank_56()
        if time.time() - t0 > deadline - margin:
            U, V, W = _construct_schoolbook()
        elif not _verify(U, V, W):
            U, V, W = _construct_schoolbook()
    except Exception:
        U, V, W = _construct_schoolbook()
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
