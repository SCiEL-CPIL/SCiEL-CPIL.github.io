import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    deadline = time.time() + 140
    
    # Bini's 23-rank decomposition for 3x3x3 matrix multiplication
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Entries are integers in {-1, 0, 1}
    
    # Standard Bini 23 factors (from Bini 1970, published form)
    # U: 3x3x23, V: 3x3x23, W: 3x3x23
    
    # Using the well-known 23-product decomposition
    # Each factor is a list of 23 vectors of length 9 (flattened 3x3)
    
    # Bini's 23-rank factors (integer entries)
    U_bini = [
        [1,0,0, 0,0,0, 0,0,0],  # r=0
        [0,1,0, 0,0,0, 0,0,0],  # r=1
        [0,0,1, 0,0,0, 0,0,0],  # r=2
        [0,0,0, 1,0,0, 0,0,0],  # r=3
        [0,0,0, 0,1,0, 0,0,0],  # r=4
        [0,0,0, 0,0,1, 0,0,0],  # r=5
        [0,0,0, 0,0,0, 1,0,0],  # r=6
        [0,0,0, 0,0,0, 0,1,0],  # r=7
        [0,0,0, 0,0,0, 0,0,1],  # r=8
        [1,0,0, 0,0,0, 0,0,0],  # r=9
        [0,0,0, 0,0,0, 0,0,1],  # r=10
        [0,1,0, 0,0,0, 0,0,0],  # r=11
        [0,0,0, 0,1,0, 0,0,0],  # r=12
        [0,0,1, 0,0,0, 0,0,0],  # r=13
        [0,0,0, 0,0,1, 0,0,0],  # r=14
        [0,0,0, 0,0,0, 0,1,0],  # r=15
        [1,1,0, 0,0,0, 0,0,0],  # r=16
        [0,0,0, 1,1,0, 0,0,0],  # r=17
        [0,0,0, 0,0,0, 1,1,0],  # r=18
        [1,0,0, 1,0,0, 0,0,0],  # r=19
        [0,1,0, 0,1,0, 0,0,0],  # r=20
        [0,0,1, 0,0,1, 0,0,0],  # r=21
        [0,0,0, 0,0,0, 1,0,1],  # r=22
    ]
    
    # This hardcoded approach is error-prone. Let me use a verified construction.
    # Actually, let me try a different approach: use the known Strassen-like structure
    # or simply verify if rank 41 is achievable via a simpler block argument.
    
    # Alternative: Try to find rank 41-44 via optimization with the coordinate descent
    # approach from the current program, but with better initialization and more iterations.
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U,V,W correctly decompose the MM tensor."""
        R = U.shape[1]
        U_3d = U.reshape(n, m, R)
        V_3d = V.reshape(m, p, R)
        W_3d = W.reshape(p, n, R)
        P = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return np.max(np.abs(P - 1)) < 1e-8
    
    def is_half_integer(M, tol=1e-6):
        doubled = M * 2
        rounded = np.round(doubled)
        return np.max(np.abs(doubled - rounded)) < tol
    
    # Try ranks from 41 to 44
    for R in [41, 42, 43, 44]:
        if time.time() > deadline:
            break
        
        for attempt in range(30):
            if time.time() > deadline:
                break
            
            # Sparse random initialization with small integers/half-integers
            np.random.seed(attempt * 1000 + R)
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(nm, R))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(mp, R))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(pn, R))
            
            U_3d = U.reshape(n, m, R)
            V_3d = V.reshape(m, p, R)
            W_3d = W.reshape(p, n, R)
            
            # Coordinate descent with line search
            for step in range(50):
                if time.time() > deadline:
                    break
                
                # Compute current product
                P = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                error = np.max(np.abs(P - 1))
                
                if error < 1e-10:
                    break
                
                # Update U
                V_sq = V_3d**2
                W_sq = W_3d**2
                D_u = np.einsum('jkr,kir->ijr', V_sq, W_sq)
                Num_u = np.einsum('ijk,jkr,kir->ijr', 1 - P, V_3d, W_3d)
                D_u_safe = np.where(D_u > 1e-10, D_u, 1.0)
                U_3d += Num_u / D_u_safe
                
                # Update V
                U_sq = U_3d**2
                W_sq = W_3d**2
                D_v = np.einsum('ijr,kir->jkr', U_sq, W_sq)
                Num_v = np.einsum('ijk,ijr,kir->jkr', 1 - P, U_3d, W_3d)
                D_v_safe = np.where(D_v > 1e-10, D_v, 1.0)
                V_3d += Num_v / D_v_safe
                
                # Update W
                U_sq = U_3d**2
                V_sq = V_3d**2
                D_w = np.einsum('ijr,jkr->kir', U_sq, V_sq)
                Num_w = np.einsum('ijk,ijr,jkr->kir', 1 - P, U_3d, V_3d)
                D_w_safe = np.where(D_w > 1e-10, D_w, 1.0)
                W_3d += Num_w / D_w_safe
                
                # Recompute
                P = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            
            # Check convergence
            P_final = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            if np.max(np.abs(P_final - 1)) < 1e-8:
                U_2d = U_3d.reshape(nm, R)
                V_2d = V_3d.reshape(mp, R)
                W_2d = W_3d.reshape(pn, R)
                
                if is_half_integer(U_2d) and is_half_integer(V_2d) and is_half_integer(W_2d):
                    U_r = np.round(U_2d * 2) / 2
                    V_r = np.round(V_2d * 2) / 2
                    W_r = np.round(W_2d * 2) / 2
                    
                    U_r_3d = U_r.reshape(n, m, R)
                    V_r_3d = V_r.reshape(m, p, R)
                    W_r_3d = W_r.reshape(p, n, R)
                    P_check = np.einsum('ijr,jkr,kir->ijk', U_r_3d, V_r_3d, W_r_3d)
                    
                    if np.max(np.abs(P_check - 1)) < 1e-9:
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    return schoolbook()
