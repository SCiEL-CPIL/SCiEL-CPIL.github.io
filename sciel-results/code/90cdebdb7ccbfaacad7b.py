import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    
    # Helper to verify a decomposition
    def verify(U, V, W, n, m, p):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Helper to find a decomposition for 2x2xK using optimization
    def find_22xK_decomp(K, target_rank):
        # Tensor 2x2xK has 4*K entries, all 1.
        # U: 4 x r, V: 2K x r, W: 2K x r
        # Wait, W is indexed by k*n+i. For 2x2xK, n=2, so W is 2K x r.
        # V is indexed by j*p+k. For 2x2xK, p=K? No, the dimensions are 2, 2, K.
        # Let's map: i in 0..1, j in 0..1, k in 0..K-1.
        # U[i,j], V[j,k], W[k,i].
        # U size 4x r.
        # V size 2K x r.
        # W size 2K x r.
        
        def func(params):
            U_flat = params[:4*target_rank]
            V_flat = params[4*target_rank:4*target_rank + 2*K*target_rank]
            W_flat = params[4*target_rank + 2*K*target_rank:]
            
            U = U_flat.reshape(4, target_rank)
            V = V_flat.reshape(2*K, target_rank)
            W = W_flat.reshape(2*K, target_rank)
            
            errors = []
            for i in range(2):
                for j in range(2):
                    for k in range(K):
                        s = 0
                        for c in range(target_rank):
                            s += U[i*2+j, c] * V[j*K+k, c] * W[k*2+i, c]
                        errors.append(s - 1.0)
            return errors

        # Random initialization
        best_sol = None
        best_err = float('inf')
        
        for restart in range(5):
            x0 = np.random.randn(4*target_rank + 4*K*target_rank) * 0.1
            try:
                res = least_squares(func, x0, method='lm', max_nfev=1000)
                if res.success and res.cost < best_err:
                    best_err = res.cost
                    best_sol = res.x
            except Exception:
                continue
        
        if best_sol is None:
            return None
            
        U_flat = best_sol[:4*target_rank]
        V_flat = best_sol[4*target_rank:4*target_rank + 2*K*target_rank]
        W_flat = best_sol[4*target_rank + 2*K*target_rank:]
        
        U = np.round(U_flat.reshape(4, target_rank))
        V = np.round(V_flat.reshape(2*K, target_rank))
        W = np.round(W_flat.reshape(2*K, target_rank))
        
        # Verify
        if verify(U.tolist(), V.tolist(), W.tolist(), 2, 2, K):
            return U, V, W
        return None

    # Strassen 2x2x2 decomposition
    def get_strassen_222():
        # Returns U, V, W for 2x2x2
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # m1
        U[0,0] = 1; U[3,0] = 1
        V[0,0] = 1; V[3,0] = 1
        W[0,0] = 1; W[3,0] = 1
        # m2
        U[2,1] = 1; U[3,1] = 1
        V[0,1] = 1
        W[2,1] = 1; W[3,1] = -1
        # m3
        U[0,2] = 1
        V[1,2] = 1; V[3,2] = -1
        W[1,2] = 1; W[3,2] = 1
        # m4
        U[3,3] = 1
        V[2,3] = 1; V[0,3] = -1
        W[0,3] = 1; W[2,3] = 1
        # m5
        U[0,4] = 1; U[1,4] = 1
        V[3,4] = 1
        W[0,4] = -1; W[1,4] = 1
        # m6
        U[2,5] = 1; U[0,5] = -1
        V[0,5] = 1; V[1,5] = 1
        W[3,5] = 1
        # m7
        U[1,6] = 1; U[3,6] = -1
        V[2,6] = 1; V[3,6] = 1
        W[0,6] = 1
        
        return U, V, W

    # Try to find 2x2x3 decomposition
    decomp_223 = None
    for r in [10, 11, 12]:
        decomp_223 = find_22xK_decomp(3, r)
        if decomp_223:
            break
            
    if decomp_223:
        U223, V223, W223 = decomp_223
        r223 = U223.shape[1]
        
        # Assemble 2x4x5
        # Structure:
        # j in {0,1}:
        #   k in {0,1,2}: Use 223 decomp
        #   k in {3,4}: Use Strassen 222
        # j in {2,3}:
        #   k in {0,1,2}: Use 223 decomp
        #   k in {3,4}: Use Strassen 222
        
        U_str, V_str, W_str = get_strassen_222()
        r_str = 7
        
        total_rank = 2 * (r223 + r_str)
        
        U = np.zeros((8, total_rank))
        V = np.zeros((20, total_rank))
        W = np.zeros((10, total_rank))
        
        col = 0
        for j_block in range(2):
            # 223 part
            for i in range(2):
                for j_local in range(2):
                    j = j_block * 2 + j_local
                    for c in range(r223):
                        U[i*4+j, col+c] = U223[i*2+j_local, c]
                        
            for j_local in range(2):
                j = j_block * 2 + j_local
                for k_local in range(3):
                    k = k_local
                    for c in range(r223):
                        V[j*5+k, col+c] = V223[j_local*3+k_local, c]
                        
            for k_local in range(3):
                k = k_local
                for i in range(2):
                    for c in range(r223):
                        W[k*2+i, col+c] = W223[k_local*2+i, c]
                        
            col += r223
            
            # Strassen part (for k=3,4)
            for i in range(2):
                for j_local in range(2):
                    j = j_block * 2 + j_local
                    for c in range(r_str):
                        U[i*4+j, col+c] = U_str[i*2+j_local, c]
                        
            for j_local in range(2):
                j = j_block * 2 + j_local
                for k_local in range(2):
                    k = 3 + k_local
                    for c in range(r_str):
                        V[j*5+k, col+c] = V_str[j_local*2+k_local, c]
                        
            for k_local in range(2):
                k = 3 + k_local
                for i in range(2):
                    for c in range(r_str):
                        W[k*2+i, col+c] = W_str[k_local*2+i, c]
                        
            col += r_str
            
        if verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback: Schoolbook
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}
