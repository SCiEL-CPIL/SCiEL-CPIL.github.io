import time
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin
    
    # Build the multiplication tensor T[a, b, c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        T_recon = np.einsum('ad,bd,cd->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_recon, T, atol=tol)
    
    def try_als(r, seed, max_iters=500):
        """Alternating least squares for rank r decomposition."""
        np.random.seed(seed)
        # Initialize with random values
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            # Update U
            TW = V[:, np.newaxis, :] * W[np.newaxis, :, :]  # (m*p, p*n, r)
            h_U = np.einsum('abc,bcd->ad', T, TW)  # (n*m, r)
            G_U = np.einsum('bcd,bce->de', TW, TW)  # (r, r)
            
            try:
                U_new = np.linalg.lstsq(G_U, h_U.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                return None
            
            # Update V
            UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]  # (n*m, p*n, r)
            h_V = np.einsum('abc,acd->bd', T, UW)  # (m*p, r)
            G_V = np.einsum('acd,ace->de', UW, UW)  # (r, r)
            
            try:
                V_new = np.linalg.lstsq(G_V, h_V.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                return None
            
            # Update W
            UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (n*m, m*p, r)
            h_W = np.einsum('abc,abd->cd', T, UV)  # (p*n, r)
            G_W = np.einsum('abd,abe->de', UV, UV)  # (r, r)
            
            try:
                W_new = np.linalg.lstsq(G_W, h_W.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                return None
            
            U, V, W = U_new, V_new, W_new
            
            # Calculate error
            T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
            err = np.linalg.norm(T_recon - T)
            
            if err < 1e-10:
                break
                
            if err > prev_err * 1.1:
                # Diverging, break
                break
            prev_err = err
        
        # Try rounding to half-integers
        for scale in [2, 1]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            
            if verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                return (U_r, V_r, W_r)
        
        return None

    # Try to construct a rank-15 decomposition using known structure
    # For 3x3x5, the minimal rank is 15
    # Use a structured initialization based on the bilinear algorithm
    
    def try_structured_rank15():
        """Try to construct a rank-15 decomposition using structured initialization."""
        # The multiplication tensor for 3x3 * 3x5 can be decomposed
        # using a combination of block structure and known small-matrix results
        
        # Try various structured initializations
        inits = []
        
        # Init 1: Block-diagonal structure
        U1 = np.zeros((9, 15))
        V1 = np.zeros((15, 15))
        W1 = np.zeros((15, 15))
        for i in range(3):
            for j in range(3):
                for k in range(5):
                    idx = i * 15 + j * 5 + k
                    U1[i * 3 + j, idx] = 1
                    V1[j * 5 + k, idx] = 1
                    W1[k * 3 + i, idx] = 1
        inits.append((U1, V1, W1))
        
        # Init 2: Random with structure
        for seed in range(5):
            np.random.seed(seed)
            U2 = np.random.randn(9, 15)
            V2 = np.random.randn(15, 15)
            W2 = np.random.randn(15, 15)
            inits.append((U2, V2, W2))
        
        # Try each initialization with ALS refinement
        for U_init, V_init, W_init in inits:
            if time.time() > deadline - 5:
                break
            
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
            prev_err = np.inf
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                TW = V[:, np.newaxis, :] * W[np.newaxis, :, :]
                h_U = np.einsum('abc,bcd->ad', T, TW)
                G_U = np.einsum('bcd,bce->de', TW, TW)
                try:
                    U_new = np.linalg.lstsq(G_U, h_U.T, rcond=None)[0].T
                except:
                    break
                
                # Update V
                UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]
                h_V = np.einsum('abc,acd->bd', T, UW)
                G_V = np.einsum('acd,ace->de', UW, UW)
                try:
                    V_new = np.linalg.lstsq(G_V, h_V.T, rcond=None)[0].T
                except:
                    break
                
                # Update W
                UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]
                h_W = np.einsum('abc,abd->cd', T, UV)
                G_W = np.einsum('abd,abe->de', UV, UV)
                try:
                    W_new = np.linalg.lstsq(G_W, h_W.T, rcond=None)[0].T
                except:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
                err = np.linalg.norm(T_recon - T)
                
                if err < 1e-10:
                    break
                if err > prev_err * 1.1:
                    break
                prev_err = err
            
            # Try rounding
            for scale in [2, 1]:
                U_r = np.round(U * scale) / scale
                V_r = np.round(V * scale) / scale
                W_r = np.round(W * scale) / scale
                
                if verify_decomposition(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                    return (U_r, V_r, W_r)
        
        return None
    
    # First try structured rank-15
    result15 = try_structured_rank15()
    if result15 is not None:
        U_f, V_f, W_f = result15
        best_rank = 15
        best_result = {"U": [[float(x) for x in row] for row in U_f],
                       "V": [[float(x) for x in row] for row in V_f],
                       "W": [[float(x) for x in row] for row in W_f]}
    
    # If rank 15 not found, try ranks 16-20 with ALS
    if best_rank > 15:
        ranks_to_try = list(range(16, min(21, best_rank)))
        seeds = [42, 0, 1, 2, 3, 7, 13, 99, 123, 456]
        
        for r in ranks_to_try:
            if time.time() > deadline - 5:
                break
            if r >= best_rank:
                break
            
            for seed in seeds:
                if time.time() > deadline - 5:
                    break
                
                result = try_als(r, seed)
                if result is not None:
                    U_f, V_f, W_f = result
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": [[float(x) for x in row] for row in U_f],
                                       "V": [[float(x) for x in row] for row in V_f],
                                       "W": [[float(x) for x in row] for row in W_f]}
                    break
    
    return best_result
