# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    schoolbook_rank = n * m * p  # 54
    
    # Precompute tensor targets
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    non_zeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                non_zeros.append((a, b, c))
                
    nz_a = np.array([x[0] for x in non_zeros])
    nz_b = np.array([x[1] for x in non_zeros])
    nz_c = np.array([x[2] for x in non_zeros])
    num_nz = len(non_zeros)

    # Precompute constant target matrices for ALS
    # These are sparse, but we use dense for simplicity as sizes are small
    # Targets_U shape: (n*m, num_nz)
    Targets_U = np.zeros((n * m, num_nz))
    Targets_V = np.zeros((m * p, num_nz))
    Targets_W = np.zeros((p * n, num_nz))
    
    for idx, (a, b, c) in enumerate(non_zeros):
        Targets_U[a, idx] = 1.0
        Targets_V[b, idx] = 1.0
        Targets_W[c, idx] = 1.0

    def try_decomposition(r, num_restarts=5, max_iter=100, tol=1e-10):
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        start_time = time.time()
        time_limit_per_restart = 15 # seconds
        
        for restart in range(num_restarts):
            if time.time() - start_time > 100: # Total limit check
                break
                
            np.random.seed(restart * 1000 + r)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            restart_start = time.time()
            
            for it in range(max_iter):
                if time.time() - restart_start > time_limit_per_restart:
                    break
                
                # Update U
                # D_U[b * (p*n) + c, d] = V[b, d] * W[c, d]
                # We need to map (b, c) to a single index for lstsq
                # b in 0..mp-1, c in 0..pn-1
                # Total indices: mp * pn
                D_U = np.einsum('bd, cd -> bcd', V, W).reshape(m * p * p * n, r)
                # Targets_U is (nm, num_nz). We need to map nz to D_U columns
                # D_U columns are indexed by b * (pn) + c
                # nz_b and nz_c give the indices
                D_U_nz = D_U[nz_b * (p * n) + nz_c, :]
                # Solve D_U_nz @ U.T = Targets_U.T
                U, _ = np.linalg.lstsq(D_U_nz, Targets_U.T, rcond=None)
                U = U.T
                
                # Update V
                D_V = np.einsum('ad, cd -> acd', U, W).reshape(n * m * p * n, r)
                D_V_nz = D_V[nz_a * (p * n) + nz_c, :]
                V, _ = np.linalg.lstsq(D_V_nz, Targets_V.T, rcond=None)
                V = V.T
                
                # Update W
                D_W = np.einsum('ad, bd -> abd', U, V).reshape(n * m * m * p, r)
                D_W_nz = D_W[nz_a * (m * p) + nz_b, :]
                W, _ = np.linalg.lstsq(D_W_nz, Targets_W.T, rcond=None)
                W = W.T
                
                # Check error
                U_A = U[nz_a, :]
                V_B = V[nz_b, :]
                W_C = W[nz_c, :]
                Prods = U_A * V_B * W_C
                Sums = Prods.sum(axis=1)
                err = np.sum((Sums - 1.0) ** 2)
                
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                if err < tol:
                    break
                    
        return best_err, best_U, best_V, best_W

    start_time = time.time()
    time_limit = 110 # Safety margin for 120s limit
    
    # Try ranks 53, 52, 51, 50
    for r in range(schoolbook_rank - 1, schoolbook_rank - 5, -1):
        if time.time() - start_time > time_limit:
            break
        err, U, V, W = try_decomposition(r, num_restarts=5, max_iter=100, tol=1e-10)
        if err < 1e-8:
            # Round to nearest 0.5
            U_rounded = [[round(U[a, d] * 2) / 2 for d in range(r)] for a in range(n * m)]
            V_rounded = [[round(V[b, d] * 2) / 2 for d in range(r)] for b in range(m * p)]
            W_rounded = [[round(W[c, d] * 2) / 2 for d in range(r)] for c in range(p * n)]
            
            # Verify
            verified = True
            for a, b, c in non_zeros:
                val = sum(U_rounded[a][d] * V_rounded[b][d] * W_rounded[c][d] for d in range(r))
                if abs(val - 1.0) > 1e-6:
                    verified = False
                    break
            
            if verified:
                return {"U": U_rounded, "V": V_rounded, "W": W_rounded}

    # Return schoolbook
    rank = schoolbook_rank
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
