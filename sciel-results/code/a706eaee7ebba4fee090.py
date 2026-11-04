import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # 1. Construct Schoolbook Decomposition (Fallback)
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    best_rank = rank_schoolbook

    # 2. Attempt to find a lower rank decomposition using ALS
    # We target ranks lower than schoolbook.
    # Given the complexity, we try a few ranks and restarts.
    
    deadline = time.time() + 140.0 # Leave 10s margin for safety
    
    # Try ranks from 49 down to 45
    for target_rank in range(rank_schoolbook - 1, 40, -1):
        if time.time() > deadline - 5.0:
            break
            
        # We try a few restarts for each rank
        # ALS is fast, so we can afford more restarts for higher ranks
        max_restarts = 20 if target_rank > 45 else 50
        
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
                
            np.random.seed(1000 + restart * target_rank)
            
            # Initialize with small random values
            U = np.random.randn(n * m, target_rank) * 0.1
            V = np.random.randn(m * p, target_rank) * 0.1
            W = np.random.randn(p * n, target_rank) * 0.1
            
            # ALS Iterations
            # Minimize sum (sum_c U_ij,c V_jk,c W_ki,c - 1)^2
            # For fixed V, W, update U_ij:
            #   System: sum_c U_ij,c * (V_jk,c * W_ki,c) = 1 for all k
            #   Let A_ij be matrix of size (p, r) where A_ij[k, c] = V[j*p+k, c] * W[k*n+i, c]
            #   Solve A_ij @ U_ij = ones(p)
            
            error = float('inf')
            for it in range(50): # Max ALS iterations
                if time.time() > deadline - 1.0:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # Build A: (p, r)
                        A = np.zeros((p, target_rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve A @ x = ones(p)
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                            U[i * m + j, :] = x
                        except np.linalg.LinAlgError:
                            pass

                # Update V
                for j in range(m):
                    for k in range(p):
                        # System: sum_c V_jk,c * (U_ij,c * W_ki,c) = 1 for all i
                        # Let B_jk be matrix of size (n, r) where B_jk[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        # Solve B_jk @ V_jk = ones(n)
                        B = np.zeros((n, target_rank))
                        for i in range(n):
                            B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            y, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                            V[j * p + k, :] = y
                        except np.linalg.LinAlgError:
                            pass

                # Update W
                for k in range(p):
                    for i in range(n):
                        # System: sum_c W_ki,c * (U_ij,c * V_jk,c) = 1 for all j
                        # Let C_ki be matrix of size (m, r) where C_ki[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        # Solve C_ki @ W_ki = ones(m)
                        C = np.zeros((m, target_rank))
                        for j in range(m):
                            C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            z, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                            W[k * n + i, :] = z
                        except np.linalg.LinAlgError:
                            pass
                
                # Check convergence / error
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            error += (s - 1.0)**2
                
                if error < 1e-12:
                    break
            
            if error < 1e-8:
                # Try rounding to rationals
                # The entries of the tensor are integers (1). 
                # Often decompositions use integers or half-integers.
                
                def try_round(U, V, W, denom):
                    U_r = np.round(U * denom) / denom
                    V_r = np.round(V * denom) / denom
                    W_r = np.round(W * denom) / denom
                    
                    # Verify with exact arithmetic using Fractions to be safe
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = Fraction(0)
                                for c in range(target_rank):
                                    s += Fraction(U_r[i * m + j, c].item()) * Fraction(V_r[j * p + k, c].item()) * Fraction(W_r[k * n + i, c].item())
                                if s != 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        return U_r, V_r, W_r
                    return None

                res = try_round(U, V, W, 1)
                if res:
                    best_U = res[0].tolist()
                    best_V = res[1].tolist()
                    best_W = res[2].tolist()
                    best_rank = target_rank
                    found = True
                    break # Found for this rank, try lower?
                
                res = try_round(U, V, W, 2)
                if res:
                    best_U = res[0].tolist()
                    best_V = res[1].tolist()
                    best_W = res[2].tolist()
                    best_rank = target_rank
                    found = True
                    break

        if found:
            # If we found a decomposition for this rank, we can try to find an even lower one
            # But given time constraints, we might just stop or continue if time permits.
            # Let's continue to see if we can go lower, but break if time is low.
            if time.time() > deadline - 20.0:
                break

    return {"U": best_U, "V": best_V, "W": best_W}
