import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Build schoolbook
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_schoolbook

    # Target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    T_flat = T.flatten()

    start_time = time.time()
    deadline = start_time + 140.0

    # Try to find lower rank
    for target_rank in [28, 29]:
        if time.time() > deadline:
            break
            
        found = False
        # Try a few restarts
        for restart in range(20):
            if time.time() > deadline:
                break
                
            # Initialize with schoolbook + random noise
            np.random.seed(restart + target_rank)
            U = U_sb[:, :target_rank].copy() if target_rank <= rank_schoolbook else U_sb
            V = V_sb[:, :target_rank].copy() if target_rank <= rank_schoolbook else V_sb
            W = W_sb[:, :target_rank].copy() if target_rank <= rank_schoolbook else W_sb
            
            if target_rank < rank_schoolbook:
                # Add some random columns to break symmetry
                U[:, -5:] += np.random.randn(n*m, 5) * 0.1
                V[:, -5:] += np.random.randn(m*p, 5) * 0.1
                W[:, -5:] += np.random.randn(p*n, 5) * 0.1
            else:
                U += np.random.randn(n*m, target_rank) * 0.1
                V += np.random.randn(m*p, target_rank) * 0.1
                W += np.random.randn(p*n, target_rank) * 0.1

            # ALS optimization
            for it in range(50):
                # Solve for U
                T_mat = T.reshape(n*m, m*p*p*n)
                A_mat = np.zeros((m*p*p*n, target_rank))
                for r in range(target_rank):
                    A_mat[:, r] = (V[:, r] * W[:, r]).flatten()
                try:
                    U = np.linalg.lstsq(A_mat.T, T_mat.T, rcond=None)[0].T
                except:
                    break
                
                # Solve for V
                T_mat2 = T.reshape(m*p, n*m*p*n)
                A_mat2 = np.zeros((n*m*p*n, target_rank))
                for r in range(target_rank):
                    A_mat2[:, r] = (U[:, r] * W[:, r]).flatten()
                try:
                    V = np.linalg.lstsq(A_mat2.T, T_mat2.T, rcond=None)[0].T
                except:
                    break
                    
                # Solve for W
                T_mat3 = T.reshape(p*n, n*m*m*p)
                A_mat3 = np.zeros((n*m*m*p, target_rank))
                for r in range(target_rank):
                    A_mat3[:, r] = (U[:, r] * V[:, r]).flatten()
                try:
                    W = np.linalg.lstsq(A_mat3.T, T_mat3.T, rcond=None)[0].T
                except:
                    break
            
            # Check residual
            T_recon = np.tensordot(U, np.tensordot(V, W, axes=1), axes=1)
            T_recon_flat = T_recon.flatten()
            residual = np.linalg.norm(T_recon_flat - T_flat)
            
            if residual < 1e-6:
                # Try to snap to integers/half-integers
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                T_recon_int = np.tensordot(U_int, np.tensordot(V_int, W_int, axes=1), axes=1)
                T_recon_int_flat = T_recon_int.flatten()
                residual_int = np.linalg.norm(T_recon_int_flat - T_flat)
                
                if residual_int < 1e-9:
                    best_rank = target_rank
                    best_U = U_int.tolist()
                    best_V = V_int.tolist()
                    best_W = W_int.tolist()
                    found = True
                    break

        if found:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
