import numpy as np
import time
import random
import scipy.optimize

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110 # Safety margin

    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def verify(U, V, W):
        """Check if U, V, W correctly decompose T exactly."""
        # Check if entries are close to integers or half-integers
        # The problem statement allows integer/half-integer entries.
        # We will check if the reconstructed tensor matches T exactly (within float tolerance)
        # and if the factors are "clean".
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(T_recon, T, atol=1e-8):
            return False
        
        # Check if U, V, W have integer or half-integer entries
        # If they are close to k/2, we snap them.
        for M in [U, V, W]:
            # Check distance to nearest half-integer
            half_ints = np.round(M * 2) / 2.0
            if not np.allclose(M, half_ints, atol=1e-5):
                return False
        return True

    def snap_to_half_int(M):
        return np.round(M * 2) / 2.0

    # Rank-23 decomposition (Baseline)
    rank_base = 23
    U_base = np.zeros((n*m, rank_base))
    V_base = np.zeros((m*p, rank_base))
    W_base = np.zeros((p*n, rank_base))
    col = 0

    def add_strassen_block(j_off, k_off):
        """Add a Strassen 2x2x2 block for rows j in {j_off, j_off+1}, cols k in {k_off, k_off+1}."""
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        U_base[0*m + j_off,     col] += 1; U_base[1*m + j_off + 1, col] += 1
        V_base[j_off*p + k_off, col] += 1; V_base[(j_off+1)*p + k_off+1, col] += 1
        W_base[k_off*n + 0, col] += 1; W_base[(k_off+1)*n + 1, col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U_base[0*m + j_off + 1, col] += 1; U_base[1*m + j_off + 1, col] += 1
        V_base[j_off*p + k_off, col] += 1
        W_base[k_off*n + 1, col] += 1; W_base[(k_off+1)*n + 1, col] -= 1
        col += 1
        # M3 = A00(B01-B11)
        U_base[0*m + j_off, col] += 1
        V_base[j_off*p + k_off+1, col] += 1; V_base[(j_off+1)*p + k_off+1, col] -= 1
        W_base[(k_off+1)*n + 0, col] += 1; W_base[(k_off+1)*n + 1, col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U_base[1*m + j_off + 1, col] += 1
        V_base[(j_off+1)*p + k_off, col] += 1; V_base[j_off*p + k_off, col] -= 1
        W_base[k_off*n + 0, col] += 1; W_base[(k_off+1)*n + 0, col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U_base[0*m + j_off, col] += 1; U_base[0*m + j_off + 1, col] += 1
        V_base[(j_off+1)*p + k_off+1, col] += 1
        W_base[k_off*n + 0, col] -= 1; W_base[(k_off+1)*n + 0, col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U_base[0*m + j_off + 1, col] += 1; U_base[0*m + j_off, col] -= 1
        V_base[j_off*p + k_off, col] += 1; V_base[j_off*p + k_off+1, col] += 1
        W_base[(k_off+1)*n + 1, col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U_base[0*m + j_off + 1, col] += 1; U_base[1*m + j_off + 1, col] -= 1
        V_base[(j_off+1)*p + k_off, col] += 1; V_base[(j_off+1)*p + k_off+1, col] += 1
        W_base[k_off*n + 0, col] += 1
        col += 1

    add_strassen_block(0, 0)
    add_strassen_block(0, 2)

    # Block 3: j in {0,1}, k = 4 (Schoolbook 2x2x1, rank 4)
    for i in range(n):
        for j in range(2):
            U_base[i*m + j, col] = 1
            V_base[j*p + 4, col] = 1
            W_base[4*n + i, col] = 1
            col += 1

    # Block 4: j = 2, k in {0..4} (Schoolbook 2x1x5, rank 5)
    for k in range(p):
        U_base[0*m + 2, col] = 1
        U_base[1*m + 2, col] = 1
        V_base[2*p + k, col] = 1
        W_base[k*n + 0, col] = 1
        W_base[k*n + 1, col] = 1
        col += 1

    # Verify baseline
    if verify(U_base, V_base, W_base):
        best_U, best_V, best_W = snap_to_half_int(U_base), snap_to_half_int(V_base), snap_to_half_int(W_base)
        best_rank = rank_base
    else:
        # This should not happen, but handle it
        best_U, best_V, best_W = None, None, None
        best_rank = 30 # Fallback to schoolbook rank

    # Search for lower ranks: 22, 21, 20, 19
    # We use random initialization and gradient descent (L-BFGS-B or similar)
    # to minimize || T - sum(U_r V_r W_r) ||^2
    
    # To speed up, we can use the fact that the tensor is sparse.
    # But scipy.optimize works on flattened vectors.
    
    def objective(params, rank):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank : ].reshape(p*n, rank)
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((T_recon - T)**2)
        return err

    found_lower = False
    for target_rank in [22, 21, 20, 19]:
        if time.time() > deadline:
            break
            
        # Try a few random restarts
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Initialize with random small values
            # Scaling is important. The entries of T are 1.
            # U, V, W should be of order 1.
            init_U = np.random.randn(n*m, target_rank) * 0.5
            init_V = np.random.randn(m*p, target_rank) * 0.5
            init_W = np.random.randn(p*n, target_rank) * 0.5
            
            # Flatten
            x0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            # Use L-BFGS-B
            try:
                res = scipy.optimize.minimize(objective, x0, args=(target_rank,), method='L-BFGS-B', 
                                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.success or res.fun < 1e-8:
                    # Extract U, V, W
                    U_sol = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_sol = res.x[n*m*target_rank + m*p*target_rank : ].reshape(p*n, target_rank)
                    
                    if verify(U_sol, V_sol, W_sol):
                        best_U = snap_to_half_int(U_sol)
                        best_V = snap_to_half_int(V_sol)
                        best_W = snap_to_half_int(W_sol)
                        best_rank = target_rank
                        found_lower = True
                        break # Break restart loop
            except Exception:
                pass
                
        if found_lower:
            break # Break rank loop

    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to rank 30 schoolbook if something went wrong with rank 23 verification
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb))
        V_fb = np.zeros((m*p, rank_fb))
        W_fb = np.zeros((p*n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
