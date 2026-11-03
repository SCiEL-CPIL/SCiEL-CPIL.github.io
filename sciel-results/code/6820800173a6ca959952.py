import time
import numpy as np
from fractions import Fraction
import sys

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s to be safe within 150s limit

    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 21
    pn = p * n  # 14

    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m + j, j*p + k, k*n + i] = 1.0
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    # Schoolbook decomposition as baseline
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol

    def polish(U, V, W, max_iter=50, tol=1e-12):
        """
        Use Newton's method to polish the decomposition to high precision.
        This helps if ALS converges to a point very close to the solution 
        but with small numerical errors.
        """
        nm, mp, pn = U.shape[0], V.shape[0], W.shape[0]
        R = U.shape[1]
        
        # Flatten indices for easier handling
        # Residual R[i,j,k] = T[i,j,k] - sum_c U[i,c]V[j,c]W[k,c]
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Compute current residual
            T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
            Res = T - T_recon
            err = np.max(np.abs(Res))
            if err < tol:
                break
            
            # Solve for deltas dU, dV, dW using linearization
            # dT = dU * V * W + U * dV * W + U * V * dW
            # This is a linear system. We can solve it iteratively or via least squares on flattened system.
            # For speed, we use a Gauss-Seidel-like update or simple least squares on one block at a time.
            
            # Update U
            # Res[i] = sum_{j,k} Res[i,j,k] e_j e_k
            # We want to find dU such that dU @ (V @ W^T) approximates Res
            # Let M_U[i, c] = sum_{j,k} Res[i,j,k] V[j,c] W[k,c]
            # This is equivalent to:
            # Res_flat[i, j*p+k] = sum_c dU[i,c] * (V[j,c] * W[k,c])
            # Let A_U[i, c] = sum_{j,k} Res[i,j,k] * V[j,c] * W[k,c]
            # This is not quite right. The linearization is:
            # Res[i,j,k] = - sum_c (dU[i,c] V[j,c] W[k,c] + U[i,c] dV[j,c] W[k,c] + U[i,c] V[j,c] dW[k,c])
            # If we fix V and W, we can solve for dU:
            # sum_c dU[i,c] (V[j,c] W[k,c]) = -Res[i,j,k]
            # This is a linear system for each i.
            # Let's construct the matrix for U update.
            # For each i, we have a vector of unknowns dU[i, :] of size R.
            # The equations are for all j, k.
            # Sum_c dU[i,c] * (V[j,c] * W[k,c]) = -Res[i,j,k]
            
            # We can solve this using least squares for each i.
            # Matrix A_i has shape (mp*pn, R) where A_i[j*p+k, c] = V[j,c] * W[k,c]
            # Vector b_i has shape (mp*pn) where b_i[j*p+k] = -Res[i,j,k]
            
            # To save time, we can do this in blocks or use a simpler gradient step if Newton is too slow.
            # Given R is small (~30), mp*pn is ~300, solving 6 systems of size 300x30 is fast.
            
            VW = np.einsum('jc,kc->jk c', V, W) # Shape (mp, pn, R)
            VW_flat = VW.reshape(mp * pn, R)
            
            U_new = U.copy()
            for i in range(nm):
                b = -Res[i].flatten()
                # Solve U_new[i] = argmin || A @ x - b ||
                # A is VW_flat
                try:
                    U_new[i] = np.linalg.lstsq(VW_flat, b, rcond=None)[0]
                except:
                    pass
            
            # Similarly for V
            UW = np.einsum('ic,kc->ik c', U_new, W) # Shape (nm, pn, R)
            UW_flat = UW.reshape(nm * pn, R)
            V_new = V.copy()
            for j in range(mp):
                b = -Res[:, j, :].flatten()
                try:
                    V_new[j] = np.linalg.lstsq(UW_flat, b, rcond=None)[0]
                except:
                    pass
            
            # Similarly for W
            UV = np.einsum('ic,jc->ij c', U_new, V_new) # Shape (nm, mp, R)
            UV_flat = UV.reshape(nm * mp, R)
            W_new = W.copy()
            for k in range(pn):
                b = -Res[:, :, k].flatten()
                try:
                    W_new[k] = np.linalg.lstsq(UV_flat, b, rcond=None)[0]
                except:
                    pass
            
            U, V, W = U_new, V_new, W_new
            
        return U, V, W

    def to_list(mat):
        """Convert numpy array to list of lists with clean values."""
        result = []
        for row in mat:
            result.append([])
            for x in row:
                if abs(x) < 1e-10:
                    result[-1].append(0)
                elif abs(x - 1) < 1e-10:
                    result[-1].append(1)
                elif abs(x + 1) < 1e-10:
                    result[-1].append(-1)
                elif abs(x - 0.5) < 1e-10:
                    result[-1].append(0.5)
                elif abs(x + 0.5) < 1e-10:
                    result[-1].append(-0.5)
                else:
                    # Try to find a simple fraction
                    try:
                        frac = Fraction(x).limit_denominator(1000)
                        if abs(x - float(frac)) < 1e-8:
                            if frac.denominator == 1:
                                result[-1].append(frac.numerator)
                            else:
                                result[-1].append(float(frac))
                        else:
                            result[-1].append(float(x))
                    except:
                        result[-1].append(float(x))
        return result

    # Search Strategy:
    # 1. Try structured sparse initializations for ranks from 35 down to 25.
    # 2. If a low-rank solution is found, polish it.
    
    # Pre-compute some structure for initialization
    # We can use random sparse matrices with entries in {-1, 0, 1}
    
    found = False
    for target_rank in range(35, 24, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random sparse initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            # Initialize U, V, W with sparse random values
            # Use a density that is not too low
            U = np.random.choice([-1, 0, 1], size=(nm, target_rank), p=[0.25, 0.5, 0.25])
            V = np.random.choice([-1, 0, 1], size=(mp, target_rank), p=[0.25, 0.5, 0.25])
            W = np.random.choice([-1, 0, 1], size=(pn, target_rank), p=[0.25, 0.5, 0.25])
            
            # Add a small random noise to break symmetries if needed, but keep it mostly sparse
            U += np.random.randn(nm, target_rank) * 0.01
            V += np.random.randn(mp, target_rank) * 0.01
            W += np.random.randn(pn, target_rank) * 0.01
            
            # Run a few iterations of ALS to get close
            # Simple ALS
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                # Update U
                VW = np.einsum('jc,kc->jk c', V, W).reshape(mp * pn, target_rank)
                for i in range(nm):
                    U[i] = np.linalg.lstsq(VW, T[i].flatten(), rcond=None)[0]
                # Update V
                UW = np.einsum('ic,kc->ik c', U, W).reshape(nm * pn, target_rank)
                for j in range(mp):
                    V[j] = np.linalg.lstsq(UW, T[:, j, :].flatten(), rcond=None)[0]
                # Update W
                UV = np.einsum('ic,jc->ij c', U, V).reshape(nm * mp, target_rank)
                for k in range(pn):
                    W[k] = np.linalg.lstsq(UV, T[:, :, k].flatten(), rcond=None)[0]
            
            # Check error
            T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
            err = np.max(np.abs(T_recon - T))
            
            if err < 1e-6:
                # Polish
                U, V, W = polish(U, V, W)
                T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.max(np.abs(T_recon - T))
                if err < 1e-9:
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
        
        if found:
            # Try to go lower
            continue
        else:
            # If we failed at this rank, it might be possible that lower ranks are also hard,
            # but sometimes a specific structure is needed. 
            # We break to save time for potentially easier lower ranks if we think rank is monotonic?
            # Actually, if we fail at R, we might succeed at R-1 with different init.
            # But to be safe with time, we break if we've spent too much time.
            pass

    # If we found a good one, try to reduce further by 1 or 2
    if found and time.time() < deadline - 20:
        for target_rank in range(best_rank - 1, max(20, best_rank - 3), -1):
            if time.time() > deadline - 10:
                break
            success = False
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                U = np.random.choice([-1, 0, 1], size=(nm, target_rank), p=[0.25, 0.5, 0.25])
                V = np.random.choice([-1, 0, 1], size=(mp, target_rank), p=[0.25, 0.5, 0.25])
                W = np.random.choice([-1, 0, 1], size=(pn, target_rank), p=[0.25, 0.5, 0.25])
                
                for _ in range(30):
                    if time.time() > deadline - 5:
                        break
                    VW = np.einsum('jc,kc->jk c', V, W).reshape(mp * pn, target_rank)
                    for i in range(nm):
                        U[i] = np.linalg.lstsq(VW, T[i].flatten(), rcond=None)[0]
                    UW = np.einsum('ic,kc->ik c', U, W).reshape(nm * pn, target_rank)
                    for j in range(mp):
                        V[j] = np.linalg.lstsq(UW, T[:, j, :].flatten(), rcond=None)[0]
                    UV = np.einsum('ic,jc->ij c', U, V).reshape(nm * mp, target_rank)
                    for k in range(pn):
                        W[k] = np.linalg.lstsq(UV, T[:, :, k].flatten(), rcond=None)[0]
                
                T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.max(np.abs(T_recon - T))
                
                if err < 1e-6:
                    U, V, W = polish(U, V, W)
                    T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
                    err = np.max(np.abs(T_recon - T))
                    if err < 1e-9:
                        best_rank = target_rank
                        best_U = U
                        best_V = V
                        best_W = W
                        success = True
                        break
            if not success:
                break

    # Convert to lists
    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
