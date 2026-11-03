"""Program to find a rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication."""
import time
import itertools
from fractions import Fraction

import numpy as np
import scipy.linalg as sla

DIMENSIONS = [3, 3, 5]
n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n
RANK_SCHOOLBOOK = NM * MP  # 45

_DEADLINE = 145.0  # seconds, safety margin from 150s kill


def _build_schoolbook():
    rank = RANK_SCHOOLBOOK
    U = np.zeros((NM, rank))
    V = np.zeros((MP, rank))
    W = np.zeros((PN, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _target_tensor():
    T = np.zeros((NM, MP, PN))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T


def _verify(U, V, W, tol=1e-6):
    r = U.shape[1]
    # Sum over rank: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Efficient check: for each r, outer product
    # T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    # But that's expensive; instead check via contraction
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Check: for each r, the rank-1 contribution is outer(U[:,r], outer(V[:,r], W[:,r]))
    # Sum them up
    T_recon = np.zeros((NM, MP, PN))
    for r in range(r):
        T_recon += np.outer(U[:, r], np.outer(V[:, r], W[:, r]))
    T_target = _target_tensor()
    return np.max(np.abs(T_recon - T_target)) < tol


def _try_remove_one_term():
    """Try to find a decomposition of rank 44 by removing one term from schoolbook
    and checking if the remaining structure still spans the target tensor."""
    U0, V0, W0 = _build_schoolbook()
    rank = RANK_SCHOOLBOOK
    T = _target_tensor()
    
    best = (U0.copy(), V0.copy(), W0.copy())
    
    for remove_col in range(rank):
        if time.time() > _DEADLINE:
            break
        # Remove this column
        mask = np.ones(rank, dtype=bool)
        mask[remove_col] = False
        U = U0[:, mask]
        V = V0[:, mask]
        W = W0[:, mask]
        
        # The missing term contributed: U[:,remove_col] ⊗ V[:,remove_col] ⊗ W[:,remove_col]
        # This is a rank-1 tensor. We need to see if the remaining terms can span it.
        # Actually, we need to check if the remaining 44 rank-1 tensors span the full 45-dimensional
        # space of the target tensor. Since the target tensor is exactly the sum of 45 rank-1 tensors,
        # and we removed one, the remaining sum is T - T_removed.
        # For the decomposition to be valid with 44 terms, we need T = sum of 44 rank-1 tensors.
        # This is generally not possible by just removing one term.
        # Skip this approach - it won't work directly.
        pass
    
    return best


def _randomized_search(max_iter=2000):
    """Try to find a lower-rank decomposition via random initialization and projection."""
    np.random.seed(42)
    best_U, best_V, best_W = _build_schoolbook()
    best_rank = RANK_SCHOOLBOOK
    T = _target_tensor()
    
    # Try rank 44
    for trial_rank in [44]:
        for attempt in range(max_iter // 2):
            if time.time() > _DEADLINE:
                return best_U, best_V, best_W
            
            r = trial_rank
            U = np.random.randn(NM, r) * 0.1
            V = np.random.randn(MP, r) * 0.1
            W = np.random.randn(PN, r) * 0.1
            
            # Iterate: fix two, update the third via least squares
            for inner in range(50):
                if time.time() > _DEADLINE:
                    return best_U, best_V, best_W
                
                # Update W: for each c, T[:, :, c] = sum_r U[:,r]*V[:,r]*W[c,r]
                # T[:, :, c] flattened = sum_r (U*V)[:,r] * W[c,r]
                # So T_flat[:, c] = UV^T @ W[:, c] where UV is NM*MP x r
                UV = (U[:, :, None] * V[None, :, :]).reshape(NM * MP, r)
                # T reshaped: (NM*MP, PN)
                T_mat = T.reshape(NM * MP, PN)
                # T_mat = UV @ W^T  =>  W^T = UV^+ @ T_mat
                W_new = sla.lstsq(UV, T_mat)[0].T  # (r, PN) -> need (PN, r)
                W_new = W_new.T  # (PN, r)
                
                # Update U: T_mat = (U*V) @ W^T
                # T_mat = UV @ W^T where UV[a*MP+b, r] = U[a,r]*V[b,r]
                # T_mat[am+b, c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # For fixed b,c: sum_r U[a,r] * (V[b,r]*W[c,r]) = T_mat[a*MP+b, c]
                # So T_mat[:, :] = U @ diag stuff... 
                # Actually: T_mat[a*MP+b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # = sum_r U[a,r] * (V[b,r]*W[c,r])
                # Let M_bc = V[b,r]*W[c,r] for all r, then T_mat[a*MP+b, c] = U[a,:] dot M_bc
                # This is a 3-way contraction.
                # Easier: for each (b,c), we have U @ M_bc = T_mat[:, a*MP+b, c]... 
                # Let's use: reshape approach
                # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
                # Fix V and W, solve for U:
                # For each (b,c): T[:, b, c] = U @ (V[:,b] * W[:,c] ... no
                # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # Let Q[r] = V[b,r]*W[c,r], then T[:,b,c] = U @ Q
                # But Q depends on (b,c). We can vectorize:
                # T reshaped as (NM, MP*PN): T2[a, b*PN+c] = sum_r U[a,r] * (V[b,r]*W[c,r])
                # Let B[r] = V[:,r] * W[:,r] ... no, V is (MP, r), W is (PN, r)
                # V[b,r]*W[c,r] for each (b,c,r)
                # T2[a, b*PN+c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # Define M[r] as a matrix (MP, PN): M[r, b, c] = V[b,r]*W[c,r]
                # Then T2 = U @ M_flat where M_flat[r, b*PN+c] = V[b,r]*W[c,r]
                M_flat = np.zeros((r, MP * PN))
                for rr in range(r):
                    M_flat[rr, :] = (V[:, rr] * W[:, rr].reshape(PN, 1)).T.reshape(-1)
                    # Actually V[:,rr] is (MP,), W[:,rr] is (PN,)
                    # outer(V[:,rr], W[:,rr]) is (MP, PN), flatten to (MP*PN,)
                M_flat = np.zeros((r, MP * PN))
                for rr in range(r):
                    M_flat[rr, :] = np.outer(V[:, rr], W[:, rr]).reshape(-1)
                
                T2 = T.reshape(NM, MP * PN)
                U_new = sla.lstsq(M_flat.T, T2.T)[0].T  # Solve M_flat.T @ U_new.T = T2.T
                # M_flat.T is (MP*PN, r), T2.T is (MP*PN, NM)
                # U_new.T = M_flat^+ @ T2.T, so U_new = (M_flat^+ @ T2.T)^T = T2 @ (M_flat^+)^T
                # Just use lstsq properly
                U_new = sla.lstsq(M_flat.T, T2.T)[0].T
                
                # Update V similarly
                T3 = T.reshape(MP, NM * PN)
                # T3[b, a*PN+c] = sum_r V[b,r] * U[a,r] * W[c,r]
                N_flat = np.zeros((r, NM * PN))
                for rr in range(r):
                    N_flat[rr, :] = np.outer(U[:, rr], W[:, rr]).reshape(-1)
                V_new = sla.lstsq(N_flat.T, T3.T)[0].T
                
                U, V, W = U_new, V_new, W_new
            
            # Check residual
            T_recon = np.zeros((NM, MP, PN))
            for rr in range(r):
                T_recon += np.outer(U[:, rr], np.outer(V[:, rr], W[:, rr]))
            err = np.max(np.abs(T_recon - T))
            if err < 1e-4:
                return U, V, W
    
    return best_U, best_V, best_W


def construct():
    start_time = time.time()
    
    # Start with schoolbook (guaranteed valid)
    U, V, W = _build_schoolbook()
    
    # Verify schoolbook
    if not _verify(U, V, W):
        # Should not happen, but just in case
        pass
    
    # Try to find lower rank
    U, V, W = _randomized_search(max_iter=500)
    
    # Final verification
    if not _verify(U, V, W, tol=1e-4):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook()
    
    # Convert to lists with integer/half-integer entries if possible
    def to_fraction_list(arr):
        r, c = arr.shape
        result = []
        for i in range(r):
            row = []
            for j in range(c):
                val = arr[i, j]
                # Round to nearest half-integer if close
                val_rounded = round(val * 2) / 2
                if abs(val - val_rounded) < 1e-6:
                    # Represent as integer or half-integer
                    if val_rounded == int(val_rounded):
                        row.append(int(val_rounded))
                    else:
                        row.append(Fraction(val_rounded * 2, 2))
                else:
                    row.append(Fraction(str(val)))
            result.append(row)
        return result
    
    U_list = to_fraction_list(U)
    V_list = to_fraction_list(V)
    W_list = to_fraction_list(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
