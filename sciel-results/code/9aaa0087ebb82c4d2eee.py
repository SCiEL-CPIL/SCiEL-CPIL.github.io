"""Matrix multiplication 3x5x7 - attempting rank-minimal decomposition."""
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n

    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We build a decomposition with rank r, and try to find a low-rank exact solution.
    #
    # Strategy: Use a greedy randomized search over rational (half-integer) entries.
    # We start with the schoolbook rank-105 decomposition and attempt to reduce rank
    # by finding linear dependencies / cancellations.
    #
    # For small dimensions like 3x5x7, the known lower bound is 3*5*7 = 105 for
    # generic entries, but exact decompositions may exist with rank < 105 for
    # specific structured tensors. However, the matrix multiplication tensor
    # T_{i,j,k} = delta_{j, j} delta_{k,k} (i.e., the identity in the middle index)
    # is NOT a generic tensor — it's a structured one.
    #
    # Actually, the matrix multiplication tensor for (n,m,p) has entries:
    #   T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0.
    # This is the standard multiplication tensor. Its rank is known to be
    # n*m*p for the "trivial" decomposition, but can be lower.
    #
    # For 3x3x3, the best known is 23 (Strassen-like). For 3x5x7, we don't have
    # published sub-105 results easily accessible, so we attempt a computational
    # search.
    #
    # Approach: Build the full tensor T as a 3D array, flatten to a vector of length
    # nm*mp*pn = 105*35*21 = 77175. A rank-1 term is u⊗v⊗w, which contributes
    # u[i]*v[j]*w[k] to T[i,j,k]. We need sum_{r} u_r[i]*v_r[j]*w_r[k] = T[i,j,k].
    #
    # This is a rank-constrained tensor decomposition. We can use the following:
    # Fix U and V (nm x r, mp x r matrices), then W must satisfy:
    #   sum_r U[i,r]*V[j,r]*W[k,r] = T[i,j,k]  for all i,j,k.
    # For fixed i,j: sum_r (U[i,r]*V[j,r]) * W[k,r] = T[i,j,k] for all k.
    # This is a linear system in W: (U[i,:] * V[j,:])^T * W^T = T[i,j,:].
    # Equivalently, for each pair (i,j), the vector T[i,j,:] (length pn) must be
    # in the column space of M_{ij} = U[i,:] ⊙ V[j,:] (element-wise product, length r),
    # i.e., there exists W[:,r] such that sum_r M_{ij}[r]*W[k,r] = T[i,j,k].
    # This is: W^T * M_{ij} = T[i,j,:]^T, where W^T is (pn x r).
    #
    # So: for all (i,j), W^T * (U[i,:] ⊙ V[j,:]) = T[i,j,:]^T.
    # Let B_{ij} = U[i,:] ⊙ V[j,:]. Then W^T * B_{ij} = T[i,j,:]^T.
    # Stacking all (i,j): W^T * [B_{00}, B_{01}, ..., B_{(nm-1)(mp-1)}] = T_flat^T
    # where the stacked matrix is (r x nm*mp) and T_flat is (pn x nm*mp).
    #
    # So W^T = T_flat^T * B^+ (pseudoinverse) if the system is consistent.
    #
    # Alternative: use alternating least squares.
    #
    # Given the 120s budget and the need for exactness with integer/half-integer entries,
    # we try a structured approach:
    #
    # 1. Start with schoolbook (rank 105, all ones).
    # 2. Try to merge/cancel pairs of rank-1 terms that sum to something simpler.
    # 3. Use Strassen's trick for the 3x3 blocks.
    #
    # For 3x5x7: decompose into 5x7 = 5*(3+4) or 3x5 = 3*(1+1+1) etc.
    #
    # Let's try: 3x5x7 = 3*(5x7). The 5x7 matrix multiplication can be done with
    # rank at most 5*7 = 35. So total 3*35 = 105.
    #
    # Can we do better? 5x7: lower bound is 5*7 = 35? No, lower bound for n x p
    # multiplication tensor rank is n*p (since each output element depends on distinct
    # products). Actually the rank is exactly n*p for the "outer" product structure.
    #
    # I'll try a computational search for rank < 105 using ALS with rational rounding.

    # Build the target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    # Try to find a decomposition with rank r using ALS
    best_rank = None
    best_result = None

    # Start from r = 105 and try to reduce
    # Due to time constraints, try a few specific ranks
    for r in [105, 104, 103, 102, 100, 95, 90]:
        if r > nm * mp:  # r must be at least max(nm, mp, pn) for the ALS to work well
            pass
        
        np.random.seed(42)
        U = np.random.randn(nm, r) * 0.1
        V = np.random.randn(mp, r) * 0.1
        W = np.random.randn(pn, r) * 0.1

        # ALS iterations
        converged = False
        for it in range(200):
            # Update W: for each i,j, solve W^T * (U[i]⊙V[j]) = T[i,j,:]
            B = np.zeros((r, nm * mp))
            for idx, (i, j) in enumerate(product(range(nm), range(mp))):
                B[:, idx] = U[i, :] * V[j, :]
            
            T_flat = T.reshape(pn, nm * mp)
            # W^T * B = T_flat
            # B^T * W = T_flat^T
            # W = (B^T)^+ * T_flat^T
            try:
                W_new, _, _, _ = np.linalg.lstsq(B.T, T_flat.T, rcond=None)
                W = W_new.T
            except:
                break

            # Update V: for each i,k, solve V^T * (U[i]⊙W[k]) = T[i,:,k]
            B2 = np.zeros((r, nm * pn))
            for idx, (i, k) in enumerate(product(range(nm), range(pn))):
                B2[:, idx] = U[i, :] * W[k, :]
            
            T_flat2 = T.reshape(nm, mp * pn).T.reshape(pn, nm * mp).T  # This is getting complex
            # T[i, j, k] -> for fixed i,k, we need T[i, :, k] which is length mp
            T_ik = T.transpose(0, 2, 1).reshape(nm * pn, mp)
            # V^T * B2 = T_ik^T ? No.
            # sum_r U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
            # For fixed i,k: sum_r (U[i,r]*W[k,r]) * V[j,r] = T[i,j,k] for all j
            # So: V^T * (U[i]⊙W[k]) = T[i,:,k]^T
            B2 = np.zeros((r, nm * pn))
            for idx, (i, k) in enumerate(product(range(nm), range(pn))):
                B2[:, idx] = U[i, :] * W[k, :]
            T_ik_flat = T.transpose(0, 2, 1).reshape(nm * pn, mp)
            try:
                V_new, _, _, _ = np.linalg.lstsq(B2.T, T_ik_flat.T, rcond=None)
                V = V_new.T
            except:
                break

            # Update U: for each j,k, solve U^T * (V[j]⊙W[k]) = T[:,j,k]
            B3 = np.zeros((r, mp * pn))
            for idx, (j, k) in enumerate(product(range(mp), range(pn))):
                B3[:, idx] = V[j, :] * W[k, :]
            T_jk_flat = T.transpose(1, 2, 0).reshape(mp * pn, nm)
            try:
                U_new, _, _, _ = np.linalg.lstsq(B3.T, T_jk_flat.T, rcond=None)
                U = U_new.T
            except:
                break

            # Check convergence
            recon = np.einsum('ir,jr,kr->ijk', U, V, W)
            err = np.linalg.norm(recon - T)
            if err < 1e-10:
                converged = True
                break

        if converged:
            # Try to round to half-integers
            U_half = np.round(U * 2) / 2
            V_half = np.round(V * 2) / 2
            W_half = np.round(W * 2) / 2
            recon_half = np.einsum('ir,jr,kr->ijk', U_half, V_half, W_half)
            if np.allclose(recon_half, T, atol=1e-9):
                best_rank = r
                best_result = (U_half, V_half, W_half)
                break

    if best_result is not None:
        U, V, W = best_result
    else:
        # Fallback to schoolbook
        U = np.zeros((nm, nm * mp))
        V = np.zeros((mp, nm * mp))
        W = np.zeros((pn, nm * mp))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1

    # Convert to list of lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    # Ensure entries are integer or half-integer (multiply by 2 and check)
    # They should be already from rounding

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
