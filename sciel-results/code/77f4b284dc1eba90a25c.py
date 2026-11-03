import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Try nptools
    try:
        import nptools
        # Construct tensor
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0

        # Attempt decomposition
        # nptools.tensor.decomposition.CP
        # We want a low rank. We don't know the rank, so we might have to guess or let it find it?
        # CP in nptools usually requires a rank parameter.
        # If we don't know the rank, we can try a range of ranks below schoolbook.
        # Or use a method that finds the rank?
        # nptools has `nptools.tensor.decomposition.CP` which takes `rank`.
        # There is also `nptools.tensor.decomposition.CP` with `method='ftd'`?
        # Actually, FTD is often implemented as `nptools.tensor.decomposition.CP` with specific params or a separate function.
        # Let's try ranks from 71 down to maybe 60? That's 12 calls.
        # Each call might be slow.
        # Better: try a few likely candidates.
        
        best_rank = rank_schoolbook
        best_factors = None

        # Try a few ranks below schoolbook
        # Heuristic: Strassen-like reductions might save some rank.
        # For 2x2x2, 8->7.
        # For 3x4x6, maybe 72 -> ?
        # Let's try 71, 70, 69, 68, 67, 66.
        
        for r in range(rank_schoolbook - 1, max(rank_schoolbook - 10, 0), -1):
            try:
                # Use CP decomposition
                # nptools.tensor.decomposition.CP
                # The signature is CP(tensor, rank, ...)
                # We need to check the exact API.
                # Assuming standard nptools API:
                # from nptools.tensor.decomposition import CP
                # res = CP(T, r)
                # res.factors -> list of arrays
                from nptools.tensor.decomposition import CP
                res = CP(T, r, method='ftd') # FTD might be a method option
                # If 'ftd' is not a method, it might fail.
                # Let's catch exceptions.
                
                U, V, W = res.factors
                # Convert to numpy arrays
                U = np.array(U)
                V = np.array(V)
                W = np.array(W)
                
                # Round to nearest integer or half-integer
                # Check if values are close to 0.5 * int
                # Multiply by 2, round, check if close to integer
                U2 = np.round(U * 2)
                V2 = np.round(V * 2)
                W2 = np.round(W * 2)
                
                # Check if rounding is close
                if np.allclose(U * 2, U2) and np.allclose(V * 2, V2) and np.allclose(W * 2, W2):
                    U_int = U2 / 2.0
                    V_int = V2 / 2.0
                    W_int = W2 / 2.0
                    
                    # Verify decomposition
                    # Sum over r of U[:, t] V[:, t] W[:, t]
                    # T_recon = np.einsum('it,jt,kt->ijk', U_int, V_int, W_int)
                    # Or simpler:
                    recon = np.tensordot(U_int, V_int, axes=1)
                    recon = np.tensordot(recon, W_int, axes=1)
                    
                    if np.allclose(recon, T, atol=1e-9):
                        best_rank = r
                        best_factors = (U_int, V_int, W_int)
                        break # Found a valid lower rank, stop
            except Exception:
                continue

        if best_factors is not None:
            U, V, W = best_factors
            # Convert to lists of lists as per seed
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    except Exception:
        pass

    return {"U": U_school, "V": V_school, "W": W_school}
