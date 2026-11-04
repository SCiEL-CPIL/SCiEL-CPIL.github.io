<<<<<<< SEARCH
    # Try to absorb the a=2 part into these 28 terms
    # We need to find U_ext (2 x 28) and V_ext (8 x 28) such that:
    # sum_r U_ext[i, r] * V_ext[k, r] * W_base[k*n+i, r] == 1 for all i, k
    
    # Set U_ext[i, r] = 1 for all i, r.
    # Then for each k, we need sum_r V_ext[k, r] W_base[k*n+i, r] = 1 for all i.
    # This is a system of n equations for rank_base unknowns for each k.
    
    U_ext = [[1] * rank_base for _ in range(n)]
    V_ext = [[0] * rank_base for _ in range(p)]
    
    success = True
    for k in range(p):
        # Build system for this k
        # A_k is n x rank_base
        A_k = np.zeros((n, rank_base))
        for i in range(n):
            row_idx = k * n + i
            for r in range(rank_base):
                A_k[i, r] = W_base[row_idx][r]
        
        b_k = np.ones(n)
        
        try:
            # Solve least squares
            v_k, residuals, rank_A, sv = np.linalg.lstsq(A_k, b_k, rcond=None)
        except:
            success = False
            break
            
        # Round to nearest half-integer
        v_k_rounded = np.round(v_k * 2) / 2
        
        # Check if rounded solution is exact
        for i in range(n):
            acc = 0.0
            row_idx = k * n + i
            for r in range(rank_base):
                acc += v_k_rounded[r] * W_base[row_idx][r]
            if abs(acc - 1.0) > 1e-9:
                success = False
                break
        if not success:
            break
        V_ext[k] = v_k_rounded.tolist()
        
    if success:
        # Construct the final candidate with rank 28
        U_final = [row[:] for row in U_base]
        V_final = [row[:] for row in V_base]
        W_final = [row[:] for row in W_base]
        
        # Add U_ext to U_final (indices for a=2)
        for i in range(n):
            for r in range(rank_base):
                U_final[i * m + 2][r] = U_ext[i][r]
                
        # Add V_ext to V_final (indices for a=2)
        for k in range(p):
            for r in range(rank_base):
                V_final[2 * p + k][r] = V_ext[k][r]
                
        candidate = {"U": U_final, "V": V_final, "W": W_final}
        
        if _validate(candidate):
            return candidate
        else:
            # Fallback to 44 if validation fails due to precision or logic error
            return _schoolbook()
    else:
        # Fallback to 44
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Copy base
        for r in range(rank_base):
            for i in range(n):
                for a in range(2):
                    U[i*m+a][r] = U_base[i*m+a][r]
            for a in range(2):
                for k in range(p):
                    V[a*p+k][r] = V_base[a*p+k][r]
            for k in range(p):
                for i in range(n):
                    W[k*n+i][r] = W_base[k*n+i][r]
                    
        col = rank_base
        for i in range(n):
            for k in range(p):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if not _elapsed_ok():
                    return _schoolbook()

        candidate = {"U": U, "V": V, "W": W}
        if _validate(candidate):
            return candidate
        else:
            return _schoolbook()
=======
    # Try to absorb the a=2 part into the Strassen terms with increasing rank
    # Strategy: for rank R >= 28, use 28 Strassen terms + (R-28) schoolbook terms for a=2
    # But also try to see if we can do better by mixing
    
    def _try_rank(R):
        """Try to construct a valid decomposition of rank R."""
        if R < 28:
            return None
        
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        # Copy the 28 Strassen terms
        for r in range(rank_base):
            for i in range(n):
                for a in range(2):
                    U[i*m+a][r] = U_base[i*m+a][r]
            for a in range(2):
                for k in range(p):
                    V[a*p+k][r] = V_base[a*p+k][r]
            for k in range(p):
                for i in range(n):
                    W[k*n+i][r] = W_base[k*n+i][r]
        
        # Remaining terms: R - 28 schoolbook terms for a=2
        extra = R - rank_base
        col = rank_base
        for i in range(n):
            for k in range(p):
                if col >= R:
                    break
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
            if col >= R:
                break
        
        # We need exactly n*p = 16 schoolbook terms for a=2 if we use pure schoolbook
        # But if R - 28 < 16, we can't cover all (i,k) pairs with schoolbook
        # So this only works if R >= 28 + 16 = 44
        if R < 44:
            # Try the absorption approach for the remaining terms
            # We have 28 Strassen terms, and we need to add (R-28) more terms
            # that cover the a=2 part
            # For each term r in [28, R), we need to assign U[i*m+2, r], V[2*p+k, r], W[k*n+i, r]
            # such that the total contribution to each (i,k) is 1
            # This is a hard problem, so let's just return None for now
            return None
        
        candidate = {"U": U, "V": V, "W": W}
        if _validate(candidate):
            return candidate
        return None
    
    # Try ranks from 28 up to 44
    for R in range(28, 45):
        if not _elapsed_ok():
            break
        if R >= 44:
            # Pure Strassen + schoolbook
            U = [[0] * R for _ in range(n * m)]
            V = [[0] * R for _ in range(m * p)]
            W = [[0] * R for _ in range(p * n)]
            
            for r in range(rank_base):
                for i in range(n):
                    for a in range(2):
                        U[i*m+a][r] = U_base[i*m+a][r]
                for a in range(2):
                    for k in range(p):
                        V[a*p+k][r] = V_base[a*p+k][r]
                for k in range(p):
                    for i in range(n):
                        W[k*n+i][r] = W_base[k*n+i][r]
                        
            col = rank_base
            for i in range(n):
                for k in range(p):
                    if col >= R:
                        break
                    U[i * m + 2][col] = 1
                    V[2 * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
                if col >= R:
                    break
            
            candidate = {"U": U, "V": V, "W": W}
            if _validate(candidate):
                return candidate
        else:
            # For R < 44, try the absorption approach
            # Use 28 Strassen terms and try to solve for the a=2 part
            # with the constraint that we only have R-28 additional "slots"
            # But since the Strassen terms already have non-zero W entries,
            # we can modify U and V for the a=2 indices in the existing 28 terms
            # and add (R-28) new terms
            
            U_final = [row[:] for row in U_base]
            V_final = [row[:] for row in V_base]
            W_final = [row[:] for row in W_base]
            
            # Extend to rank R
            for i in range(n * m):
                U_final[i].extend([0] * (R - rank_base))
            for a in range(m * p):
                V_final[a].extend([0] * (R - rank_base))
            for k in range(p * n):
                W_final[k].extend([0] * (R - rank_base))
            
            # Try to solve for U_ext (2 x R) and V_ext (8 x R)
            # such that sum_r U_ext[i,r] * V_ext[k,r] * W_final[k*n+i, r] = 1
            # For r < 28, W_final[k*n+i, r] is already set
            # For r >= 28, we can set W_final[k*n+i, r] = 1 for all i,k (schoolbook style)
            
            # Set W for new terms to schoolbook pattern
            for r in range(rank_base, R):
                idx = r - rank_base
                i = idx % n
                k = idx // n
                W_final[k * n + i][r] = 1
            
            # Now solve for U_ext and V_ext
            # For simplicity, try U_ext[i,r] = 1 for all i,r
            U_ext = [[1] * R for _ in range(n)]
            V_ext = [[0] * R for _ in range(p)]
            
            success = True
            for k in range(p):
                A_k = np.zeros((n, R))
                for i in range(n):
                    row_idx = k * n + i
                    for r in range(R):
                        A_k[i, r] = W_final[row_idx][r]
                
                b_k = np.ones(n)
                
                try:
                    v_k, residuals, rank_A, sv = np.linalg.lstsq(A_k, b_k, rcond=None)
                except:
                    success = False
                    break
                    
                # Round to nearest integer (since we're using 0/1 entries)
                v_k_rounded = np.round(v_k)
                
                # Check if rounded solution is exact
                for i in range(n):
                    acc = 0.0
                    row_idx = k * n + i
                    for r in range(R):
                        acc += v_k_rounded[r] * W_final[row_idx][r]
                    if abs(acc - 1.0) > 1e-9:
                        success = False
                        break
                if not success:
                    break
                V_ext[k] = v_k_rounded.tolist()
            
            if success:
                for i in range(n):
                    for r in range(R):
                        U_final[i * m + 2][r] = U_ext[i][r]
                for k in range(p):
                    for r in range(R):
                        V_final[2 * p + k][r] = V_ext[k][r]
                
                candidate = {"U": U_final, "V": V_final, "W": W_final}
                if _validate(candidate):
                    return candidate
    
    # Fallback to schoolbook
    return _schoolbook()
>>>>>>> REPLACE

Wait, I realize my approach has issues. Let me reconsider. The problem with trying R < 44 is that the system is underdetermined in a complex way. Let me take a simpler, more reliable approach: just try ranks 44, 43, 42, ... down to 28, and for each rank, construct a valid decomposition if possible.

For rank R where 28 ≤ R ≤ 43, we have 28 Strassen terms plus (R-28) additional terms. The additional terms can be used to cover the a=2 part. Since we need n*p = 16 contributions for a=2, and each additional term can contribute to at most one (i,k) pair if we use schoolbook-style entries, we need at least 16 additional terms, so R ≥ 44.

But we could potentially do better if the additional terms have non-trivial W entries that allow one term to contribute to multiple (i,k) pairs. However, this makes the problem much harder to solve.

Given the time constraint, let me just keep the current approach but fix the validation and make sure we return the best valid result. The current code already tries rank 28 and falls back to 44. Let me verify that the rank 44 construction is correct and validated.

Actually, looking more carefully at the current code, the fallback to rank 44 should work. The issue might be that `_validate` is slow or there's a bug. Let me simplify and make sure the rank 44 case is robust.

Let me try a different improvement: instead of the complex absorption approach, just directly construct rank 44 and validate it. If that works, we get score 1.0909 (same as current). But maybe we can find a rank 43 or lower decomposition.

Let me try: use 4 Strassen blocks for the 2×2×8 part (rank 28), and for the 2×1×8 part, use a smarter decomposition. The 2×1×8 part is just: for each i in {0,1} and k in {0,...,7}, we need U[i*3+2, r] * V[2*8+k, r] * W[k*2+i, r] to sum to 1.

If we use schoolbook, that's 16 terms. But maybe we can do it in fewer terms if we allow the W entries to be non-1. For example, if W[k*2+i, r] = 1 for all k,i in a term, then we need U[i*3+2, r] * V[2*8+k, r] to be 1 for all i,k, which forces U and V to be 1, giving 16 terms.

But if we allow W entries to be different, we might be able to share terms. For example, a single term with W[k*2+0, r] = 1 and W[k*2+1, r] = 1 for all k would require U[0*3+2, r] * V[2*8+k, r] = 1 and U[1*3+2, r] * V[2*8+k, r] = 1 for all k, which means U[2, r] = U[5, r] and V[16+k, r] = 1/U[2, r]. This still requires 8 terms (one per k) if we want to cover all k.

I think rank 44 is likely optimal for this construction family. Let me just make the code more robust and ensure it returns a valid rank-44 decomposition.

Let me simplify the code to be more reliable:

