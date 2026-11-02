import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_entries = nm * mp * pn
    
    # Build the target tensor T flattened
    # T[a, b, c] = 1 if indices are consistent
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T_flat = np.zeros(total_entries, dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx = (c * mp + b) * nm + a  # Row-major: c is slowest, a is fastest?
                # Let's use a consistent flattening:
                # Index = c * (mp * nm) + b * nm + a
                T_flat[idx] = 1

    def verify(U, V, W):
        """Check if U, V, W exactly decompose T."""
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Compute sum_r U[:,r] V[:,r]^T W[c,r]
        # This is equivalent to:
        # For each c, T_c = sum_r W[c,r] (U[:,r] V[:,r]^T)
        # Flatten T_c to vector, compare with T_flat segment
        
        # Efficient check:
        # Compute M = U @ V.T  (nm x mp) for each r? No, sum weighted by W.
        # Result[c, a, b] = sum_r U[a,r] V[b,r] W[c,r]
        
        # Vectorized:
        # Reshape U to (nm, 1, R), V to (1, mp, R), W to (pn, 1, 1, R)?
        # Simpler:
        # Product = np.einsum('ar,br,cr->abc', U, V, W)
        # But this is slow for large R.
        
        # Alternative:
        # For each r, term_r = np.outer(U[:,r], V[:,r]) # nm x mp
        # Sum over r weighted by W[c,r]
        
        # Let's do:
        # Temp = U @ V.T # No, this sums over r without W.
        
        # We can compute:
        # P = np.einsum('ar,cr->ac', U, W) # (nm, pn)
        # Then Result[c, a, b] = sum_a P[a, c] V[b, r]? No.
        
        # Correct einsum:
        # 'ar,br,cr->abc'
        # This creates a (n*m, m*p, p*n) array.
        # Then flatten and compare to T_flat.
        
        try:
            # Use float64 for precision during check
            U_f = U.astype(np.float64)
            V_f = V.astype(np.float64)
            W_f = W.astype(np.float64)
            
            # If R is large, this might be slow, but R < 90 is small.
            prod = np.einsum('ar,br,cr->abc', U_f, V_f, W_f, optimize=True)
            prod_flat = prod.flatten()
            
            # Check if prod_flat matches T_flat
            # T_flat is int8, prod_flat is float
            # They should be exactly 0 or 1
            return np.allclose(prod_flat, T_flat, atol=1e-9)
        except Exception:
            return False

    def solve_for_W(U, V):
        """
        Given U (nm x R) and V (mp x R), find W (pn x R) such that
        sum_r W[c,r] U[a,r] V[b,r] = T[a,b,c]
        This is a linear system Z w = b.
        """
        R = U.shape[1]
        # Construct Z: (total_entries x R)
        # Z[idx, r] = U[a,r] * V[b,r]
        # where idx corresponds to (c, b, a)
        
        # Vectorized construction of Z
        # a indices: 0..nm-1
        # b indices: 0..mp-1
        # c indices: 0..pn-1
        
        # We need to map (c, b, a) to idx.
        # idx = c * (mp * nm) + b * nm + a
        
        # Create arrays for a, b, c
        a_idx = np.arange(nm)
        b_idx = np.arange(mp)
        c_idx = np.arange(pn)
        
        # Meshgrid
        # To save memory, we might not want to create full Z if R is large, but R is small.
        # Z shape: (pn * mp * nm, R)
        
        # We can build Z column by column or use broadcasting.
        # Let's build Z using broadcasting.
        # U shape (nm, R) -> (1, 1, nm, R)
        # V shape (mp, R) -> (1, mp, 1, R)
        # Product shape (1, mp, nm, R)
        # We need to tile this for c.
        
        # Actually, Z[c, b, a, r] = U[a, r] * V[b, r]
        # This is independent of c!
        # So Z is a block diagonal structure if we group by c?
        # No, the equation is:
        # For each c, sum_r W[c,r] (U[a,r] V[b,r]) = T[c,b,a]
        
        # So for a fixed c, we have a system:
        # sum_r W[c,r] Z_c[b,a,r] = T_c[b,a]
        # where Z_c[b,a,r] = U[a,r] V[b,r]
        
        # Since Z_c is the same for all c (depends only on U, V), let M_r = U[:,r] V[:,r]^T.
        # We need sum_r W[c,r] M_r = T_c (flattened).
        
        # Let's form the matrix Z_full of size (nm * mp) x R.
        # Z_full[b*nm+a, r] = U[a,r] V[b,r]
        
        Z_block = np.zeros((nm * mp, R))
        for r in range(R):
            Z_block[:, r] = np.outer(V[:, r], U[:, r]).flatten() # V[b,r] U[a,r]
            # Note: flatten order must match T_flat.
            # T_flat index: c * (mp*nm) + b * nm + a
            # So within a c-block, index is b * nm + a.
            # np.outer(V, U) has shape (mp, nm). Flatten is b-major, then a. Correct.
        
        # Now, for each c, we have:
        # Z_block @ W[c, :] = T_c_flat
        # T_c_flat is T_flat[c * (mp*nm) : (c+1) * (mp*nm)]
        
        W_solutions = []
        for c in range(pn):
            T_c = T_flat[c * (nm * mp) : (c + 1) * (nm * mp)]
            # Solve Z_block @ w = T_c
            # Use least squares
            w, residuals, rank, s = np.linalg.lstsq(Z_block, T_c, rcond=None)
            W_solutions.append(w)
            
        W = np.array(W_solutions).T # Shape (pn, R)
        return W

    def try_rank(R, num_trials=10):
        for _ in range(num_trials):
            if time.time() > deadline:
                return None
            
            # Random integer U, V in {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(nm, R))
            V = np.random.randint(-1, 2, size=(mp, R))
            
            # Solve for W
            W = solve_for_W(U, V)
            
            # Check if W is close to integer/half-integer
            # Round to nearest 0.5
            W_rounded = np.round(W * 2) / 2.0
            
            # Verify with rounded W
            if verify(U, V, W_rounded):
                return U, V, W_rounded
                
        return None

    # Try ranks from lower bound to schoolbook
    # Lower bound is max(nm, mp, pn) = 30
    # We'll try a few specific ranks
    ranks_to_try = [35, 40, 45, 50, 60, 70, 80]
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        result = try_rank(R, num_trials=20)
        if result is not None:
            U, V, W = result
            # Convert to lists of lists as required
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }

    # Fallback to schoolbook
    rank = n * m * p
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
