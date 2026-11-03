import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    total_eqs = nm * mp * pn
    
    # Build target tensor T_flat
    # T is 1 at (alpha, beta, gamma) if alpha=i*m+j, beta=j*p+k, gamma=k*n+i
    # 0 otherwise
    T_flat = np.zeros(total_eqs)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                idx = alpha * mp * pn + beta * pn + gamma
                T_flat[idx] = 1.0
                
    # Try ranks
    for rank in [16, 15, 14, 13, 12]:
        # Time limit check
        start_time = time.time()
        max_time = 10 # seconds per rank attempt?
        
        for _ in range(10000): # iterations
            if time.time() - start_time > max_time:
                break
            
            # Random U, V with half-integer entries
            # Use -1, -0.5, 0, 0.5, 1
            vals = [-1, -0.5, 0, 0.5, 1]
            U = np.random.choice(vals, size=(nm, rank))
            V = np.random.choice(vals, size=(mp, rank))
            
            # Construct M matrix
            # M has shape (total_eqs, rank * pn)
            # M[idx, c * pn + gamma] = U[alpha, c] * V[beta, c]
            # This is too big to construct explicitly for every iteration if slow.
            # But total_eqs is 972. rank is 16. 972 * 96 ~ 93k entries.
            # It's fine.
            
            M = np.zeros((total_eqs, rank * pn))
            for c in range(rank):
                # u_c is vector of length nm
                # v_c is vector of length mp
                # outer product u_c v_c is nm x mp
                # We need to map (alpha, beta) to idx part
                # idx = alpha * mp * pn + beta * pn + gamma
                # For fixed c and gamma, the column is c * pn + gamma
                # The entry is U[alpha, c] * V[beta, c]
                
                # Vectorize:
                # For each alpha, beta:
                # val = U[alpha, c] * V[beta, c]
                # M[alpha * mp * pn + beta * pn : alpha * mp * pn + beta * pn + pn, c * pn : c * pn + pn] = val
                
                # This loop is slow in Python.
                # Better to construct M using broadcasting or Kronecker products?
                # M is sum over c of (U_c \otimes V_c) \otimes I_pn ? No.
                
                # Let's just use a faster construction or smaller search space.
                # Actually, 10000 iterations * 972*96 operations might be too slow in Python.
                # 10000 * 100k = 1e9 ops. Too slow.
                
                # Optimization:
                # Only check a subset of equations? No, must be exact.
                # Maybe reduce iterations or use C-optimized numpy operations.
                
                # Let's try to construct M more efficiently.
                # M[:, c*pn : (c+1)*pn] = np.kron(U[:, c], np.eye(mp)) ? No.
                # The structure is:
                # Row idx corresponds to (alpha, beta, gamma).
                # Col col corresponds to (c, gamma').
                # M[idx, col] = U[alpha, c] * V[beta, c] * (1 if gamma == gamma' else 0).
                # So M is block diagonal in gamma?
                # For each gamma, the block is U \otimes V (reshaped).
                # M_gamma is nm*mp x rank.
                # M_gamma[alpha*mp+beta, c] = U[alpha, c] * V[beta, c].
                # This is the element-wise product of U and V broadcast?
                # U is nm x rank. V is mp x rank.
                # We want a matrix of size nm*mp x rank.
                # M_gamma = np.kron(U, V)? No.
                # M_gamma[alpha*mp+beta, c] = U[alpha, c] * V[beta, c].
                # This is simply (U \otimes V) if we reshape?
                # Let's check.
                # U[alpha, c] V[beta, c].
                # This is the c-th column of the outer product of U and V?
                # No, it's the element-wise product of U and V for each c.
                # So M_gamma = np.repeat(U, mp, axis=0) * np.tile(V, nm, axis=0)?
                # Yes.
                
                # So we can construct M_gamma quickly.
                # M = stack of M_gamma for gamma in 0..pn-1.
                # But M is total_eqs x rank*pn.
                # M[idx, c*pn + gamma] = M_gamma[alpha*mp+beta, c].
                
                # We can build M by filling blocks.
                pass # Skip explicit M construction in inner loop if possible.
                
                # Alternative:
                # Solve for W using least squares on the fly?
                # Or just check the 18 non-zero equations?
                # If we satisfy the 18 non-zero equations, do we satisfy the zeros?
                # Not necessarily.
                # But if we find a solution with small norm, it might work.
                
                # Let's try a different approach:
                # Use the fact that T is sparse.
                # We only need to satisfy the 18 equations where T=1.
                # And we hope the zeros are satisfied automatically or we check them.
                # If we check all 972, it's expensive.
                # Let's check only the 18 non-zero ones in the search,
                # and verify all 972 at the end.
                
                # So, minimize || A W - T_nonzero ||^2.
                # A has 18 rows.
                # A[idx, c*pn + gamma] = U[alpha, c] V[beta, c] if gamma matches.
                # For the 18 non-zero entries, gamma is fixed.
                # So for each of the 18 equations, we only have 1 non-zero column block?
                # No, for a fixed (alpha, beta, gamma), the column is c*pn + gamma.
                # So each row of A has only `rank` non-zero entries (one for each c, at col c*pn+gamma).
                # This makes A very sparse.
                
                # We can solve this 18 x (rank*pn) system.
                # But it's underdetermined (18 < 16*6=96).
                # We can find the minimum norm solution.
                # Then check if it satisfies the zero equations.
                
                # This is much faster.
                
                # Let's implement this.
                
                pass # Placeholder for logic
            
            # Logic:
            # 1. Pick 18 non-zero indices.
            # 2. For given U, V, construct A (18 x rank*pn) sparse.
            # 3. Solve A W_flat = T_nonzero (1).
            # 4. Get W_flat.
            # 5. Check if W_flat satisfies zero equations.
            # 6. If yes, snap to half-integers and verify.
            
            # To make it faster, we can precompute the indices.
            
            pass # End of loop logic placeholder

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
