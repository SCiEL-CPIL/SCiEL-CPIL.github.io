import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    deadline = time.time() + 110  # safety margin
    
    def try_rank(r, max_trials=50):
        nonlocal best_rank, best_U, best_V, best_W
        if r >= best_rank:
            return False
        
        # Pre-compute indices for vectorization
        # We need to build A (7056 x 21*r) and b (7056,)
        # A[row, col] where row=(j,k,i,jp,kp,ip), col=(kp,ip,c)
        # A[row, col] = U[i*m+j, c] * V[jp*p+k, c] if row's kp,ip matches col's kp,ip?
        # No, col is (kp, ip, c). Row is (j, k, i, jp, kp, ip).
        # The variable is W[kp, ip, c].
        # The coefficient in the equation for (j,k,i,jp,kp,ip) for variable W[kp', ip', c'] is:
        # Non-zero only if kp'=kp, ip'=ip, c'=c.
        # Value: U[i, j, c] * V[jp, k, c]
        
        # So A is block diagonal in (kp, ip) blocks?
        # For each (kp, ip), we have a block of size (m*p*n*m) x r?
        # Rows for fixed (kp, ip): (j, k, i, jp)
        # Cols for fixed (kp, ip): (c)
        # Wait, the row index includes kp, ip.
        # So for a specific row (j, k, i, jp, kp, ip), it only has non-zeros in columns (kp, ip, c) for all c.
        # So the matrix is block diagonal with respect to the (kp, ip) grouping of rows and columns.
        # Block size: (m*p*n*m) x (p*n*r)? No.
        # Rows are indexed by (j, k, i, jp, kp, ip).
        # Cols are indexed by (kp, ip, c).
        # If we group rows by (kp, ip), there are m*p*n*m rows for each (kp, ip).
        # If we group cols by (kp, ip), there are r cols for each (kp, ip).
        # So we have p*n blocks, each of size (m*p*n*m) x r.
        # This is great! We can solve p*n independent systems?
        # No, the RHS b depends on the row.
        # b[row] = 1 if j==jp and k==kp and i==ip else 0.
        # So for a specific block (kp, ip), the RHS is non-zero only for rows where j==jp, k==kp, i==ip.
        # There is only ONE such row in the block: (j=kp? No, j is 0..m-1, kp is 0..p-1).
        # Wait, j and kp are different indices.
        # The condition is j==jp, k==kp, i==ip.
        # In the block for (kp, ip), the rows are (j, k, i, jp).
        # The RHS is 1 if j==jp AND k==kp AND i==ip.
        # Note that kp and ip are FIXED for the block.
        # So the condition is j==jp, k==kp, i==ip.
        # So for each block (kp, ip), there is exactly one row (j=kp? No, j is free, but must equal jp. k must equal kp. i must equal ip.)
        # Wait, j ranges 0..m-1, jp ranges 0..m-1.
        # k ranges 0..p-1, kp ranges 0..p-1.
        # i ranges 0..n-1, ip ranges 0..n-1.
        # So for a fixed block (kp, ip), the row (j, k, i, jp) has RHS=1 iff j==jp, k==kp, i==ip.
        # So yes, exactly one row per block has RHS=1, the others 0.
        # And that row is (j=kp? No, j is independent. The row index is (j, k, i, jp).
        # The condition is j==jp, k==kp, i==ip.
        # So the specific row is (j=jp, k=kp, i=ip).
        
        # So we have p*n independent linear systems!
        # Each system is of size (m*p*n*m) x r.
        # RHS is a vector with a single 1.
        
        # Let's verify this decomposition.
        # Original equation:
        # Sum_c U[i, j, c] V[jp, k, c] W[kp, ip, c] = delta(j, jp) delta(k, kp) delta(i, ip)
        # For fixed (kp, ip), let's look at the variables W[kp, ip, c].
        # The equations involving these variables are for all (j, k, i, jp).
        # But the LHS only depends on W[kp, ip, c] if the row's kp, ip match the variable's kp, ip.
        # Yes, the row index contains kp, ip.
        # So for a fixed (kp, ip), we consider all rows (j, k, i, jp, kp, ip).
        # The equation is:
        # Sum_c U[i, j, c] V[jp, k, c] W[kp, ip, c] = delta(j, jp) delta(k, kp) delta(i, ip)
        
        # This is a linear system in W[kp, ip, c].
        # Number of equations: m * p * n * m = m^2 p n.
        # Number of unknowns: r.
        
        # We can solve this for each (kp, ip) independently!
        # This is a massive speedup.
        
        for trial in range(max_trials):
            if time.time() > deadline:
                return False
            
            # Random integer U and V
            U = np.random.randint(-2, 3, size=(n * m, r))
            V = np.random.randint(-2, 3, size=(m * p, r))
            
            # Reshape U and V for easy indexing
            # U[i, j, c]
            U_reshaped = U.reshape(n, m, r)
            # V[jp, k, c]
            V_reshaped = V.reshape(m, p, r)
            
            W_flat = np.zeros(p * n * r)
            valid = True
            
            # Iterate over all (kp, ip) blocks
            for kp in range(p):
                for ip in range(n):
                    # Build the system A_block * x = b_block
                    # x is W[kp, ip, c] for c in 0..r-1
                    # Equations for (j, k, i, jp)
                    # A_block[row_idx, c] = U[i, j, c] * V[jp, k, c]
                    # b_block[row_idx] = 1 if j==jp and k==kp and i==ip else 0
                    
                    # We can build this block efficiently.
                    # Rows: j, k, i, jp
                    # Let's iterate over j, k, i, jp
                    A_block_rows = []
                    b_block_vals = []
                    
                    for j in range(m):
                        for k in range(p):
                            for i in range(n):
                                for jp in range(m):
                                    # Coefficients for c
                                    # coeff_c = U[i, j, c] * V[jp, k, c]
                                    coeffs = U_reshaped[i, j, :] * V_reshaped[jp, k, :]
                                    A_block_rows.append(coeffs)
                                    rhs = 1.0 if (j == jp and k == kp and i == ip) else 0.0
                                    b_block_vals.append(rhs)
                    
                    A_block = np.array(A_block_rows) # Shape (m^2 p n, r)
                    b_block = np.array(b_block_vals)
                    
                    # Solve least squares
                    # Since the system is overdetermined (usually), use lstsq
                    try:
                        x, residuals, rank, sv = np.linalg.lstsq(A_block, b_block, rcond=None)
                        # Check residual
                        residual = A_block @ x - b_block
                        if np.max(np.abs(residual)) < 1e-6:
                            # Check integrality
                            if np.all(np.abs(x - np.round(x)) < 1e-4):
                                W_block = np.round(x).astype(int)
                                W_flat[kp * n * r + ip * r : (kp * n + ip + 1) * r] = W_block
                            else:
                                # Allow half-integers?
                                x_half = 2 * x
                                if np.all(np.abs(x_half - np.round(x_half)) < 1e-4):
                                    W_block = np.round(x_half).astype(int) / 2.0
                                    W_flat[kp * n * r + ip * r : (kp * n + ip + 1) * r] = W_block
                                else:
                                    valid = False
                                    break
                        else:
                            valid = False
                            break
                    except:
                        valid = False
                        break
                
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition
                W_mat = W_flat.reshape(p * n, r)
                U_int = U.astype(int)
                V_int = V.astype(int)
                
                # W might be float if we used half-integers
                # The problem allows integer/half-integer.
                # Let's verify with floats for now, then convert if needed.
                ok = True
                for j in range(m):
                    for k in range(p):
                        for i in range(n):
                            for jp in range(m):
                                for kp in range(p):
                                    for ip in range(n):
                                        s = 0.0
                                        for c in range(r):
                                            s += U_int[i*m+j, c] * V_int[jp*p+k, c] * W_mat[kp*n+ip, c]
                                        expected = 1.0 if (j == jp and k == kp and i == ip) else 0.0
                                        if abs(s - expected) > 1e-6:
                                            ok = False
                                            break
                                    if not ok:
                                        break
                                if not ok:
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    best_rank = r
                    best_U = U_int
                    best_V = V_int
                    # Convert W to integer if possible, else keep float?
                    # The interface expects integer/half-integer.
                    # If W_mat has halves, we can't just astype(int).
                    # But the return type is list.
                    # We can return floats if they are exact halves.
                    best_W = W_mat
                    return True
        
        return False
    
    # Try ranks from 7 upward
    # Theoretical lower bound for 3x4x7 is likely > 20
    for r in range(20, best_rank + 1):
        if time.time() > deadline:
            break
        if try_rank(r, max_trials=50):
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
