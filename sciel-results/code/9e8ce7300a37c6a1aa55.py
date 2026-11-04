import numpy as np
import time

DIMENSIONS = [4, 5, 5]

def verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                # Sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                actual = 0
                for r in range(rank):
                    actual += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(actual - expected) > 1e-10:
                    return False
    return True

def get_rank(U):
    return len(U[0]) if U else 0

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline for 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Attempt to reduce rank through elimination
    # Try to find columns that can be eliminated
    
    def try_eliminate_column(U, V, W, col_idx, n, m, p):
        """Try to eliminate a column by expressing it as combination of others."""
        rank = get_rank(U)
        if rank <= 1:
            return None, None, None, False
        
        # Get the column
        u_col = [U[i][col_idx] for i in range(nm)]
        v_col = [V[i][col_idx] for i in range(mp)]
        w_col = [W[i][col_idx] for i in range(pn)]
        
        # Check if this column is all zeros
        if all(x == 0 for x in u_col) and all(x == 0 for x in v_col) and all(x == 0 for x in w_col):
            # Remove this column
            new_U = [[U[i][j] for j in range(rank) if j != col_idx] for i in range(nm)]
            new_V = [[V[i][j] for j in range(rank) if j != col_idx] for i in range(mp)]
            new_W = [[W[i][j] for j in range(rank) if j != col_idx] for i in range(pn)]
            return new_U, new_V, new_W, True
        
        # Try to express this column as a linear combination of other columns
        # For each tensor coordinate (i,j,k), we need:
        # u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i] = sum_r c_r * U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        
        # Build system of equations
        # For each (i,j,k) where the product is non-zero, we get an equation
        equations = []  # (coefficients, rhs)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = u_col[i*m + j]
                    v_val = v_col[j*p + k]
                    w_val = w_col[k*n + i]
                    rhs = u_val * v_val * w_val
                    if rhs != 0:
                        coeffs = []
                        for r in range(rank):
                            if r == col_idx:
                                continue
                            coeffs.append(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r])
                        equations.append((coeffs, rhs))
        
        if not equations:
            return None, None, None, False
        
        # Solve for coefficients using numpy
        A = np.array([eq[0] for eq in equations], dtype=float)
        b = np.array([eq[1] for eq in equations], dtype=float)
        
        # Check if system is consistent and has a solution
        try:
            # Use least squares
            coeffs, residuals, rank_sys, _ = np.linalg.lstsq(A, b, rcond=None)
            if len(residuals) == 0 or np.max(np.abs(residuals)) < 1e-10:
                # Verify solution
                A_test = A @ coeffs
                if np.max(np.abs(A_test - b)) < 1e-10:
                    # Create new matrices without this column
                    new_U = [[U[i][j] for j in range(rank) if j != col_idx] for i in range(nm)]
                    new_V = [[V[i][j] for j in range(rank) if j != col_idx] for i in range(mp)]
                    new_W = [[W[i][j] for j in range(rank) if j != col_idx] for i in range(pn)]
                    
                    # Add the combination
                    for idx, c in enumerate(coeffs):
                        if abs(c) > 1e-15:
                            new_col_idx = idx if idx < col_idx else idx - 1
                            for i in range(nm):
                                new_U[i][new_col_idx] += c * U[i][col_idx]
                            for i in range(mp):
                                new_V[i][new_col_idx] += c * V[i][col_idx]
                            for i in range(pn):
                                new_W[i][new_col_idx] += c * W[i][col_idx]
                    
                    # Verify
                    if verify_decomposition(new_U, new_V, new_W, n, m, p):
                        return new_U, new_V, new_W, True
        except:
            pass
        
        return None, None, None, False
    
    # Try to eliminate columns
    max_iterations = 500
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        current_rank = get_rank(best_U)
        found_elimination = False
        
        for col in range(current_rank):
            if time.time() > deadline:
                break
            
            new_U, new_V, new_W, success = try_eliminate_column(
                best_U, best_V, best_W, col, n, m, p
            )
            
            if success:
                best_U, best_V, best_W = new_U, new_V, new_W
                best_rank = get_rank(best_U)
                found_elimination = True
                break
        
        if not found_elimination:
            break
    
    # Final verification
    if verify_decomposition(best_U, best_V, best_W, n, m, p):
        return {'U': best_U, 'V': best_V, 'W': best_W}
    
    # Fallback to schoolbook
    return {'U': U, 'V': V, 'W': W}
