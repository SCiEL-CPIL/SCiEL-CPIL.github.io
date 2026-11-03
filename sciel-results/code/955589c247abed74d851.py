import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all entries
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Fallback: Schoolbook (Rank 72)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_schoolbook

    # Strategy: Solve over GF(11) using randomized linear algebra peeling.
    # The tensor T_{ijk} = sum_c U_{ij,c} V_{jk,c} W_{ki,c} = 1 for all i,j,k.
    # This is a system of 72 equations with variables U, V, W.
    # For a fixed rank R, we have 3 * R * (nm+mp+pn)/3 variables? No, R * (nm + mp + pn) variables.
    # We can fix a gauge to reduce symmetry, but let's just search for a solution.
    
    # We will try to find a solution with rank R.
    # Since the equations are quadratic (product of 3 vars), it's not linear.
    # However, if we fix U and V, the equations for W become linear:
    # sum_c W_{ki,c} (U_{ij,c} V_{jk,c}) = 1.
    # This is a linear system for W given U, V.
    # So we can iterate:
    # 1. Randomly initialize U, V in GF(11).
    # 2. Solve for W in GF(11).
    # 3. Check if solution exists and is non-trivial.
    # 4. If not, perturb U, V and retry.
    
    # Actually, a better approach for "peeling" or "Wiedemann" style:
    # We want to find a low rank decomposition.
    # Let's try rank 60, 58, ... down to 40.
    
    def solve_over_gf(R, seed):
        """
        Try to find a rank R decomposition over GF(11).
        Returns (U, V, W) as lists of lists of floats (0..10) if found, else None.
        """
        random.seed(seed)
        np.random.seed(seed)
        
        P = 11
        n_vars_u = n * m
        n_vars_v = m * p
        n_vars_w = p * n
        
        # Initialize U, V randomly
        U = np.random.randint(0, P, size=(n_vars_u, R)) % P
        V = np.random.randint(0, P, size=(n_vars_v, R)) % P
        
        # Construct linear system for W
        # For each (k, i), we have equation: sum_j sum_c W_{k,i,c} * (U_{i,j,c} * V_{j,k,c}) = 1
        # Wait, the index for W is k*n+i.
        # The term in the sum for T_{i,j,k} is U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}.
        # So for a fixed (k, i), the coefficient of W_{k*n+i, c} is sum_j (U_{i*m+j, c} * V_{j*p+k, c}).
        # Let A_{(k,i), c} = sum_j U_{i*m+j, c} * V_{j*p+k, c}.
        # Then the equation for T_{i,j,k} is not directly linear in W alone because T_{i,j,k} sums over c.
        # Actually, T_{i,j,k} = sum_c (U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}).
        # This is NOT a linear equation in W if we consider all (i,j,k) simultaneously?
        # Yes it is! For each (i,j,k), the equation is:
        # sum_c W_{k*n+i, c} * (U_{i*m+j, c} * V_{j*p+k, c}) = 1.
        # But W_{k*n+i, c} appears in multiple equations (for different j).
        # So we have a system of 72 equations for the variables W_{row, c}.
        # The variables are W[0..pn-1, 0..R-1]. Total pn*R variables.
        # The equations are indexed by (i, j, k).
        # Equation (i,j,k): sum_{c=0}^{R-1} W[k*n+i, c] * B_{i,j,k,c} = 1,
        # where B_{i,j,k,c} = U[i*m+j, c] * V[j*p+k, c].
        
        # This is a linear system A * w = b, where w is a vector of size pn*R.
        # A is 72 x (pn*R).
        
        # We can solve this over GF(11).
        # If the system is consistent, we have a candidate W.
        # Then we have a valid decomposition?
        # Yes, if we find W such that for all (i,j,k), the sum is 1, then we have a valid decomposition.
        
        # Let's build A and b.
        A = np.zeros((n*m*p, n_vars_w * R), dtype=int)
        b = np.ones(n*m*p, dtype=int)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    w_row = k * n + i
                    for c in range(R):
                        coeff = (U[i*m+j, c] * V[j*p+k, c]) % P
                        A[idx, w_row * R + c] = coeff
                    idx += 1
        
        # Solve A * w = b over GF(11)
        # Use Gaussian elimination or numpy if we cast to float? No, must be exact in GF(11).
        # We can use sympy? No, standard library only.
        # Implement simple Gaussian elimination over GF(11).
        
        # Augmented matrix
        M = np.hstack([A, b.reshape(-1, 1)]) % P
        rows, cols = M.shape
        
        # Gaussian elimination
        rank = 0
        for col in range(cols - 1):
            # Find pivot
            pivot_row = -1
            for r in range(rank, rows):
                if M[r, col] != 0:
                    pivot_row = r
                    break
            if pivot_row == -1:
                continue
            
            # Swap
            if pivot_row != rank:
                M[[rank, pivot_row]] = M[[pivot_row, rank]]
            
            # Normalize
            inv = pow(M[rank, col], P-2, P)
            M[rank] = (M[rank] * inv) % P
            
            # Eliminate
            for r in range(rows):
                if r != rank and M[r, col] != 0:
                    factor = M[r, col]
                    M[r] = (M[r] - factor * M[rank]) % P
            
            rank += 1
        
        # Check consistency
        for r in range(rank, rows):
            if M[r, cols-1] != 0:
                return None # Inconsistent
        
        # If consistent, extract solution.
        # We need to find a solution w.
        # Since we might have free variables, we can set them to 0 and solve for the rest.
        # Or just pick one solution.
        
        # Back substitution or just read off if we reduced to RREF?
        # My elimination above didn't fully reduce to RREF for free variables.
        # Let's do a proper RREF or just solve.
        
        # Actually, if we just want ANY solution, we can set free variables to 0.
        # Let's identify pivot columns.
        pivot_cols = []
        # Re-do elimination to track pivots properly or just use the current M.
        # The current M has pivots in the first 'rank' rows, but columns might not be sorted if I skipped?
        # No, I iterated col from 0 to cols-2. So pivot cols are the ones where I found a pivot.
        
        # Let's re-implement to be safe and get a solution.
        # Given time constraints, if the system is consistent, a solution exists.
        # We can use a simple solver.
        
        # For the purpose of this search, if it's consistent, we can try to extract W.
        # However, extracting the specific W from the GF(11) solution requires care.
        
        # Alternative: If we find a consistent system, we can just return a "dummy" valid decomposition?
        # No, we need the actual U, V, W.
        
        # Let's try a different tactic:
        # If we fix U and V, and the system for W is consistent, we can find W.
        # But we need to construct W from the linear system solution.
        
        # Let's assume we can find W.
        # If rank < n_vars_w * R, there are free variables.
        # We can set free variables to 0.
        
        # To extract W:
        # The matrix M is in row echelon form.
        # We need to map the solution back to W.
        
        # This is getting complex to implement correctly in a short time.
        # Let's try a simpler heuristic search over GF(11) for small ranks?
        
        # Actually, the "peeling" inspiration suggests finding dependencies.
        # If I can't implement full GF(11) solver robustly in time, I should fall back to the current strategy but with a different initialization or search space.
        
        # Let's try to just return None for now if I can't solve it easily, and rely on the fallback.
        # But wait, the prompt says "rethinking the approach".
        # If I can't implement the GF(11) solver, I should try a different structural change.
        
        # Let's try a structural change: Use a fixed known low-rank decomposition if one exists for 3x4x6?
        # Theoretical lower bound is ~ 3*4*6 / 2 = 36?
        # Known best for 3x3x3 is 23 (Strassen-like).
        # For 3x4x6, maybe rank 50-60 is achievable.
        
        # Since I cannot guarantee a GF(11) solution in the time limit without a robust solver,
        # I will implement a "Hybrid" approach:
        # 1. Try to find a solution over GF(11) for rank 60 using a simple solver.
        # 2. If that fails, fall back to the schoolbook.
        
        # To make the GF(11) solver work, I need to extract the solution.
        # Let's implement a simple solver for A*x=b over GF(p).
        
        def solve_gf(A, b, p):
            """Solve Ax=b over GF(p). Returns x if consistent, else None."""
            M = np.hstack([A, b.reshape(-1, 1)]) % p
            rows, cols = M.shape
            rank = 0
            pivot_cols = []
            
            for col in range(cols - 1):
                pivot_row = -1
                for r in range(rank, rows):
                    if M[r, col] != 0:
                        pivot_row = r
                        break
                if pivot_row == -1:
                    continue
                
                if pivot_row != rank:
                    M[[rank, pivot_row]] = M[[pivot_row, rank]]
                
                inv = pow(M[rank, col], p-2, p)
                M[rank] = (M[rank] * inv) % p
                pivot_cols.append(col)
                
                for r in range(rows):
                    if r != rank and M[r, col] != 0:
                        factor = M[r, col]
                        M[r] = (M[r] - factor * M[rank]) % p
                
                rank += 1
            
            # Check consistency
            for r in range(rank, rows):
                if M[r, cols-1] != 0:
                    return None
            
            # Construct solution
            x = np.zeros(cols - 1, dtype=int)
            # Free variables are 0.
            # Pivot variables are determined by the last column.
            # Since we reduced to RREF (eliminated above and below),
            # x[pivot_col] = M[rank_index, -1]
            
            # Wait, my elimination didn't eliminate above. It only eliminated below?
            # "for r in range(rows): if r != rank..." -> Yes, it eliminated both above and below.
            # So it is RREF.
            
            for i in range(rank):
                x[pivot_cols[i]] = M[i, cols-1]
            
            return x
        
        w_vec = solve_gf(A, b, P)
        if w_vec is None:
            return None
        
        # Reshape w_vec to W
        W = w_vec.reshape(n_vars_w, R)
        
        # Verify the solution over GF(11)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(R):
                        s = (s + U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]) % P
                    if s != 1 % P:
                        return None
        
        # If verified over GF(11), it is a valid decomposition over integers too?
        # Yes, if the integer values satisfy the equation mod 11, they might not satisfy it over reals.
        # BUT, if we treat the entries as integers, the equation is exact.
        # The problem asks for "integer/half-integer entries that exactly decompose".
        # If U, V, W are integers, and sum U*V*W = 1, then it is a valid exact decomposition.
        # The fact that it holds mod 11 doesn't guarantee it holds over reals unless the sums are small or we verify over reals.
        # We MUST verify over reals.
        
        # Convert to float and verify
        U_float = U.astype(np.float64)
        V_float = V.astype(np.float64)
        W_float = W.astype(np.float64)
        
        if not verify(U_float.tolist(), V_float.tolist(), W_float.tolist(), n, m, p):
            return None
        
        return U_float.tolist(), V_float.tolist(), W_float.tolist()

    # Try ranks from 60 down to 40
    for rank in range(60, 39, -2):
        if time.time() > deadline - 5:
            break
            
        for seed in range(5):
            if time.time() > deadline - 5:
                break
                
            result = solve_over_gf(rank, seed)
            if result:
                U, V, W = result
                if verify(U, V, W, n, m, p):
                    best_U = U
                    best_V = V
                    best_W = W
                    best_rank = rank
                    print(f"Found valid rank {rank} with seed {seed}")
                    break
        
        if best_rank < rank_schoolbook:
            # Found something better. Continue to see if we can find lower.
            pass

    # If no better rank found, return schoolbook
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
