import time
import numpy as np
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = 2, 3, 5
    nm, mp, pn = n * m, m * p, p * n  # 6, 15, 10
    
    start_time = time.time()
    deadline = start_time + 118.0

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        r = len(U[0]) if U else 0
        if not U or not V or not W:
            return False
        if len(V[0]) != r or len(W[0]) != r:
            return False
        # Check that U, V, W have correct number of rows
        if len(U) != nm or len(V) != mp or len(W) != pn:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(r):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def verify_exact(U, V, W):
        """Exact verification using Fraction to avoid floating point issues."""
        r = len(U[0]) if U else 0
        if not U or not V or not W:
            return False
        if len(V[0]) != r or len(W[0]) != r:
            return False
        if len(U) != nm or len(V) != mp or len(W) != pn:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                    if s != 1:
                        return False
        return True

    def is_half_integer(x):
        """Check if x is an integer or half-integer (multiple of 0.5)."""
        return (x * 2) % 1 == 0

    def try_rank_strassen(rank):
        """
        Try to construct a decomposition using Strassen-like techniques.
        For 2x3x5, we can try to exploit the 2x2 substructure.
        """
        if rank >= 30:
            return None
        
        # Known construction: Use Strassen 2x2x2 (rank 7) for the 2x2x2 part,
        # and handle the remaining dimensions separately.
        # The 2x3x5 multiplication can be viewed as:
        # C = A * B where A is 2x3, B is 3x5.
        # We can split B into [B1, B2, B3] where B1 is 3x1, etc.
        # Or use the fact that 2x3x5 = 2x3x3 + 2x3x2 (but these aren't independent).
        
        # Let's try a direct algebraic construction for rank 28.
        # One known approach: the tensor rank of 2x3x5 is at most 28.
        
        # For now, let's try to find a decomposition by solving the linear system
        # with a structured ansatz.
        
        # Ansatz: U has rows that are sparse vectors with entries in {0, 1, -1}
        # V has rows that are sparse vectors with entries in {0, 1, -1}
        # W is solved from the linear system and should be in {0, 1, -1, 1/2, -1/2}
        
        # We'll use a randomized local search with the constraint that U and V
        # have limited density.
        
        max_restarts = 30
        rng = np.random.default_rng(42)
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                return None
            
            # Random initialization of U and V with controlled density
            density_u = 0.4 + 0.1 * (restart % 3)
            density_v = 0.5 + 0.1 * (restart % 3)
            
            U = np.zeros((nm, rank), dtype=int)
            V = np.zeros((mp, rank), dtype=int)
            
            mask_u = rng.random((nm, rank)) < density_u
            mask_v = rng.random((mp, rank)) < density_v
            
            # Ensure each column has at least one non-zero in U and V
            for c in range(rank):
                if not mask_u[:, c].any():
                    mask_u[rng.integers(0, nm), c] = True
                if not mask_v[:, c].any():
                    mask_v[rng.integers(0, mp), c] = True
            
            U_vals = rng.choice([-1, 1], size=(nm, rank))
            V_vals = rng.choice([-1, 1], size=(mp, rank))
            
            U[mask_u] = U_vals[mask_u]
            V[mask_v] = V_vals[mask_v]
            
            # Construct the linear system for W
            # Equation: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
            num_eqs = n * m * p
            A_mat = np.zeros((num_eqs, pn * rank))
            b = np.ones(num_eqs)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_row = U[i * m + j, :].astype(float)
                        v_row = V[j * p + k, :].astype(float)
                        coeffs = u_row * v_row
                        start_idx = (k * n + i) * rank
                        A_mat[idx, start_idx:start_idx + rank] = coeffs
                        idx += 1
            
            try:
                x, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b, rcond=None)
            except:
                continue
            
            W_sol = x.reshape(pn, rank)
            
            # Check if W entries are close to half-integers
            W_rounded = np.round(W_sol * 2) / 2
            if np.allclose(W_rounded, W_sol, atol=1e-4):
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = [[float(x) for x in row] for row in W_rounded.tolist()]
                # Convert to clean integers/half-integers
                W_clean = []
                for row in W_list:
                    clean_row = []
                    for val in row:
                        val2 = val * 2
                        if abs(val2 - round(val2)) < 1e-6:
                            clean_row.append(int(round(val2)) / 2)
                        else:
                            clean_row.append(val)
                    W_clean.append(clean_row)
                if verify_exact(U_list, V_list, W_clean):
                    return {"U": U_list, "V": V_list, "W": W_clean}
            
            # Local search: flip random entries in U and V
            for _ in range(20):
                if time.time() > deadline:
                    return None
                
                # Flip a random entry in U
                if rng.random() < 0.5:
                    r_idx = rng.integers(0, nm)
                    c_idx = rng.integers(0, rank)
                    if U[r_idx, c_idx] != 0:
                        U[r_idx, c_idx] = -U[r_idx, c_idx]
                    else:
                        U[r_idx, c_idx] = rng.choice([-1, 1])
                else:
                    r_idx = rng.integers(0, mp)
                    c_idx = rng.integers(0, rank)
                    if V[r_idx, c_idx] != 0:
                        V[r_idx, c_idx] = -V[r_idx, c_idx]
                    else:
                        V[r_idx, c_idx] = rng.choice([-1, 1])
                
                # Re-solve for W
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_row = U[i * m + j, :].astype(float)
                            v_row = V[j * p + k, :].astype(float)
                            coeffs = u_row * v_row
                            start_idx = (k * n + i) * rank
                            A_mat[idx, start_idx:start_idx + rank] = coeffs
                            idx += 1
                
                try:
                    x, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b, rcond=None)
                except:
                    continue
                
                W_sol = x.reshape(pn, rank)
                W_rounded = np.round(W_sol * 2) / 2
                if np.allclose(W_rounded, W_sol, atol=1e-4):
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = [[float(x) for x in row] for row in W_rounded.tolist()]
                    W_clean = []
                    for row in W_list:
                        clean_row = []
                        for val in row:
                            val2 = val * 2
                            if abs(val2 - round(val2)) < 1e-6:
                                clean_row.append(int(round(val2)) / 2)
                            else:
                                clean_row.append(val)
                        W_clean.append(clean_row)
                    if verify_exact(U_list, V_list, W_clean):
                        return {"U": U_list, "V": V_list, "W": W_clean}
        
        return None

    # Try ranks from low to high
    for r in range(20, 30):
        if time.time() > deadline:
            break
        res = try_rank_strassen(r)
        if res:
            return res

    return schoolbook()
