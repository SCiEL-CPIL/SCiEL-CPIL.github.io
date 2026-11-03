import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify exact decomposition using the trace convention W[k*n+i, r]."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_ilp(n, m, p, rank, deadline):
    """
    Solve for V and W given a fixed U using Integer Linear Programming.
    The constraint sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 is linear in V,W
    if U is fixed (coefficients become U[i,j,c]).
    We assume U entries are integers in {-2, -1, 1, 2}.
    We enumerate a subset of U patterns or use random initialization for U
    and solve for V, W.
    """
    if time.time() > deadline:
        return None
    
    nU = n * m
    nV = m * p
    nW = p * n
    nT = n * m * p
    
    # Variables: V (nV * rank) and W (nW * rank)
    # Total variables = (nV + nW) * rank
    n_vars = (nV + nW) * rank
    
    # We will try several random U configurations
    # U is fixed for each ILP run.
    # We restrict U to small integers to keep the problem tractable.
    
    num_restarts = 5
    random.seed(42 + rank)
    
    for attempt in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate a random U with entries in {-2, -1, 0, 1, 2}
        # Ensure no row is all zeros to allow solutions
        U = []
        for idx in range(nU):
            row = [random.randint(-2, 2) for _ in range(rank)]
            # Force at least one non-zero
            if all(v == 0 for v in row):
                row[random.randint(0, rank-1)] = 1
            U.append(row)
        
        # Construct ILP
        # Minimize 0 subject to:
        # For each (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # Wait, milp in scipy is for linear constraints.
        # The term V * W is bilinear. This is NOT linear.
        # My previous thought was flawed: fixing U makes it V*W = 1/U, which is still bilinear.
        # ILP cannot handle bilinear constraints directly.
        
        # Correction: We can linearize if we fix V as well? No, we need to find V and W.
        # However, if we fix V, we can solve for W linearly?
        # sum_c U V W = 1 => W is involved linearly if U*V is fixed.
        # So: Fix U and V, solve for W?
        # Constraint: sum_c (U[i,j,c] * V[j,k,c]) * W[k,i,c] = 1.
        # This is linear in W.
        # So we can:
        # 1. Randomly initialize U and V.
        # 2. Solve for W using linear programming (or just direct substitution if square? No, underdetermined).
        # Actually, if we fix U and V, the system for W is:
        # For each (i, k): sum_j sum_c (U[i,j,c] V[j,k,c]) W[k,i,c] ... wait.
        # The tensor is T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
        # If U and V are fixed, let A[i,k] = sum_c U[i,j,c] V[j,k,c] ... No, j is summed in the matrix product, but here j is an index of the tensor.
        # The tensor equation is for ALL i,j,k.
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
        # If U and V are fixed, this is a system of linear equations in W.
        # Let coeff[i,j,k]_c = U[i,j,c] * V[j,k,c].
        # We need sum_c coeff[i,j,k]_c * W[k,i,c] = 1 for all i,j,k.
        # This is a linear system: M * W_flat = b.
        # Where M is a (n*m*p) x (p*n*rank) matrix.
        # We can solve this using least squares or exact integer solve if possible.
        
        # Strategy:
        # 1. Randomly generate U and V with small integers.
        # 2. Solve M * W_flat = 1 for W_flat using np.linalg.lstsq.
        # 3. Round W_flat to nearest integers.
        # 4. Verify.
        
        # This is essentially what the NLP approach did, but using linear algebra directly.
        # Let's implement this "Fix U, V -> Solve W" approach.
        
        pass # Placeholder, logic below

def _try_rank_fix_uv_solve_w(n, m, p, rank, deadline):
    """
    Fix U and V to random small integers.
    Solve for W using least squares.
    Round W to integers.
    Verify.
    """
    if time.time() > deadline:
        return None
    
    nU = n * m
    nV = m * p
    nW = p * n
    nT = n * m * p
    
    num_restarts = 100
    random.seed(12345 + rank)
    
    for attempt in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random U and V
        U = np.array([[random.randint(-2, 2) for _ in range(rank)] for _ in range(nU)])
        V = np.array([[random.randint(-2, 2) for _ in range(rank)] for _ in range(nV)])
        
        # Ensure some non-zeros
        if np.all(U == 0):
            continue
        if np.all(V == 0):
            continue
            
        # Build linear system for W
        # Equation for (i, j, k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # Let x be the vector of W entries. x is size nW * rank.
        # Index of W[k*n+i, c] in flat vector: (k*n+i)*rank + c
        
        M = np.zeros((nT, nW * rank))
        b = np.ones(nT)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for c in range(rank):
                        coeff = U[i*m+j, c] * V[j*p+k, c]
                        if coeff != 0:
                            w_idx = (k*n+i)*rank + c
                            M[idx, w_idx] = coeff
                    idx += 1
        
        # Solve M x = b
        # Use least squares
        try:
            x, residuals, rank_mat, sv = np.linalg.lstsq(M, b, rcond=None)
        except Exception:
            continue
        
        # Round x to integers
        x_int = np.round(x)
        
        # Check if x_int satisfies M x_int = b exactly
        if np.allclose(M @ x_int, b, atol=1e-6):
            # Reconstruct W
            W_flat = x_int
            W = np.zeros((nW, rank))
            for r in range(nW):
                for c in range(rank):
                    W[r, c] = W_flat[r*rank + c]
            
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Also try rounding to half-integers?
        # The problem allows integer/half-integer.
        # If we scale W by 2, we can search for integer solutions for 2W.
        # But M (2W) = b => M W = b/2.
        # Let's just stick to integers for W for now, or scale the whole system.
        
        # Alternative: Scale U and V by 2? No, that changes the product.
        # If we want half-integers in W, we can set b = 1 and allow W to be half-integers.
        # Let Y = 2W. Then M Y = 2.
        # Solve M Y = 2. Round Y to integers. W = Y/2.
        
        try:
            y, _, _, _ = np.linalg.lstsq(M, 2*b, rcond=None)
            y_int = np.round(y)
            if np.allclose(M @ y_int, 2*b, atol=1e-6):
                W_flat = y_int / 2.0
                W = np.zeros((nW, rank))
                for r in range(nW):
                    for c in range(rank):
                        W[r, c] = W_flat[r*rank + c]
                
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass

    return None

def _try_rank_symmetric_search(n, m, p, rank, deadline):
    """
    Exploit symmetry or structure.
    For 2x3 * 3x5, maybe we can find a decomposition where U and V have specific sparsity patterns.
    """
    if time.time() > deadline:
        return None
    
    # Try to find a decomposition where U has only 1s and 0s, and V has only 1s and 0s.
    # This is equivalent to finding a 0-1 matrix factorization of the tensor.
    # This is hard.
    
    # Instead, let's try a more targeted local search on the "Fix U, Solve W" approach.
    # We can iterate:
    # 1. Fix U, V. Solve W.
    # 2. Fix U, W. Solve V.
    # 3. Fix V, W. Solve U.
    # This is like alternating minimization.
    
    num_restarts = 20
    random.seed(999 + rank)
    
    for attempt in range(num_restarts):
        if time.time() > deadline:
            break
            
        U = np.array([[random.randint(-2, 2) for _ in range(rank)] for _ in range(n*m)])
        V = np.array([[random.randint(-2, 2) for _ in range(rank)] for _ in range(m*p)])
        W = np.array([[random.randint(-2, 2) for _ in range(rank)] for _ in range(p*n)])
        
        for step in range(50):
            if time.time() > deadline:
                break
            
            # Solve for W given U, V
            M_w = np.zeros((n*m*p, p*n*rank))
            b = np.ones(n*m*p)
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            coeff = U[i*m+j, c] * V[j*p+k, c]
                            if coeff != 0:
                                M_w[idx, (k*n+i)*rank + c] = coeff
                        idx += 1
            
            try:
                x_w, _, _, _ = np.linalg.lstsq(M_w, b, rcond=None)
                W = np.round(x_w).reshape(p*n, rank)
            except:
                pass
            
            # Solve for V given U, W
            M_v = np.zeros((n*m*p, m*p*rank))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            coeff = U[i*m+j, c] * W[k*n+i, c]
                            if coeff != 0:
                                M_v[idx, (j*p+k)*rank + c] = coeff
                        idx += 1
            
            try:
                x_v, _, _, _ = np.linalg.lstsq(M_v, b, rcond=None)
                V = np.round(x_v).reshape(m*p, rank)
            except:
                pass
            
            # Solve for U given V, W
            M_u = np.zeros((n*m*p, n*m*rank))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for c in range(rank):
                            coeff = V[j*p+k, c] * W[k*n+i, c]
                            if coeff != 0:
                                M_u[idx, (i*m+j)*rank + c] = coeff
                        idx += 1
            
            try:
                x_u, _, _, _ = np.linalg.lstsq(M_u, b, rcond=None)
                U = np.round(x_u).reshape(n*m, rank)
            except:
                pass
            
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 30
    
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    deadline = time.time() + 140
    
    # Try ranks from 16 to 29
    for rank in range(16, min(schoolbook_rank, 30)):
        if time.time() > deadline - 5:
            break
        
        # Strategy 1: Alternating Minimization (Fix 2, Solve 3rd)
        result = _try_rank_symmetric_search(n, m, p, rank, deadline - 5)
        if result is not None:
            best = result
            best_rank = rank
            break
            
        # Strategy 2: Fix U, V, Solve W (with rounding)
        result = _try_rank_fix_uv_solve_w(n, m, p, rank, deadline - 5)
        if result is not None:
            best = result
            best_rank = rank
            break

    return best

# EVOLVE-BLOCK-END
