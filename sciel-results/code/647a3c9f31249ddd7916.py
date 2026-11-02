import time
import itertools
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _build_target(n, m, p):
    """Return the matrix-multiplication tensor as a dict of 1s (sparse)."""
    tgt = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tgt[(i * m + j, j * p + k, k * n + i)] = 1
    return tgt

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
    return U, V, W

def _to_frac(x):
    if isinstance(x, Fraction):
        return x
    if isinstance(x, (int,)):
        return Fraction(x)
    if isinstance(x, float):
        r = round(x * 2) / 2
        return Fraction(r).limit_denominator(1000000)
    return Fraction(x)

def _verify(U, V, W, tgt, n, m, p):
    """Exact check that sum_r U[a,r]V[b,r]W[c,r] reproduces the target."""
    R = len(U[0])
    def col_dict(mat, r):
        d = {}
        for a in range(len(mat)):
            v = mat[a][r]
            if v != 0:
                d[a] = v
        return d
    cols = []
    for r in range(R):
        cols.append((col_dict(U, r), col_dict(V, r), col_dict(W, r)))
    acc = {}
    for cu, cv, cw in cols:
        for a, ua in cu.items():
            for b, vb in cv.items():
                uv = ua * vb
                for c, wc in cw.items():
                    acc[(a, b, c)] = acc.get((a, b, c), 0) + uv * wc
    for key, val in tgt.items():
        if acc.get(key, 0) != val:
            return False
    for key, val in acc.items():
        if val != 0 and key not in tgt:
            return False
    return True

def _try_find_lower_rank(n, m, p, target_rank, deadline):
    """
    Attempts to find a decomposition of rank < current best.
    Uses random initialization of U and V, then solves for W linearly.
    """
    U_dim, V_dim, W_dim = n * m, m * p, p * n
    tgt = _build_target(n, m, p)
    
    # We try to find a decomposition of exactly target_rank.
    # If we fix U and V, the problem is linear in W.
    # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
    # For each c, we have a system:
    # sum_r (U_{a,r} V_{b,r}) W_{c,r} = T_{a,b,c}
    # This is a matrix equation: M_c W_c = T_c
    # where M_c is a (U_dim * V_dim) x R matrix, and W_c is a R x 1 vector.
    # Specifically, row index is (a,b), col index is r.
    # M_c[a * V_dim + b, r] = U_{a,r} V_{b,r}
    # T_c[a * V_dim + b] = T_{a,b,c}
    
    import random
    rng = random.Random(42)
    
    # Try a few random sparse U and V
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Initialize U and V with small integers or 0s
        # To make it sparse, we pick a few non-zeros per column
        U = [[0] * target_rank for _ in range(U_dim)]
        V = [[0] * target_rank for _ in range(V_dim)]
        
        for r in range(target_rank):
            # Pick 2 non-zero entries in U[:, r]
            u_indices = rng.sample(range(U_dim), 2)
            for idx in u_indices:
                U[idx][r] = rng.choice([-1, 1])
            
            v_indices = rng.sample(range(V_dim), 2)
            for idx in v_indices:
                V[idx][r] = rng.choice([-1, 1])
        
        # Construct the linear system for W
        # We need to solve for W[c, r] for all c, r.
        # The equations are coupled? No, for each c, the equations for W[:, c] are independent?
        # T_{a,b,c} depends only on W[:, c].
        # So we can solve for each c separately.
        
        W = [[0] * target_rank for _ in range(W_dim)]
        
        valid = True
        for c in range(W_dim):
            # Build M_c and T_c
            # M_c is (U_dim * V_dim) x target_rank
            # We only need rows (a,b) where T_{a,b,c} != 0?
            # No, T_{a,b,c} is 0 for most (a,b).
            # But if T_{a,b,c} = 0, we still have the equation sum_r ... = 0.
            # However, we only have 72 non-zero entries in T total.
            # For a fixed c, how many non-zero T_{a,b,c}?
            # c = k*n + i.
            # T_{a,b,c} = 1 if a = i*m + j, b = j*p + k.
            # So for fixed c (fixed i, k), the non-zero T are for j=0..m-1.
            # There are m non-zero entries for each c.
            # Total non-zero entries in T is n*m*p = 72.
            # For each c, there are m=4 non-zero equations.
            # We have R=70 unknowns W[c, 0..69].
            # We have 4 equations. Under-determined.
            # We can set most W[c, r] to 0 and solve for the rest?
            # Or we can just check if a solution exists.
            # Since it's under-determined, a solution always exists (e.g., W=0 works if T=0, but T!=0).
            # Wait, if we fix U and V, M_c is fixed.
            # We need M_c W_c = T_c.
            # Since it's under-determined, we can find a solution using Least Squares.
            # If the residual is 0, it's a valid solution.
            
            # Let's collect the non-zero (a,b) for this c
            rows = []
            t_vals = []
            for a in range(U_dim):
                for b in range(V_dim):
                    if tgt.get((a, b, c), 0) != 0:
                        rows.append(a * V_dim + b)
                        t_vals.append(tgt[(a, b, c)])
            
            if not rows:
                continue # No constraints for this c? Impossible, T has 72 entries.
            
            # Build matrix M_sub (len(rows) x target_rank)
            M_sub = np.zeros((len(rows), target_rank))
            for i_row, row_idx in enumerate(rows):
                a = row_idx // V_dim
                b = row_idx % V_dim
                for r in range(target_rank):
                    M_sub[i_row, r] = U[a][r] * V[b][r]
            
            t_vec = np.array(t_vals)
            
            # Solve M_sub W_c = t_vec
            # Use numpy lstsq
            try:
                sol, residuals, rank, sv = np.linalg.lstsq(M_sub, t_vec, rcond=None)
                # Check if residual is close to 0
                if residuals and residuals[0] > 1e-9:
                    valid = False
                    break
                # Assign solution to W
                for r in range(target_rank):
                    W[c][r] = int(round(sol[r]))
            except Exception:
                valid = False
                break
        
        if not valid:
            continue
            
        # Verify the integer solution
        if _verify(U, V, W, tgt, n, m, p):
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115  # Keep margin

    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try to find lower rank
    # Try ranks 71, 70, ...
    for r in range(best_rank - 1, best_rank - 10, -1):
        if time.time() > deadline:
            break
        found = _try_find_lower_rank(n, m, p, r, deadline)
        if found is not None:
            U, V, W = found
            best_rank = r
            break # Found one, stop (or continue to find even lower?)
            # Let's try to find even lower if time permits
            # But usually the first success is good enough for a jump.
            # We can continue the loop to try r-1, etc.
            # But the loop structure above breaks. Let's not break, just update best.
            # Actually, if we find rank 70, we should try 69?
            # The loop goes down. So we can just continue.
            # But we need to update U,V,W.
            # Let's just return the first found to save time for verification?
            # No, lower is better.
            # Let's continue searching.

    return {"U": U, "V": W, "W": W} # Wait, U, V, W order.
    # Return U, V, W correctly.
    
# Correct return
def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115

    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p

    for r in range(best_rank - 1, best_rank - 10, -1):
        if time.time() > deadline:
            break
        found = _try_find_lower_rank(n, m, p, r, deadline)
        if found is not None:
            U, V, W = found
            best_rank = r
            # Continue to see if we can go lower
            # But we should probably stop if we get a significant drop?
            # Let's just return the best found so far.
            # Actually, the loop will try lower ranks next.
            # If we find rank 70, we will try 69.
            # If 69 fails, we return 70.
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
