"""Search-based exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
import time
import random
import itertools

DIMENSIONS = [4, 4, 5]

def _time_left(deadline):
    return deadline - time.time()

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _greedy_build(n, m, p, deadline, max_rank=80):
    """Greedy set-cover style approach."""
    nm, mp, pn = n * m, m * p, p * n
    # Remaining tensor values
    remaining = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                remaining[(i, j, k)] = 1.0
    
    U = []
    V = []
    W = []
    
    for r in range(max_rank):
        if _time_left(deadline) < 5:
            break
        
        # Find the best rank-1 term to add
        # A rank-1 term is defined by (u, v, w) where:
        # u is a vector of length nm, v of length mp, w of length pn
        # Contribution to (i,j,k) is u[i*m+j] * v[j*p+k] * w[k*n+i]
        
        # For simplicity, try terms with small integer/half-integer entries
        # Try all combinations of small supports
        best_term = None
        best_coverage = -1
        
        # Try terms with support size 1-3 in each dimension
        for u_size in [1, 2, 3]:
            for v_size in [1, 2, 3]:
                for w_size in [1, 2, 3]:
                    if _time_left(deadline) < 2:
                        break
                    # Sample some random supports
                    for _ in range(50):
                        u_idx = random.sample(range(nm), min(u_size, nm))
                        v_idx = random.sample(range(mp), min(v_size, mp))
                        w_idx = random.sample(range(pn), min(w_size, pn))
                        
                        # Try different value patterns
                        for u_vals in [[1]*u_size, [1, 1], [1, -1]][:1]:
                            for v_vals in [[1]*v_size, [1, 1], [1, -1]][:1]:
                                for w_vals in [[1]*w_size, [1, 1], [1, -1]][:1]:
                                    # Build the term
                                    u = [0] * nm
                                    v = [0] * mp
                                    w = [0] * pn
                                    for idx, val in zip(u_idx, u_vals):
                                        u[idx] = val
                                    for idx, val in zip(v_idx, v_vals):
                                        v[idx] = val
                                    for idx, val in zip(w_idx, w_vals):
                                        w[idx] = val
                                    
                                    # Calculate coverage
                                    coverage = 0
                                    for (i, j, k), rem in remaining.items():
                                        contrib = u[i*m+j] * v[j*p+k] * w[k*n+i]
                                        if contrib != 0:
                                            # Check if this helps
                                            if abs(contrib) <= abs(rem) + 1e-9:
                                                coverage += 1
                                    
                                    if coverage > best_coverage:
                                        best_coverage = coverage
                                        best_term = (u, v, w)
        
        if best_term is None or best_coverage == 0:
            break
        
        u, v, w = best_term
        U.append(u)
        V.append(v)
        W.append(w)
        
        # Update remaining
        for (i, j, k) in list(remaining.keys()):
            contrib = u[i*m+j] * v[j*p+k] * w[k*n+i]
            remaining[(i, j, k)] -= contrib
            if abs(remaining[(i, j, k)]) < 1e-9:
                del remaining[(i, j, k)]
        
        if not remaining:
            break
    
    if remaining:
        return None
    
    # Convert to matrix form
    rank = len(U)
    U_mat = [[0] * rank for _ in range(nm)]
    V_mat = [[0] * rank for _ in range(mp)]
    W_mat = [[0] * rank for _ in range(pn)]
    for r in range(rank):
        for idx in range(nm):
            U_mat[idx][r] = U[r][idx]
        for idx in range(mp):
            V_mat[idx][r] = V[r][idx]
        for idx in range(pn):
            W_mat[idx][r] = W[r][idx]
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try greedy search
    if _time_left(deadline) > 10:
        result = _greedy_build(n, m, p, deadline, max_rank=best_rank - 1)
        if result is not None:
            # MUST VALIDATE THE RESULT
            if _is_valid(result["U"], result["V"], result["W"], n, m, p):
                rank = len(result["U"][0])
                if rank < best_rank:
                    best = result
                    best_rank = rank
    
    return best
