import numpy as np
from itertools import product
import time

def construct():
    n, m, p = 2, 3, 3
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    nm, mp, pn = n * m, m * p, p * n  # 6, 9, 6
    
    # Try ranks from 12 upward
    for rank in range(12, 19):
        if time.time() > deadline - 5:
            break
        result = _try_rank(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook (rank 18)
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank decomposition with integer/half-integer entries."""
    np.random.seed(42)
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute the tensor structure for fast checking
    # For each (i, j, jp, l), we need sum_c U[i*m+j,c]*V[jp*p+l,c]*W[l*n+i,c] = 1 if j==jp else 0
    # Let's precompute which (i,j,jp,l) tuples need to be 1 and which need to be 0
    
    # There are n*m*m*p = 2*3*3*3 = 54 constraints
    # For each constraint (i,j,jp,l), the relevant rows are:
    #   U row: i*m+j
    #   V row: jp*p+l  
    #   W row: l*n+i
    
    # Build list of constraints
    constraints = []
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for l in range(p):
                    u_row = i * m + j
                    v_row = jp * p + l
                    w_row = l * n + i
                    expected = 1.0 if j == jp else 0.0
                    constraints.append((u_row, v_row, w_row, expected))
    
    # Strategy: Start with schoolbook-like structure and try to merge terms
    # For rank r, we have r columns. Each column c defines a "pattern" of which
    # (u_row, v_row, w_row) triples are active.
    
    # Better strategy: Use the fact that for matrix multiplication, we can exploit
    # the structure. Let's try a parametric family.
    
    # For 2x3x3, known minimum rank is 15 (I believe). Let's try to construct it.
    # 
    # Approach: Use 15 terms where some terms are "combined" to cover multiple
    # schoolbook positions.
    
    # Let's try a more systematic search with better initialization
    
    for attempt in range(200):
        if time.time() > deadline - 3:
            return None
        
        # Initialize with a structured pattern
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Start with a base structure: use the first min(rank, 18) schoolbook terms
        base_terms = min(rank, n * m * p)
        for c in range(base_terms):
            # Map column c to schoolbook position (i, j, k)
            i = c // (m * p)
            j = (c // p) % m
            k = c % p
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        
        # For remaining columns, add random sparse patterns
        for c in range(base_terms, rank):
            # Add sparse random entries
            nnz_U = np.random.randint(1, 4)
            nnz_V = np.random.randint(1, 4)
            nnz_W = np.random.randint(1, 4)
            
            u_idx = np.random.choice(nm, nnz_U, replace=False)
            v_idx = np.random.choice(mp, nnz_V, replace=False)
            w_idx = np.random.choice(pn, nnz_W, replace=False)
            
            U[u_idx, c] = np.random.choice([-1.0, 1.0], nnz_U)
            V[v_idx, c] = np.random.choice([-1.0, 1.0], nnz_V)
            W[w_idx, c] = np.random.choice([-1.0, 1.0], nnz_W)
        
        # Check if valid
        if _check_decomposition_fast(n, m, p, U, V, W, constraints):
            return _to_lists(U, V, W)
        
        # Local search: try modifying individual entries
        for _ in range(30):
            if time.time() > deadline - 3:
                return None
            improved = False
            
            # Try modifying random entries
            for _ in range(50):
                # Pick a random matrix, row, column
                mat_choice = np.random.choice(3)
                if mat_choice == 0:
                    row = np.random.randint(nm)
                    col = np.random.randint(rank)
                    old = U[row, col]
                    for new_val in [-1.0, 0.0, 1.0]:
                        if new_val == old:
                            continue
                        U[row, col] = new_val
                        if _check_decomposition_fast(n, m, p, U, V, W, constraints):
                            return _to_lists(U, V, W)
                        improved = True
                    U[row, col] = old
                elif mat_choice == 1:
                    row = np.random.randint(mp)
                    col = np.random.randint(rank)
                    old = V[row, col]
                    for new_val in [-1.0, 0.0, 1.0]:
                        if new_val == old:
                            continue
                        V[row, col] = new_val
                        if _check_decomposition_fast(n, m, p, U, V, W, constraints):
                            return _to_lists(U, V, W)
                        improved = True
                    V[row, col] = old
                else:
                    row = np.random.randint(pn)
                    col = np.random.randint(rank)
                    old = W[row, col]
                    for new_val in [-1.0, 0.0, 1.0]:
                        if new_val == old:
                            continue
                        W[row, col] = new_val
                        if _check_decomposition_fast(n, m, p, U, V, W, constraints):
                            return _to_lists(U, V, W)
                        improved = True
                    W[row, col] = old
            
            if not improved:
                break
    
    return None

def _check_decomposition_fast(n, m, p, U, V, W, constraints):
    """Fast check using precomputed constraints."""
    for u_row, v_row, w_row, expected in constraints:
        s = np.dot(U[u_row, :] * V[v_row, :], W[w_row, :])
        if abs(s - expected) > 1e-9:
            return False
    return True

def _check_decomposition(n, m, p, U, V, W):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for l in range(p):
                    s = np.sum(U[i * m + j, :] * V[jp * p + l, :] * W[l * n + i, :])
                    expected = 1.0 if j == jp else 0.0
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists."""
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
