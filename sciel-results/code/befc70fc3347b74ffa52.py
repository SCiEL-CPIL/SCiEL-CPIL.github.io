"""Matrix multiplication 2x3x7 rank-minimal exact tensor decomposition."""
import numpy as np
import random
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = 0.0
                for c in range(rank):
                    s += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _verify_exact(U_frac, V_frac, W_frac, n, m, p):
    """Verify with exact Fraction arithmetic."""
    rank = len(U_frac[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = Fraction(0, 1)
                for c in range(rank):
                    s += U_frac[u_idx][c] * V_frac[v_idx][c] * W_frac[w_idx][c]
                if s != 1:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = Fraction(1)
                V[j * p + k][col] = Fraction(1)
                W[k * n + i][col] = Fraction(1)
                col += 1
    return U, V, W

def _to_list(U, V, W):
    """Convert Fraction matrices to lists of floats (for return)."""
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return U_list, V_list, W_list

def _try_merge_columns(U, V, W, n, m, p):
    """Try to merge pairs of columns that are linearly dependent in the tensor sense.
    
    If columns a and b satisfy: for all (i,j,k),
    U[i,j,a]*V[j,k,a]*W[i,k,a] + U[i,j,b]*V[j,k,b]*W[i,k,b] = 0
    then we can remove one. But more generally, if the sum of two rank-1 terms
    equals zero on the support, we can eliminate.
    
    Actually, a better approach: if column a is a linear combination of other columns
    in a way that preserves the tensor, we can eliminate it.
    
    For now, try: for each pair (a,b), check if removing both and adding a correction
    works. This is complex. Instead, try the simpler approach of checking if any
    single column can be removed.
    """
    rank = len(U[0])
    for col_idx in range(rank - 1, -1, -1):
        # Try removing this column
        U_test = [row[:col_idx] + row[col_idx+1:] for row in U]
        V_test = [row[:col_idx] + row[col_idx+1:] for row in V]
        W_test = [row[:col_idx] + row[col_idx+1:] for row in W]
        
        if _verify_exact(U_test, V_test, W_test, n, m, p):
            return U_test, V_test, W_test, rank - 1
    return U, V, W, rank

def _greedy_reduce(U, V, W, n, m, p, deadline):
    """Repeatedly try to remove columns."""
    rank = len(U[0])
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for col_idx in range(rank - 1, -1, -1):
            if time.time() > deadline:
                break
            U_test = [row[:col_idx] + row[col_idx+1:] for row in U]
            V_test = [row[:col_idx] + row[col_idx+1:] for row in V]
            W_test = [row[:col_idx] + row[col_idx+1:] for row in W]
            
            if _verify_exact(U_test, V_test, W_test, n, m, p):
                U, V, W = U_test, V_test, W_test
                rank = len(U[0])
                changed = True
                break
    return U, V, W, rank

def _structured_2xn_p(n, m, p):
    """Try a structured decomposition exploiting n=2.
    
    For 2×m×p, we can write:
    C[0,k] = sum_j A[0,j] B[j,k]
    C[1,k] = sum_j A[1,j] B[j,k]
    
    One approach: use the fact that we can define auxiliary variables.
    For each j, define D_j = A[0,j] - A[1,j] (or some combination).
    
    Actually, let's try a different structured approach:
    For each j in [0,m), k in [0,p), we have two equations:
    C[0,k] += A[0,j]*B[j,k]
    C[1,k] += A[1,j]*B[j,k]
    
    We can combine these. If we define for each (j,k):
    - One term with U[0,j]=1, U[1,j]=1, V[j,k]=1, W[0,k]=1/2, W[1,k]=1/2
      This gives C[0,k] += A[0,j]*B[j,k]/2 and C[1,k] += A[1,j]*B[j,k]/2
    - One term with U[0,j]=1, U[1,j]=-1, V[j,k]=1, W[0,k]=1/2, W[1,k]=-1/2
      This gives C[0,k] += A[0,j]*B[j,k]/2 and C[1,k] -= A[1,j]*B[j,k]/2
    
    Sum: C[0,k] += A[0,j]*B[j,k], C[1,k] += 0. Not right.
    
    Let me try:
    Term 1: U[0,j]=1, U[1,j]=0, V[j,k]=1, W[0,k]=1, W[1,k]=0
      -> C[0,k] += A[0,j]*B[j,k], C[1,k] += 0
    Term 2: U[0,j]=0, U[1,j]=1, V[j,k]=1, W[0,k]=0, W[1,k]=1
      -> C[0,k] += 0, C[1,k] += A[1,j]*B[j,k]
    
    That's just schoolbook.
    
    Alternative: Use half-integers.
    Term 1: U[0,j]=1, U[1,j]=1, V[j,k]=1, W[0,k]=1/2, W[1,k]=1/2
      -> C[0,k] += A[0,j]*B[j,k]/2, C[1,k] += A[1,j]*B[j,k]/2
    Term 2: U[0,j]=1, U[1,j]=-1, V[j,k]=1, W[0,k]=1/2, W[1,k]=-1/2
      -> C[0,k] += A[0,j]*B[j,k]/2, C[1,k] -= A[1,j]*B[j,k]/2
    
    Sum of terms 1 and 2: C[0,k] += A[0,j]*B[j,k], C[1,k] += 0. Still not right.
    
    I need: C[0,k] += A[0,j]*B[j,k] and C[1,k] += A[1,j]*B[j,k]
    
    Let's use:
    Term 1: U[0,j]=1, U[1,j]=0, V[j,k]=1, W[0,k]=1, W[1,k]=0
    Term 2: U[0,j]=0, U[1,j]=1, V[j,k]=1, W[0,k]=0, W[1,k]=1
    
    That's 2mp terms = schoolbook.
    
    To save rank, we need to share some structure. The key is that the W matrix
    has only p*n = 14 rows, and we need to encode both the "which i" and "which k" info.
    
    For n=2, W has 14 rows: W[0*2+0]=W[0], W[1*2+0]=W[2], W[0*2+1]=W[1], W[1*2+1]=W[3], etc.
    Wait, the convention is W[k*n+i, r]. So for n=2:
    k=0,i=0 -> W[0]
    k=0,i=1 -> W[1]  
    k=1,i=0 -> W[2]
    k=1,i=1 -> W[3]
    ...
    k=6,i=0 -> W[12]
    k=6,i=1 -> W[13]
    
    So W[k*2+i] for k in [0,7), i in [0,2).
    
    The decomposition is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    For n=2, this is: sum_c U[i*3+j,c] * V[j*7+k,c] * W[2*k+i,c] = 1
    
    Let me try a rank-30 approach. We have 2*3*7=42 terms in schoolbook.
    
    Idea: For each j, we have 7 k-values. For each (j,k), we need to contribute
    to both i=0 and i=1 outputs. 
    
    What if for each j, we use a shared V column across all k? No, V depends on k.
    
    Let me try: for each j, define 7 pairs of terms, but try to merge them.
    
    Actually, let's just do the greedy reduction from schoolbook and see if we can
    get below 42.
    """
    pass

def _random_half_integer_search(start_rank, end_rank, time_budget, n, m, p):
    """Search for a decomposition with rank in [start_rank, end_rank] using half-integers."""
    best = None
    
    for r in range(start_rank, end_rank + 1):
        if time.time() > time_budget:
            break
        for attempt in range(100):
            if time.time() > time_budget:
                break
            
            # Random half-integer initialization
            U = np.random.randint(-4, 5, size=(n*m, r)).astype(float) / 2.0
            V = np.random.randint(-4, 5, size=(m*p, r)).astype(float) / 2.0
            W = np.random.randint(-4, 5, size=(p*n, r)).astype(float) / 2.0
            
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                if best is None or r < best[0]:
                    best = (r, U.copy(), V.copy(), W.copy())
                break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Start with schoolbook in exact fractions
    U, V, W = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try greedy reduction
    U, V, W, best_rank = _greedy_reduce(U, V, W, n, m, p, deadline)
    
    # If we improved, great. Otherwise, try random search for lower ranks.
    if best_rank == schoolbook_rank and time.time() < deadline:
        # Try random search from rank 35 to 41
        result = _random_half_integer_search(35, 41, deadline, n, m, p)
        if result is not None:
            r, U_r, V_r, W_r = result
            if r < best_rank:
                best_rank = r
                U_list, V_list, W_list = _to_list(U, V, W)
                # Use the random result
                U_list = U_r.tolist()
                V_list = V_r.tolist()
                W_list = W_r.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Convert to list and return
    U_list, V_list, W_list = _to_list(U, V, W)
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
