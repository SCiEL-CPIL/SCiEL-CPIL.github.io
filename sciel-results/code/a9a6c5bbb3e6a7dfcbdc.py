"""Rewritten program: matrix multiplication 2x3 x 3x5 tensor decomposition."""
import time
import random
import numpy as np

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

def _try_rank_random_local_search(n, m, p, rank, deadline):
    """
    Randomized local search with integer values in {-2,-1,0,1,2}.
    Multiple restarts with greedy entry-level adjustments.
    """
    if time.time() > deadline:
        return None
    
    best_result = None
    num_restarts = 30
    
    for attempt in range(num_restarts):
        if time.time() > deadline:
            break
        
        seed = attempt * 7919 + rank * 104729
        random.seed(seed)
        
        # Initialize with small random integers
        U = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(n * m)]
        V = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(m * p)]
        W = [[random.randint(-2, 2) for _ in range(rank)] for _ in range(p * n)]
        
        # Quick check
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Local search: iteratively fix worst constraint
        for step in range(2000):
            if time.time() > deadline:
                break
            
            # Compute all constraint sums
            sums = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(rank):
                            s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                        sums[(i, j, k)] = s
            
            # Find worst error
            worst_ijk = None
            worst_err = 0
            for (i, j, k), s in sums.items():
                err = abs(s - 1)
                if err > worst_err:
                    worst_err = err
                    worst_ijk = (i, j, k)
            
            if worst_err == 0:
                return {"U": U, "V": V, "W": W}
            
            if worst_ijk is None:
                break
            
            i, j, k = worst_ijk
            target = 1
            current = sums[(i, j, k)]
            
            # Try adjusting entries in U, V, W for this constraint
            improved = False
            for matrix, idx in [('U', i*m+j), ('V', j*p+k), ('W', k*n+i)]:
                M = {'U': U, 'V': V, 'W': W}[matrix]
                for c in range(rank):
                    old = M[idx][c]
                    for new_val in [-3, -2, -1, 0, 1, 2, 3]:
                        if new_val == old:
                            continue
                        M[idx][c] = new_val
                        # Check if this improves the worst constraint
                        new_s = 0
                        for cc in range(rank):
                            new_s += U[i*m + j][cc] * V[j*p + k][cc] * W[k*n + i][cc]
                        if abs(new_s - 1) < worst_err:
                            improved = True
                            break
                        M[idx][c] = old
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Random perturbation to escape local optimum
                if step % 50 == 0:
                    for _ in range(3):
                        mat = random.choice([U, V, W])
                        idx = random.randint(0, len(mat) - 1)
                        c = random.randint(0, rank - 1)
                        mat[idx][c] += random.choice([-1, 1])
    
    return None

def _try_rank_structured(n, m, p, rank, deadline):
    """
    Structured approach: exploit the 2x3 x 3x5 structure.
    Use Kronecker-like decomposition or block structure.
    """
    if time.time() > deadline:
        return None
    
    # For 2x3 * 3x5, we can think of it as:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # where i in {0,1}, j in {0,1,2}, k in {0,...,4}
    
    # Try a decomposition where we share terms across the 3 middle indices
    # Structure: for each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
    
    # A known approach: use the fact that 3 = 2+1 to split
    # But let's try a more direct search with structure
    
    # Initialize with a structured pattern
    if rank < 15:
        return None  # Below lower bound max(6,15,10)=15
    
    random.seed(42 + rank)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Structured initialization: assign values based on position
        U = []
        V = []
        W = []
        
        for i in range(n):
            for j in range(m):
                row = []
                for c in range(rank):
                    # Pattern: use modular arithmetic for structure
                    val = (i * 7 + j * 3 + c * 11) % 5 - 2
                    row.append(val)
                U.append(row)
        
        for j in range(m):
            for k in range(p):
                row = []
                for c in range(rank):
                    val = (j * 5 + k * 3 + c * 7) % 5 - 2
                    row.append(val)
                V.append(row)
        
        for k in range(p):
            for i in range(n):
                row = []
                for c in range(rank):
                    val = (k * 3 + i * 5 + c * 13) % 5 - 2
                    row.append(val)
                W.append(row)
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Local refinement
        for step in range(1000):
            if time.time() > deadline:
                break
            
            # Find constraints
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                        errors.append((abs(s-1), i, j, k))
            
            if not errors or errors[0][0] == 0:
                if all(e[0] == 0 for e in errors):
                    return {"U": U, "V": V, "W": W}
                break
            
            errors.sort(reverse=True)
            _, i, j, k = errors[0]
            
            # Try to fix
            for matrix, idx in [('U', i*m+j), ('V', j*p+k), ('W', k*n+i)]:
                M = {'U': U, 'V': V, 'W': W}[matrix]
                for c in range(rank):
                    old = M[idx][c]
                    for nv in [old-1, old+1, -old, 0, 1, -1]:
                        if nv == old:
                            continue
                        M[idx][c] = nv
                        new_s = sum(U[i*m+j][cc]*V[j*p+k][cc]*W[k*n+i][cc] for cc in range(rank))
                        if abs(new_s - 1) < errors[0][0]:
                            break
                        M[idx][c] = old
                    M[idx][c] = old
    
    return None

def _try_rank_nlp(n, m, p, rank, deadline):
    """
    Use scipy.optimize to solve for real-valued entries, then round.
    The tensor equation is cubic (trilinear), but we can set up
    a least-squares problem over the flattened tensor.
    """
    if time.time() > deadline:
        return None
    
    from scipy.optimize import minimize
    
    # Total number of variables
    nU = n * m * rank
    nV = m * p * rank
    nW = p * n * rank
    nvars = nU + nV + nW
    
    def objective(x):
        if time.time() > deadline:
            return float('inf')
        
        U_flat = x[:nU]
        V_flat = x[nU:nU+nV]
        W_flat = x[nU+nV:]
        
        U_arr = U_flat.reshape(n*m, rank)
        V_arr = V_flat.reshape(m*p, rank)
        W_arr = W_flat.reshape(p*n, rank)
        
        # Compute all tensor entries and compare to 1
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    err += (s - 1.0) ** 2
        return err
    
    # Initial guess
    x0 = np.random.uniform(-1, 1, nvars)
    x0[::max(1, nvars // rank)] = 1.0  # Some ones for structure
    
    best_result = None
    best_obj = float('inf')
    
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        np.random.seed(attempt * 12345 + rank)
        x0 = np.random.uniform(-2, 2, nvars)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8,
                                      'adaptive': True})
        except Exception:
            continue
        
        if result.fun < best_obj:
            best_obj = result.fun
            if result.fun < 1e-10:
                # Round to nearest integers
                x = np.round(result.x)
                U_flat = x[:nU]
                V_flat = x[nU:nU+nV]
                W_flat = x[nU+nV:]
                
                U = [list(map(int, U_arr_row)) for U_arr_row in U_flat.reshape(n*m, rank)]
                V = [list(map(int, V_arr_row)) for V_arr_row in V_flat.reshape(m*p, rank)]
                W = [list(map(int, W_arr_row)) for W_arr_row in W_flat.reshape(p*n, rank)]
                
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
                
                # Try half-integer rounding
                for scale in [2]:
                    x2 = np.round(result.x * scale) / scale
                    U_flat2 = x2[:nU]
                    V_flat2 = x2[nU:nU+nV]
                    W_flat2 = x2[nU+nV:]
                    
                    U2 = [list(U_arr_row) for U_arr_row in U_flat2.reshape(n*m, rank)]
                    V2 = [list(V_arr_row) for V_arr_row in V_flat2.reshape(m*p, rank)]
                    W2 = [list(W_arr_row) for W_arr_row in W_flat2.reshape(p*n, rank)]
                    
                    if _verify(U2, V2, W2, n, m, p):
                        return {"U": U2, "V": V2, "W": W2}
    
    return None

def _try_rank_ipa(n, m, p, rank, deadline):
    """
    Integer Programming Approach: Use a greedy column-by-column construction.
    Each column c contributes a rank-1 tensor: U_c[i,j] * V_c[j,k] * W_c[i,k].
    We need the sum over c to equal 1 for all (i,j,k).
    
    Strategy: build columns one at a time, each column is a "basis" of the
    space of rank-1 tensors. We use a greedy set-cover-like approach.
    """
    if time.time() > deadline:
        return None
    
    nU = n * m
    nV = m * p
    nW = p * n
    nT = n * m * p  # total tensor entries
    
    # Precompute: for each column c, the contribution to tensor entry (i,j,k)
    # is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # We want to find rank columns such that sum_c contrib_c[(i,j,k)] = 1 for all (i,j,k)
    
    # Greedy: at each step, pick the column that best reduces the residual
    # This is like a greedy approximation to a covering problem
    
    best_result = None
    num_restarts = 10
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart * 31337 + rank)
        
        # Start with zero residual (we need to build up to 1)
        residual = [[1 for _ in range(p)] for _ in range(m)]  # residual[i][j][k]
        residual = [[[1 for _ in range(p)] for _ in range(m)] for _ in range(n)]
        
        U_cols = []
        V_cols = []
        W_cols = []
        
        for col in range(rank):
            if time.time() > deadline:
                break
            
            # Find the column (U_c, V_c, W_c) that best reduces total error
            best_err = float('inf')
            best_U = None
            best_V = None
            best_W = None
            
            num_candidates = 20
            for cand in range(num_candidates):
                # Random candidate column
                Uc = [random.randint(-2, 2) for _ in range(nU)]
                Vc = [random.randint(-2, 2) for _ in range(nV)]
                Wc = [random.randint(-2, 2) for _ in range(nW)]
                
                # Compute new residual
                new_res = [row[:] for row in residual]  # deep copy
                total_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            contrib = Uc[i*m+j] * Vc[j*p+k] * Wc[k*n+i]
                            new_res[i][j][k] -= contrib
                            total_err += new_res[i][j][k] ** 2
                
                if total_err < best_err:
                    best_err = total_err
                    best_U = Uc
                    best_V = Vc
                    best_W = Wc
            
            if best_U is None:
                break
            
            U_cols.append(best_U)
            V_cols.append(best_V)
            W_cols.append(best_W)
            
            # Update residual
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[i][j][k] -= best_U[i*m+j] * best_V[j*p+k] * best_W[k*n+i]
            
            # Check if done
            if all(residual[i][j][k] == 0 for i in range(n) for j in range(m) for k in range(p)):
                # Build final matrices
                U = [[U_cols[c][i*m+j] for c in range(col+1)] for i in range(n) for j in range(m)]
                V = [[V_cols[c][j*p+k] for c in range(col+1)] for j in range(m) for k in range(p)]
                W = [[W_cols[c][k*n+i] for c in range(col+1)] for k in range(p) for i in range(n)]
                
                if _verify(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
        
        # If we didn't finish in 'rank' columns, try to verify what we have
        # (in case we used fewer columns than rank)
    
    return None

def _try_rank_enumerate_small(n, m, p, rank, deadline):
    """
    For small ranks, try to enumerate structured decompositions.
    Key insight: for 2x3*3x5, we can use the structure where
    the 3 middle dimension is split into 2+1.
    
    A[i,:] * B[:,k] for i in {0,1}, k in {0..4}
    
    We can write B as [B_top; B_bottom] where B_top is 2x5, B_bottom is 1x5.
    Then C[i,k] = A[i,0:2] @ B_top[:,k] + A[i,2] * B_bottom[:,k]
    
    This gives a structured decomposition.
    """
    if time.time() > deadline:
        return None
    
    # The splitting approach:
    # A is 2x3, B is 3x5
    # Split j into j in {0,1} (top) and j=2 (bottom)
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # For the 2x2 part (i in {0,1}, k in {0,1} of top part), we can use rank-2
    # For cross terms, we need additional rank
    
    # Let's try: rank = 2*5 + 2*3 + 2*5 - something
    # Actually, let's just do a focused search for the specific structure
    
    if rank < 18:
        return None
    
    random.seed(999 + rank)
    
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        # Structured init: group columns by (i, k) pair
        # There are 2*5 = 10 output entries, each needs sum_j contribution
        # Use 3 columns per output entry as a starting structure
        
        U = []
        V = []
        W = []
        
        for i in range(n):
            for j in range(m):
                row = [0] * rank
                # Assign based on which (i,k) groups this j contributes to
                for k in range(p):
                    # Each (i,k) needs rank/10 columns
                    base = (i * p + k) * (rank // 10)
                    for offset in range(rank // 10):
                        c = base + offset
                        if c < rank:
                            row[c] = 1 if j == 0 else (1 if j == 1 else 0)
                U.append(row)
        
        for j in range(m):
            for k in range(p):
                row = [0] * rank
                for i in range(n):
                    base = (i * p + k) * (rank // 10)
                    for offset in range(rank // 10):
                        c = base + offset
                        if c < rank:
                            row[c] = 1 if j == k % 3 else 0
                V.append(row)
        
        for k in range(p):
            for i in range(n):
                row = [0] * rank
                for j in range(m):
                    base = (i * p + k) * (rank // 10)
                    for offset in range(rank // 10):
                        c = base + offset
                        if c < rank:
                            row[c] = 1
                W.append(row)
        
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
        # Local search from this structured init
        for step in range(500):
            if time.time() > deadline:
                break
            
            # Find worst constraint
            worst = 0
            worst_ijk = None
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = sum(U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c] for c in range(rank))
                        e = abs(s - 1)
                        if e > worst:
                            worst = e
                            worst_ijk = (i, j, k)
            
            if worst == 0:
                return {"U": U, "V": V, "W": W}
            
            if worst_ijk is None:
                break
            
            i, j, k = worst_ijk
            # Try small adjustments
            for matrix, idx in [('U', i*m+j), ('V', j*p+k), ('W', k*n+i)]:
                M = {'U': U, 'V': V, 'W': W}[matrix]
                for c in range(rank):
                    old = M[idx][c]
                    for nv in [old-1, old+1]:
                        M[idx][c] = nv
                        new_s = sum(U[i*m+j][cc]*V[j*p+k][cc]*W[k*n+i][cc] for cc in range(rank))
                        if abs(new_s - 1) < worst:
                            break
                        M[idx][c] = old
                    M[idx][c] = old
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 30
    
    # Get schoolbook as guaranteed fallback
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Try various strategies from lower ranks upward
    # Lower bound: max(n*m, m*p, n*p) = max(6, 15, 10) = 15
    # Known results suggest rank is around 24-26 for this size
    
    # Strategy 1: NLP (continuous relaxation then round)
    for rank in range(16, min(schoolbook_rank, 28)):
        if time.time() > deadline - 20:
            break
        
        result = _try_rank_nlp(n, m, p, rank, deadline - 20)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    # Strategy 2: Structured enumeration
    for rank in range(16, min(schoolbook_rank, 28)):
        if time.time() > deadline - 15:
            break
        
        result = _try_rank_enumerate_small(n, m, p, rank, deadline - 15)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    # Strategy 3: Random local search
    for rank in range(16, min(schoolbook_rank, 28)):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank_random_local_search(n, m, p, rank, deadline - 10)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    # Strategy 4: Greedy column construction (IPA-like)
    for rank in range(16, min(schoolbook_rank, 28)):
        if time.time() > deadline - 5:
            break
        
        result = _try_rank_ipa(n, m, p, rank, deadline - 5)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    # Strategy 5: General structured search
    for rank in range(16, min(schoolbook_rank, 28)):
        if time.time() > deadline - 2:
            break
        
        result = _try_rank_structured(n, m, p, rank, deadline - 2)
        if result is not None:
            best = result
            best_rank = rank
            break
    
    return best
# EVOLVE-BLOCK-END
