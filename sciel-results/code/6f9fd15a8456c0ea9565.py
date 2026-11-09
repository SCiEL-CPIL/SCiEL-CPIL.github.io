"""Matrix multiplication 3x4x8 tensor decomposition search."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    rank_trivial = n * m * p  # 96
    triples = [(i, k, j) for i in range(n) for k in range(m) for j in range(p)]
    T = len(triples)  # 96
    
    idx_U = np.array([i*m+k for i,k,j in triples])
    idx_V = np.array([k*p+j for i,k,j in triples])
    idx_W = np.array([j*n+i for i,k,j in triples])
    
    def _verify_fast(U, V, W, tol=1e-9):
        """Verify decomposition using vectorized operations."""
        r = U.shape[1]
        U_exp = U[idx_U, :]
        V_exp = V[idx_V, :]
        W_exp = W[idx_W, :]
        vals = np.einsum('tr,tr,tr->t', U_exp, V_exp, W_exp)
        return bool(np.all(np.abs(vals - 1.0) < tol))
    
    def _verify_exact(U, V, W):
        """Exact verification using Fraction arithmetic."""
        U_frac = [[Fraction(int(round(x*2)), 2) for x in row] for row in U]
        V_frac = [[Fraction(int(round(x*2)), 2) for x in row] for row in V]
        W_frac = [[Fraction(int(round(x*2)), 2) for x in row] for row in W]
        
        for t in range(T):
            s = Fraction(0)
            for r in range(len(U_frac[0])):
                s += U_frac[idx_U[t]][r] * V_frac[idx_V[t]][r] * W_frac[idx_W[t]][r]
            if s != Fraction(1):
                return False
        return True
    
    def _clean_output(U, V, W):
        """Convert to clean integer/half-integer representation."""
        def _clean_mat(M):
            result = []
            for row in M:
                new_row = []
                for val in row:
                    v2 = round(val * 2)
                    if v2 % 2 == 0:
                        new_row.append(int(v2 // 2))
                    else:
                        new_row.append(v2 / 2.0)
                result.append(new_row)
            return result
        return {"U": _clean_mat(U), "V": _clean_mat(V), "W": _clean_mat(W)}
    
    best_result = None
    best_rank = rank_trivial
    
    lower_bound = max(n*m, m*p, n*p)  # 32
    
    # Key structural change: systematic search over U, V with small entries,
    # solve linear system for W, verify half-integer constraint.
    # For each rank r, try many (U,V) pairs from {-1,0,1}^r, solve A@W=b,
    # check if solution is half-integer.
    
    # Use a more efficient approach: iterate over possible ranks from lower_bound up
    # and for each rank, try random U,V and solve for W
    
    for rank in range(lower_bound, min(rank_trivial + 1, 50)):
        if time.time() > deadline - 5.0:
            break
        
        found = False
        # Number of attempts: more for smaller ranks
        if rank <= 35:
            attempts = 100
        elif rank <= 40:
            attempts = 50
        elif rank <= 45:
            attempts = 30
        else:
            attempts = 20
        
        for attempt in range(attempts):
            if time.time() > deadline - 5.0:
                break
            
            rng = np.random.default_rng(seed=rank * 100000 + attempt * 7)
            
            # Generate random U, V with entries from {-1, 0, 1}
            U = rng.choice([-1, 0, 1], size=(n*m, rank)).astype(float)
            V = rng.choice([-1, 0, 1], size=(m*p, rank)).astype(float)
            
            # Build the linear system: for each triple (i,k,j),
            # sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = 1
            # This is linear in W.
            num_unknowns = p * n * rank  # 24 * rank
            A = np.zeros((T, num_unknowns))
            b = np.ones(T)
            
            for t, (i, k, j) in enumerate(triples):
                for r in range(rank):
                    col = (j * n + i) * rank + r
                    A[t, col] = U[i*m+k, r] * V[k*p+j, r]
            
            # Solve least squares
            try:
                W_sol, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                W_sol = W_sol.reshape(p*n, rank)
                
                # Check if W_sol is close to half-integers
                W_proj = np.round(W_sol * 2) / 2
                
                # Quick verification
                if _verify_fast(U, V, W_proj, 1e-8):
                    # Exact verification
                    if _verify_exact(U, V, W_proj):
                        best_result = _clean_output(U, V, W_proj)
                        best_rank = rank
                        found = True
                        break
            except np.linalg.LinAlgError:
                continue
        
        if found:
            break
    
    # If the above didn't work, try a structured approach exploiting 8 = 2*4
    if best_result is None or best_rank == rank_trivial:
        if time.time() < deadline - 10.0:
            for rank in range(lower_bound, min(rank_trivial + 1, 45)):
                if time.time() > deadline - 5.0:
                    break
                
                found = False
                attempts = 30
                
                for attempt in range(attempts):
                    if time.time() > deadline - 5.0:
                        break
                    
                    rng = np.random.default_rng(seed=rank * 200000 + attempt * 13)
                    
                    U = rng.choice([-1, 0, 1], size=(n*m, rank)).astype(float)
                    
                    # Structured V: same values for j and j+4
                    V = np.zeros((m*p, rank))
                    for k in range(m):
                        for j in range(4):
                            val = rng.choice([-1, 0, 1], size=(rank,))
                            V[k*p+j, :] = val
                            V[k*p+(j+4), :] = val
                    
                    num_unknowns = p * n * rank
                    A = np.zeros((T, num_unknowns))
                    b = np.ones(T)
                    
                    for t, (i, k, j) in enumerate(triples):
                        for r in range(rank):
                            col = (j * n + i) * rank + r
                            A[t, col] = U[i*m+k, r] * V[k*p+j, r]
                    
                    try:
                        W_sol, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
                        W_sol = W_sol.reshape(p*n, rank)
                        W_proj = np.round(W_sol * 2) / 2
                        
                        if _verify_fast(U, V, W_proj, 1e-8):
                            if _verify_exact(U, V, W_proj):
                                best_result = _clean_output(U, V, W_proj)
                                best_rank = rank
                                found = True
                                break
                    except np.linalg.LinAlgError:
                        continue
                
                if found:
                    break
    
    # Fallback: return schoolbook decomposition
    if best_result is None:
        U_out = [[0]*rank_trivial for _ in range(n*m)]
        V_out = [[0]*rank_trivial for _ in range(m*p)]
        W_out = [[0]*rank_trivial for _ in range(p*n)]
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    r = i*(m*p) + k*p + j
                    U_out[i*m+k][r] = 1
                    V_out[k*p+j][r] = 1
                    W_out[j*n+i][r] = 1
        best_result = {"U": U_out, "V": V_out, "W": W_out}
    
    return best_result
# EVOLVE-BLOCK-END
