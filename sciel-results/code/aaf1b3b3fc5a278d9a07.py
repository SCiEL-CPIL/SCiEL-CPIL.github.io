"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                ui = U[i*m+j]
                vj = V[j*p+k]
                wk = W[k*n+i]
                for c in range(rank):
                    s += ui[c] * vj[c] * wk[c]
                if s != 1:
                    return False
    return True

def _verify_float(U, V, W, n, m, p, tol=1e-10):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                ui = U[i*m+j]
                vj = V[j*p+k]
                wk = W[k*n+i]
                for c in range(rank):
                    s += ui[c] * vj[c] * wk[c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _solve_rational(U_support, V_support, W_support, n, m, p):
    """Given binary support patterns, solve for rational coefficients."""
    rank = len(U_support[0])
    # Build linear system: for each (i,j,k), sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    # Unknowns: U[i,j,c] for all i,j,c; V[j,k,c] for all j,k,c; W[k,i,c] for all k,i,c
    # This is nonlinear! Need different approach.
    
    # Alternative: fix V and W to 1 where support exists, solve for U
    # Or use a structured approach.
    
    # Let's try: assume V[j,k,c] = 1 if V_support[j,k,c]=1 else 0
    #            assume W[k,i,c] = 1 if W_support[k,i,c]=1 else 0
    # Then U[i,j,c] = 1 / (V[j,k,c] * W[k,i,c]) for the (i,j,k) term
    # But each U[i,j,c] appears in multiple equations (for different k)
    
    # Better: use the fact that we can scale columns.
    # For each column c, we have freedom to scale U[:,c], V[:,c], W[:,c]
    # such that the product scales appropriately.
    
    # Let's set up: for column c, let a_c = product of all U entries in col c
    #             b_c = product of all V entries in col c  
    #             c_c = product of all W entries in col c
    # We can normalize so that a_c * b_c * c_c = 1 (or some fixed value)
    
    # Simpler approach: just try all-ones for supported entries and verify
    U = [[1 if U_support[i*m+j][c] else 0 for c in range(rank)] for i in range(n*m)]
    V = [[1 if V_support[j*p+k][c] else 0 for c in range(rank)] for j in range(m*p)]
    W = [[1 if W_support[k*n+i][c] else 0 for c in range(rank)] for k in range(p*n)]
    
    if _verify(U, V, W, n, m, p):
        return U, V, W
    
    return None

def _find_support_f2(n, m, p, max_rank=48, max_iters=5000):
    """Search for sparse support over F_2."""
    # The multiplication tensor has n*m*p = 48 nonzero entries
    # Each entry (i,j,k) must be covered by at least one column c where
    # U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1
    
    # Represent supports as bitmasks
    U_size = n * m  # 12
    V_size = m * p  # 16
    W_size = p * n  # 12
    
    best_solution = None
    best_rank = max_rank + 1
    
    for rank in range(24, max_rank + 1):
        if rank >= best_rank:
            break
        found = False
        for it in range(max_iters):
            if time.time() > time.time() + 0:  # placeholder
                pass
            
            # Random support generation
            U_sup = [0] * U_size  # bitmask of which columns have 1
            V_sup = [0] * V_size
            W_sup = [0] * W_size
            
            for c in range(rank):
                # Each column picks random subsets
                for idx in range(U_size):
                    if random.random() < 0.3:
                        U_sup[idx] |= (1 << c)
                for idx in range(V_size):
                    if random.random() < 0.3:
                        V_sup[idx] |= (1 << c)
                for idx in range(W_size):
                    if random.random() < 0.3:
                        W_sup[idx] |= (1 << c)
            
            # Check coverage: for each (i,j,k), exists c with U[i,j,c]&V[j,k,c]&W[k,i,c]
            covered = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        ui = U_sup[i*m+j]
                        vj = V_sup[j*p+k]
                        wk = W_sup[k*n+i]
                        if (ui & vj & wk) == 0:
                            covered = False
                            break
                    if not covered:
                        break
                if not covered:
                    break
            
            if covered:
                # Try to solve for rational coefficients
                U_list = [[(U_sup[i*m+j] >> c) & 1 for c in range(rank)] for i in range(U_size)]
                V_list = [[(V_sup[j*p+k] >> c) & 1 for c in range(rank)] for j in range(V_size)]
                W_list = [[(W_sup[k*n+i] >> c) & 1 for c in range(rank)] for k in range(W_size)]
                
                sol = _solve_rational(U_list, V_list, W_list, n, m, p)
                if sol is not None:
                    best_solution = sol
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    return best_solution, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Greedy column removal
    while time.time() < deadline - 30:
        found = False
        current_rank = len(U[0])
        for c in range(current_rank):
            if time.time() > deadline - 30:
                break
            U_new = [row[:c] + row[c+1:] for row in U]
            V_new = [row[:c] + row[c+1:] for row in V]
            W_new = [row[:c] + row[c+1:] for row in W]
            if _verify(U_new, V_new, W_new, n, m, p):
                U, V, W = U_new, V_new, W_new
                found = True
                if len(U[0]) < best_rank:
                    best_rank = len(U[0])
                    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
                break
        if not found:
            break
    
    # Try finite-field support search
    if time.time() < deadline - 5:
        sol, fr = _find_support_f2(n, m, p, max_rank=best_rank - 1, max_iters=2000)
        if sol is not None and fr < best_rank:
            best_rank = fr
            best_U, best_V, best_W = sol
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
