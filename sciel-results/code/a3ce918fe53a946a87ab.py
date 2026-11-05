"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, rank):
    """Verify exact decomposition."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _is_half_int(x):
    """Check if x is integer or half-integer."""
    x2 = x * 2.0
    r = round(x2)
    return abs(x2 - r) < 1e-9

def _snap_to_half_int(x):
    """Snap x to nearest integer or half-integer."""
    x2 = x * 2.0
    r = round(x2)
    return r / 2.0

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook solution as guaranteed fallback
    rank_school = n * m * p
    U_sch = [[0] * rank_school for _ in range(nm)]
    V_sch = [[0] * rank_school for _ in range(mp)]
    W_sch = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m+j][col] = 1
                V_sch[j*p+k][col] = 1
                W_sch[k*n+i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sch, V_sch, W_sch
    best_rank = rank_school
    
    # Try to find lower-rank solutions
    lb = max(n*m, m*p, p*n)
    
    for rank in range(lb, best_rank):
        if time.time() > deadline - 20:
            break
        
        found = False
        for trial in range(30):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(rank * 10000 + trial)
            U = np.random.randn(nm, rank)
            V = np.random.randn(mp, rank)
            W = np.random.randn(pn, rank)
            
            # ALS iterations
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        M = np.array([[V[j*p+k, c] * W[k*n+i, c] for c in range(rank)] for k in range(p)])
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        M = np.array([[U[i*m+j, c] * W[k*n+i, c] for c in range(rank)] for i in range(n)])
                        try:
                            V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        M = np.array([[U[i*m+j, c] * V[j*p+k, c] for c in range(rank)] for j in range(m)])
                        try:
                            W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence periodically
                if it % 10 == 9:
                    ok = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                                if abs(s - 1.0) > 1e-4:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if not ok:
                            break
                    if ok:
                        # Try to snap to half-integers and verify
                        U_snapped = [[_snap_to_half_int(U[i, c]) for c in range(rank)] for i in range(nm)]
                        V_snapped = [[_snap_to_half_int(V[i, c]) for c in range(rank)] for i in range(mp)]
                        W_snapped = [[_snap_to_half_int(W[i, c]) for c in range(rank)] for i in range(pn)]
                        
                        if _verify(U_snapped, V_snapped, W_snapped, n, m, p, rank):
                            best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
                            best_rank = rank
                            found = True
                            break
            
            if found:
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
