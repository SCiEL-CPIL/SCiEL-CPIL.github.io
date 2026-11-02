import numpy as np
import time

def construct():
    n, m, p = 3, 5, 7
    nrm, mpr, pnr = n*m, m*p, p*n
    
    # Target tensor T[i, j, k] = 1 if indices match multiplication, else 0
    targets = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                targets.append((u_idx, v_idx, w_idx, 1.0))
    
    n_targets = len(targets)
    
    deadline = time.time() + 110
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook (rank 105)
    U_sch = np.zeros((nrm, best_rank))
    V_sch = np.zeros((mpr, best_rank))
    W_sch = np.zeros((pnr, best_rank))
    for idx, (u, v, w, val) in enumerate(targets):
        U_sch[u, idx] = 1.0
        V_sch[v, idx] = 1.0
        W_sch[w, idx] = 1.0
    best_U, best_V, best_W = U_sch, V_sch, W_sch
    
    # Structured initialization based on grouping by middle index j
    # For each j, the sub-problem is: row i of A times column k of B
    # This gives a natural rank-m decomposition where each "block" handles one j
    
    def check_error(U, V, W, rank):
        """Compute sum of squared errors"""
        err = 0.0
        for u, v, w, val in targets:
            s = 0.0
            for c in range(rank):
                s += U[u, c] * V[v, c] * W[w, c]
            err += (s - val) ** 2
        return err
    
    def snap_and_check(U, V, W, rank):
        """Snap to half-integers and verify"""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        err = 0.0
        for u, v, w, val in targets:
            s = 0.0
            for c in range(rank):
                s += U_snap[u, c] * V_snap[v, c] * W_snap[w, c]
            err += (s - val) ** 2
        return U_snap, V_snap, W_snap, err
    
    def als_solve(U, V, W, rank, max_iters=80, tol=1e-10):
        """Alternating least squares with structured updates"""
        for it in range(max_iters):
            if time.time() > deadline - 8:
                return U, V, W
            
            # Solve for U
            A_U = np.zeros((n_targets, nrm * rank))
            b = np.ones(n_targets)
            for t, (u, v, w, val) in enumerate(targets):
                for c in range(rank):
                    A_U[t, u * rank + c] = V[v, c] * W[w, c]
            U_flat, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
            U = U_flat.reshape(nrm, rank)
            
            # Solve for V
            A_V = np.zeros((n_targets, mpr * rank))
            for t, (u, v, w, val) in enumerate(targets):
                for c in range(rank):
                    A_V[t, v * rank + c] = U[u, c] * W[w, c]
            V_flat, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
            V = V_flat.reshape(mpr, rank)
            
            # Solve for W
            A_W = np.zeros((n_targets, pnr * rank))
            for t, (u, v, w, val) in enumerate(targets):
                for c in range(rank):
                    A_W[t, w * rank + c] = U[u, c] * V[v, c]
            W_flat, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
            W = W_flat.reshape(pnr, rank)
            
            # Check convergence
            err = check_error(U, V, W, rank)
            if err < tol:
                break
        
        return U, V, W
    
    def structured_init(rank):
        """Initialize with structure: group terms by middle index j"""
        U = np.zeros((nrm, rank))
        V = np.zeros((mpr, rank))
        W = np.zeros((pnr, rank))
        
        # Distribute rank among the m=5 middle indices
        # Each j gets rank//m or rank//m + 1 terms
        base = rank // m
        extra = rank % m
        
        col = 0
        for j in range(m):
            n_terms = base + (1 if j < extra else 0)
            # For this j, we need to decompose the n x p = 3x7 = 21 entries
            # into n_terms rank-1 terms
            # Simple: use random init for this block
            for c in range(n_terms):
                # U[i*m+j, col] for i in 0..2
                for i in range(n):
                    U[i*m+j, col] = np.random.randn() * 0.5
                # V[j*p+k, col] for k in 0..6
                for k in range(p):
                    V[j*p+k, col] = np.random.randn() * 0.5
                # W[k*n+i, col] for i in 0..2, k in 0..6
                for k in range(p):
                    for i in range(n):
                        W[k*n+i, col] = np.random.randn() * 0.5
                col += 1
        
        return U, V, W
    
    def random_init(rank):
        """Pure random initialization"""
        U = np.random.randn(nrm, rank) * 0.5
        V = np.random.randn(mpr, rank) * 0.5
        W = np.random.randn(pnr, rank) * 0.5
        return U, V, W
    
    # Search ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # Try a range of ranks
    ranks_to_try = list(range(35, 76))  # 35 to 75
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        found = False
        n_attempts = 15 if rank <= 50 else 8
        
        for attempt in range(n_attempts):
            if time.time() > deadline - 5:
                break
            
            # Mix of initialization strategies
            if attempt % 3 == 0:
                U, V, W = structured_init(rank)
            else:
                U, V, W = random_init(rank)
            
            U, V, W = als_solve(U, V, W, rank, max_iters=60)
            
            err = check_error(U, V, W, rank)
            
            if err < 1e-6:
                U_snap, V_snap, W_snap, err_snap = snap_and_check(U, V, W, rank)
                if err_snap < 1e-9:
                    best_rank = rank
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    found = True
                    break
            
            # Also try with tighter ALS if close
            if err < 1e-3:
                U, V, W = als_solve(U, V, W, rank, max_iters=100, tol=1e-12)
                err = check_error(U, V, W, rank)
                if err < 1e-6:
                    U_snap, V_snap, W_snap, err_snap = snap_and_check(U, V, W, rank)
                    if err_snap < 1e-9:
                        best_rank = rank
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        found = True
                        break
        
        if found:
            break
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
