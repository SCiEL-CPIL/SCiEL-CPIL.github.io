import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # 115s limit, leaving 5s margin
    
    # Trivial solution (rank 90)
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
                
    best_U, best_V, best_W = U_trivial, V_trivial, W_trivial
    best_rank = rank_trivial
    
    # Strassen 2x2x2 decomposition (rank 7)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    def strassen_decomposition():
        """Return Strassen 2x2x2 decomposition as (U, V, W) with shapes (4,7), (4,7), (4,7)"""
        # U indexed by (i,j) for i,j in {0,1} -> row i*2+j
        # V indexed by (j,k) for j,k in {0,1} -> row j*2+k
        # W indexed by (k,i) for k,i in {0,1} -> row k*2+i (trace convention)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1  # A11
        U[3, 0] = 1  # A22
        V[0, 0] = 1  # B11
        V[3, 0] = 1  # B22
        W[0, 0] = 1  # C11 contribution (k=0,i=0)
        W[1, 1] = 1  # C22 contribution (k=1,i=1)
        
        # M2 = (A21+A22)B11
        U[1, 1] = 1  # A21 (row 1*2+1=3? No, row i*2+j, so A21 is i=1,j=0 -> row 2)
        # Wait, let me be careful. A[i,j] where i,j in {0,1}
        # A11 = A[0,0] -> row 0
        # A12 = A[0,1] -> row 1
        # A21 = A[1,0] -> row 2
        # A22 = A[1,1] -> row 3
        # B11 = B[0,0] -> row 0
        # B12 = B[0,1] -> row 1
        # B21 = B[1,0] -> row 2
        # B22 = B[1,1] -> row 3
        # C11 = C[0,0] -> W[k*n+i] = W[0*2+0] = W[0]
        # C12 = C[0,1] -> W[1*2+0] = W[2]
        # C21 = C[1,0] -> W[0*2+1] = W[1]
        # C22 = C[1,1] -> W[1*2+1] = W[3]
        
        # Redo:
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1  # A11
        U[3, 0] = 1  # A22
        V[0, 0] = 1  # B11
        V[3, 0] = 1  # B22
        W[0, 0] = 1  # C11
        W[3, 0] = 1  # C22
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1  # A21
        U[3, 1] = 1  # A22
        V[0, 1] = 1  # B11
        W[1, 1] = 1  # C21
        
        # M3 = A11(B21-B22)
        U[0, 2] = 1  # A11
        V[2, 2] = 1  # B21
        V[3, 2] = -1 # B22
        W[2, 2] = 1  # C12
        
        # M4 = A22(B11-B21)
        U[3, 3] = 1  # A22
        V[0, 3] = 1  # B11
        V[2, 3] = -1 # B21
        W[0, 3] = 1  # C11
        W[1, 3] = 1  # C21
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1  # A11
        U[1, 4] = 1  # A12
        V[3, 4] = 1  # B22
        W[0, 4] = -1 # C11 (negative)
        W[2, 4] = 1  # C12
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1  # A21
        U[0, 5] = -1 # A11
        V[0, 5] = 1  # B11
        V[1, 5] = 1  # B12
        W[3, 5] = 1  # C22
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1  # A12
        U[3, 6] = -1 # A22
        V[2, 6] = 1  # B21
        V[3, 6] = 1  # B22
        W[0, 6] = 1  # C11
        W[3, 6] = 1  # C22
        
        return U, V, W
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact"""
        Ur = U.reshape(n, m, -1)
        Vr = V.reshape(m, p, -1)
        Wr = W.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
        target = np.ones((n, m, p))
        return np.allclose(T, target, atol=1e-9)
    
    def try_optimization(rank, max_iter=200, lr_init=0.1):
        """Try to find a rank-rank decomposition using L-BFGS-B"""
        if time.time() > deadline - 5:
            return None
        
        # Initialize with structured approach: start from trivial and project
        # Or use random init with proper scaling
        scale = 1.0 / np.sqrt(rank)
        U = np.random.randn(n * m, rank) * scale
        V = np.random.randn(m * p, rank) * scale
        W = np.random.randn(p * n, rank) * scale
        
        try:
            from scipy.optimize import minimize
        except ImportError:
            return None
        
        def objective(params):
            U_flat = params[:n*m*rank].reshape(n*m, rank)
            V_flat = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_flat = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            Ur = U_flat.reshape(n, m, -1)
            Vr = V_flat.reshape(m, p, -1)
            Wr = W_flat.reshape(p, n, -1)
            
            T = np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
            target = np.ones((n, m, p))
            error = T - target
            loss = np.sum(error**2)
            
            dU = np.einsum('ijk,jkr,kir->ijr', error, Vr, Wr).reshape(-1)
            dV = np.einsum('ijk,ijr,kir->jkr', error, Ur, Wr).reshape(-1)
            dW = np.einsum('ijk,ijr,jkr->kir', error, Ur, Vr).reshape(-1)
            
            grad = np.concatenate([dU, dV, dW])
            return loss, grad
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=True, 
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000})
            
            if res.fun < 1e-6:
                U_opt = res.x[:n*m*rank].reshape(n*m, rank)
                V_opt = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_opt = res.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Try snapping to half-integers
                for mult in [2, 4, 1]:  # Try different precisions
                    U_snap = np.round(U_opt * mult) / mult
                    V_snap = np.round(V_opt * mult) / mult
                    W_snap = np.round(W_opt * mult) / mult
                    
                    if verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                        return U_snap, V_snap, W_snap
                
                # If snapping fails, check if original is close enough
                if verify_decomposition(U_opt, V_opt, W_opt, n, m, p):
                    return U_opt, V_opt, W_opt
                    
        except Exception:
            pass
        
        return None
    
    # Search for better ranks
    # Known: rank(3x5x6) is between 76 and 78 based on literature
    # Try from 76 upward
    for rank in range(76, 85):
        if time.time() > deadline - 10:
            break
            
        found = False
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
                
            res = try_optimization(rank, max_iter=100)
            if res is not None:
                U_opt, V_opt, W_opt = res
                if rank < best_rank:
                    best_U = U_opt.tolist()
                    best_V = V_opt.tolist()
                    best_W = W_opt.tolist()
                    best_rank = rank
                    found = True
                break
        
        if found:
            # Found a valid decomposition, try lower ranks
            break
    
    # If optimization didn't find anything, try a structured block approach
    if best_rank == rank_trivial:
        # Try a simpler structured approach: use the fact that we can
        # decompose 3x5x6 as (2+1)x5x6
        # Split into 2x5x6 and 1x5x6
        # 1x5x6 has rank 30 (trivial)
        # 2x5x6: try to find good rank
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
