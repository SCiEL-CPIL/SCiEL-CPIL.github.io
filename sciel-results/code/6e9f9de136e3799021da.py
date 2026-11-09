"""Matrix multiplication 2x3x5 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    def schoolbook():
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
    
    def verify_exact(U, V, W, tol=1e-9):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Ua[i*m+j, :], Va[j*p+k, :] * Wa[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to construct a decomposition with reduced rank using algebraic structure
    # For 2x3x5, we can try rank 15 by pairing up the 30 terms
    
    # Strategy: For each pair of j values, combine them
    # C[i,k] = sum_j A[i,j]B[j,k]
    # Group j=0,1 together and j=2 separately
    # A[i,0]B[0,k] + A[i,1]B[1,k] = (A[i,0]+A[i,1])/2 * (B[0,k]+B[1,k]) + (A[i,0]-A[i,1])/2 * (B[0,k]-B[1,k])
    # This uses 2 terms for 2 original terms - no gain
    # But if we share across i values...
    
    # Better: use the 2x2 block structure in i dimension
    # For fixed j,k: we have A[0,j]B[j,k] and A[1,j]B[j,k]
    # These are independent, so no sharing there
    
    # Let me try a direct computational search with better initialization
    # Use the structure: initialize from schoolbook and merge columns
    
    def try_merge(rank_target):
        """Try to merge schoolbook columns to achieve target rank."""
        n, m, p = DIMENSIONS
        nm, mp, pn = n*m, m*p, p*n
        
        # Start from schoolbook
        U = np.zeros((nm, nm*mp//mn if False else n*m*p))
        V = np.zeros((mp, n*m*p))
        W = np.zeros((pn, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        
        # Try to find linear dependencies and reduce rank
        # This is complex; instead, let's try a parametric family
        
        # Family: for each (i,j,k) triple, allow U[i*m+j,c], V[j*p+k,c], W[k*n+i,c] to be non-zero
        # But share across k for fixed (i,j) or across j for fixed (i,k)
        
        # Try: group by (i, j) and use 5 parameters for k dimension with a low-rank structure
        # For fixed (i,j), C[i,k] += A[i,j] * B[j,k]
        # If we use a rank-2 decomposition of the k-dimension...
        
        return None
    
    # Let's try a direct approach: assume a specific structure and solve for coefficients
    # Structure: rank = 15, with pairs of schoolbook terms merged
    
    # For each (i,j), we have p=5 terms in k. If we can represent the k-dependence 
    # with fewer parameters using a shared structure across i...
    
    # Alternative: just return schoolbook if we can't find better quickly
    # But let's try one more thing: use the known result that for 2x3x3 rank is 7
    
    # Decompose 2x3x5 as: (2x3x3) x (3x5) in some sense?
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Split B into B1 (3x3) and B2 (3x2): B[j,k] = B1[j,k] + B2[j,k] for k<3, B2[j,k] for k>=3
    # C[i,k] = sum_j A[i,j] B1[j,k] + sum_j A[i,j] B2[j,k]
    # First term is 2x3x3 multiplication, second is 2x3x2 multiplication
    # Rank(2x3x3) = 7, Rank(2x3x2) = ?
    # 2x3x2: C[i,k] = sum_j A[i,j] B[j,k], i=0,1; j=0,1,2; k=0,1
    # Schoolbook: 12. Can we do better?
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    # C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
    # C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
    # This is like 2 separate 1x3x2 multiplications, or we can share B[j,k] terms
    # Actually, for fixed j: A[0,j] and A[1,j] multiply B[j,0] and B[j,1]
    # This is a 2x2 matrix times a 2-vector... 
    # C[:,0] = A[:,0]B[0,0] + A[:,1]B[1,0] + A[:,2]B[2,0] (vector equation)
    # C[:,1] = A[:,0]B[0,1] + A[:,1]B[1,1] + A[:,2]B[2,1]
    # So C = A * B where A is 2x3, B is 3x2
    # Rank of M_{2,3,2}: schoolbook is 12. 
    # Can we do better? Each output C[i,k] depends on 3 products.
    # Lower bound: at least max(n*p, m*p, n*m) = max(4,6,6) = 6? No, that's not right.
    # The rank must be at least the dimension of any slice. 
    # For M_{2,3,2}: slice at j=0 gives a 2x2 matrix (rank 2), so rank >= 2.
    # Actually, the rank is at least the max of the ranks of all slices, but that's weak.
    # I think M_{2,3,2} has rank 10 or so. Let me not go down this path.
    
    # Given time constraints, let me try a numerical search for rank 20, 21, 22, etc.
    # with a smarter objective
    
    def search_rank(r, n_restarts=10):
        n, m, p = DIMENSIONS
        nm, mp, pn = n*m, m*p, p*n
        total_vars = (nm + mp + pn) * r
        
        best_err = float('inf')
        best_x = None
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            np.random.seed(1000 + restart * 777)
            # Initialize with small random values
            scale = 1.0 / (r ** 0.5)
            U0 = np.random.randn(nm, r) * scale
            V0 = np.random.randn(mp, r) * scale
            W0 = np.random.randn(pn, r) * scale
            x0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
            
            def obj(x):
                Uf = x[:nm*r].reshape(nm, r)
                Vf = x[nm*r:nm*r+mp*r].reshape(mp, r)
                Wf = x[nm*r+mp*r:].reshape(pn, r)
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(Uf[i*m+j, :], Vf[j*p+k, :] * Wf[k*n+i, :])
                            err += (val - 1.0) ** 2
                return err
            
            # Use Nelder-Mead with reasonable limits
            from scipy.optimize import minimize
            res = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x.copy()
        
        if best_err < 1e-4 and best_x is not None:
            Uf = best_x[:nm*r].reshape(nm, r)
            Vf = best_x[nm*r:nm*r+mp*r].reshape(mp, r)
            Wf = best_x[nm*r+mp*r:].reshape(pn, r)
            
            # Snap to simple values
            def snap_matrix(M):
                Ms = M.copy()
                for idx in np.ndindex(M.shape):
                    v = M[idx]
                    for den in [1, 2, 4]:
                        num = round(v * den) / den
                        if abs(v - num) < 1e-5:
                            Ms[idx] = num
                            break
                return Ms
            
            Us = snap_matrix(Uf)
            Vs = snap_matrix(Vf)
            Ws = snap_matrix(Wf)
            
            if verify_exact(Us.tolist(), Vs.tolist(), Ws.tolist(), tol=1e-4):
                return Us.tolist(), Vs.tolist(), Ws.tolist()
            
            # Try with more precision
            U_r = np.round(Uf, 10).tolist()
            V_r = np.round(Vf, 10).tolist()
            W_r = np.round(Wf, 10).tolist()
            if verify_exact(U_r, V_r, W_r, tol=1e-6):
                return U_r, V_r, W_r
            
            return None
        return None
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try ranks from 18 to 29 (skip some to save time)
    for r in [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
        if time.time() > deadline - 20:
            break
        result = search_rank(r, n_restarts=5)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            best_rank = r
            break
    
    return best_result
# EVOLVE-BLOCK-END
