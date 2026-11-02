"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    schoolbook_rank = n * m * p  # 32
    deadline = time.time() + 140  # safety margin before 150s kill

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-8):
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    def strassen_2x2_decomp():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        U[0,0]=1.0; U[3,0]=1.0; V[0,0]=1.0; V[3,0]=1.0; W[0,0]=1.0; W[3,0]=1.0
        U[2,1]=1.0; U[3,1]=1.0; V[0,1]=1.0; W[1,1]=1.0; W[3,1]=-1.0
        U[0,2]=1.0; V[1,2]=1.0; V[3,2]=-1.0; W[2,2]=1.0; W[3,2]=1.0
        U[3,3]=1.0; V[2,3]=1.0; V[0,3]=-1.0; W[0,3]=1.0; W[1,3]=1.0
        U[0,4]=1.0; U[1,4]=1.0; V[3,4]=1.0; W[0,4]=-1.0; W[2,4]=1.0
        U[2,5]=1.0; U[0,5]=-1.0; V[0,5]=1.0; V[1,5]=1.0; W[3,5]=1.0
        U[1,6]=1.0; U[3,6]=-1.0; V[2,6]=1.0; V[3,6]=1.0; W[0,6]=1.0
        return U, V, W

    def build_strassen28():
        U_s, V_s, W_s = strassen_2x2_decomp()
        full_rank = 28
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))
        blocks = [
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+lk,          lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+lk,      lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+(lk+2),      lambda lk,li: (lk+2)*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+(lk+2),  lambda lk,li: (lk+2)*2+li),
        ]
        for bi,(ga,gb,gc) in enumerate(blocks):
            for t in range(7):
                col = bi*7+t
                for li in range(2):
                    for lj in range(2):
                        U_full[ga(li,lj), col] = U_s[li*2+lj, t]
                for lj in range(2):
                    for lk in range(2):
                        V_full[gb(lj,lk), col] = V_s[lj*2+lk, t]
                for lk in range(2):
                    for li in range(2):
                        W_full[gc(lk,li), col] = W_s[lk*2+li, t]
        return U_full, V_full, W_full

    U28, V28, W28 = build_strassen28()
    if not verify(U28, V28, W28, tol=1e-8):
        U = np.zeros((n*m, schoolbook_rank)); V = np.zeros((m*p, schoolbook_rank)); W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col]=1.0; V[j*p+k, col]=1.0; W[k*n+i, col]=1.0; col+=1
        return to_lists(U, V, W, schoolbook_rank)

    # Tensor slicing: fix A to standard basis vectors e_a (a=0..7)
    # For each a, the output is B * W_a where W_a is a 4x4 matrix
    # The tensor rank r requires that all W_a be in the span of {V[:,t] outer W[:,t]^T}
    # So the minimum rank is at least the dimension of span{W_0, ..., W_7}
    
    # Extract the 8 output matrices W_a (each 4x4)
    # T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    # For fixed a=i0*m+j0: T[a,b,c]=1 iff j=j0, b=j0*p+k, c=k*n+i0
    # So W_a is a 4x4 matrix where W_a[k, i0] = 1 if b = j0*p+k
    
    W_slices = []
    for a in range(n * m):
        i0 = a // m
        j0 = a % m
        # W_slice is 4x4: rows indexed by k (0..3), cols by i (0..1)
        # But we need a 4x4 matrix in the output space
        # Actually, the output C = A*B is 2x4, so C[i,k] for i=0,1; k=0,1,2,3
        # For fixed A = e_a (1 at position i0,j0), C = e_a * B
        # C[i,k] = sum_j A[i,j] * B[j,k] = A[i0,j0] * B[j0,k] = B[j0,k] if i=i0, else 0
        # So the output is a 2x4 matrix with row i0 = B[j0,:] and other rows = 0
        # In the flattened output (p*n = 8), the nonzero entries are at positions k*n+i0 for k=0..3
        # The "matrix" in output space is 4x4 if we think of it as mapping B to C
        # Let's think of it differently: the linear map from B (flattened 16-dim) to C (flattened 8-dim)
        # For fixed A=e_a, the map is: B -> C where C[k*n+i0] = B[j0*p+k] for k=0..3
        # This is a 8x16 matrix (or we can view it as a set of 4 linear functionals on B)
        
        # The tensor T[a, b, c] gives the coefficient. For fixed a, the slice T[a, :, :] is an (m*p) x (p*n) matrix
        # This matrix has rank at most 1 (since it's outer product of two vectors in the tensor decomposition)
        # Actually, for the matrix multiplication tensor, T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
        # For fixed a=i0*m+j0: T[a, b, c] = 1 iff b=j0*p+k AND c=k*n+i0 for some k
        # So the slice is: T[a, j0*p+k, k*n+i0] = 1 for k=0..3
        # This is a sum of 4 rank-1 matrices: sum_k e_{j0*p+k} e_{k*n+i0}^T
        # The rank of this 16x8 matrix is 4 (since the 4 vectors e_{j0*p+k} are independent and the 4 vectors e_{k*n+i0} are independent)
        
        slice_mat = np.zeros((m*p, p*n))
        for k in range(p):
            b = j0 * p + k
            c = k * n + i0
            slice_mat[b, c] = 1.0
        W_slices.append(slice_mat)
    
    # The tensor rank r must be at least the maximum over a of the rank of T[a,:,:]
    # But more importantly, all slices must be in the span of the r rank-1 matrices from the decomposition
    # The dimension of the span of all 8 slices gives a lower bound on r
    
    # Stack all slices and compute the dimension of their span
    # Each slice is a 16x8 matrix, so we can flatten to 128-dim vectors
    all_slices_flat = np.array([s.flatten() for s in W_slices])
    
    # Compute the rank of the set of 8 slices
    # Use SVD to find the rank
    U_svd, svals, Vt = np.linalg.svd(all_slices_flat, full_matrices=False)
    slice_rank = np.sum(svals > 1e-10)
    
    # The tensor rank must be at least slice_rank
    # For this problem, let's see what slice_rank is
    # Each slice has rank 4, and they may or may not be independent
    
    # Now, let's try to find a decomposition of rank = slice_rank or close to it
    # The key insight: if we can find r rank-1 matrices whose span contains all 8 slices,
    # then we can construct a rank-r tensor decomposition
    
    # Let's try rank = slice_rank first, then try smaller if possible
    # Actually, the tensor rank is at least the slice rank (max rank of any slice)
    # And at least the dimension of the span of all slices
    
    # For a better approach: use the fact that for matrix multiplication,
    # the tensor rank equals the minimum number of scalar multiplications
    # Let's try to find a decomposition using the structure
    
    # Build the Strassen-28 as baseline
    # Then try to reduce rank by finding linear dependencies
    
    # Try rank 24, 25, 26, 27 with randomized optimization
    def try_rank_optimized(r, seed):
        rng = np.random.RandomState(seed)
        
        # Initialize from Strassen-28 with some columns removed and noise
        if r <= 28:
            U = U28[:, :min(r, 28)].copy()
            V = V28[:, :min(r, 28)].copy()
            W = W28[:, :min(r, 28)].copy()
            if r < 28:
                # Add random columns to fill up to r
                extra = r - min(r, 28)
                if extra > 0:
                    U_extra = rng.randn(n*m, extra) * 0.5
                    V_extra = rng.randn(m*p, extra) * 0.5
                    W_extra = rng.randn(p*n, extra) * 0.5
                    U = np.hstack([U, U_extra])
                    V = np.hstack([V, V_extra])
                    W = np.hstack([W, W_extra])
        else:
            U = rng.randn(n*m, r) * 0.5
            V = rng.randn(m*p, r) * 0.5
            W = rng.randn(p*n, r) * 0.5
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective_and_grad(x):
            U = x[:n*m*r].reshape(n*m, r)
            V = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
            W = x[n*m*r + m*p*r:].reshape(p*n, r)
            
            recon = np.einsum('at,bt,ct->abc', U, V, W)
            F = recon - T
            loss = float(np.sum(F * F))
            
            gU = np.einsum('abc,bt,ct->at', F, V, W)
            gV = np.einsum('abc,at,ct->bt', F, U, W)
            gW = np.einsum('abc,at,bt->ct', F, U, V)
            
            grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            return loss, grad

        try:
            from scipy.optimize import minimize
            res = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B', 
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14, 'maxfun': 300})
            
            if res.fun < 1e-10:
                x_opt = res.x
                U_opt = x_opt[:n*m*r].reshape(n*m, r)
                V_opt = x_opt[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_opt = x_opt[n*m*r + m*p*r:].reshape(p*n, r)
                
                # Try snapping to integers first, then half-integers
                U_int = np.round(U_opt)
                V_int = np.round(V_opt)
                W_int = np.round(W_opt)
                if verify(U_int, V_int, W_int, tol=1e-8):
                    return U_int, V_int, W_int
                
                U_half = np.round(U_opt * 2) / 2
                V_half = np.round(V_opt * 2) / 2
                W_half = np.round(W_opt * 2) / 2
                if verify(U_half, V_half, W_half, tol=1e-8):
                    return U_half, V_half, W_half
        except Exception:
            pass
        
        return None

    # Try ranks from 24 to 27
    for r in [24, 25, 26, 27]:
        if time.time() > deadline - 30:
            break
        seeds = list(range(0, 20, 3))
        for seed in seeds:
            if time.time() > deadline - 15:
                break
            res = try_rank_optimized(r, seed)
            if res is not None:
                Uf, Vf, Wf = res
                return to_lists(Uf, Vf, Wf, r)

    # Fallback: verified rank-28 Strassen
    return to_lists(U28, V28, W28, 28)
# EVOLVE-BLOCK-END
