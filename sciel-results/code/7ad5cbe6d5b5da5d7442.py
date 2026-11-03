import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill
    
    nm, mp, pn = n * m, m * p, p * n  # 6, 30, 20
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Precompute Kronecker structures for fast ALS
    # For updating U: B_U[a, r] uses V and W
    # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Flatten: T[a].flat = sum_r U[a,r] * (V[:,r] * W[:,r]).flat
    # So B_U[:, r] = kron(V[:,r], W[:,r])  (shape mp*pn)
    # Similarly for V: B_V[:, r] = kron(U[:,r], W[:,r])  (shape nm*pn)
    # And for W: B_W[:, r] = kron(U[:,r], V[:,r])  (shape nm*mp)
    
    def kron_vw(V, W, r):
        """Build B_U matrix: (mp*pn) x r where B[:,r] = kron(V[:,r], W[:,r])"""
        B = np.zeros((mp * pn, r))
        for idx in range(r):
            B[:, idx] = np.kron(V[:, idx], W[:, idx])
        return B
    
    def kron_uw(U, W, r):
        """Build B_V matrix: (nm*pn) x r where B[:,r] = kron(U[:,r], W[:,r])"""
        B = np.zeros((nm * pn, r))
        for idx in range(r):
            B[:, idx] = np.kron(U[:, idx], W[:, idx])
        return B
    
    def kron_uv(U, V, r):
        """Build B_W matrix: (nm*mp) x r where B[:,r] = kron(U[:,r], V[:,r])"""
        B = np.zeros((nm * mp, r))
        for idx in range(r):
            B[:, idx] = np.kron(U[:, idx], V[:, idx])
        return B
    
    # Build a WFTA-based solution: treat m=3 as polynomial over Z_3
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # This is equivalent to: for each (i,k), the "polynomial" in j must satisfy
    # sum over j of A[i,j] * B[j,k] = C[i,k]
    # Using DFT over Z_3: let omega = exp(2*pi*i/3)
    # A_hat[i, r] = sum_j A[i,j] * omega^(j*r) for r in {0,1,2}
    # B_hat[k, r] = sum_j B[j,k] * omega^(-j*r) for r in {0,1,2}
    # Then C[i,k] = (1/3) * sum_r A_hat[i,r] * B_hat[k,r]
    # So we need 3 * p * n = 3 * 10 * 2 = 60 rank-1 terms with complex coefficients
    # But we can use real/integer representations
    # Actually, let's think differently: use the fact that multiplication in Z_3[x]/(x^3-1)
    # has rank equal to the number of distinct roots, which is 3 (over C)
    # So total rank = 3 * n * p = 3 * 2 * 10 = 60... that's worse.
    # Let me reconsider: the current Strassen approach gives 55.
    # Let me try a different decomposition using Winograd for the 3x3 part.
    
    # Alternative: use the fact that C = A * B where A is 2x3, B is 3x10
    # We can view this as: for each k, C[:,k] = A * B[:,k]
    # A is 2x3, so A * x for x in R^3 can be computed with... well, it's just 6 mults normally.
    # But we can use precomputation: let a0 = A[0,:], a1 = A[1,:]
    # C[0,k] = a0[0]*B[0,k] + a0[1]*B[1,k] + a0[2]*B[2,k]
    # C[1,k] = a1[0]*B[0,k] + a1[1]*B[1,k] + a1[2]*B[2,k]
    # This is 6 multiplications per k, total 6*10 = 60.
    
    # Let me try to use a block approach: group the 10 k's into blocks of 2
    # and use Strassen-like tricks on the 3-dimension.
    
    # Actually, let me just keep the existing rank-55 and try to optimize it.
    def build_rank55():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        c = 0
        # Part A: j=0 (first column of A times first row of B)
        for i in range(n):
            for k in range(p):
                U[i*m+0, c] = 1
                V[0*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
        
        # Part B: A1*B1 + A2*B2 using Strassen for 2x2 blocks
        for b in range(5):
            k0 = 2*b
            k1 = 2*b+1
            
            u11, u12 = 0*m+1, 0*m+2
            u21, u22 = 1*m+1, 1*m+2
            
            v11, v12 = 1*p+k0, 1*p+k1
            v21, v22 = 2*p+k0, 2*p+k1
            
            w00, w01 = k0*n+0, k1*n+0
            w10, w11 = k0*n+1, k1*n+1
            
            # M1 = (a11+a22)(b11+b22)
            U[u11,c]+=1; U[u22,c]+=1; V[v11,c]+=1; V[v22,c]+=1; W[w00,c]+=1; W[w11,c]+=1; c+=1
            # M2 = (a21+a22)b11
            U[u21,c]+=1; U[u22,c]+=1; V[v11,c]+=1; W[w10,c]+=1; W[w11,c]+=-1; c+=1
            # M3 = a11(b12-b22)
            U[u11,c]+=1; V[v12,c]+=1; V[v22,c]+=-1; W[w01,c]+=1; W[w11,c]+=1; c+=1
            # M4 = a22(b21-b11)
            U[u22,c]+=1; V[v21,c]+=1; V[v11,c]+=-1; W[w00,c]+=1; W[w10,c]+=1; c+=1
            # M5 = (a11+a12)b22
            U[u11,c]+=1; U[u12,c]+=1; V[v22,c]+=1; W[w00,c]+=-1; W[w01,c]+=1; c+=1
            # M6 = (a21-a11)(b11+b12)
            U[u21,c]+=1; U[u11,c]+=-1; V[v11,c]+=1; V[v12,c]+=1; W[w11,c]+=1; c+=1
            # M7 = (a12-a22)(b21+b22)
            U[u12,c]+=1; U[u22,c]+=-1; V[v21,c]+=1; V[v22,c]+=1; W[w00,c]+=1; c+=1
        
        return U, V, W, rank
    
    # Verify a decomposition - check ALL entries
    def verify(U, V, W, tol=1e-6):
        r = U.shape[1]
        # Compute full tensor product
        # T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Efficient: for each r, outer product
        T_recon = np.zeros((nm, mp, pn))
        for r_idx in range(r):
            T_recon += np.einsum('a,b,c->abc', U[:, r_idx], V[:, r_idx], W[:, r_idx])
        
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if abs(T_recon[i*m+j, j*p+k, k*n+i] - 1.0) > tol:
                        return False
        # Check zero entries
        zero_mask = (T == 0)
        max_err = np.max(np.abs(T_recon[zero_mask]))
        return max_err < tol
    
    # Fast vectorized ALS with improved convergence
    def als_optimize(U_init, V_init, W_init, target_rank, max_iters=300, tol=1e-10):
        U = U_init[:, :target_rank].copy()
        V = V_init[:, :target_rank].copy()
        W = W_init[:, :target_rank].copy()
        
        prev_err = float('inf')
        stall_count = 0
        
        for iteration in range(max_iters):
            if time.time() > deadline - 8:
                break
            
            # Update U: T[a] = B_U @ U[a,:] for all a simultaneously
            B_U = kron_vw(V, W, target_rank)
            T_flat = T.reshape(nm, mp * pn)
            try:
                U_T, _, _, _ = np.linalg.lstsq(B_U.T, T_flat.T, rcond=None)
                U_new = U_T.T
                # Check if update is stable
                if np.all(np.isfinite(U_new)):
                    U = U_new
            except:
                pass
            
            # Update V
            B_V = kron_uw(U, W, target_rank)
            T_flat_v = np.zeros((nm * pn, mp))
            for b in range(mp):
                T_flat_v[:, b] = T[:, b, :].flatten()
            try:
                V_T, _, _, _ = np.linalg.lstsq(B_V.T, T_flat_v, rcond=None)
                V_new = V_T.T
                if np.all(np.isfinite(V_new)):
                    V = V_new
            except:
                pass
            
            # Update W
            B_W = kron_uv(U, V, target_rank)
            T_flat_w = np.zeros((nm * mp, pn))
            for c in range(pn):
                T_flat_w[:, c] = T[:, :, c].flatten()
            try:
                W_T, _, _, _ = np.linalg.lstsq(B_W.T, T_flat_w, rcond=None)
                W_new = W_T.T
                if np.all(np.isfinite(W_new)):
                    W = W_new
            except:
                pass
            
            # Compute error on target entries
            if iteration % 3 == 0 or iteration == max_iters - 1:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            err += (val - 1.0)**2
                if err < tol:
                    break
                if abs(prev_err - err) < 1e-12:
                    stall_count += 1
                    if stall_count > 10:
                        break
                else:
                    stall_count = 0
                prev_err = err
        
        return U, V, W
    
    # Exact rational polish: round to nearest 0.5 and verify
    def polish_exact(U, V, W):
        """Round to nearest half-integer and check exactness"""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        if verify(U_r, V_r, W_r, tol=1e-9):
            return U_r, V_r, W_r
        # Try rounding to nearest integer
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if verify(U_i, V_i, W_i, tol=1e-9):
            return U_i, V_i, W_i
        return None
    
    # Get initial rank-55 solution
    U55, V55, W55, rank55 = build_rank55()
    if not verify(U55, V55, W55, tol=1e-9):
        # Fallback: return schoolbook rank 60
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W, best_rank = U55, V55, W55, rank55
    
    # Try to find lower-rank solutions using fast ALS
    for target_rank in range(rank55 - 1, 40, -1):
        if time.time() > deadline - 15:
            break
        
        # Multiple random restarts with diverse initialization strategies
        for restart in range(8):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(12345 + restart * 13 + target_rank * 31)
            
            if restart == 0:
                # Truncate the rank-55 solution and add small noise
                U_init = U55[:, :target_rank].copy() + 0.05 * np.random.randn(nm, target_rank)
                V_init = V55[:, :target_rank].copy() + 0.05 * np.random.randn(mp, target_rank)
                W_init = W55[:, :target_rank].copy() + 0.05 * np.random.randn(pn, target_rank)
            elif restart == 1:
                # Random initialization
                U_init = np.random.randn(nm, target_rank) * 0.5
                V_init = np.random.randn(mp, target_rank) * 0.5
                W_init = np.random.randn(pn, target_rank) * 0.5
            elif restart == 2:
                # Use best found so far
                U_init = best_U[:, :target_rank].copy() + 0.02 * np.random.randn(nm, target_rank)
                V_init = best_V[:, :target_rank].copy() + 0.02 * np.random.randn(mp, target_rank)
                W_init = best_W[:, :target_rank].copy() + 0.02 * np.random.randn(pn, target_rank)
            elif restart == 3:
                # Different random
                U_init = np.random.randn(nm, target_rank)
                V_init = np.random.randn(mp, target_rank)
                W_init = np.random.randn(pn, target_rank)
            elif restart == 4:
                # Sparse initialization: start with a subset of rank-55 terms
                idx = np.random.choice(rank55, target_rank, replace=False)
                U_init = U55[:, idx].copy() + 0.1 * np.random.randn(nm, target_rank)
                V_init = V55[:, idx].copy() + 0.1 * np.random.randn(mp, target_rank)
                W_init = W55[:, idx].copy() + 0.1 * np.random.randn(pn, target_rank)
            elif restart == 5:
                # Structured initialization based on block patterns
                U_init = np.zeros((nm, target_rank))
                V_init = np.zeros((mp, target_rank))
                W_init = np.zeros((pn, target_rank))
                # Initialize with a subset of the schoolbook terms
                for t in range(target_rank):
                    i = t % n
                    j = (t // n) % m
                    k = (t // (n*m)) % p
                    U_init[i*m+j, t] = 1.0
                    V_init[j*p+k, t] = 1.0
                    W_init[k*n+i, t] = 1.0
                U_init += 0.1 * np.random.randn(nm, target_rank)
                V_init += 0.1 * np.random.randn(mp, target_rank)
                W_init += 0.1 * np.random.randn(pn, target_rank)
            elif restart == 6:
                # Random permutation of rank-55 terms
                perm = np.random.permutation(rank55)
                idx = perm[:target_rank]
                U_init = U55[:, idx].copy() + 0.08 * np.random.randn(nm, target_rank)
                V_init = V55[:, idx].copy() + 0.08 * np.random.randn(mp, target_rank)
                W_init = W55[:, idx].copy() + 0.08 * np.random.randn(pn, target_rank)
            else:
                # Very small noise perturbation of truncated solution
                U_init = U55[:, :target_rank].copy() + 0.01 * np.random.randn(nm, target_rank)
                V_init = V55[:, :target_rank].copy() + 0.01 * np.random.randn(mp, target_rank)
                W_init = W55[:, :target_rank].copy() + 0.01 * np.random.randn(pn, target_rank)
            
            U_opt, V_opt, W_opt = als_optimize(U_init, V_init, W_init, target_rank, max_iters=300)
            
            # Try exact polish
            polished = polish_exact(U_opt, V_opt, W_opt)
            if polished is not None:
                best_U, best_V, best_W, best_rank = polished
                break
            
            # Also check with looser tolerance
            if verify(U_opt, V_opt, W_opt, tol=1e-6):
                # Try to polish
                polished = polish_exact(U_opt, V_opt, W_opt)
                if polished is not None:
                    best_U, best_V, best_W, best_rank = polished
                    break
                else:
                    # Use the floating point solution
                    best_U, best_V, best_W, best_rank = U_opt, V_opt, W_opt, target_rank
                    break
        
        if best_rank < rank55:
            continue
    
    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-6):
        best_U, best_V, best_W, best_rank = U55, V55, W55, rank55
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
