"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _build_tensor(n, m, p):
    """Build the matmul tensor T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'.
    T[a,b,c] = 1 if j=j' and i=i' (consistent indices), else 0."""
    # Reshape into 4D: T[i, j, k, ...] but really it's sparse
    # Actually T has shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] equals the matmul tensor."""
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Compute tensor product sum
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    # Check against known tensor
    T_exact = _build_tensor(n, m, p)
    return np.allclose(T, T_exact, atol=tol)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank using iterative refinement
    # Strategy: start from a slightly lower rank and try to fit
    
    # Lower bound: for matmul (n,m,p), rank >= max(n*p, m*p, n*m) is not quite right
    # A known lower bound is rank >= n*p if m is large enough... 
    # Actually, the rank of M_{n,m,p} is at least max(n*p, m*p, n*m) / something
    # Let's just try various ranks
    
    # Try ranks from a reasonable lower bound up
    # Known: rank of M_{2,2,2} = 7. For M_{2,4,8}, let's estimate.
    # The tensor has 8*32*16 = 4096 entries, but only 64 are nonzero.
    
    # Let me try a structured approach:
    # Group by the middle index j. For each j, we have a (n x p) sub-tensor
    # that's rank-n (since C[i,k] = sum_j A[i,j]*B[j,k], for fixed j it's outer product of A[:,j] and B[j,:])
    
    # Actually for fixed j, the contribution is: for all i,k, A[i,j]*B[j,k]*C[k,i]
    # This is the outer product of (A[:,j] ⊗ B[j,:]) reshaped... 
    # For fixed j: u = e_j in middle... no.
    
    # Let me think differently. For each j, define:
    #   A_j = A[:,j] (vector in R^n)
    #   B_j = B[j,:] (vector in R^p)
    # Then C = sum_j A_j * B_j (outer product summed)
    # The tensor for fixed j is: sum_{i,k} A_j[i] * B_j[k] * (E_{i,k})
    # where E_{i,k} is the matrix unit. In our indexing, the C-index for (i,k) is k*n+i.
    
    # So for fixed j, the tensor slice is: u_j ⊗ v_j ⊗ w_j where
    #   u_j has 1 at position i*m+j for each i... no, u_j is a vector in R^{n*m}
    #   u_j[i*m+j] = A_j[i]... but we want integer coefficients.
    
    # Hmm, this is getting complicated. Let me just try numerical optimization.
    
    def try_rank(R, max_iters=500, tol=1e-6):
        """Try to find a rank-R decomposition."""
        np.random.seed(42)
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        for it in range(max_iters):
            if time.time() > deadline - 15:
                return None
            
            # Update U: for each a, minimize ||T[a,:,:] - sum_r U[a,r]*V[:,r]⊗W[:,r]||
            # This is: T[a,:,:] = sum_r U[a,r] * (V[:,r] ⊗ W[:,r])
            # So for fixed a, U[a,:] is the least squares solution
            # Reshape T[a,:,:] to (m*p, p*n) and project onto span{V[:,r]⊗W[:,r]}
            
            # For each a, compute the residual and update U[a,:]
            for a in range(n*m):
                # T[a,:,:] is (m*p, p*n)
                Ta = T[a, :, :]
                # Build design matrix: each column r is V[:,r] ⊗ W[:,r] reshaped to (m*p*p*n,)
                # But this is huge. Instead, use the fact that:
                # <Ta, V[:,r]⊗W[:,r]> = sum_{b,c} Ta[b,c] * V[b,r] * W[c,r]
                # = sum_b V[b,r] * sum_c Ta[b,c]*W[c,r]
                # = V[:,r]^T @ (Ta @ W[:,r])... wait
                # sum_{b,c} Ta[b,c] * V[b,r] * W[c,r] = (V[:,r] * (Ta @ W[:,r]))^T @ ones... 
                # Actually: = V[:,r].T @ (Ta @ W[:,r])... no.
                # = sum_b V[b,r] * (Ta @ W[:,r])[b] = V[:,r].T @ (Ta @ W[:,r])
                
                # So the normal equations for U[a,:]:
                # M U[a,:] = g where M[r,s] = <V[:,r]⊗W[:,s], V[:,r]⊗W[:,s]>... 
                # M[r,s] = sum_{b,c} V[b,r]*W[c,r]*V[b,s]*W[c,s] = (sum_b V[b,r]*V[b,s]) * (sum_c W[c,r]*W[c,s])
                # = (V.T @ V)[r,s] * (W.T @ W)[r,s]... no, that's elementwise.
                # M[r,s] = (V.T @ V)[r,s] * (W.T @ W)[r,s] -- NO!
                # M[r,s] = sum_{b,c} V[b,r]*W[c,r]*V[b,s]*W[c,s]
                # = (sum_b V[b,r]*V[b,s]) * (sum_c W[c,r]*W[c,s])
                # = (V.T V)[r,s] * (W.T W)[r,s] -- this is WRONG. It's the elementwise product of the Gram matrices.
                
                # Actually yes! sum_{b,c} f(b,r)*g(c,r)*f(b,s)*g(c,s) = (sum_b f(b,r)f(b,s))(sum_c g(c,r)g(c,s))
                # So M = (V.T@V) * (W.T@W) elementwise (Hadamard product)
                
                Vg = V.T @ V  # (R,R)
                Wg = W.T @ W  # (R,R)
                M = Vg * Wg  # Hadamard product
                
                # g[r] = <T[a,:,:], V[:,r]⊗W[:,r]> = sum_{b,c} T[a,b,c]*V[b,r]*W[c,r]
                # = sum_b V[b,r] * sum_c T[a,b,c]*W[c,r]
                # = V[:,r].T @ (T[a] @ W[:,r])... 
                # = (T[a] @ (V[:,r]*W[:,r]))... no
                # Let me compute: for each r, g_r = sum_{b,c} Ta[b,c] * V[b,r] * W[c,r]
                # = sum_b V[b,r] * (Ta @ W[:,r])[b]
                # So g = (Ta @ (V*W elementwise?))... 
                # Actually: g_r = sum_b V[b,r] * sum_c Ta[b,c]*W[c,r]
                # Let M1 = Ta @ W  # (m*p, R), M1[b,r] = sum_c Ta[b,c]*W[c,r]
                # Then g_r = sum_b V[b,r] * M1[b,r] = elementwise sum of V and M1... 
                # g = np.einsum('br,br->r', V, Ta @ W)
                
                M1 = Ta @ W  # (m*p, R)
                g = np.einsum('br,br->r', V, M1)
                
                # Solve M @ U[a,:] = g
                try:
                    U[a, :] = np.linalg.solve(M + 1e-8*np.eye(R), g)
                except:
                    U[a, :] = np.linalg.lstsq(M, g, rcond=None)[0]
            
            # Similarly update V and W
            for b in range(m*p):
                Tb = T[:, b, :]  # (n*m, p*n)
                # For fixed b: sum_r U[:,r] * V[b,r] * W[:,r] = Tb[:,r]... 
                # Tb[:,c] = sum_r U[:,r] * V[b,r] * W[c,r] for each c
                # So Tb[:, :] = sum_r U[:,r] * W[:,r] * V[b,r]
                # For each c: Tb[:,c] = (U * W[:,c] elementwise) @ V[b,:]... 
                # Let's do: for fixed b, minimize over V[b,:]:
                # Residual: for each c, Tb[:,c] - sum_r U[:,r]*W[c,r]*V[b,r]
                # = Tb[:,c] - (U * W[c] elementwise) @ V[b,:]
                # So: stack all c: design matrix has columns (U[:,r]*W[:,r] for r=0..R-1) but each is (n*m,)
                # And target is Tb reshaped to (n*m, p*n)
                
                # M2[r,s] = sum_c U[:,r]*W[c,r]*U[:,s]*W[c,s] = (U.T@U)[r,s] * (sum_c W[c,r]*W[c,s])... 
                # No: sum_c U[i,r]*W[c,r]*U[i,s]*W[c,s] summed over i and c
                # = sum_i U[i,r]*U[i,s] * sum_c W[c,r]*W[c,s]
                # = (U.T@U)[r,s] * (W.T@W)[r,s]
                
                Ug = U.T @ U
                Wg2 = W.T @ W
                M2 = Ug * Wg2  # Hadamard
                
                # g2[r] = sum_{i,c} Tb[i,c] * U[i,r] * W[c,r]
                # = sum_i U[i,r] * sum_c Tb[i,c]*W[c,r]
                # = sum_i U[i,r] * (Tb @ W[:,r])[i]
                # M2r = Tb @ W  # (n*m, R), M2r[i,r] = sum_c Tb[i,c]*W[c,r]
                # g2[r] = sum_i U[i,r] * M2r[i,r] = einsum('ir,ir->r', U, M2r)
                
                M2r = Tb @ W  # (n*m, R)
                g2 = np.einsum('ir,ir->r', U, M2r)
                
                try:
                    V[b, :] = np.linalg.solve(M2 + 1e-8*np.eye(R), g2)
                except:
                    V[b, :] = np.linalg.lstsq(M2, g2, rcond=None)[0]
            
            for c in range(p*n):
                Tc = T[:, :, c]  # (n*m, m*p)
                # Similar to V update but with roles of U and V swapped, and Tc instead of Tb
                Ug2 = U.T @ U
                Vg2 = V.T @ V
                M3 = Ug2 * Vg2  # Hadamard
                
                # g3[r] = sum_{i,j} Tc[i,j] * U[i,r] * V[j,r]
                # = sum_i U[i,r] * sum_j Tc[i,j]*V[j,r]
                # M3r = Tc @ V  # (n*m, R), M3r[i,r] = sum_j Tc[i,j]*V[j,r]
                # g3[r] = einsum('ir,ir->r', U, M3r)
                
                M3r = Tc @ V  # (n*m, R)
                g3 = np.einsum('ir,ir->r', U, M3r)
                
                try:
                    W[c, :] = np.linalg.solve(M3 + 1e-8*np.eye(R), g3)
                except:
                    W[c, :] = np.linalg.lstsq(M3, g3, rcond=None)[0]
            
            # Check convergence
            if it % 50 == 0:
                T_recon = np.zeros((n*m, m*p, p*n))
                for r in range(R):
                    T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                err = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
                if err < 1e-8:
                    break
        
        # Final check
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(R):
            T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        err = np.linalg.norm(T_recon - T) / np.linalg.norm(T)
        
        if err < 1e-6:
            # Try to round to integers/half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            T_check = np.zeros((n*m, m*p, p*n))
            for r in range(R):
                T_check += np.einsum('a,b,c->abc', U_round[:,r], V_round[:,r], W_round[:,r])
            err_round = np.linalg.norm(T_check - T) / np.linalg.norm(T)
            
            if err_round < 1e-6:
                return U_round, V_round, W_round
            else:
                # Try rounding to integers only
                U_round = np.round(U)
                V_round = np.round(V)
                W_round = np.round(W)
                T_check2 = np.zeros((n*m, m*p, p*n))
                for r in range(R):
                    T_check2 += np.einsum('a,b,c->abc', U_round[:,r], V_round[:,r], W_round[:,r])
                err_int = np.linalg.norm(T_check2 - T) / np.linalg.norm(T)
                if err_int < 1e-6:
                    return U_round, V_round, W_round
                else:
                    return None
        return None
    
    # Try ranks from 48 up (below 64)
    for R in range(48, 64):
        if time.time() > deadline - 10:
            break
        result = try_rank(R)
        if result is not None:
            U_res, V_res, W_res = result
            if _verify(U_res, V_res, W_res, n, m, p):
                # Convert to lists
                best_rank = R
                best_U = U_res
                best_V = V_res
                best_W = W_res
                break  # Found lowest rank in this range
    
    # If we found something better, return it; else return schoolbook
    if best_rank < rank_sb:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    else:
        U_out = best_U.tolist()
        V_out = best_V.tolist()
        W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
