import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # n=2, m=2, p=12

    # The tensor for 2x2 x 2x12 matrix multiplication has dimensions:
    # U: n*m = 4, V: m*p = 24, W: p*n = 24
    # We want rank r such that T = sum_i U_i * V_i * W_i (outer products)
    # T[i,j,k] = U[i,ci] * V[j,ci] * W[k,ci] where ci indexes the rank factor
    
    # The matrix multiplication tensor T for C = A * B:
    # A is 2x2 (indices: row 0..1, col 0..1) -> flattened to n*m=4
    # B is 2x12 (indices: row 0..1, col 0..11) -> flattened to m*p=24
    # C is 2x12 (indices: row 0..1, col 0..11) -> flattened to p*n=24
    # T[i,j,k] = 1 if i corresponds to A[row_a, col_a], j to B[row_b, col_b], k to C[row_c, col_c]
    # and col_a == row_b, row_c == row_a, col_c == col_b, else 0.
    
    # Let's index:
    # i = row_a * m + col_a = row_a * 2 + col_a, so row_a = i//2, col_a = i%2
    # j = row_b * p + col_b = row_b * 12 + col_b, so row_b = j//12, col_b = j%12
    # k = row_c * p + col_c = row_c * 12 + col_c, so row_c = k//12, col_c = k%12
    
    # T[i,j,k] = 1 if (i%2 == j//12) and (i//2 == k//12) and (j%12 == k%12)
    
    # Build the target tensor (sparse, but we'll work with it)
    # Actually, for the decomposition we just need to match the tensor entries.
    # The tensor has exactly 1s at positions where the above conditions hold.
    # Number of 1s: for each of the 4 values of i, 12 values of col_b (=col_c), 
    # but j is determined by row_b=col_a and col_b, and k by row_c=row_a and col_c=col_b.
    # So for each i (4 choices) and each col_b (12 choices), there's exactly one (j,k).
    # Total: 4 * 12 = 48 ones.
    
    # We'll use a random initialization and try to minimize the tensor error.
    # But first, let's think about the theoretical minimum rank.
    # For 2x2 x 2x2, rank is 7 (Strassen).
    # For 2x2 x 2x12, we can group the 12 columns into 6 groups of 2, each needing rank 7 -> 42.
    # But maybe we can do better by sharing structure across groups.
    
    # Alternative: treat it as a single tensor and try to find a lower rank decomposition.
    # The tensor is very sparse (48 non-zeros out of 4*24*24 = 2304 entries).
    
    # Let's try a direct approach: parameterize U, V, W with rank r and minimize ||T - T_decomp||^2.
    # We'll try several ranks and use random restarts.
    
    # Build target tensor as a dense array (it's small enough)
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n*m):
        row_a = i // m
        col_a = i % m
        for col_b in range(p):
            row_b = col_a  # must equal col_a
            j = row_b * p + col_b
            row_c = row_a  # must equal row_a
            col_c = col_b  # must equal col_b
            k = row_c * p + col_c
            T[i, j, k] = 1.0
    
    def tensor_decompose(U, V, W):
        """Compute the decomposed tensor from U, V, W."""
        # U: (nm, r), V: (mp, r), W: (pn, r)
        r = U.shape[1]
        # T_decomp[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
        # This can be computed efficiently:
        # T_decomp = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.einsum('ic,jc,kc->ijk', U, V, W)
    
    def objective(params, r):
        """Objective function: sum of squared differences."""
        nm, mp, pn = n*m, m*p, p*n
        U = params[0:nm*r].reshape(nm, r)
        V = params[nm*r:nm*r+mp*r].reshape(mp, r)
        W = params[nm*r+mp*r:nm*r+mp*r+pn*r].reshape(pn, r)
        T_decomp = tensor_decompose(U, V, W)
        err = T - T_decomp
        return np.sum(err * err)
    
    def gradient(params, r):
        """Gradient of the objective (approximate or exact)."""
        nm, mp, pn = n*m, m*p, p*n
        U = params[0:nm*r].reshape(nm, r)
        V = params[nm*r:nm*r+mp*r].reshape(mp, r)
        W = params[nm*r+mp*r:nm*r+mp*r+pn*r].reshape(pn, r)
        T_decomp = tensor_decompose(U, V, W)
        diff = T_decomp - T  # gradient is 2 * diff for each factor
        
        # dT/dU[i,c] = sum_jk diff[i,j,k] * V[j,c] * W[k,c]
        # = sum_j diff[i,j,:] * (V[j,:] * W[k,:] summed over k) -- not quite
        # Actually: dT/dU[i,c] = sum_{j,k} diff[i,j,k] * V[j,c] * W[k,c]
        # Let's compute it as:
        # temp1[i,j] = sum_k diff[i,j,k] * W[k,c] for each c... 
        # Better: 
        # dU[i,c] = sum_jk diff[i,j,k] V[j,c] W[k,c]
        # = sum_j (sum_k diff[i,j,k] W[k,c]) V[j,c]
        # = sum_j (diff[i,j,:] * W[:,c]) V[j,c]
        
        # For efficiency, compute:
        # diff_U[i,c] = sum_jk diff[i,j,k] V[j,c] W[k,c]
        # We can do: 
        # tmp1 = np.einsum('ijk,ic->ijk', diff, W)  # no
        # Let's use:
        # For fixed c: diff_U[:,c] = sum_jk diff[:,:,?] ... 
        # Actually:
        # diff_U[i,c] = sum_j sum_k diff[i,j,k] V[j,c] W[k,c]
        # = sum_j V[j,c] * (sum_k diff[i,j,k] W[k,c])
        # = sum_j V[j,c] * (diff[i,j,:] @ W[:,c])
        
        # Let's compute:
        # DW = diff @ W  -> shape (nm, mp, r)? No.
        # diff is (nm, mp, pn), W is (pn, r)
        # diff @ W would be (nm, mp, r) if we contract the last axis:
        # tmp1 = np.einsum('ijk,kr->ijr', diff, W)  # (nm, mp, r)
        # Then diff_U = np.einsum('ijr,jc->ic', tmp1, V)  # but c and r are same
        
        tmp1 = np.einsum('ijk,kr->ijr', diff, W)  # (nm, mp, r)
        dU = np.einsum('ijr,jr->ir', tmp1, V)  # (nm, r)
        
        # Similarly:
        # dV[j,c] = sum_ik diff[i,j,k] U[i,c] W[k,c]
        tmp2 = np.einsum('ijk,ir->jk r'.replace(' ',''), diff, U)  # (mp, pn, r)
        tmp2 = np.einsum('ijk,ir->jkr', diff, U)
        dV = np.einsum('jkr,kr->jr', tmp2, W)  # (mp, r)
        
        # dW[k,c] = sum_ij diff[i,j,k] U[i,c] V[j,c]
        tmp3 = np.einsum('ijk,jr->ikr', diff, V)  # (nm, pn, r)
        dW = np.einsum('ikr,ir->kr', tmp3, U)  # (pn, r)
        
        # The actual gradient of sum(diff^2) is 2 * (dU, dV, dW) where diff = T_decomp - T
        # Since we defined diff = T_decomp - T, the gradient of sum(diff^2) wrt params is:
        # 2 * (dT_decomp/dU * diff, ...) but dT_decomp/dU[i,c] = V[j,c]W[k,c] summed...
        # Actually, the gradient we computed is d(sum diff^2)/dU = 2 * sum_jk diff[i,j,k] V[j,c] W[k,c]
        # Wait, let me re-derive:
        # f = sum_ijk (T_decomp[i,j,k] - T[i,j,k])^2
        # df/dU[i,c] = 2 * sum_ijk (T_decomp[i,j,k] - T[i,j,k]) * dT_decomp[i,j,k]/dU[i,c]
        # dT_decomp[i,j,k]/dU[i,c] = V[j,c] * W[k,c]
        # So df/dU[i,c] = 2 * sum_jk diff[i,j,k] * V[j,c] * W[k,c]
        # which is 2 * dU as computed.
        
        gU = 2.0 * dU
        gV = 2.0 * dV
        gW = 2.0 * dW
        
        grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        return grad
    
    best_result = None
    best_rank = 999
    
    # Try ranks from a lower bound upward
    # Lower bound: The tensor has 48 non-zero entries, but rank can be as low as ~7 or so?
    # Actually, for 2x2 x 2xp, the rank is at least 7 (from the 2x2x2 case embedded).
    # Let's try ranks 7, 8, 9, ..., up to maybe 30.
    
    np.random.seed(42)
    
    for r in range(7, 35):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            nm, mp, pn = n*m, m*p, p*n
            scale = 0.5
            params0 = np.random.randn(nm*r + mp*r + pn*r) * scale
            
            res = minimize(objective, params0, args=(r,), method='L-BFGS-B', 
                          jac=gradient, options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if res.success or (res.fun < 1e-6):
                U = res.x[0:nm*r].reshape(nm, r)
                V = res.x[nm*r:nm*r+mp*r].reshape(mp, r)
                W = res.x[nm*r+mp*r:nm*r+mp*r+pn*r].reshape(pn, r)
                
                # Verify
                T_decomp = tensor_decompose(U, V, W)
                max_err = np.max(np.abs(T - T_decomp))
                
                if max_err < 1e-4:
                    if r < best_rank:
                        best_rank = r
                        best_result = (U, V, W)
                        print(f"Found decomposition with rank {r}, max_err={max_err}")
                        break  # No need to try more restarts for this rank
        else:
            continue
        break
    
    if best_result is None:
        # Fallback: use the Strassen-based rank-42 decomposition
        rank = 42
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        
        num_blocks = p // 2  # 6
        for t in range(num_blocks):
            if time.time() > deadline:
                break
            base = t * 7
            k0 = 2 * t
            k1 = 2 * t + 1
            
            c = base + 0
            U[0, c] += 1
            U[3, c] += 1
            V[0*p + k0, c] += 1
            V[1*p + k1, c] += 1
            W[k0*n + 0, c] += 1
            W[k1*n + 1, c] += 1
            
            c = base + 1
            U[2, c] += 1
            U[3, c] += 1
            V[0*p + k0, c] += 1
            W[k0*n + 1, c] += 1
            W[k1*n + 1, c] += -1
            
            c = base + 2
            U[0, c] += 1
            V[0*p + k1, c] += 1
            V[1*p + k1, c] += -1
            W[k1*n + 0, c] += 1
            W[k1*n + 1, c] += 1
            
            c = base + 3
            U[3, c] += 1
            V[1*p + k0, c] += 1
            V[0*p + k0, c] += -1
            W[k0*n + 0, c] += 1
            W[k0*n + 1, c] += 1
            
            c = base + 4
            U[0, c] += 1
            U[1, c] += 1
            V[1*p + k1, c] += 1
            W[k0*n + 0, c] += -1
            W[k1*n + 0, c] += 1
            
            c = base + 5
            U[2, c] += 1
            U[0, c] += -1
            V[0*p + k0, c] += 1
            V[0*p + k1, c] += 1
            W[k1*n + 1, c] += 1
            
            c = base + 6
            U[1, c] += 1
            U[3, c] += -1
            V[1*p + k0, c] += 1
            V[1*p + k1, c] += 1
            W[k0*n + 0, c] += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
