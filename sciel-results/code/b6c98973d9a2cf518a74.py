"""Program for rank-minimal tensor decomposition of 4x5 * 5x5 matrix multiplication."""
DIMENSIONS = [4, 5, 5]
import numpy as np
import time
from fractions import Fraction
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Schoolbook decomposition
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    # Target Tensor Construction
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # We store non-zero indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
    
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    def check_exact(U, V, W):
        """Check if U, V, W exactly decompose the tensor."""
        # T[a, b, c] should be sum_r U[a,r] V[b,r] W[c,r]
        # For all (a,b,c) in indices, sum should be 1.
        # For all other (a,b,c), sum should be 0.
        # Checking all is too slow. We check the non-zeros and a sample of zeros.
        # Actually, for exactness, we need to verify the whole tensor.
        # But we can verify the non-zeros are 1.
        # And we can check the rank of the reconstruction vs original.
        
        # Let's reconstruct the tensor as a dict for non-zeros
        recon = {}
        for r in range(U.shape[1]):
            ur = U[:, r]
            vr = V[:, r]
            wr = W[:, r]
            # Add outer product
            for a in range(dim_u):
                if ur[a] == 0: continue
                for b in range(dim_v):
                    if vr[b] == 0: continue
                    val_ab = ur[a] * vr[b]
                    if val_ab == 0: continue
                    for c in range(dim_w):
                        if wr[c] == 0: continue
                        val = val_ab * wr[c]
                        if val == 0: continue
                        if (a, b, c) in recon:
                            recon[(a, b, c)] += val
                        else:
                            recon[(a, b, c)] = val
        
        # Check non-zeros
        for a, b, c in indices:
            if recon.get((a, b, c), 0) != 1.0:
                return False
        
        # Check a few random zeros to ensure no spurious non-zeros
        # If the rank is low, spurious non-zeros are likely.
        # We need to ensure the support is exactly the indices.
        # This is expensive. 
        # Alternative: Check if the matrix multiplication is correct for random matrices.
        # If U,V,W define a valid decomposition, then for random A, B:
        # C = A @ B should equal sum_r (A * U_r) * (B * V_r) * W_r ?
        # No, the decomposition is:
        # (A*B)[i,k] = sum_j A[i,j] B[j,k]
        # = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_k B[j,k] V[j*p+k, r]) * W[k*n+i, r] ??
        # No, the standard decomposition is:
        # T = sum_r u_r \otimes v_r \otimes w_r
        # (A*B)[i,k] = sum_{i,j,k} A[i,j] B[j,k] T[i*m+j, j*p+k, k*n+i]
        # = sum_r (sum_{i,j} A[i,j] u_r[i*m+j]) (sum_{j,k} B[j,k] v_r[j*p+k]) w_r[k*n+i] ??
        # This form is not quite right for direct multiplication.
        # The correct check is:
        # Construct the tensor from U,V,W and compare with T.
        # We already did that in `recon`.
        # We need to ensure `recon` has NO other keys with non-zero values.
        
        for key, val in recon.items():
            if key not in set(indices) and val != 0.0:
                return False
            if key in set(indices) and val != 1.0:
                return False
        
        return True

    def als_decompose(rank, max_iters=50, tol=1e-8):
        """
        Perform ALS to find U, V, W of given rank that approximate the tensor.
        """
        # Initialize
        U = np.random.randn(dim_u, rank) * 0.1
        V = np.random.randn(dim_v, rank) * 0.1
        W = np.random.randn(dim_w, rank) * 0.1
        
        # Precompute indices for faster access?
        # We will build the linear system matrices on the fly.
        
        prev_err = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Update U
            # For each a, solve min_Ua sum_{b,c} (T[a,b,c] - sum_r Ua[r] V[b,r] W[c,r])^2
            # Let Z_a be dim_v*dim_w x rank matrix
            # Z_a[b*dim_w + c, r] = V[b,r] * W[c,r]
            # T_a is dim_v*dim_w vector
            # Ua = T_a @ Z_a.T @ inv(Z_a @ Z_a.T + lambda I)
            
            lambda_reg = 1e-6
            for a in range(dim_u):
                # Build T_a
                T_a = np.zeros(dim_v * dim_w)
                # Find non-zeros for this a
                # a = i*m+j -> i = a//m, j = a%m
                i = a // m
                j = a % m
                # b = j*p+k -> k = b%p
                # c = k*n+i
                # For fixed a, j is fixed.
                # b ranges over all j'*p+k. Only b with j'=j are non-zero in T?
                # T[a,b,c]=1 iff b=j*p+k and c=k*n+i.
                # So for fixed a (i,j), non-zero T entries are at:
                # b = j*p + k, c = k*n + i for k in 0..p-1.
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    T_a[b * dim_w + c] = 1.0
                
                # Build Z_a
                # Z_a[b*dim_w + c, r] = V[b,r] * W[c,r]
                # This is (V[:, :, None] * W[None, :, :]).reshape(dim_v*dim_w, rank)
                # But we only need to solve for the non-zero parts?
                # No, the sum is over ALL b,c.
                # However, T_a is zero for most b,c.
                # The gradient is sum_{b,c} (T_a[b,c] - (Z_a @ Ua)[b,c]) * Z_a[:, r]
                # = sum_{b,c} T_a[b,c] Z_a[:, r] - Z_a.T @ (Z_a @ Ua)
                # We can compute Z_a.T @ Z_a and Z_a.T @ T_a.
                
                # Efficiently compute Z_a.T @ Z_a
                # (V[b,r] W[c,r])^T (V[b',r'] W[c',r'])
                # sum_{b,c} V[b,r] W[c,r] V[b,r'] W[c,r']
                # = (sum_b V[b,r] V[b,r']) * (sum_c W[c,r] W[c,r'])
                # = (V.T @ V)[r,r'] * (W.T @ W)[r,r']
                
                VtV = V.T @ V
                WtW = W.T @ W
                # ZtZ = VtV \otimes WtW ? No, element-wise product?
                # Let's check dimensions.
                # (Z_a.T @ Z_a)[r, r'] = sum_{b,c} Z_a[b,c,r] Z_a[b,c,r']
                # = sum_{b,c} V[b,r] W[c,r] V[b,r'] W[c,r']
                # = (sum_b V[b,r] V[b,r']) * (sum_c W[c,r] W[c,r'])
                # Yes, it is the Hadamard product of VtV and WtW.
                ZtZ = VtV * WtW
                
                # Compute Z_a.T @ T_a
                # sum_{b,c} T_a[b,c] V[b,r] W[c,r]
                # T_a is non-zero only for specific (b,c) pairs.
                # b = j*p+k, c = k*n+i
                ZtT = np.zeros(rank)
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    # T_a[b,c] = 1
                    # V[b,r] W[c,r]
                    ZtT += V[b, :] * W[c, :]
                
                # Solve (ZtZ + lambda I) Ua = ZtT
                try:
                    U[a, :] = np.linalg.solve(ZtZ + lambda_reg * np.eye(rank), ZtT)
                except np.linalg.LinAlgError:
                    pass
            
            # Update V (similar logic)
            # For each b, solve min_Vb sum_{a,c} (T[a,b,c] - sum_r U[a,r] Vb[r] W[c,r])^2
            # Z_b[a*dim_w + c, r] = U[a,r] W[c,r]
            # UUt = U.T @ U
            # WtW already computed
            # ZtZ = UUt * WtW
            # ZtT = sum_{a,c} T[a,b,c] U[a,r] W[c,r]
            
            UUt = U.T @ U
            for b in range(dim_v):
                # b = j*p+k -> j = b//p, k = b%p
                j = b // p
                k = b % p
                # Non-zeros for fixed b:
                # a = i*m+j, c = k*n+i for i in 0..n-1
                T_b = np.zeros(dim_u * dim_w) # Not needed directly, just ZtT
                ZtT = np.zeros(rank)
                for i in range(n):
                    a = i * m + j
                    c = k * n + i
                    ZtT += U[a, :] * W[c, :]
                
                ZtZ = UUt * WtW
                try:
                    V[b, :] = np.linalg.solve(ZtZ + lambda_reg * np.eye(rank), ZtT)
                except np.linalg.LinAlgError:
                    pass
            
            # Update W
            # For each c, solve min_Wc sum_{a,b} (T[a,b,c] - sum_r U[a,r] V[b,r] Wc[r])^2
            # Z_c[a*dim_v + b, r] = U[a,r] V[b,r]
            # UUt * VtV
            # ZtT = sum_{a,b} T[a,b,c] U[a,r] V[b,r]
            
            VtV = V.T @ V
            for c in range(dim_w):
                # c = k*n+i -> k = c//n, i = c%n
                k = c // n
                i = c % n
                # Non-zeros for fixed c:
                # a = i*m+j, b = j*p+k for j in 0..m-1
                ZtT = np.zeros(rank)
                for j in range(m):
                    a = i * m + j
                    b = j * p + k
                    ZtT += U[a, :] * V[b, :]
                
                ZtZ = UUt * VtV
                try:
                    W[c, :] = np.linalg.solve(ZtZ + lambda_reg * np.eye(rank), ZtT)
                except np.linalg.LinAlgError:
                    pass
            
            # Calculate error
            # Error = sum_{indices} (1 - sum_r U[a,r]V[b,r]W[c,r])^2
            err = 0.0
            for a, b, c in indices:
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                err += (1.0 - val) ** 2
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
        # Round to nearest 0.5
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if check_exact(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
        
        return None

    # Try different ranks
    # Lower bound is max(n, m, p) = 5.
    # Schoolbook is 100.
    # Let's try ranks: 10, 15, 20, 30, 50
    ranks_to_try = [10, 15, 20, 30, 50]
    
    best_result = None
    
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        for restart in range(3):
            if time.time() > deadline - 10:
                break
                
            result = als_decompose(rank, max_iters=100)
            if result is not None:
                U, V, W = result
                # Verify again just in case
                if check_exact(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
    # If no exact solution found, return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
