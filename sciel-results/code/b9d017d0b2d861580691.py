import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # 100s out of 120s
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i for some i,j,k
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Precompute indices of non-zero entries in T
    nz_u, nz_v, nz_w = np.nonzero(T)
    nz_count = len(nz_u)
    nz_u = nz_u.astype(int)
    nz_v = nz_v.astype(int)
    nz_w = nz_w.astype(int)
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    
    def verify(U, V, W, rank):
        """Verify that U, V, W exactly decompose the tensor."""
        # Compute S = sum_r U[:,r] V[:,r] W[:,r] using outer products
        # This is the tensor reconstruction
        S = np.zeros((n * m, m * p, p * n), dtype=float)
        for r in range(rank):
            S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        # Check that S matches T at non-zero positions and is 0 elsewhere
        # Check non-zero positions
        S_nz = S[nz_u, nz_v, nz_w]
        if not np.allclose(S_nz, 1.0, atol=1e-9):
            return False
        # Check zero positions
        S_full = S.flatten()
        T_flat = T.flatten()
        zero_mask = T_flat == 0
        if not np.allclose(S_full[zero_mask], 0.0, atol=1e-9):
            return False
        return True
    
    def verify_rational(U, V, W, rank):
        """Verify using Fraction for exactness."""
        # Convert to Fraction
        U_frac = np.zeros((n * m, rank), dtype=object)
        V_frac = np.zeros((m * p, rank), dtype=object)
        W_frac = np.zeros((p * n, rank), dtype=object)
        for i in range(U.shape[0]):
            for r in range(rank):
                U_frac[i, r] = Fraction(U[i, r]).limit_denominator(10**6)
                V_frac[i, r] = Fraction(V[i, r]).limit_denominator(10**6)
        for i in range(V.shape[0]):
            for r in range(rank):
                V_frac[i, r] = Fraction(V[i, r]).limit_denominator(10**6)
        for i in range(W.shape[0]):
            for r in range(rank):
                W_frac[i, r] = Fraction(W[i, r]).limit_denominator(10**6)
        
        # Check each non-zero entry
        for idx in range(nz_count):
            u, v, w = nz_u[idx], nz_v[idx], nz_w[idx]
            total = Fraction(0)
            for r in range(rank):
                total += U_frac[u, r] * V_frac[v, r] * W_frac[w, r]
            if total != 1:
                return False
        
        # Check a sample of zero entries
        # Get indices of zero entries in the flattened tensor
        # This is expensive, so we check a random sample
        rng = np.random.RandomState(42)
        total_entries = n * m * m * p * p * n
        zero_indices = []
        for _ in range(200):
            u = rng.randint(0, n * m)
            v = rng.randint(0, m * p)
            w = rng.randint(0, p * n)
            if T[u, v, w] == 0:
                zero_indices.append((u, v, w))
        
        for u, v, w in zero_indices[:50]:
            total = Fraction(0)
            for r in range(rank):
                total += U_frac[u, r] * V_frac[v, r] * W_frac[w, r]
            if total != 0:
                return False
        return True
    
    # Strategy: Use randomized tensor decomposition with local search
    # We try to find a rank R decomposition by iteratively optimizing U, V, W
    # using a Gauss-Seidel-like approach on the tensor
    
    best_U = np.zeros((n * m, rank_schoolbook))
    best_V = np.zeros((m * p, rank_schoolbook))
    best_W = np.zeros((p * n, rank_schoolbook))
    best_rank = rank_schoolbook
    
    # Initialize schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                best_U[u, col] = 1
                best_V[v, col] = 1
                best_W[w, col] = 1
                col += 1
    
    # Try to find lower rank decompositions
    # Use a structured approach: try different rank values and use
    # alternating least squares (ALS) for tensor decomposition
    
    for target_rank in range(78, rank_schoolbook):
        if time.time() > deadline:
            break
        
        if target_rank >= best_rank:
            continue
        
        # Multiple random restarts
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(42 + restart * 1000 + target_rank)
            
            # Initialize with random values, but scale appropriately
            U = rng.randn(n * m, target_rank) * 0.1
            V = rng.randn(m * p, target_rank) * 0.1
            W = rng.randn(p * n, target_rank) * 0.1
            
            # Add some structure: initialize with a mix of schoolbook-like and random
            # This helps convergence
            for r in range(min(target_rank, rank_schoolbook)):
                if r < rank_schoolbook:
                    i, j, k = divmod(r, m * p)
                    # Actually let's just use the schoolbook pattern for first rank_schoolbook entries
                    pass
            
            # Better initialization: use the schoolbook solution as a starting point
            # for the first min(target_rank, 84) columns, then add random columns
            U_init = np.zeros((n * m, target_rank))
            V_init = np.zeros((m * p, target_rank))
            W_init = np.zeros((p * n, target_rank))
            
            num_school = min(target_rank, rank_schoolbook)
            col2 = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col2 < target_rank:
                            u = i * m + j
                            v = j * p + k
                            w = k * n + i
                            U_init[u, col2] = 1
                            V_init[v, col2] = 1
                            W_init[w, col2] = 1
                            col2 += 1
                        else:
                            break
                    if col2 >= target_rank:
                        break
                if col2 >= target_rank:
                    break
            if col2 >= target_rank:
                break
            
            # For remaining columns, use random initialization
            for r in range(num_school, target_rank):
                U_init[:, r] = rng.randn(n * m) * 0.01
                V_init[:, r] = rng.randn(m * p) * 0.01
                W_init[:, r] = rng.randn(p * n) * 0.01
            
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
            
            # Iterative optimization: minimize ||sum_r U_r V_r W_r - T||^2
            # Using alternating updates
            # For each factor, we solve a linear system
            
            max_iters = 200
            prev_error = float('inf')
            converged = False
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U: For each column i of U, we want to minimize
                # sum over (v,w) of (sum_r U[i,r] V[v,r] W[w,r] - T[i,v,w])^2
                # This gives a linear system for each i
                # But this is expensive. Instead, use a simpler approach:
                # Gradient descent with step size adaptation
                
                # Compute gradient for U
                # grad_U[i,r] = sum_{v,w} 2*(S[i,v,w] - T[i,v,w]) * V[v,r] * W[w,r]
                # where S[i,v,w] = sum_r' U[i,r'] V[v,r'] W[w,r']
                
                # This is expensive to compute. Let's use a different approach:
                # ALS (Alternating Least Squares) for tensor decomposition
                
                # Actually, let's use a simpler iterative approach:
                # For each factor, treat the other two as fixed and solve a least squares problem
                
                # Update U:
                # For each row i of U, we have:
                # min_{u_i} sum_{v,w} (sum_r u_i[r] V[v,r] W[w,r] - T[i,v,w])^2
                # This is: min_{u_i} ||V^T W^T u_i - T[i,:,:]||^2
                # where V^T W^T is the matrix with entries (V^T W^T)[r, (v,w)] = V[v,r] W[w,r]
                # Actually, it's: min_{u_i} ||sum_r u_i[r] (V[:,r] * W[:,r]) - T[i,:,:]||^2
                # where V[:,r] * W[:,r] is the outer product vectorized
                # So we have a linear system: M @ u_i = t_i
                # where M[r, (v,w)] = V[v,r] * W[w,r]
                # and t_i[(v,w)] = T[i,v,w]
                
                # But M is target_rank x (m*p * p*n) which is 78 x (28*21) = 78 x 588
                # This is manageable.
                
                # Build M for U update
                # M shape: (target_rank, m*p * p*n)
                # M[r, v * (p*n) + w] = V[v, r] * W[w, r]
                MW = np.einsum('vr,wr->vwr', V, W)  # shape (m*p, p*n, target_rank)
                MW_flat = MW.transpose(2, 0, 1).reshape(target_rank, -1)  # shape (target_rank, m*p * p*n)
                
                # For each i, solve MW_flat^T MW_flat @ u_i = MW_flat^T T[i,:,:].flatten()
                # But this is the same for all i, so we can precompute
                A_U = MW_flat.T @ MW_flat  # (target_rank, target_rank)
                A_U += 1e-8 * np.eye(target_rank)  # Regularization
                
                # Solve for all i at once
                T_flat = T.reshape(n * m, -1)  # (n*m, m*p * p*n)
                b_U = T_flat @ MW_flat  # (n*m, target_rank)
                
                # Solve A_U @ U[i,:]^T = b_U[i,:]^T for each i
                # Or equivalently, A_U @ U^T = b_U^T
                try:
                    U_new = np.linalg.solve(A_U, b_U.T).T
                    U = U_new
                except np.linalg.LinAlgError:
                    U = np.linalg.lstsq(A_U, b_U.T, rcond=None)[0].T
                
                # Update V similarly
                # M[r, (u,w)] = U[u,r] * W[w,r]
                UW = np.einsum('ur,wr->uwr', U, W)  # shape (n*m, p*n, target_rank)
                UW_flat = UW.transpose(2, 0, 1).reshape(target_rank, -1)
                
                A_V = UW_flat.T @ UW_flat
                A_V += 1e-8 * np.eye(target_rank)
                
                T_flat_V = T.transpose(1, 0, 2).reshape(m * p, -1)  # (m*p, n*m * p*n)
                b_V = T_flat_V @ UW_flat
                
                try:
                    V_new = np.linalg.solve(A_V, b_V.T).T
                    V = V_new
                except np.linalg.LinAlgError:
                    V = np.linalg.lstsq(A_V, b_V.T, rcond=None)[0].T
                
                # Update W similarly
                # M[r, (u,v)] = U[u,r] * V[v,r]
                UV = np.einsum('ur,vr->uvr', U, V)  # shape (n*m, m*p, target_rank)
                UV_flat = UV.transpose(2, 0, 1).reshape(target_rank, -1)
                
                A_W = UV_flat.T @ UV_flat
                A_W += 1e-8 * np.eye(target_rank)
                
                T_flat_W = T.transpose(2, 0, 1).reshape(p * n, -1)  # (p*n, n*m * m*p)
                b_W = T_flat_W @ UV_flat
                
                try:
                    W_new = np.linalg.solve(A_W, b_W.T).T
                    W = W_new
                except np.linalg.LinAlgError:
                    W = np.linalg.lstsq(A_W, b_W.T, rcond=None)[0].T
                
                # Compute error
                S = np.einsum('ir,jr,kr->ijk', U, V, W)
                error = np.sum((S - T)**2)
                
                if abs(prev_error - error) < 1e-10:
                    converged = True
                    break
                prev_error = error
            
            # Check if we found a valid decomposition
            if verify(U, V, W, target_rank):
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                break  # Break out of restarts for this rank
        
        if best_rank <= target_rank:
            break
    
    # If we found a better solution, return it
    if best_rank < rank_schoolbook:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

