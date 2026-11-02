"""Combined approach: structured rank-1 tensor search with coefficient solving."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Verify by computing actual multiplication for random matrices
    np.random.seed(12345)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for c in range(rank):
            uc = U_arr[:, c]
            vc = V_arr[:, c]
            wc = W_arr[:, c]
            # Reconstruct A and B factors
            A_c = uc.reshape(n, m)
            B_c = vc.reshape(m, p)
            # The contribution to C from this rank-1 term
            # C[i,k] gets contribution: wc[k*n+i] * sum_j A[i,j]*B[j,k] for this component
            # Actually, the rank-1 term contributes: A_c @ B_c * wc[k*n+i] for each (i,k)
            # But wait - the tensor decomposition means:
            # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
            # No, that's not right either. Let me reconsider.
            
            # The standard tensor decomposition for matrix multiplication:
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # = sum_c sum_j U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]
            # This doesn't quite work as stated. Let me think again.
            
            # Actually, the correct interpretation is:
            # We want: C[i,k] = sum_c W[k*n+i, c] * (U_c^T A) * (B V_c)
            # where U_c is the c-th column of U reshaped as (n,m), etc.
            
            # More precisely, for each rank-1 term c:
            # It contributes to C[i,k] the value: W[k*n+i, c] * sum_j U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k]
            
            # This can be rewritten as:
            # For each c, define:
            #   A_c[i,j] = U[i*m+j, c]  (this scales A)
            #   B_c[j,k] = V[j*p+k, c]  (this scales B)
            #   Then the contribution is W[k*n+i, c] * (A * A_c) @ (B * B_c) [i,k]
            # where * is element-wise multiplication
            
            # So: C_computed[i,k] += W[k*n+i, c] * sum_j (A[i,j]*U[i*m+j,c]) * (B[j,k]*V[j*p+k,c])
            
            A_scaled = A * A_c
            B_scaled = B * B_c
            C_computed += wc.reshape(p, n).T * (A_scaled @ B_scaled)
        
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition."""
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

def _generate_structured_tensors(n, m, p, pool_size=500):
    """Generate a pool of structured rank-1 tensors with small integer entries."""
    tensors = []
    np.random.seed(42)
    
    # Generate random structured tensors with entries in {-1, 0, 1}
    for _ in range(pool_size):
        # Create sparse patterns
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        
        # Randomly set some entries to ±1
        n_u = np.random.randint(1, n * m + 1)
        n_v = np.random.randint(1, m * p + 1)
        n_w = np.random.randint(1, p * n + 1)
        
        u_indices = np.random.choice(n * m, n_u, replace=False)
        v_indices = np.random.choice(m * p, n_v, replace=False)
        w_indices = np.random.choice(p * n, n_w, replace=False)
        
        u[u_indices] = np.random.choice([-1, 1], n_u)
        v[v_indices] = np.random.choice([-1, 1], n_v)
        w[w_indices] = np.random.choice([-1, 1], n_w)
        
        tensors.append((u, v, w))
    
    return tensors

def _try_structured_decomposition(n, m, p, max_rank=None, deadline=None):
    """
    Try to find a lower-rank decomposition using structured rank-1 tensors.
    Uses a greedy approach: iteratively select the best tensor and solve for coefficients.
    """
    if max_rank is None:
        max_rank = n * m * p
    if deadline is None:
        deadline = time.time() + (DEADLINE_SECONDS - 5)
    
    start_time = time.time()
    best_result = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # The multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We want: T[i,j,k] = sum_c alpha_c * u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i]
    
    # Generate a pool of candidate tensors
    pool = _generate_structured_tensors(n, m, p, pool_size=1000)
    
    # Greedy selection: at each step, find the tensor that best explains the residual
    residual = np.ones((n, m, p))
    selected = []
    coefficients = []
    
    for rank_attempt in range(1, max_rank + 1):
        if time.time() > deadline:
            break
        
        # Find the best tensor from the pool
        best_tensor = None
        best_coeff = 0
        best_residual_norm = np.linalg.norm(residual)
        
        # Sample a subset of the pool for speed
        sample_size = min(100, len(pool))
        sample_indices = np.random.choice(len(pool), sample_size, replace=False)
        
        for idx in sample_indices:
            u, v, w = pool[idx]
            
            # Compute the tensor this represents
            tensor_val = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_val[i, j, k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
            
            # Find optimal coefficient: alpha = <residual, tensor> / <tensor, tensor>
            dot_res = np.sum(residual * tensor_val)
            dot_ten = np.sum(tensor_val * tensor_val)
            
            if dot_ten > 1e-12:
                alpha = dot_res / dot_ten
                new_residual = residual - alpha * tensor_val
                new_norm = np.linalg.norm(new_residual)
                
                if new_norm < best_residual_norm * 0.99:  # Require significant improvement
                    best_residual_norm = new_norm
                    best_tensor = (u, v, w)
                    best_coeff = alpha
        
        if best_tensor is None:
            break
        
        selected.append(best_tensor)
        coefficients.append(best_coeff)
        residual -= best_coeff * np.array([
            [u[i*m+j] * v[j*p+k] * w[k*n+i] for k in range(p)] for j in range(m) for i in range(n)
        ]).reshape(n, m, p)
        
        # Check if residual is small enough
        if np.linalg.norm(residual) < 1e-8:
            break
        
        # Check if we can construct a valid decomposition
        if len(selected) > 0:
            # Try to round coefficients to nice values
            U_mat = np.zeros((n*m, len(selected)))
            V_mat = np.zeros((m*p, len(selected)))
            W_mat = np.zeros((p*n, len(selected)))
            
            for c, (u, v, w) in enumerate(selected):
                U_mat[:, c] = u
                V_mat[:, c] = v
                W_mat[:, c] = w * coefficients[c]  # Absorb coefficient into W
            
            # Check if W has nice values
            W_check = W_mat * 2
            if np.allclose(W_check, np.round(W_check), atol=1e-6):
                # Convert to lists
                U_list = [[int(round(U_mat[i, c])) for c in range(len(selected))] for i in range(n*m)]
                V_list = [[int(round(V_mat[i, c])) for c in range(len(selected))] for i in range(m*p)]
                W_list = [[float(np.round(W_mat[i, c] * 2)) / 2.0 for c in range(len(selected))] for i in range(p*n)]
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    result = {"U": U_list, "V": V_list, "W": W_list}
                    if len(selected) < best_rank:
                        best_rank = len(selected)
                        best_result = result
    
    return best_result

def _try_fixed_UV_solve(n, m, p, target_rank, deadline):
    """
    Try fixing U and V with random small integer entries, then solve for W.
    """
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Generate random U and V with small integer entries
        U_rand = np.random.randint(-2, 3, size=(n*m, target_rank))
        V_rand = np.random.randint(-2, 3, size=(m*p, target_rank))
        
        # Compute W: for each (k,i), W[k*n+i, c] = 1 / sum_j U[i*m+j, c] * V[j*p+k, c]
        W_rand = np.zeros((p*n, target_rank))
        valid = True
        for c in range(target_rank):
            for k in range(p):
                for i in range(n):
                    denom = 0.0
                    for j in range(m):
                        denom += U_rand[i*m+j, c] * V_rand[j*p+k, c]
                    if abs(denom) < 1e-12:
                        valid = False
                        break
                    W_rand[k*n+i, c] = 1.0 / denom
                if not valid:
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Check if W has nice values (close to integers or half-integers)
        W_check = W_rand * 2
        if not np.allclose(W_check, np.round(W_check), atol=1e-6):
            continue
        
        # Verify the decomposition
        U_list = [[int(round(U_rand[i, c])) for c in range(target_rank)] for i in range(n*m)]
        V_list = [[int(round(V_rand[i, c])) for c in range(target_rank)] for i in range(m*p)]
        W_list = [[float(np.round(W_rand[i, c] * 2)) / 2.0 for c in range(target_rank)] for i in range(p*n)]
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    deadline = start_time + DEADLINE_SECONDS
    
    # Try structured decomposition search
    try:
        result = _try_structured_decomposition(n, m, p, max_rank=best_rank, deadline=deadline)
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank and _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = rank
    except Exception:
        pass
    
    # Try fixed UV solve for various ranks
    if time.time() < deadline - 10:
        for target_rank in range(max(20, best_rank - 10), best_rank):
            if time.time() > deadline - 5:
                break
            try:
                result = _try_fixed_UV_solve(n, m, p, target_rank, deadline)
                if result is not None:
                    rank = len(result["U"][0])
                    if rank < best_rank:
                        best = result
                        best_rank = rank
            except Exception:
                pass
    
    return best

# EVOLVE-BLOCK-END
