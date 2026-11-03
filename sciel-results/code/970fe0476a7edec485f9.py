"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Find a low-rank exact tensor decomposition for 4x4 x 4x7 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    # Start with schoolbook rank 112
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    # Try to find lower rank via structured Winograd-style construction
    # For 4x4 x 4x7, try rank values from 100 down to 80
    for target_rank in range(100, 79, -2):
        if time.time() > deadline - 30:
            break
        result = _try_winograd_decomposition(n, m, p, target_rank, deadline)
        if result is not None:
            best_rank = target_rank
            best_result = result
            # Don't break - try to go lower
        if time.time() > deadline - 20:
            break
    
    # Try even lower ranks
    for target_rank in range(78, 63, -1):
        if time.time() > deadline - 15:
            break
        result = _try_winograd_decomposition(n, m, p, target_rank, deadline)
        if result is not None:
            best_rank = target_rank
            best_result = result
        if time.time() > deadline - 10:
            break
    
    return best_result


def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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


def _try_winograd_decomposition(n, m, p, target_rank, deadline):
    """
    Attempt to find a rank-`target_rank` decomposition using a structured
    Winograd-style approach with random initialization and iterative refinement.
    """
    np.random.seed(42)
    
    # The tensor T[i*m+j, j*p+k, k*n+i] = delta (it's the matmul tensor)
    # We need: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    
    # Strategy: use a structured initialization based on separating the i,j,k indices
    # and then use ALS (alternating least squares) to refine
    
    U = np.random.randn(n * m, target_rank) * 0.5
    V = np.random.randn(m * p, target_rank) * 0.5
    W = np.random.randn(p * n, target_rank) * 0.5
    
    # Build the target tensor as a 3D array
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # ALS iterations
    max_iters = 50
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Update U: for each a, U[a,:] should satisfy
        # sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c] for all b,c
        # This is: U[a,:] @ (V * W) = T[a,:,:].reshape(-1)
        # where (V*W)[b,c,r] = V[b,r]*W[c,r]
        
        # Compute the "contracted" matrix for each a
        # For fixed a, we need U[a,:] such that:
        # sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
        # Let M[a,b,c] = T[a,b,c], and we form a linear system
        # Reshape: for each a, the equation is U[a,:] @ B_a = t_a
        # where B_a[b_idx, r] = V[b_idx, r] * W[c_idx, r] ... this is tricky
        
        # Better: use the fact that for fixed a, 
        # U[a,:] is a vector of length target_rank
        # and the constraint is: for all (b,c), sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
        # This is a linear system: (V ⊙ W^T) flattened... 
        
        # Let's use a different approach: treat it as a bilinear system
        # For fixed U, update V: for each b, sum_r V[b,r] U[a,r] W[c,r] = T[a,b,c]
        # => V[b,:] @ (U * W) = T[:,b,:].reshape(-1)
        # where (U*W)[a,c,r] = U[a,r]*W[c,r]
        
        # Actually, let's use a simpler ALS:
        # Fix V and W, solve for U:
        # For each a: U[a,:] * (V ⊗ W) = T[a,:,:]
        # The matrix (V ⊗ W) has shape (m*p * p*n, target_rank)
        # where element [b*p*n + c, r] = V[b,r] * W[c,r]
        
        # This is expensive. Let's use a smarter approach.
        
        # For each a (i*m+j), the constraint involves only specific (b,c) pairs.
        # Specifically, T[a,b,c] = 1 only when b = j*p+k and c = k*n+i for the same i,j,k.
        # Wait no, T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
        # So for fixed a = i*m+j, T[a, j*p+k, k*n+i] = 1 for all k, and 0 otherwise.
        
        # So for fixed a = i*m+j:
        # sum_r U[a,r] V[j*p+k, r] W[k*n+i, r] = 1 for k=0..p-1
        # sum_r U[a,r] V[b, r] W[c, r] = 0 for other (b,c)
        
        # This gives us (p) nonzero constraints and (m*p*p*n - p) zero constraints.
        # That's a lot of constraints. Let's just use numpy lstsq on the full system.
        
        # Build the design matrix for U:
        # For each a, we have (m*p)*(p*n) equations in target_rank unknowns
        # This is too large. Let's use a subsampled approach or gradient descent.
        
        # Use gradient descent with projected updates
        if it % 3 == 0:
            # Update U
            for a in range(n * m):
                # Compute gradient: dL/du = sum_{b,c} (pred - T) * V[b,:] * W[c,:]
                # pred = U[a,:] @ (V[b,:]*W[c,:])
                # This is expensive. Use random sampling.
                sample_size = min(200, m * p * p * n)
                b_samples = np.random.randint(0, m * p, sample_size)
                c_samples = np.random.randint(0, p * n, sample_size)
                
                # Compute predictions
                # pred[s] = U[a,:] @ (V[b_samples[s],:] * W[c_samples[s],:])
                prod = V[b_samples, :] * W[c_samples, :]  # (sample_size, target_rank)
                preds = prod @ U[a, :]  # (sample_size,)
                targets = T[a, b_samples, c_samples]
                
                # Gradient
                grad = prod.T @ (preds - targets)  # (target_rank,)
                U[a, :] -= 0.01 * grad
        
        elif it % 3 == 1:
            # Update V
            for b in range(m * p):
                sample_size = min(200, n * m * p * n)
                a_samples = np.random.randint(0, n * m, sample_size)
                c_samples = np.random.randint(0, p * n, sample_size)
                
                prod = U[a_samples, :] * W[c_samples, :]  # (sample_size, target_rank)
                preds = prod @ V[b, :]
                targets = T[a_samples, b, c_samples]
                
                grad = prod.T @ (preds - targets)
                V[b, :] -= 0.01 * grad
        
        else:
            # Update W
            for c in range(p * n):
                sample_size = min(200, n * m * m * p)
                a_samples = np.random.randint(0, n * m, sample_size)
                b_samples = np.random.randint(0, m * p, sample_size)
                
                prod = U[a_samples, :] * V[b_samples, :]
                preds = prod @ W[c, :]
                targets = T[a_samples, b_samples, c]
                
                grad = prod.T @ (preds - targets)
                W[c, :] -= 0.01 * grad
        
        # Check convergence
        if it % 5 == 4:
            if _check_validity(U, V, W, n, m, p, tol=1e-6):
                # Snap to nearest half-integers if close
                U_snapped, V_snapped, W_snapped = _snap_to_half_integers(U, V, W)
                if _check_validity(U_snapped, V_snapped, W_snapped, n, m, p, tol=1e-9):
                    return _to_lists(U_snapped, V_snapped, W_snapped)
                elif _check_validity(U, V, W, n, m, p, tol=1e-9):
                    return _to_lists(U, V, W)
    
    return None


def _check_validity(U, V, W, n, m, p, tol=1e-6):
    """Check if the decomposition is valid."""
    # T[i*m+j, j*p+k, k*n+i] should equal sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j], V[j * p + k])
                val = np.dot(val, W[k * n + i])  # This is wrong, need element-wise then sum
                # Actually: sum_r U[a,r] * V[b,r] * W[c,r]
                val = np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i])
                if abs(val - 1.0) > tol:
                    return False
    # Check zeros: for random (a,b,c) that don't correspond to valid (i,j,k)
    # This is expensive, so just check a sample
    for _ in range(100):
        a = np.random.randint(0, n * m)
        b = np.random.randint(0, m * p)
        c = np.random.randint(0, p * n)
        # Check if this corresponds to a valid triple
        i_a, j_a = divmod(a, m)
        j_b, k_b = divmod(b, p)
        k_c, i_c = divmod(c, n)
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue  # This is a valid position, should be 1
        val = np.sum(U[a] * V[b] * W[c])
        if abs(val) > tol:
            return False
    return True


def _snap_to_half_integers(U, V, W):
    """Snap values to nearest half-integer if close."""
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    return U_snap, V_snap, W_snap


def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists."""
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
