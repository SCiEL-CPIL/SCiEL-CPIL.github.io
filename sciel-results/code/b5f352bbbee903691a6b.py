import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _strassen_44():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 2, 0, 2)
    ]
    
    col = 0
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
            for (li, lj), val in u_coeffs.items():
                gi = li + xi
                gj = lj + xj
                U[gi * m + gj][col] += val
            for (lj, lk), val in v_coeffs.items():
                gj = lj + yj
                gk = lk + yk
                V[gj * p + gk][col] += val
            for (li, lk), val in w_coeffs.items():
                gi = li + zi
                gk = lk + zk
                W[gk * n + gi][col] += val
            col += 1
    
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] += 1
            V[j * p + k][col] += 1
            W[k * n + 2][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

def _try_reduce_to_rank43(U, V, W, n, m, p, deadline):
    """Try to find a rank-43 decomposition by searching near the rank-44 solution."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    if rank < 43:
        return None
    
    # Strategy: try to find a linear dependence among the rank-1 terms
    # that allows us to drop one term while maintaining the decomposition
    
    # For each pair of columns, check if one can be expressed as a combination
    # of the others (with the constraint that the tensor equation still holds)
    
    # A more practical approach: try random projections and see if we can
    # reduce the effective rank
    
    best_result = None
    best_rank = rank
    
    # Try to find if any column is redundant
    for c_drop in range(rank):
        if time.time() > deadline:
            break
        
        # Try dropping column c_drop and see if we can adjust others
        U_test = np.delete(U_arr, c_drop, axis=1)
        V_test = np.delete(V_arr, c_drop, axis=1)
        W_test = np.delete(W_arr, c_drop, axis=1)
        
        # Check if the tensor equation still holds (it won't, but check error)
        U_3d = U_test.reshape(n, m, -1)
        V_3d = V_test.reshape(m, p, -1)
        W_3d = W_test.reshape(p, n, -1)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        err = np.max(np.abs(T - 1.0))
        
        if err < 1e-9:
            return {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test.tolist()}
    
    # Try a smoother approach: parameterize a small perturbation
    # and use gradient descent to find a rank-43 solution
    
    # Create a rank-43 parameterization: take the first 43 columns and
    # allow a small perturbation
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Start with a random subset of 43 columns
        cols = list(range(rank))
        np.random.shuffle(cols)
        cols = cols[:43]
        
        U_43 = U_arr[:, cols].copy()
        V_43 = V_arr[:, cols].copy()
        W_43 = W_arr[:, cols].copy()
        
        # Add small random perturbations
        np.random.seed(attempt)
        U_43 += 0.01 * np.random.randn(*U_43.shape)
        V_43 += 0.01 * np.random.randn(*V_43.shape)
        W_43 += 0.01 * np.random.randn(*W_43.shape)
        
        # Verify
        U_3d = U_43.reshape(n, m, 43)
        V_3d = V_43.reshape(m, p, 43)
        W_3d = W_43.reshape(p, n, 43)
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        err = np.max(np.abs(T - 1.0))
        
        if err < 1e-9:
            return {"U": U_43.tolist(), "V": V_43.tolist(), "W": W_43.tolist()}
    
    return None

def _structured_block_search(n, m, p, deadline):
    """Try different block structures to find a lower rank decomposition."""
    best_result = None
    best_rank = n * m * p
    
    # Try: 3×2×2 blocks (3 blocks for the 4 columns of B, but 3 rows of A)
    # This gives 3 × 7 = 21 for a 3×4×4 if we can do it
    
    # Actually, let's try a 3×4×4 specific approach:
    # Partition into 3×2×2 blocks: 2 blocks × 7 = 14, plus handle the rest
    
    # Try a hybrid: use Strassen for 2×2×2 sub-problems but with different groupings
    
    # Approach: treat the problem as multiplying a 3×4 matrix by a 4×4 matrix
    # Partition A into rows: A[0:2,:] and A[2:3,:]
    # For A[0:2,:] × B: use 2×4×4 Strassen-like (rank 14 per 2×2×2 block, 4 blocks = 56? No)
    
    # Actually the current approach already does this optimally for the 2×4×4 part.
    
    # Let's try a different angle: use the fact that n=3 to create a
    # 3×3×3 core (rank 23 via Bini) plus a 3×4×1 tail (rank 9)
    # Total: 32? Let's try to construct this.
    
    # Bini et al. 3×3×3 algorithm (rank 23) - simplified version
    # This is complex; let's try a simpler structured search
    
    # Try: 1×4×4 naive for each row (rank 16×3 = 48) - that's schoolbook
    # Try: 2×4×4 Strassen (rank 14) × 2 + 1×4×4 naive (rank 16) = 44 (current)
    # Try: 3×3×3 Bini (rank 23) + 3×4×1 (rank 9) + 3×1×3 (rank 9) = 41?
    
    # The 3×4×4 can be decomposed as:
    # C = A[:, :3] × B[:, :3] + A[:, :3] × B[:, 3:4] + A[:, 3:4] × B[:, :3] + A[:, 3:4] × B[:, 3:4]
    # = (3×3×3) + (3×3×1) + (3×1×3) + (3×1×1)
    # Ranks: 23 + 9 + 9 + 3 = 44 (same as current!)
    # Or: 23 + 6 + 6 + 3 = 38 if we can do better on the small parts
    
    # For 3×3×1: C[i,k] = sum_j A[i,j] * B[j,0] for k=0 only
    # This is rank 9 schoolbook, but can we do less? 
    # It's equivalent to computing 3 dot products of length 3, so rank 9 is minimal
    # unless we share structure with the 3×3×3 part
    
    # For 3×1×3: C[i,k] = A[i,3] * B[3,k], rank 3 (each term is rank 1)
    # For 3×1×1: C[i,3] = A[i,3] * B[3,3], rank 1
    
    # So the theoretical minimum with this decomposition is 23 + 9 + 3 + 1 = 36
    # But we need to handle the coupling between terms
    
    # This is getting complex. Let's just try a direct optimization.
    
    return None

def _gradient_refine(U, V, W, n, m, p, target_rank, deadline):
    """Use gradient-based optimization to find a lower-rank decomposition."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    current_rank = U_arr.shape[1]
    
    if current_rank <= target_rank:
        return None
    
    # Try to find a target_rank decomposition by optimization
    # Initialize with a random projection of the current solution
    
    np.random.seed(42)
    
    for trial in range(10):
        if time.time() > deadline:
            break
        
        # Initialize with random combination of current columns
        U_init = np.zeros((n*m, target_rank))
        V_init = np.zeros((m*p, target_rank))
        W_init = np.zeros((p*n, target_rank))
        
        for c in range(target_rank):
            # Random linear combination of 2-3 current columns
            num_parents = min(2, current_rank)
            parents = np.random.choice(current_rank, size=num_parents, replace=False)
            weights = np.random.randn(num_parents)
            weights /= np.linalg.norm(weights)
            
            for p_idx, w in zip(parents, weights):
                U_init[:, c] += w * U_arr[:, p_idx]
                V_init[:, c] += w * V_arr[:, p_idx]
                W_init[:, c] += w * W_arr[:, p_idx]
        
        # Gradient descent on the loss: sum of (T[i,j,k] - 1)^2
        lr = 0.001
        for step in range(200):
            if time.time() > deadline:
                break
            
            # Compute tensor
            U_3d = U_init.reshape(n, m, target_rank)
            V_3d = V_init.reshape(m, p, target_rank)
            W_3d = W_init.reshape(p, n, target_rank)
            T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            err = T - 1.0
            
            # Compute gradients
            # dL/dU[i,j,c] = sum_{k} V[j,k,c] * W[k,i,c] * err[i,j,k]
            dU = np.zeros_like(U_init)
            dV = np.zeros_like(V_init)
            dW = np.zeros_like(W_init)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = err[i, j, k]
                        for c in range(target_rank):
                            dU[i*m+j, c] += V_init[j*p+k, c] * W_init[k*n+i, c] * e
                            dV[j*p+k, c] += U_init[i*m+j, c] * W_init[k*n+i, c] * e
                            dW[k*n+i, c] += U_init[i*m+j, c] * V_init[j*p+k, c] * e
            
            # Update
            U_init -= lr * dU
            V_init -= lr * dV
            W_init -= lr * dW
            
            # Check convergence
            max_err = np.max(np.abs(T - 1.0))
            if max_err < 1e-9:
                return {"U": U_init.tolist(), "V": V_init.tolist(), "W": W_init.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Start with the known rank-44 decomposition
    result_44 = _strassen_44()
    ok, err = _verify_fast(result_44["U"], result_44["V"], result_44["W"], n, m, p)
    
    if not ok:
        return _schoolbook()
    
    best_result = result_44
    best_rank = 44
    
    # Try to reduce to rank 43
    if time.time() < deadline - 20:
        reduced = _try_reduce_to_rank43(
            result_44["U"], result_44["V"], result_44["W"], n, m, p, deadline - 10
        )
        if reduced is not None:
            r = len(reduced["U"][0])
            if r < best_rank:
                best_rank = r
                best_result = reduced
    
    # Try gradient-based refinement for rank 43
    if time.time() < deadline - 10 and best_rank > 43:
        refined = _gradient_refine(
            result_44["U"], result_44["V"], result_44["W"], n, m, p, 43, deadline - 5
        )
        if refined is not None:
            r = len(refined["U"][0])
            if r < best_rank:
                best_rank = r
                best_result = refined
    
    return best_result
