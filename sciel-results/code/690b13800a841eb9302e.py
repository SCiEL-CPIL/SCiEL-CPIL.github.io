"""Matrix multiplication 3x4x8 rank-minimal decomposition search."""
import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """Verify that U, V, W form a valid tensor decomposition."""
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The target tensor is T[i,j,k] = 1 for all i,j,k
    rank = U.shape[1]
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_reshaped = np.array(U).reshape(n, m, rank)
    V_reshaped = np.array(V).reshape(m, p, rank)
    W_reshaped = np.array(W).reshape(p, n, rank)
    
    # Compute the reconstructed tensor
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # This is a tensor contraction
    T_recon = np.zeros((n, m, p))
    for c in range(rank):
        T_recon += U_reshaped[:,:,c][:, :, np.newaxis] * V_reshaped[:,:,c][np.newaxis, :, :] * W_reshaped[:,:,c][np.newaxis, np.newaxis, :]
    
    # The target is all ones
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _verify_fast(U, V, W, n=3, m=4, p=8, tol=1e-6):
    """Faster verification using einsum."""
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    T_recon = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    if not np.allclose(T_recon, 1.0, atol=tol):
        return False
    return True

def _round_to_halfint(arr, tol=1e-4):
    """Round array entries to nearest half-integer (0.5 increments)."""
    rounded = np.round(arr * 2) / 2
    # Check if rounding was close enough
    if np.max(np.abs(arr - rounded)) > tol:
        return None
    return rounded

def _try_rank_fixed(rank, n=3, m=4, p=8, max_iter=5000, seed=42):
    """Try to find a decomposition of given rank using iterative optimization."""
    np.random.seed(seed)
    
    # Initialize U, V, W randomly
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    lr = 0.01
    for it in range(max_iter):
        # Compute T_recon
        T_recon = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        T_target = np.ones((n, m, p))
        
        # Loss
        loss = np.sum((T_recon - T_target) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-10:
            break
        
        # Compute gradients
        # dL/dU[i,j,c] = 2 * (T_recon[i,j,k] - 1) * V[j,k,c] * W[k,i,c] summed over k
        # dL/dV[j,k,c] = 2 * (T_recon[i,j,k] - 1) * U[i,j,c] * W[k,i,c] summed over i
        # dL/dW[k,i,c] = 2 * (T_recon[i,j,k] - 1) * U[i,j,c] * V[j,k,c] summed over j
        
        diff = T_recon - T_target  # (n, m, p)
        
        # grad_U[i,j,c] = 2 * sum_k diff[i,j,k] * V[j,k,c] * W[k,i,c]
        grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', diff, V_r, W_r)
        # grad_V[j,k,c] = 2 * sum_i diff[i,j,k] * U[i,j,c] * W[k,i,c]
        grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', diff, U_r, W_r)
        # grad_W[k,i,c] = 2 * sum_j diff[i,j,k] * U[i,j,c] * V[j,k,c]
        grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', diff, U_r, V_r)
        
        # Update
        U_r -= lr * grad_U
        V_r -= lr * grad_V
        W_r -= lr * grad_W
        
        # Decay learning rate
        if it % 500 == 499:
            lr *= 0.9
    
    # Check if we found a valid solution
    U_final = best_U.reshape(n, m, rank)
    V_final = best_V.reshape(m, p, rank)
    W_final = best_W.reshape(p, n, rank)
    T_final = np.einsum('ijc,jkc,kic->ijk', U_final, V_final, W_final)
    
    if np.allclose(T_final, 1.0, atol=1e-5):
        # Try to round to half-integers
        U_rounded = _round_to_halfint(best_U)
        V_rounded = _round_to_halfint(best_V)
        W_rounded = _round_to_halfint(best_W)
        
        if U_rounded is not None and V_rounded is not None and W_rounded is not None:
            if _verify_fast(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        # Return as-is if valid
        if _verify_fast(best_U.tolist(), best_V.tolist(), best_W.tolist()):
            return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def _strassen_block_decomposition(n=3, m=4, p=8):
    """
    Try a block-based decomposition.
    Split A (3x4) into blocks: A1 (3x2), A2 (3x2)
    Split B (4x8) into blocks: B1 (2x8), B2 (2x8)
    
    C = A1*B1 + A2*B2
    
    For each 3x2 * 2x8 product, use a decomposition.
    The tensor rank of 3x2*2x8 is at most 3*2*8=48, but can be less.
    
    Actually, let's think about this differently.
    We can use the fact that matrix multiplication C = A*B where A is n×m and B is m×p.
    
    Let's try a simpler approach: use known small decompositions.
    For 2x2x2, Strassen gives rank 7.
    
    Let me try a different strategy: enumerate possible low-rank decompositions
    using a structured approach.
    """
    pass

def _greedy_decomposition(n=3, m=4, p=8, max_rank=50):
    """
    Greedy approach: build the decomposition one term at a time.
    Each term is a rank-1 tensor U_c ⊗ V_c ⊗ W_c.
    We want sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1 for all i,j,k.
    """
    U = np.zeros((n * m, max_rank))
    V = np.zeros((m * p, max_rank))
    W = np.zeros((p * n, max_rank))
    
    U_r = U.reshape(n, m, max_rank)
    V_r = V.reshape(m, p, max_rank)
    W_r = W.reshape(p, n, max_rank)
    
    # Target tensor
    T_target = np.ones((n, m, p))
    T_recon = np.zeros((n, m, p))
    
    for c in range(max_rank):
        diff = T_target - T_recon
        
        # Find the best rank-1 approximation to diff
        # diff[i,j,k] ≈ a[i,j] * b[j,k] * c[k,i]
        # This is a 3-way tensor decomposition problem.
        # Use ALS (Alternating Least Squares) for one iteration.
        
        # Initialize
        a = np.random.randn(n, m) * 0.1
        b = np.random.randn(m, p) * 0.1
        cc = np.random.randn(p, n) * 0.1
        
        for _ in range(20):
            # Update a: a[i,j] = sum_{k} diff[i,j,k] * b[j,k] * cc[k,i] / sum_{k} (b[j,k] * cc[k,i])^2
            denom_a = np.einsum('jk,ki->ij', b**2, cc**2)
            num_a = np.einsum('ijk,jk,ki->ij', diff, b, cc)
            a = num_a / (denom_a + 1e-8)
            
            # Update b: b[j,k] = sum_{i} diff[i,j,k] * a[i,j] * cc[k,i] / sum_{i} (a[i,j] * cc[k,i])^2
            denom_b = np.einsum('ij,ki->jk', a**2, cc**2)
            num_b = np.einsum('ijk,ij,ki->jk', diff, a, cc)
            b = num_b / (denom_b + 1e-8)
            
            # Update cc: cc[k,i] = sum_{j} diff[i,j,k] * a[i,j] * b[j,k] / sum_{j} (a[i,j] * b[j,k])^2
            denom_c = np.einsum('ij,jk->ik', a**2, b**2)
            num_c = np.einsum('ijk,ij,jk->ik', diff, a, b)
            cc = num_c / (denom_c + 1e-8)
        
        # Set the c-th column
        U_r[:, :, c] = a
        V_r[:, :, c] = b
        W_r[:, :, c] = cc
        
        # Update T_recon
        T_recon += a[:, :, np.newaxis] * b[np.newaxis, :, :] * cc[np.newaxis, np.newaxis, :]
        
        # Check if we're done
        if np.allclose(T_recon, T_target, atol=1e-6):
            # Trim to rank c+1
            U_trim = U[:, :c+1]
            V_trim = V[:, :c+1]
            W_trim = W[:, :c+1]
            
            # Try to round to half-integers
            U_rounded = _round_to_halfint(U_trim, tol=1e-3)
            V_rounded = _round_to_halfint(V_trim, tol=1e-3)
            W_rounded = _round_to_halfint(W_trim, tol=1e-3)
            
            if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                if _verify_fast(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), c+1
            
            if _verify_fast(U_trim.tolist(), V_trim.tolist(), W_trim.tolist()):
                return U_trim.tolist(), V_trim.tolist(), W_trim.tolist(), c+1
    
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin
    
    best_result = None
    best_rank = float('inf')
    
    # Start with schoolbook as fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = rank_schoolbook
    
    # Try greedy decomposition
    if time.time() < deadline:
        result = _greedy_decomposition(n, m, p, max_rank=40)
        if result is not None:
            U_g, V_g, W_g, rank_g = result
            if rank_g < best_rank:
                best_result = {"U": U_g, "V": V_g, "W": W_g}
                best_rank = rank_g
    
    # Try fixed-rank optimization for various ranks
    for rank in [20, 24, 28, 32, 36]:
        if time.time() > deadline - 10:
            break
        if rank >= best_rank:
            continue
        for seed in range(3):
            if time.time() > deadline - 10:
                break
            result = _try_rank_fixed(rank, n, m, p, max_iter=2000, seed=seed)
            if result is not None:
                U_t, V_t, W_t = result
                if rank < best_rank:
                    best_result = {"U": U_t, "V": V_t, "W": W_t}
                    best_rank = rank
                    break
    
    return best_result
# EVOLVE-BLOCK-END
