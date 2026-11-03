import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U,V,W decompose the matmul tensor exactly."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Check all target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :]))
                if abs(val - 1.0) > tol:
                    return False
    
    # Check that non-target entries are 0
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_set.add((a, b, c))
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in target_set:
                    val = np.dot(U_arr[a, :], np.multiply(V_arr[b, :], W_arr[c, :]))
                    if abs(val) > tol:
                        return False
    return True

def _construct_rank_36(n, m, p):
    """Schoolbook rank-36 decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _find_block_decomposition(n, m, p_block, max_iters=5000, lr=0.05, n_restarts=10, seed=42):
    """
    Find a low-rank decomposition of the n x m x p_block matmul tensor.
    Returns (U, V, W, rank) or None.
    """
    np.random.seed(seed)
    
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p_block):
                a = i * m + j
                b = j * p_block + k
                c = k * n + i
                triples.append((a, b, c))
    
    target_set = set(triples)
    
    best = None
    best_loss = float('inf')
    
    for attempt in range(n_restarts):
        if _time_left() < 10:
            break
        
        # Try rank from 3 up
        for r in range(3, 7):
            if _time_left() < 5:
                break
            
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p_block, r) * 0.5
            W = np.random.randn(p_block * n, r) * 0.5
            
            current_loss = float('inf')
            patience = 0
            max_patience = 100
            
            for it in range(max_iters):
                if _time_left() < 5:
                    break
                
                loss = 0.0
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                # Target entries should be 1
                for (a, b, c) in triples:
                    current = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                    error = current - 1.0
                    loss += error * error
                    for rr in range(r):
                        dU = error * V[b, rr] * W[c, rr]
                        dV = error * U[a, rr] * W[c, rr]
                        dW = error * U[a, rr] * V[b, rr]
                        grad_U[a, rr] += dU
                        grad_V[b, rr] += dV
                        grad_W[c, rr] += dW
                
                # Non-target entries should be 0
                for a in range(n * m):
                    for b in range(m * p_block):
                        for c in range(p_block * n):
                            if (a, b, c) not in target_set:
                                current = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                                loss += current * current
                                for rr in range(r):
                                    dU = current * V[b, rr] * W[c, rr]
                                    dV = current * U[a, rr] * W[c, rr]
                                    dW = current * U[a, rr] * V[b, rr]
                                    grad_U[a, rr] += dU
                                    grad_V[b, rr] += dV
                                    grad_W[c, rr] += dW
                
                grad_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if grad_norm > 1e-12:
                    step = lr / (1.0 + it * 0.01)
                    U -= step * grad_U
                    V -= step * grad_V
                    W -= step * grad_W
                
                if loss < current_loss * 0.99:
                    current_loss = loss
                    patience = 0
                else:
                    patience += 1
                
                if loss < 1e-14:
                    break
                if patience > max_patience:
                    break
            
            # Verify
            def _verify_block(Ub, Vb, Wb, n, m, pb, tol=1e-9):
                for i in range(n):
                    for j in range(m):
                        for k in range(pb):
                            a = i * m + j
                            b = j * pb + k
                            c = k * n + i
                            val = np.dot(Ub[a, :], np.multiply(Vb[b, :], Wb[c, :]))
                            if abs(val - 1.0) > tol:
                                return False
                ts = set()
                for i in range(n):
                    for j in range(m):
                        for k in range(pb):
                            ts.add((i*m+j, j*pb+k, k*n+i))
                for a in range(n*m):
                    for b in range(m*pb):
                        for c in range(pb*n):
                            if (a,b,c) not in ts:
                                val = np.dot(Ub[a, :], np.multiply(Vb[b, :], Wb[c, :]))
                                if abs(val) > tol:
                                    return False
                return True
            
            if _verify_block(U, V, W, n, m, p_block):
                return (U, V, W, r)
            
            if loss < best_loss:
                best_loss = loss
                best = (U.copy(), V.copy(), W.copy(), r)
    
    return best

def _construct_block_decomposition(n, m, p):
    """
    Construct rank-18 decomposition by partitioning into 3 blocks of 2 columns each,
    using a rank-6 decomposition of the 2x3x2 matmul tensor per block.
    """
    p_block = 2
    n_blocks = p // p_block  # 3 blocks
    
    # Find decomposition for 2x3x2 matmul
    block_result = _find_block_decomposition(n, m, p_block, max_iters=3000, lr=0.05, n_restarts=8, seed=42)
    
    if block_result is None:
        # Fallback: use schoolbook for each block (rank 6 per block)
        U_block = np.zeros((n * m, n * m * p_block))
        V_block = np.zeros((m * p_block, n * m * p_block))
        W_block = np.zeros((p_block * n, n * m * p_block))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p_block):
                    U_block[i * m + j, col] = 1
                    V_block[j * p_block + k, col] = 1
                    W_block[k * n + i, col] = 1
                    col += 1
        block_result = (U_block, V_block, W_block, n * m * p_block)
    
    U_b, V_b, W_b, r_block = block_result
    
    # Assemble full decomposition
    total_rank = r_block * n_blocks
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    for b in range(n_blocks):
        col_offset = b * r_block
        # U is the same for all blocks
        U[:, col_offset:col_offset + r_block] = U_b
        # V: block b uses columns 2b, 2b+1 of B, so V rows j*p + 2b + k_local
        for j in range(m):
            for k_local in range(p_block):
                v_local_row = j * p_block + k_local
                v_global_row = j * p + 2 * b + k_local
                V[v_global_row, col_offset:col_offset + r_block] = V_b[v_local_row, :]
        # W: block b uses k = 2b + k_local, so W rows (2b+k_local)*n + i
        for k_local in range(p_block):
            for i in range(n):
                w_local_row = k_local * n + i
                w_global_row = (2 * b + k_local) * n + i
                W[w_global_row, col_offset:col_offset + r_block] = W_b[w_local_row, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = 2, 3, 6
    
    # Try block decomposition approach (target rank 18 or lower)
    if _time_left() > 30:
        result = _construct_block_decomposition(n, m, p)
        if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
            return result
    
    # Fallback to schoolbook rank 36
    return _construct_rank_36(n, m, p)
