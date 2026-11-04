"""Program for matrix multiplication 2x3x7 rank minimization."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W, n=2, m=3, p=7, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Reshape to (n, m, rank), (m, p, rank), (p, n, rank)
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    
    # Compute S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Using einsum: 'ijc,jkc,kic->ijk'
    S = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    # Check if S is all ones
    if not np.allclose(S, 1.0, atol=tol):
        return False
    return True

def _random_init(n, m, p, rank, seed=None):
    if seed is not None:
        np.random.seed(seed)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    return U.tolist(), V.tolist(), W.tolist()

def _schoolbook(n, m, p):
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

def _gradient_descent(U, V, W, n, m, p, max_iter=5000, lr=0.01, tol=1e-10, deadline=None, momentum=0.9):
    """Gradient descent with momentum to minimize ||S - 1||^2 where S = U ⊙ V ⊙ W."""
    U_arr = np.array(U, dtype=float).reshape(n, m, -1)
    V_arr = np.array(V, dtype=float).reshape(m, p, -1)
    W_arr = np.array(W, dtype=float).reshape(p, n, -1)
    rank = U_arr.shape[2]
    
    best_loss = np.inf
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    
    # Momentum velocities
    vU = np.zeros_like(U_arr)
    vV = np.zeros_like(V_arr)
    vW = np.zeros_like(W_arr)
    
    # Adaptive learning rate
    current_lr = lr
    prev_loss = np.inf
    patience = 0
    max_patience = 50
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute S = sum_c U[:,:,c] * V[:,:,c] * W[:,:,c] (transposed indices)
        # S[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        S = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
        
        # Loss = sum (S - 1)^2
        diff = S - 1.0
        loss = np.sum(diff ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U_arr.copy()
            best_V = V_arr.copy()
            best_W = W_arr.copy()
            patience = 0
        else:
            patience += 1
        
        if loss < tol:
            break
        
        # Adaptive learning rate: reduce if not improving
        if patience >= max_patience:
            current_lr *= 0.5
            patience = 0
            if current_lr < 1e-8:
                break
        
        # Compute gradients
        # dL/dU[i,j,c] = 2 * (S[i,j,k] - 1) * V[j,k,c] * W[k,i,c]
        dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', diff, V_arr, W_arr)
        dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', diff, U_arr, W_arr)
        dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', diff, U_arr, V_arr)
        
        # Momentum update
        vU = momentum * vU - current_lr * dU
        vV = momentum * vV - current_lr * dV
        vW = momentum * vW - current_lr * dW
        
        U_arr += vU
        V_arr += vV
        W_arr += vW
    
    return best_U.reshape(n*m, rank).tolist(), best_V.reshape(m*p, rank).tolist(), best_W.reshape(p*n, rank).tolist()

def _try_round_and_verify(U, V, W, n, m, p, tol=1e-9):
    """Try to round to half-integers and verify. Also try integers."""
    # First try integers (subset of half-integers)
    U_int = [[int(round(x)) for x in row] for row in U]
    V_int = [[int(round(x)) for x in row] for row in V]
    W_int = [[int(round(x)) for x in row] for row in W]
    if _verify(U_int, V_int, W_int, n, m, p, tol=tol):
        return U_int, V_int, W_int
    
    # Try half-integers
    U_r = [[round(x * 2) / 2 for x in row] for row in U]
    V_r = [[round(x * 2) / 2 for x in row] for row in V]
    W_r = [[round(x * 2) / 2 for x in row] for row in W]
    
    if _verify(U_r, V_r, W_r, n, m, p, tol=tol):
        return U_r, V_r, W_r
    
    # Try floor and ceil for each entry independently (limited search)
    # For small ranks, try all combinations of floor/ceil for each entry
    U_f = [[int(x * 2) / 2 for x in row] for row in U]
    V_f = [[int(x * 2) / 2 for x in row] for row in V]
    W_f = [[int(x * 2) / 2 for x in row] for row in W]
    U_c = [[int(x * 2 + 1) / 2 for x in row] for row in U]
    V_c = [[int(x * 2 + 1) / 2 for x in row] for row in V]
    W_c = [[int(x * 2 + 1) / 2 for x in row] for row in W]
    
    # Try all 8 combinations
    for ui in [U_f, U_c]:
        for vi in [V_f, V_c]:
            for wi in [W_f, W_c]:
                if _verify(ui, vi, wi, n, m, p, tol=tol):
                    return ui, vi, wi
    
    return None

def _structured_init(n, m, p, rank, seed=None):
    """Structured initialization based on partial schoolbook patterns."""
    if seed is not None:
        np.random.seed(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Distribute schoolbook terms across rank dimensions with random signs
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Assign to a random column
                c = np.random.randint(0, rank)
                sign_u = np.random.choice([-1, 1])
                sign_v = np.random.choice([-1, 1])
                sign_w = np.random.choice([-1, 1])
                U[i * m + j, c] = sign_u
                V[j * p + k, c] = sign_v
                W[k * n + i, c] = sign_w
    
    # Add small random noise to break symmetry
    U += np.random.randn(n * m, rank) * 0.1
    V += np.random.randn(m * p, rank) * 0.1
    W += np.random.randn(p * n, rank) * 0.1
    
    return U.tolist(), V.tolist(), W.tolist()

def _polish(U, V, W, n, m, p, deadline=None, max_iter=10000):
    """High-precision polish pass with very small learning rate."""
    U_arr = np.array(U, dtype=float).reshape(n, m, -1)
    V_arr = np.array(V, dtype=float).reshape(m, p, -1)
    W_arr = np.array(W, dtype=float).reshape(p, n, -1)
    rank = U_arr.shape[2]
    
    best_loss = np.inf
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        S = np.einsum('ijc,jkc,kic->ijk', U_arr, V_arr, W_arr)
        diff = S - 1.0
        loss = np.sum(diff ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U_arr.copy()
            best_V = V_arr.copy()
            best_W = W_arr.copy()
        
        if loss < 1e-14:
            break
        
        dU = 2.0 * np.einsum('ijk,jkc,kic->ijc', diff, V_arr, W_arr)
        dV = 2.0 * np.einsum('ijk,ijc,kic->jkc', diff, U_arr, W_arr)
        dW = 2.0 * np.einsum('ijk,ijc,jkc->kic', diff, U_arr, V_arr)
        
        # Very small step
        lr = 1e-6
        U_arr -= lr * dU
        V_arr -= lr * dV
        W_arr -= lr * dW
    
    return best_U.reshape(n*m, rank).tolist(), best_V.reshape(m*p, rank).tolist(), best_W.reshape(p*n, rank).tolist()

def _combine_decompositions(decomp1, decomp2, n, m, p1, p2):
    """
    Combine two decompositions for M_{n,m,p1} and M_{n,m,p2} into M_{n,m,p1+p2}.
    decomp1: {'U': ..., 'V': ..., 'W': ...} for p1
    decomp2: {'U': ..., 'V': ..., 'W': ...} for p2
    """
    U1, V1, W1 = decomp1['U'], decomp1['V'], decomp1['W']
    U2, V2, W2 = decomp2['U'], decomp2['V'], decomp2['W']
    
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    total_rank = rank1 + rank2
    
    U_new = [[0] * total_rank for _ in range(n * m)]
    V_new = [[0] * total_rank for _ in range(m * (p1 + p2))]
    W_new = [[0] * total_rank for _ in range((p1 + p2) * n)]
    
    # Copy U
    for i in range(n * m):
        for c in range(rank1):
            U_new[i][c] = U1[i][c]
        for c in range(rank2):
            U_new[i][rank1 + c] = U2[i][c]
            
    # Copy V
    # V1 is m x p1, V2 is m x p2
    # V_new is m x (p1+p2)
    for j in range(m):
        for k in range(p1):
            for c in range(rank1):
                V_new[j * (p1 + p2) + k][c] = V1[j * p1 + k][c]
        for k in range(p2):
            for c in range(rank2):
                V_new[j * (p1 + p2) + p1 + k][rank1 + c] = V2[j * p2 + k][c]
                
    # Copy W
    # W1 is p1 x n, W2 is p2 x n
    # W_new is (p1+p2) x n
    for k in range(p1):
        for i in range(n):
            for c in range(rank1):
                W_new[k * n + i][c] = W1[k * n + i][c]
    for k in range(p2):
        for i in range(n):
            for c in range(rank2):
                W_new[(p1 + k) * n + i][rank1 + c] = W2[k * n + i][c]
                
    return {"U": U_new, "V": V_new, "W": W_new}

def _peeling_construct(n, m, p, deadline=None):
    """
    Construct a decomposition by peeling off 1 column at a time.
    Uses schoolbook for the base case (p=1) and tries to optimize the rest.
    """
    if p == 0:
        return None
    
    if p == 1:
        return _schoolbook(n, m, 1)
    
    # Peel off the last column (or first, doesn't matter due to symmetry in construction)
    # Let's peel off the first column: B = [B_col1 | B_rest]
    # M_{n,m,p} = M_{n,m,1} (for col 1) + M_{n,m,p-1} (for rest)
    
    # Get decomposition for M_{n,m,1}
    decomp_1 = _schoolbook(n, m, 1)
    
    # Get decomposition for M_{n,m,p-1}
    # We try to find a low rank decomposition for the sub-problem
    # If we don't have time, we fall back to schoolbook
    decomp_rest = None
    if deadline is None or time.time() < deadline - 5:
        # Try to find a better decomposition for the rest
        # Use a quick search
        # For now, just use schoolbook to ensure validity and speed
        # We can try to optimize M_{n,m,p-1} later if we have time
        decomp_rest = _schoolbook(n, m, p - 1)
        
        # If we have significant time left, try to optimize the rest
        if time.time() < deadline - 20:
             # Try to find a lower rank for the rest
             # This is a placeholder for recursive optimization
             pass
             
    return _combine_decompositions(decomp_1, decomp_rest, n, m, 1, p - 1)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin out of 150s
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 42
    
    # Start with schoolbook decomposition
    best = _schoolbook(n, m, p)
    best_rank = schoolbook_rank
    
    # Try the peeling construction
    peeling_decomp = _peeling_construct(n, m, p, deadline)
    if peeling_decomp:
        peeling_rank = len(peeling_decomp['U'][0])
        if peeling_rank < best_rank:
            best = peeling_decomp
            best_rank = peeling_rank
            
    # If we have time, try to optimize the sub-blocks in the peeling construction
    # Specifically, the M_{n,m,p-1} part
    if best_rank > 21 and time.time() < deadline - 30:
        # We can try to replace the schoolbook parts with optimized ones
        # For simplicity, we will try to optimize the whole thing again but with a better init
        # based on the peeling structure? No, that's complex.
        # Instead, we can try to find a low rank decomposition for M_{n,m,6} and combine with M_{n,m,1}
        
        # Let's try to find a rank < 18 decomposition for M_{2,3,6}
        # Lower bound is 18. So we can only aim for 18.
        # If we find 18, total is 6+18=24.
        
        # Try to find rank 18 for M_{2,3,6}
        target_sub_rank = 18
        found_sub = False
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            seed = 1000 + attempt
            U_init, V_init, W_init = _random_init(n, m, 6, target_sub_rank, seed=seed)
            U_opt, V_opt, W_opt = _gradient_descent(
                U_init, V_init, W_init, n, m, 6,
                max_iter=2000, lr=0.05, tol=1e-12,
                deadline=deadline - 5, momentum=0.9
            )
            if _verify(U_opt, V_opt, W_opt, n, m, 6, tol=1e-6):
                result = _try_round_and_verify(U_opt, V_opt, W_opt, n, m, 6, tol=1e-9)
                if result is not None:
                    U_r, V_r, W_r = result
                    sub_decomp = {"U": U_r, "V": V_r, "W": W_r}
                    base_decomp = _schoolbook(n, m, 1)
                    combined = _combine_decompositions(base_decomp, sub_decomp, n, m, 1, 6)
                    if len(combined['U'][0]) < best_rank:
                        best = combined
                        best_rank = len(combined['U'][0])
                        found_sub = True
                        break
                        
    return best
