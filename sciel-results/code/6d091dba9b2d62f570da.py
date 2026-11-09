"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
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
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    nm = n * m
    mp = m * p
    pn = p * n
    expected = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m + j, j*p + k, k*n + i] = 1.0
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    max_err = np.max(np.abs(T - expected))
    return max_err < tol


def _slice_rank_lower_bound(n, m, p):
    """Compute the slice rank lower bound by fixing the third index.
    
    For each k in [0,p), fixing W's index to k*n+i for varying i gives a matrix
    whose rank provides a lower bound. The slice rank is the minimum over all
    such slices.
    """
    # The tensor T[a,b,c] = 1 if a%m==b//p, b%p==c//n, a//m==c%n, else 0
    # Fix c = k*n + i for a specific k, varying i.
    # For fixed k, the slice T[:, :, k*n + i] as a function of (a, b, i)
    # We want to view this as a matrix for each i, then find the max rank.
    
    # Actually, the slice rank lower bound: fix one mode, get a matrix, find its rank.
    # T[a, b, c] where a=i*m+j, b=j'*p+k', c=k''*n+i''
    # T is non-zero when j=j', k'=k'', i=i''
    
    # Fix c = k*n + i (so k''=k, i''=i)
    # T[a, b, k*n+i] = 1 if a%m == b//p, b%p == k, a//m == i
    # So for fixed k, i: T[i*m+j, j*p+k, k*n+i] = 1, all else 0
    # This is a rank-1 slice for each (k,i) pair... 
    
    # Let me think differently. The "slice rank" as described in the directive:
    # Fix the third index to generate a matrix and use its rank.
    
    # The tensor can be seen as a matrix M_{(a,b), c} or M_{a, (b,c)} etc.
    # The rank of the tensor is at least the rank of any such matrix unfolding.
    
    # Unfold along the third mode: M is (n*m * m*p) x (p*n)
    # M[a*mp + b, c] = T[a, b, c]
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    M = np.zeros((nm * mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                M[a * mp + b, c] = 1.0
    
    rank = np.linalg.matrix_rank(M)
    return rank


def _try_gradient_descent(n, m, p, rank, max_iters=500, lr_init=0.05, seed=42, time_budget=20.0):
    """Try to find a rank-r decomposition using gradient descent with adaptive learning rate."""
    start_time = time.time()
    rng = np.random.RandomState(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    expected = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m + j, j*p + k, k*n + i] = 1.0
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        
        U = rng.randn(nm, rank) * 0.1
        V = rng.randn(mp, rank) * 0.1
        W = rng.randn(pn, rank) * 0.1
        
        lr = lr_init
        for step in range(200):
            if time.time() - start_time > time_budget:
                break
            
            T = np.einsum('ar,br,cr->abc', U, V, W)
            err = T - expected
            loss = np.sum(err**2)
            
            if loss < best_err:
                best_err = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if loss < 1e-12:
                break
            
            dU = 2 * np.einsum('abc,br,cr->ar', err, V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', err, U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', err, U, V)
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm > 0:
                effective_lr = lr * 10 / (10 + grad_norm)
            else:
                effective_lr = lr
            
            U -= effective_lr * dU
            V -= effective_lr * dV
            W -= effective_lr * dW
            
            # Decay learning rate
            lr *= 0.999
    
    if best_err < 1e-9 and best_U is not None:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s wall-clock limit
    
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Start with schoolbook as baseline
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p  # 100
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Compute slice rank lower bound
    lower_bound = _slice_rank_lower_bound(n, m, p)
    # Theoretical lower bound is max(n*m, m*p, p*n) = 25
    # Slice rank gives a potentially tighter bound
    search_start = max(lower_bound, 25)
    
    # Try ranks from lower bound upward
    for rank in range(search_start, best_rank):
        if time.time() > deadline - 15:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            U, V, W, err = _try_gradient_descent(
                n, m, p, rank, 
                max_iters=300, lr_init=0.05, 
                seed=rank*1000+attempt*7,
                time_budget=15.0
            )
            
            if U is not None and err < 1e-9:
                if _verify(U, V, W, n, m, p):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            break
    
    # Convert to lists for the return interface
    U_list = [[float(x) for x in row] for row in np.array(best_U).tolist()]
    V_list = [[float(x) for x in row] for row in np.array(best_V).tolist()]
    W_list = [[float(x) for x in row] for row in np.array(best_W).tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
