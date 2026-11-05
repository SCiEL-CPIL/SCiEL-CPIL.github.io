"""
Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.

We use a randomized search over candidate rank values, starting from a lower
bound and increasing until we find a valid decomposition. The search uses
random initialization and gradient-free local search (simulated annealing-like)
to find a rank-r decomposition if one exists.

Key insight: For matrix multiplication C = A*B where A is n x m and B is m x p,
the tensor is T[i,j,k] = sum_l A[i,l]*B[l,k]. We need U, V, W of shapes
(n*m, r), (m*p, r), (p*n, r) such that for all i,j,k:
  sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,j,k} (the tensor entry)

Wait, that's not right. Let me reconsider.

The matrix multiplication tensor T has entries T[i,j,k,l] = A[i,j]*B[j,k] -> C[i,k]
Actually, the standard formulation: we want to compute C = A * B where:
- A is n x m (indices i=0..n-1, j=0..m-1)  
- B is m x p (indices j=0..m-1, k=0..p-1)
- C is n x p (indices i=0..n-1, k=0..p-1)

C[i,k] = sum_j A[i,j] * B[j,k]

The tensor of bilinear forms: T is a 3-tensor with indices (i, j, k) where
T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].

Actually, the standard tensor decomposition for matrix multiplication:
We have a tensor T with dimensions (n*m) x (m*p) x (n*p) where
T[(i,j), (j',k), (i',k')] = 1 if i=i' and j=j' and k=k'... 

No wait. Let me think more carefully.

The matrix multiplication can be viewed as: given inputs A (n*m entries) and B (m*p entries),
compute outputs C (n*p entries). Each output C[i,k] = sum_j A[i,j]*B[j,k].

So the tensor T has shape (n*m, m*p, n*p) and T[a, b, c] where:
- a = i*m + j (input from A)
- b = j'*p + k (input from B)  
- c = i'*n + k' (output of C)

T[a, b, c] = 1 if a corresponds to (i,j), b corresponds to (j',k), c corresponds to (i',k')
AND i = i' AND j = j' AND k = k'.

Wait, that would give T[a,b,c] = 1 only when all three indices match exactly, which gives
the schoolbook rank of n*m*p.

Actually no. The correct tensor for matrix multiplication:
T[i*m+j, j'*p+k, i'*n+k'] = 1 if j = j' (the shared index matches) AND i = i' AND k = k'.

Hmm, let me reconsider. The bilinear map is:
(A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]

This is a trilinear form. The tensor T has entries T[a,b,c] where:
- a indexes A: a = i*m + j
- b indexes B: b = j'*p + k  
- c indexes C: c = i'*n + k'

T[a,b,c] = 1 if the product A[a] * B[b] contributes to C[c], i.e.,
if a = i*m+j, b = j'*p+k, c = i'*n+k' and i=i', j=j', k=k'.

So T[a,b,c] = 1 iff (i from a) = (i' from c) AND (j from a) = (j' from b) AND (k from b) = (k' from c).

This is the standard matrix multiplication tensor. The rank of this tensor is the minimum number
of rank-1 tensors needed to sum to T.

A rank-1 tensor is u * v * w where u is in R^{n*m}, v in R^{m*p}, w in R^{n*p}.
The decomposition T = sum_c U[:,c] * V[:,c] * W[:,c] means:
T[a,b,c'] = sum_c U[a,c] * V[b,c] * W[c',c]

For our problem: n=2, m=4, p=5.
- U shape: (8, r)
- V shape: (20, r) 
- W shape: (10, r)

The known lower bound for rank of matrix multiplication is at least n*p for square matrices,
but for rectangular it's more complex. For 2x4 * 4x5, the schoolbook rank is 2*4*5 = 40.

Let me try a randomized search approach. We'll try to find a decomposition of rank r for
decreasing r from some upper bound, using random initialization and local search.

Given the 150s time limit, we need to be efficient. Let's use numpy for fast operations.
"""

import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]
_START_TIME = time.time()
_TIME_LIMIT = 140  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, n*p)."""
    T = np.zeros((n * m, m * p, n * p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j       # index into A
                b = j * p + k       # index into B
                c = i * n + k       # index into C
                T[a, b, c] = 1.0
    return T

def _decompose_tensor(T, rank, n, m, p, max_iters=5000, seed=42):
    """
    Try to find a rank-`rank` decomposition of tensor T.
    Returns (U, V, W) as numpy arrays or None if not found.
    Uses random initialization + gradient descent on the loss:
      L = sum_{a,b,c} (T[a,b,c] - sum_k U[a,k]*V[b,k]*W[c,k])^2
    """
    rng = np.random.RandomState(seed)
    nm, mp, np_ = T.shape
    
    # Random initialization
    U = rng.randn(nm, rank) * 0.5
    V = rng.randn(mp, rank) * 0.5
    W = rng.randn(np_, rank) * 0.5
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    lr = 0.01
    decay = 0.999
    
    for it in range(max_iters):
        if _elapsed() > _TIME_LIMIT - 5:
            break
        
        # Compute current decomposition
        # C[a,b] = sum_k U[a,k] * V[b,k]  (contract over k)
        # Then L = sum_{a,b,c} (T[a,b,c] - C[a,b]*W[c,k]...) 
        # Actually we need: pred[a,b,c] = sum_k U[a,k]*V[b,k]*W[c,k]
        # This is expensive to compute fully. Let's use a different approach.
        
        # Compute M = U @ V.T  (nm x mp)
        M = U @ V.T  # (nm, mp)
        
        # For each c, the predicted value at (a,b,c) is sum_k U[a,k]*V[b,k]*W[c,k]
        # = (M * W[c,:].T) ... no.
        # pred[a,b,c] = sum_k U[a,k] * V[b,k] * W[c,k]
        # Let's compute this as: for each c, pred[:,:,c] = M * W[c].T (element-wise)
        # Actually: pred[a,b,c] = sum_k (U[a,k] * W[c,k]) * V[b,k]
        # = (U * W.T) @ V.T ... let's think.
        # U is (nm, rank), W is (np_, rank)
        # U * W.T is (nm, np_, rank) element-wise? No.
        # Let's do: temp[a, c, k] = U[a,k] * W[c,k], then pred[a,b,c] = sum_k temp[a,c,k] * V[b,k]
        # = (temp @ V.T)[a,c,b]... 
        # 
        # Alternatively: pred = np.einsum('ak,bk,ck->abc', U, V, W)
        # But that's (nm, mp, np_) which for our sizes is 8*20*10 = 1600, very manageable!
        
        pred = np.einsum('ak,bk,ck->abc', U, V, W)
        loss = np.sum((T - pred) ** 2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-12:
            break
        
        # Gradients
        # dL/dU[a,k] = -2 * sum_{b,c} T_residual[a,b,c] * V[b,k] * W[c,k]
        # where T_residual = T - pred
        resid = T - pred  # (nm, mp, np_)
        
        dU = -2.0 * np.einsum('abc,bk,ck->ak', resid, V, W)
        dV = -2.0 * np.einsum('abc,ak,ck->bk', resid, U, W)
        dW = -2.0 * np.einsum('abc,ak,bk->ck', resid, U, V)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        lr *= decay
    
    # Check if we found a valid decomposition
    pred_final = np.einsum('ak,bk,ck->abc', best_U, best_V, best_W)
    final_loss = np.sum((T - pred_final) ** 2)
    
    if final_loss < 1e-10:
        # Round to nearest integers or half-integers
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        # Verify rounded version
        pred_rounded = np.einsum('ak,bk,ck->abc', U_rounded, V_rounded, W_rounded)
        rounded_loss = np.sum((T - pred_rounded) ** 2)
        
        if rounded_loss < 1e-10:
            return U_rounded, V_rounded, W_rounded
        else:
            # Try with original (unrounded) values
            return best_U, best_V, best_W
    
    return None

def _try_decomposition(n, m, p, rank, max_seeds=5, max_iters=3000):
    """Try multiple random seeds for a given rank."""
    T = _build_tensor(n, m, p)
    
    for seed in range(max_seeds):
        if _elapsed() > _TIME_LIMIT - 5:
            return None
        result = _decompose_tensor(T, rank, n, m, p, max_iters=max_iters, seed=seed)
        if result is not None:
            return result
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Lower bound: at least max(n*m, m*p, n*p) / something... 
    # Actually a simple lower bound is max(n*m, m*p, n*p) but that's not tight.
    # For matrix multiplication, rank >= max(n, m, p) * min(n, m, p) is not right either.
    # Let's just try ranks from a reasonable lower bound upward.
    
    # Theoretical lower bounds: rank >= n*p (for each fixed j, we need to compute n*p products)
    # Actually no, that's not a valid lower bound.
    # A valid lower bound: rank >= max(n*m, m*p, n*p) / max(n, m, p) ... not standard.
    # Let's use: rank >= n*p = 10 as a very loose lower bound, and go up from there.
    
    # Actually, for the schoolbook, rank = n*m*p = 40.
    # Let's try ranks from 10 upward, but we need to be careful about time.
    
    # Strategy: try ranks from low to high, with limited attempts per rank.
    # If we can't find a decomposition quickly, move to next rank.
    
    # Known result: rank of 2x2 * 2x2 matrix mult is 7 (Strassen).
    # For 2x4 * 4x5, we don't know the exact rank, but it should be less than 40.
    
    # Let's try a range of ranks. Start from a lower bound.
    # A simple lower bound: rank >= n*p = 10 (since we need to produce n*p independent outputs)
    # Better: rank >= max(n*m, m*p, n*p) / min(n, m, p)... 
    
    # Let's just search from rank 10 to 40, trying a few seeds each.
    # But gradient descent might be slow. Let's be strategic.
    
    # First, let's try to see if we can find something with random search at various ranks.
    
    T = _build_tensor(n, m, p)
    nm, mp, np_ = T.shape
    
    # Try ranks from lower bound to schoolbook
    for rank in range(10, 41):
        if _elapsed() > _TIME_LIMIT - 10:
            break
        
        # Limit iterations based on remaining time
        remaining = _TIME_LIMIT - _elapsed() - 5
        if remaining < 5:
            break
        
        # Try a few seeds
        for seed in range(3):
            if _elapsed() > _TIME_LIMIT - 5:
                break
            
            rng = np.random.RandomState(seed + rank * 100)
            U = rng.randn(nm, rank) * 0.3
            V = rng.randn(mp, rank) * 0.3
            W = rng.randn(np_, rank) * 0.3
            
            lr = 0.005
            decay = 0.998
            best_loss = float('inf')
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            max_iters = min(2000, int(remaining * 50))  # adaptive
            
            for it in range(max_iters):
                if _elapsed() > _TIME_LIMIT - 3:
                    break
                    
                pred = np.einsum('ak,bk,ck->abc', U, V, W)
                resid = T - pred
                loss = np.sum(resid ** 2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if best_loss < 1e-12:
                    break
                
                dU = -2.0 * np.einsum('abc,bk,ck->ak', resid, V, W)
                dV = -2.0 * np.einsum('abc,ak,ck->bk', resid, U, W)
                dW = -2.0 * np.einsum('abc,ak,bk->ck', resid, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                lr *= decay
            
            pred_final = np.einsum('ak,bk,ck->abc', best_U, best_V, best_W)
            final_loss = np.sum((T - pred_final) ** 2)
            
            if final_loss < 1e-10:
                # Found a decomposition! Try to round
                U_r = np.round(best_U * 2) / 2
                V_r = np.round(best_V * 2) / 2
                W_r = np.round(best_W * 2) / 2
                pred_r = np.einsum('ak,bk,ck->abc', U_r, V_r, W_r)
                loss_r = np.sum((T - pred_r) ** 2)
                
                if loss_r < 1e-10:
                    U_out = U_r.tolist()
                    V_out = V_r.tolist()
                    W_out = W_r.tolist()
                else:
                    U_out = best_U.tolist()
                    V_out = best_V.tolist()
                    W_out = best_W.tolist()
                
                return {"U": U_out, "V": V_out, "W": W_out}
    
    # If we didn't find anything better, return schoolbook
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
