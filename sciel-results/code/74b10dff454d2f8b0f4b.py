"""Matrix multiplication 3,4,5 - rank-minimal exact tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def _schoolbook():
    """Return the schoolbook decomposition (rank 60)."""
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, tol=1e-10):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = 1 for the multiplication tensor
    # Decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need to check that for each (i,j,k), the sum equals 1.
    # But actually, the matrix multiplication tensor has T[i,j,k] = 1 if the
    # product C[i,k] = sum_j A[i,j]*B[j,k], so the tensor is:
    # T[i,j,k] = 1 (it's the identity in the j-sum sense)
    # Actually the multiplication tensor M has entries M[i,j,k] = 1 for all i,j,k
    # because C[i,k] = sum_j A[i,j] B[j,k].
    # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k.
    
    # Efficient check using matrix operations
    # For each c, the outer product contributes U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # Reshape U to (n,m,rank), V to (m,p,rank), W to (p,n,rank)
    Ur = Ua.reshape(n, m, rank)
    Vr = Va.reshape(m, p, rank)
    Wr = Wa.reshape(p, n, rank)
    
    # For each (i,j,k): sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # This is equivalent to: for each (i,k), sum_j sum_c Ur[i,j,c]*Vr[j,k,c]*Wr[k,i,c]
    # = sum_c (Ur[:, :, c] @ Vr[:, :, c]) ... no that's not right.
    
    # Let's just do it directly but vectorized
    # Build the full tensor check
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                max_err = max(max_err, abs(s - 1.0))
    return max_err < tol

def _try_rank(rank, time_budget, rng):
    """Try to find a decomposition of given rank using random initialization + gradient descent."""
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Use random initialization with small values
    U = rng.standard_normal((dim_U, rank)) * 0.5
    V = rng.standard_normal((dim_V, rank)) * 0.5
    W = rng.standard_normal((dim_W, rank)) * 0.5
    
    # Gradient descent on the loss: sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    # This is a non-convex problem. Use Adam-like optimizer.
    
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    mU = np.zeros_like(U)
    mV = np.zeros_like(V)
    mW = np.zeros_like(W)
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    t = 0
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    start = time.time()
    
    while time.time() - start < time_budget:
        t += 1
        lr_t = lr * (1 - 0.5 * t / max(t, 1000))  # slight decay
        
        # Compute loss and gradients
        # loss = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
        # dloss/dU[i*m+j,c] = 2 * sum_{i',j',k'} delta(i',i)delta(j',j) * (sum_c' U[i*m+j,c']*V[j*p+k',c']*W[k'*n+i,c'] - 1) * V[j*p+k,c] * W[k*n+i,c]
        # Simplified: for fixed (i,j), dloss/dU[i*m+j,c] = 2 * sum_k (pred[i,j,k] - 1) * V[j*p+k,c] * W[k*n+i,c]
        
        # Build prediction tensor
        # pred[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        Ur = U.reshape(n, m, rank)
        Vr = V.reshape(m, p, rank)
        Wr = W.reshape(p, n, rank)
        
        # pred[i,j,k] = sum_c Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
        # This is a bit tricky to vectorize. Let's use einsum.
        # pred = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        # But Wr[k,i,c] needs Wr reshaped as (p,n,rank) which it already is.
        # Actually Wr is (p, n, rank), so Wr[k,i,c] = Wr[k,i,c]. Good.
        
        pred = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
        err = pred - 1.0  # (n, m, p)
        loss = np.sum(err**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-12:
            break
        
        # Gradients
        # dloss/dU[i*m+j,c] = 2 * sum_k err[i,j,k] * V[j*p+k,c] * W[k*n+i,c]
        # = 2 * einsum('ijk,kc,ki c->...') hmm
        # Let's do: for each (i,j,c), grad_U[i*m+j,c] = 2 * sum_k err[i,j,k] * Vr[j,k,c] * Wr[k,i,c]
        # grad_U = 2 * einsum('ijk,jkc,kic->ijc', err, Vr, Wr)
        # But Vr[j,k,c] and Wr[k,i,c] - Wr is (p,n,rank), so Wr[k,i,c] is correct.
        grad_U = 2.0 * np.einsum('ijk,jkc,kic->ijc', err, Vr, Wr)
        # grad_U is (n,m,rank), reshape to (dim_U, rank)
        grad_U = grad_U.reshape(dim_U, rank)
        
        # dloss/dV[j*p+k,c] = 2 * sum_i err[i,j,k] * U[i*m+j,c] * W[k*n+i,c]
        # grad_V = 2 * einsum('ijk,ijc,kic->jkc', err, Ur, Wr)
        grad_V = 2.0 * np.einsum('ijk,ijc,kic->jkc', err, Ur, Wr)
        grad_V = grad_V.reshape(dim_V, rank)
        
        # dloss/dW[k*n+i,c] = 2 * sum_j err[i,j,k] * U[i*m+j,c] * V[j*p+k,c]
        # grad_W = 2 * einsum('ijk,ijc,jkc->kic', err, Ur, Vr)
        grad_W = 2.0 * np.einsum('ijk,ijc,jkc->kic', err, Ur, Vr)
        grad_W = grad_W.reshape(dim_W, rank)
        
        # Adam update
        mU = beta1 * mU + (1 - beta1) * grad_U
        vU = beta2 * vU + (1 - beta2) * (grad_U ** 2)
        mU_hat = mU / (1 - beta1**t)
        vU_hat = vU / (1 - beta2**t)
        U -= lr_t * mU_hat / (np.sqrt(vU_hat) + eps)
        
        mV = beta1 * mV + (1 - beta1) * grad_V
        vV = beta2 * vV + (1 - beta2) * (grad_V ** 2)
        mV_hat = mV / (1 - beta1**t)
        vV_hat = vV / (1 - beta2**t)
        V -= lr_t * mV_hat / (np.sqrt(vV_hat) + eps)
        
        mW = beta1 * mW + (1 - beta1) * grad_W
        vW = beta2 * vW + (1 - beta2) * (grad_W ** 2)
        mW_hat = mW / (1 - beta1**t)
        vW_hat = vW / (1 - beta2**t)
        W -= lr_t * mW_hat / (np.sqrt(vW_hat) + eps)
    
    # Check if we got close enough
    Ur = best_U.reshape(n, m, rank)
    Vr = best_V.reshape(m, p, rank)
    Wr = best_W.reshape(p, n, rank)
    pred = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr)
    max_err = np.max(np.abs(pred - 1.0))
    
    if max_err < 1e-8:
        # Try to snap to rational numbers
        # Round to nearest half-integer or integer
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        # Verify the rounded version
        Ur_r = U_rounded.reshape(n, m, rank)
        Vr_r = V_rounded.reshape(m, p, rank)
        Wr_r = W_rounded.reshape(p, n, rank)
        pred_r = np.einsum('ijc,jkc,kic->ijk', Ur_r, Vr_r, Wr_r)
        max_err_r = np.max(np.abs(pred_r - 1.0))
        
        if max_err_r < 1e-10:
            # Convert to lists of integers or half-integers (multiply by 2 to get integers)
            U_list = [[int(round(x * 2)) / 2 for x in row] for row in U_rounded]
            V_list = [[int(round(x * 2)) / 2 for x in row] for row in V_rounded]
            W_list = [[int(round(x * 2)) / 2 for x in row] for row in W_rounded]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # 140s out of 150s limit
    
    # Start with schoolbook
    result = _schoolbook()
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) / something
    # For 3x4x5: n*m=12, m*p=20, p*n=15
    # Known results suggest rank around 40-50 for these dimensions
    # Let's try ranks from low to high
    
    rng = random.Random(42)
    
    # Try a range of ranks, spending time proportionally
    ranks_to_try = [45, 48, 50, 52, 54, 55, 56, 57, 58, 59]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5.0:
            break
        
        # Allocate time for this rank attempt
        time_remaining = deadline - time.time() - 2.0
        if time_remaining < 3.0:
            break
        
        # Try a few restarts
        max_restarts = min(3, int(time_remaining / 10) + 1)
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5.0:
                break
            
            time_per_attempt = min(10.0, (deadline - time.time() - 5.0) / (max_restarts - restart))
            if time_per_attempt < 2.0:
                break
            
            sol = _try_rank(rank, time_per_attempt, rng)
            if sol is not None:
                # Verify
                if _verify(sol["U"], sol["V"], sol["W"]):
                    result = sol
                    # Try even lower ranks
                    break
        
        # If we found a solution at this rank, try lower
        if result != _schoolbook():
            # Check current rank
            current_rank = len(result["U"][0])
            if current_rank <= rank:
                break
    
    return result
