"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast tensor-based verification."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U_r = U.T.reshape(rank, n, m)
    V_r = V.T.reshape(rank, m, p)
    W_r = W.T.reshape(rank, p, n)
    # T[a,b,c,d,i,k] = sum_r U[r,a,b] * V[r,c,d] * W[r,k,i]
    # Should equal delta_{a,i} * delta_{b,c} * delta_{d,k}
    T = np.einsum('rab,rcd,rki->abcdik', U_r, V_r, W_r)
    # Build target
    target = np.zeros((n, m, m, p, n, p))
    for a in range(n):
        for b in range(m):
            for k in range(p):
                target[a, b, b, k, a, k] = 1.0
    return np.max(np.abs(T - target)) < tol

def _make_schoolbook(n, m, p):
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

def _gradient_descent(n, m, p, target_rank, max_steps, lr_init, seed, deadline):
    """Run gradient descent to find a rank-target_rank decomposition."""
    np.random.seed(seed)
    
    # Structured initialization: start from schoolbook and perturb
    # Or use random with structure
    U = np.random.randn(n * m, target_rank) * 0.5
    V = np.random.randn(m * p, target_rank) * 0.5
    W = np.random.randn(p * n, target_rank) * 0.5
    
    # Build target tensor
    target = np.zeros((n, m, m, p, n, p))
    for a in range(n):
        for b in range(m):
            for k in range(p):
                target[a, b, b, k, a, k] = 1.0
    
    def compute_T(U, V, W):
        U_r = U.T.reshape(target_rank, n, m)
        V_r = V.T.reshape(target_rank, m, p)
        W_r = W.T.reshape(target_rank, p, n)
        return np.einsum('rab,rcd,rki->abcdik', U_r, V_r, W_r)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    lr = lr_init
    momentum_U = np.zeros_like(U)
    momentum_V = np.zeros_like(V)
    momentum_W = np.zeros_like(W)
    beta = 0.9
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        U_r = U.T.reshape(target_rank, n, m)
        V_r = V.T.reshape(target_rank, m, p)
        W_r = W.T.reshape(target_rank, p, n)
        
        T = np.einsum('rab,rcd,rki->abcdik', U_r, V_r, W_r)
        diff = T - target
        loss = np.sum(diff**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_loss < 1e-14:
            break
        
        # Gradients
        grad_U = 2 * np.einsum('abcdik,rcd,rki->rab', diff, V_r, W_r)
        grad_V = 2 * np.einsum('abcdik,rab,rki->rcd', diff, U_r, W_r)
        grad_W = 2 * np.einsum('abcdik,rab,rcd->rki', diff, U_r, V_r)
        
        grad_U = grad_U.reshape(target_rank, n * m).T
        grad_V = grad_V.reshape(target_rank, m * p).T
        grad_W = grad_W.reshape(target_rank, p * n).T
        
        # Momentum update
        momentum_U = beta * momentum_U + (1 - beta) * grad_U
        momentum_V = beta * momentum_V + (1 - beta) * grad_V
        momentum_W = beta * momentum_W + (1 - beta) * grad_W
        
        U -= lr * momentum_U
        V -= lr * momentum_V
        W -= lr * momentum_W
        
        lr *= 0.999
    
    if best_loss < 1e-10:
        return best_U, best_V, best_W, best_loss
    return None, None, None, best_loss

def _try_strassen_embedded(n, m, p):
    """
    Try to embed Strassen's 4x4x4 rank-7 into the 2x5x5 problem.
    
    Strategy: Pad A (2x5) to (4x5) with 2 zero rows.
    Then we need 4x5 * 5x5 = 4x5 result, but we only use first 2 rows.
    
    Split 4x5 into 2x2 blocks of size 2x(2.5)... that's awkward.
    
    Better: Split into 2x2 blocks where each block is 2x2 or 2x3.
    
    Actually, let's think differently. For 2x5 * 5x5:
    - Row 0 of C: C[0,k] = sum_j A[0,j]*B[j,k] for k=0..4
    - Row 1 of C: C[1,k] = sum_j A[1,j]*B[j,k] for k=0..4
    
    Each row is a 1x5 * 5x5 = 1x5 computation, rank 25 each.
    Total schoolbook: 50.
    
    Can we share? If we group B's columns: B[:, 0:3] and B[:, 3:5]
    Then C[0, 0:3] = A[0, :] * B[:, 0:3] (1x5 * 5x3, rank 15)
    C[0, 3:5] = A[0, :] * B[:, 3:5] (1x5 * 5x2, rank 10)
    Similarly for row 1.
    
    Total: 15 + 10 + 15 + 10 = 50. No savings.
    
    What if we share the A factors?
    C[0, 0:3] and C[1, 0:3] both use B[:, 0:3].
    We can write: [C[0,0:3], C[1,0:3]] = [A[0,:], A[1,:]] * B[:, 0:3]
    This is 2x5 * 5x3 = 2x3, rank 15 (schoolbook for this sub-problem).
    
    Similarly [C[0,3:5], C[1,3:5]] = [A[0,:], A[1,:]] * B[:, 3:5], rank 10.
    
    Total: 15 + 10 = 25! That's half the schoolbook!
    
    Wait, that's just saying we compute the full 2x5 * 5x5 as two sub-products.
    The rank of 2x5 * 5x3 is 2*5*3 = 30? No, the bilinear complexity of m1 x m2 * m2 x m3 is m1*m2*m3.
    So 2*5*3 = 30 and 2*5*2 = 20. Total 50. Still schoolbook.
    
    The issue is that bilinear complexity of matrix multiplication m1xm2 * m2xm3 is exactly m1*m2*m3
    for the standard tensor. There's no free lunch from splitting.
    
    However, Strassen shows that for SQUARE matrices, we can do better.
    For 2x5 * 5x5, the tensor is not square, so Strassen doesn't directly apply.
    
    But we CAN embed: pad to 5x5 * 5x5, use a rank-r decomposition for 5x5x5,
    and the 2x5x5 is a sub-tensor. If we have a rank-r decomposition for 5x5x5,
    then 2x5x5 has rank at most r (by taking appropriate slices).
    
    The best known rank for 5x5x5 is... Strassen gives 49 for 4x4x4? No, Strassen is 7 for 2x2x2,
    which gives 49 for 4x4x4 (7^2 = 49). For 5x5x5, we'd pad to 8x8x8 and use Strassen:
    7^3 = 343, which is worse than schoolbook 125.
    
    So for 5x5x5, schoolbook (125) is better than Strassen padding (343).
    
    For 2x5x5, schoolbook is 50. The question is: is there a decomposition with rank < 50?
    
    Lower bounds: The border rank of 2x5x5 matrix multiplication is at least...
    For m1 x m2 * m2 x m3, the rank is at least max(m1*m2, m2*m3, m1*m3) = max(10, 25, 10) = 25.
    Also, rank >= m1*m2*m3 / max(m1,m2,m3) ... not a standard bound.
    
    Actually, for the matrix multiplication tensor, the rank equals the bilinear complexity.
    For rectangular matrices, it's known that the bilinear complexity of m1xm2 * m2xm3
    is exactly m1*m2*m3 for "generic" dimensions, but can be less for special cases.
    
    For 2x5 * 5x5: I believe the rank is exactly 50 (schoolbook is optimal) for these dimensions,
    but I'm not 100% sure. Let me just try to find something better via optimization.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin for 150s limit
    
    # Start with schoolbook
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    def still_going():
        return time.time() < deadline
    
    # Try gradient-based search for lower ranks
    # Theoretical lower bound is 25 (from max(m1*m2, m2*m3, m1*m3) = max(10,25,10)=25)
    # Try ranks from 35 down to 26 (skip 25 as it's very unlikely)
    
    np.random.seed(42)
    random.seed(42)
    
    # Try a few target ranks with gradient descent
    for target_rank in [45, 40, 35, 30]:
        if not still_going():
            break
        
        remaining = deadline - time.time()
        if remaining < 30:
            break
        
        # Try multiple seeds
        for seed in [42, 123, 456, 789]:
            if not still_going():
                break
            
            max_steps = int(min(remaining * 0.25, 5000))
            U, V, W, loss = _gradient_descent(n, m, p, target_rank, max_steps, 0.05, seed, deadline)
            
            if U is not None:
                # Verify
                if _verify_fast(U, V, W, n, m, p, tol=1e-8):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    break  # Found for this rank, try lower
        
        if best_rank < n * m * p:
            # Found something better, try lower ranks
            continue
    
    # Also try a structured approach: exploit the 2-row structure
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Group: for each pair of columns (k1, k2) of B, compute
    # [C[i,k1], C[i,k2]] = A[i,:] * [B[:,k1], B[:,k2]]
    # This is 1x5 * 5x2, rank 10 per pair, per row i.
    # With 5 columns, we have floor(5/2)=2 pairs + 1 single.
    # Pairs: (0,1), (2,3), single: 4
    # Per row: 10 + 10 + 5 = 25
    # Two rows: 50. Still schoolbook.
    
    # What if we share across rows?
    # [C[0,k1], C[0,k2], C[1,k1], C[1,k2]] for a pair of columns
    # = [A[0,:], A[1,:]] * [B[:,k1], B[:,k2]]
    # This is 2x5 * 5x2 = 2x2, rank 2*5*2 = 20 per column pair.
    # With 2 full pairs + 1 single column:
    # 20 + 20 + 2*5*1 = 20+20+10 = 50. Still schoolbook.
    
    # The fundamental issue: for these dimensions, schoolbook may be optimal.
    # But let's make sure our search is thorough.
    
    return best
# EVOLVE-BLOCK-END
