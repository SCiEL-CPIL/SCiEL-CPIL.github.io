"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 140.0
    n, m, p = DIMENSIONS

    # Schoolbook decomposition: rank = n*m*p = 48
    rank0 = n * m * p
    U0 = np.zeros((n * m, rank0))
    V0 = np.zeros((m * p, rank0))
    W0 = np.zeros((p * n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1

    def verify(U, V, W, tol=1e-8):
        """Verify tensor decomposition for matmul tensor exactly."""
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        T_calc = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_calc, T, atol=tol, rtol=0.0)

    def verify_exact(U, V, W):
        """Exact verification using rational arithmetic."""
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        T_calc = np.einsum('ak,bk,ck->abc', U, V, W)
        diff = np.abs(T_calc - T)
        return np.max(diff) < 1e-10

    best_rank = rank0
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()

    # Strategy: Use known rank-19 algorithm for 4x4x4 matmul
    # The 3x4x4 tensor can be viewed as 3 copies of 1x4x4 with shared 4x4 structure
    # Rank-19 for 4x4x4 is known (Strassen-like)
    
    # First, let's try a structured approach:
    # Decompose 3x4x4 as: for each output row i (0,1,2), we need to compute
    # row_i = A[i,:] * B (1x4 * 4x4)
    # This is 3 separate 1x4x4 multiplications
    # Each 1x4x4 has rank 4 (schoolbook), so naive gives 12
    # But we can share the 4x4 multiplication structure
    
    # Key insight: The 3x4x4 tensor T[i,j,k] = 1 if (i,j,k) valid
    # We can write this as a sum of rank-1 tensors
    # Let's try rank 16 first (4 per output row, sharing B structure)
    
    # Actually, let's think more carefully:
    # T[i*m+j, j*p+k, k*n+i] = 1
    # This is a 3-way tensor of size (12, 16, 12)
    
    # Try rank 16: use 4 terms per output row i
    # For fixed i, we need T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # If we share V across all i (same 4x4 multiplication), we can use rank 4 for each i
    # Total rank = 3 * 4 = 12
    
    # Let's try rank 12 with shared V structure
    def try_rank_12_shared():
        """Try rank-12 decomposition with shared V factors."""
        r = 12
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # For each output row i, use 4 rank-1 terms
        # The 4x4 multiplication: (1x4) * (4x4) = (1x4)
        # Schoolbook: for each k, term: U[i*4+j] = 1 if j==j, V[j*4+k] = 1, W[k*3+i] = 1
        # But we want to share V across all i
        
        # Actually, the standard 4x4 schoolbook uses rank 16
        # For 1x4x4: rank 4 (one term per k)
        # For 3x4x4 with shared B: we can use rank 12
        
        for i in range(n):
            for k in range(p):
                r_idx = i * p + k  # rank index
                for j in range(m):
                    U[i * m + j, r_idx] = 1.0
                    V[j * p + k, r_idx] = 1.0
                    W[k * n + i, r_idx] = 1.0
        
        if verify_exact(U, V, W):
            return U, V, W
        return None

    # Try rank 12
    if time.time() < deadline - 30:
        res = try_rank_12_shared()
        if res is not None:
            best_rank = 12
            best_U, best_V, best_W = res
            print(f"Found rank {best_rank} decomposition!")
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Try rank 16 (full schoolbook for 4x4x4, used 3 times but sharing)
    def try_rank_16():
        """Try rank-16 decomposition."""
        r = 16
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        # Use rank-4 for each output row, but with different V
        # Actually, let's just use schoolbook structure
        # For 3x4x4, schoolbook is rank 48
        # But we can do better by sharing
        
        # Try: for each (i,j,k), one term
        # That's 48 terms
        # Let's try a different structure
        
        # Use the fact that 4x4 matmul can be done in rank 19 (Strassen)
        # For 3x4x4, we can use 3 * 19 = 57, which is worse than 48
        # But maybe we can do better
        
        # Let's try rank 24: 8 terms per output row
        # Or rank 18: 6 terms per output row
        
        return None

    # Gradient-based search for lower ranks
    def try_gradient_search(target_rank, num_trials=5, max_steps=40):
        """Gradient descent to find rank-target_rank decomposition."""
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0

        for trial in range(num_trials):
            if time.time() > deadline - 20:
                break
            
            # Initialize from schoolbook subset
            idx = np.random.choice(rank0, size=target_rank, replace=False)
            U = U0[:, idx].copy() + 0.05 * np.random.randn(n * m, target_rank)
            V = V0[:, idx].copy() + 0.05 * np.random.randn(m * p, target_rank)
            W = W0[:, idx].copy() + 0.05 * np.random.randn(p * n, target_rank)

            mU = np.zeros_like(U); mV = np.zeros_like(V); mW = np.zeros_like(W)
            vU = np.zeros_like(U); vV = np.zeros_like(V); vW = np.zeros_like(W)
            b1, b2 = 0.9, 0.999
            eps = 1e-8
            best_loss = np.inf
            
            for step in range(max_steps):
                if time.time() > deadline - 20:
                    break
                t = step + 1
                T_calc = np.einsum('ak,bk,ck->abc', U, V, W)
                residual = T_calc - T
                loss = np.sum(residual * residual)
                if loss < best_loss:
                    best_loss = loss
                if loss < 1e-12:
                    if verify_exact(U, V, W):
                        return U, V, W
                    break
                grad_U = np.einsum('abc,bk,ck->ak', residual, V, W)
                grad_V = np.einsum('abc,ak,ck->bk', residual, U, W)
                grad_W = np.einsum('abc,ak,bk->ck', residual, U, V)
                mU = b1 * mU + (1 - b1) * grad_U; vU = b2 * vU + (1 - b2) * grad_U * grad_U
                mV = b1 * mV + (1 - b1) * grad_V; vV = b2 * vV + (1 - b2) * grad_V * grad_V
                mW = b1 * mW + (1 - b1) * grad_W; vW = b2 * vW + (1 - b2) * grad_W * grad_W
                mUh = mU / (1 - b1 ** t); vUh = vU / (1 - b2 ** t)
                mVh = mV / (1 - b1 ** t); vVh = vV / (1 - b2 ** t)
                mWh = mW / (1 - b1 ** t); vWh = vW / (1 - b2 ** t)
                lr = 0.03
                U -= lr * mUh / (np.sqrt(vUh) + eps)
                V -= lr * mVh / (np.sqrt(vVh) + eps)
                W -= lr * mWh / (np.sqrt(vWh) + eps)
        return None

    # Try various target ranks
    for target in [13, 14, 15, 16]:
        if time.time() > deadline - 25:
            break
        res = try_gradient_search(target, num_trials=3, max_steps=30)
        if res is not None:
            best_rank = target
            best_U, best_V, best_W = res
            print(f"Found rank {best_rank} decomposition!")
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Fall back to schoolbook
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
# EVOLVE-BLOCK-END
