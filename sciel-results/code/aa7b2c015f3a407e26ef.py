"""Tensor decomposition for 3x3 @ 3x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 9, 15, 15
    
    # Build the target tensor T (flattened)
    # T[a, b, c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i for consistent i,j,k
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank 45)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    def to_list(mat):
        return mat.tolist()
    
    def reconstruct(U, V, W):
        """Vectorized reconstruction of tensor from factors."""
        return np.einsum('ar,br,cr->abc', U, V, W)
    
    def verify(U, V, W, atol=1e-10):
        """Check if U,V,W exactly decompose T."""
        T_recon = reconstruct(U, V, W)
        return np.allclose(T_recon, T, atol=atol)
    
    # Strategy: Direct sum decomposition
    # Split the 3x3x5 tensor into 3x3x3 (rank 23, Landsberg) + 3x3x2 (rank 18, schoolbook)
    # Total rank: 23 + 18 = 41
    # 
    # The 3x3x3 part uses columns k=0,1,2 of the output
    # The 3x3x2 part uses columns k=3,4 of the output
    #
    # For the 3x3x3 part, we use the known rank-23 decomposition
    # For the 3x3x2 part, we use schoolbook (rank 18)
    
    # Landsberg's rank-23 decomposition for 3x3x3
    # This is a well-known result. Let me implement it.
    # The 23 terms for 3x3x3 matrix multiplication.
    
    # Actually, let me just try to find a good decomposition via optimization
    # with the direct sum structure as initialization.
    
    def build_direct_sum_init():
        """Build initial factors using direct sum structure."""
        # 3x3x3 part: schoolbook rank 9 (we'll try to optimize to 23 later)
        # 3x3x2 part: schoolbook rank 18
        # Total: 27 initially, try to reduce
        
        rank_total = 9 + 18  # 27
        U = np.zeros((nm, rank_total))
        V = np.zeros((mp, rank_total))
        W = np.zeros((pn, rank_total))
        
        # 3x3x3 part (k=0,1,2): schoolbook
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(3):  # only first 3 columns
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        # 3x3x2 part (k=3,4): schoolbook
        for i in range(n):
            for j in range(m):
                for k in range(3, 5):  # last 2 columns
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        return U, V, W, rank_total
    
    def adam_optimize(U, V, W, lr=0.01, steps=500, tol=1e-14):
        """Adam optimizer for tensor decomposition."""
        mU = np.zeros_like(U); vU = np.zeros_like(U)
        mV = np.zeros_like(V); vV = np.zeros_like(V)
        mW = np.zeros_like(W); vW = np.zeros_like(W)
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        prev_loss = np.sum((reconstruct(U, V, W) - T) ** 2)
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            t += 1
            T_recon = reconstruct(U, V, W)
            err = T_recon - T
            
            dU = np.einsum('abc,br,cr->ar', err, V, W)
            dV = np.einsum('abc,ar,cr->br', err, U, W)
            dW = np.einsum('abc,ar,br->cr', err, U, V)
            
            mU = b1 * mU + (1 - b1) * dU
            mV = b1 * mV + (1 - b1) * dV
            mW = b1 * mW + (1 - b1) * dW
            vU = b2 * vU + (1 - b2) * (dU ** 2)
            vV = b2 * vV + (1 - b2) * (dV ** 2)
            vW = b2 * vW + (1 - b2) * (dW ** 2)
            
            mU_hat = mU / (1 - b1 ** t)
            mV_hat = mV / (1 - b1 ** t)
            mW_hat = mW / (1 - b1 ** t)
            vU_hat = vU / (1 - b2 ** t)
            vV_hat = vV / (1 - b2 ** t)
            vW_hat = vW / (1 - b2 ** t)
            
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            loss = np.sum((reconstruct(U, V, W) - T) ** 2)
            if loss < tol:
                break
            if abs(prev_loss - loss) < 1e-16:
                break
            prev_loss = loss
        
        return U, V, W, loss
    
    # Start with schoolbook as fallback
    best_result = {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
    best_rank = rank_school
    
    # Try direct sum initialization with optimization
    U_ds, V_ds, W_ds, rank_ds = build_direct_sum_init()
    U_ds, V_ds, W_ds, loss_ds = adam_optimize(U_ds, V_ds, W_ds, lr=0.05, steps=300)
    
    if verify(U_ds, V_ds, W_ds):
        best_result = {"U": to_list(U_ds), "V": to_list(V_ds), "W": to_list(W_ds)}
        best_rank = rank_ds
    
    # Try lower ranks with random initialization
    for target_rank in range(35, min(rank_school, 45)):
        if time.time() > deadline - 5:
            break
        
        found = False
        for trial in range(3):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(42 + target_rank * 100 + trial)
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            U, V, W, loss = adam_optimize(U, V, W, lr=0.03, steps=200)
            
            if verify(U, V, W):
                best_result = {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
