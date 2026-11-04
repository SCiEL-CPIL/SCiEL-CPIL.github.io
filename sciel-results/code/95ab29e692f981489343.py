"""Matrix multiplication 3,3,5 — rank-minimal exact tensor decomposition."""
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
        rank = U.shape[1]
        # Use einsum for vectorized computation
        return np.einsum('ar,br,cr->abc', U, V, W)
    
    def verify(U, V, W, atol=1e-10):
        """Check if U,V,W exactly decompose T."""
        T_recon = reconstruct(U, V, W)
        return np.allclose(T_recon, T, atol=atol)
    
    def compute_loss(U, V, W):
        """Compute squared error loss."""
        T_recon = reconstruct(U, V, W)
        return np.sum((T_recon - T) ** 2)
    
    def compute_gradients(U, V, W):
        """Compute gradients of loss w.r.t. U, V, W using vectorized einsum."""
        T_recon = reconstruct(U, V, W)
        err = T_recon - T
        rank = U.shape[1]
        
        # dU: derivative w.r.t. U
        # dL/dU[a,r] = sum_{b,c} err[a,b,c] * V[b,r] * W[c,r]
        dU = np.einsum('abc,br,cr->ar', err, V, W)
        dV = np.einsum('abc,ar,cr->br', err, U, W)
        dW = np.einsum('abc,ar,br->cr', err, U, V)
        
        return dU, dV, dW
    
    def adam_optimize(U, V, W, lr=0.01, steps=500, tol=1e-14):
        """Adam optimizer for tensor decomposition."""
        mU = np.zeros_like(U); vU = np.zeros_like(U)
        mV = np.zeros_like(V); vV = np.zeros_like(V)
        mW = np.zeros_like(W); vW = np.zeros_like(W)
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        prev_loss = compute_loss(U, V, W)
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            t += 1
            dU, dV, dW = compute_gradients(U, V, W)
            
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
            
            loss = compute_loss(U, V, W)
            if loss < tol:
                break
            if abs(prev_loss - loss) < 1e-16:
                break
            prev_loss = loss
        
        return U, V, W, loss
    
    def structured_init(target_rank, seed=0):
        """Create structured initialization based on matrix multiplication structure."""
        np.random.seed(seed)
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        # Initialize with random but structured patterns
        for r in range(target_rank):
            # Each term picks a random (i,j,k) triple with some probability
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            
            U[i * m + j, r] = 1.0
            V[j * p + k, r] = 1.0
            W[k * n + i, r] = 1.0
            
            # Add some randomness
            if np.random.random() < 0.3:
                i2 = np.random.randint(n)
                j2 = np.random.randint(m)
                k2 = np.random.randint(p)
                U[i2 * m + j2, r] += 0.5
                V[j2 * p + k2, r] += 0.5
                W[k2 * n + i2, r] += 0.5
        
        return U, V, W
    
    best_result = {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
    best_rank = rank_school
    
    # Strategy 1: Try ranks from 35 to 44 with Adam optimization
    for target_rank in range(35, min(rank_school, 45)):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(5):
            if time.time() > deadline:
                break
            
            seed = 42 + target_rank * 100 + trial
            U, V, W = structured_init(target_rank, seed)
            U, V, W, loss = adam_optimize(U, V, W, lr=0.05, steps=300)
            
            if verify(U, V, W):
                best_result = {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # Strategy 2: Pure random initialization with Adam
    if not found and time.time() < deadline:
        for target_rank in range(35, min(rank_school, 45)):
            if time.time() > deadline:
                break
            
            found = False
            for trial in range(3):
                if time.time() > deadline:
                    break
                
                np.random.seed(1000 + target_rank * 100 + trial)
                U = np.random.randn(nm, target_rank) * 0.5
                V = np.random.randn(mp, target_rank) * 0.5
                W = np.random.randn(pn, target_rank) * 0.5
                
                U, V, W, loss = adam_optimize(U, V, W, lr=0.03, steps=400)
                
                if verify(U, V, W):
                    best_result = {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
                    best_rank = target_rank
                    found = True
                    break
            
            if found:
                break
    
    # Strategy 3: Try to find exact rational solutions via null space
    # The tensor has 45 nonzero entries. We need to find if there's a lower rank
    # decomposition. For matrix multiplication, known lower bounds exist.
    # For 3x3 * 3x5, the schoolbook is 45. Let's see if we can do better.
    # Known: omega(3,3,5) - we use the fact that this is a sub-tensor of 3x3*3x3
    
    # Try rank 40 with more aggressive optimization
    if best_rank > 40 and time.time() < deadline:
        for target_rank in [40, 41, 42]:
            if time.time() > deadline:
                break
            
            for trial in range(8):
                if time.time() > deadline:
                    break
                
                np.random.seed(2000 + target_rank * 100 + trial)
                U = np.random.randn(nm, target_rank) * 0.3
                V = np.random.randn(mp, target_rank) * 0.3
                W = np.random.randn(pn, target_rank) * 0.3
                
                # Multi-phase optimization
                for lr in [0.1, 0.05, 0.01, 0.005]:
                    U, V, W, loss = adam_optimize(U, V, W, lr=lr, steps=200)
                    if loss < 1e-10:
                        break
                
                if verify(U, V, W):
                    best_result = {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
                    best_rank = target_rank
                    break
    
    return best_result
# EVOLVE-BLOCK-END
