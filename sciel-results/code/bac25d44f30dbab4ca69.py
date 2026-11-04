import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # Compute rank of matmul tensor via bilinear rank minimization
    # We use a simple approach: try to find a decomposition using
    # random initialization and gradient descent, then snap to integers.
    
    # First, let's try a known construction:
    # For 3x3 x 3x4, the schoolbook rank is 36.
    # Let's see if we can do better with a structured approach.
    
    # Build the target tensor: T[a, b, g] = 1 if (a,b,g) corresponds to a valid term
    # a = i*m + j, b = j*p + k, g = k*n + i
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                g = k * n + i
                valid_indices.append((a, b, g))
    
    all_indices = []
    for a in range(n * m):
        for b in range(m * p):
            for g in range(p * n):
                all_indices.append((a, b, g))
    
    valid_set = set(valid_indices)
    invalid_indices = [idx for idx in all_indices if idx not in valid_set]
    
    def compute_tensor(U, V, W):
        """Compute T[a,b,g] = sum_c U[a,c]*V[b,c]*W[g,c]"""
        return np.einsum('ac,bc,gc->abg', U, V, W)
    
    def compute_loss(U, V, W):
        """Sum of squared errors for valid entries (should be 1) and invalid (should be 0)"""
        T = compute_tensor(U, V, W)
        loss = 0.0
        for (a, b, g) in valid_indices:
            loss += (T[a, b, g] - 1.0) ** 2
        for (a, b, g) in invalid_indices:
            loss += T[a, b, g] ** 2
        return loss
    
    def snap_to_halfint(U, V, W):
        """Round to nearest half-integer and verify"""
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        T = compute_tensor(U_s, V_s, W_s)
        for (a, b, g) in valid_indices:
            if abs(T[a, b, g] - 1.0) > 1e-9:
                return None
        for (a, b, g) in invalid_indices:
            if abs(T[a, b, g]) > 1e-9:
                return None
        return (U_s, V_s, W_s)
    
    def verify_integer(U, V, W):
        """Check if entries are integers (or half-integers) and decomposition is exact"""
        # Check if all entries are close to half-integers
        for mat in [U, V, W]:
            rounded = np.round(mat * 2) / 2
            if not np.allclose(mat, rounded, atol=1e-9):
                return None
        return snap_to_halfint(U, V, W)
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_rank = n * m * p  # Schoolbook rank as fallback
    best_decomp = None
    
    # Schoolbook decomposition
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                g = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[g, col] = 1
                col += 1
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Try lower ranks
    # Theoretically, for 3x3 x 3x4, the rank could be lower.
    # Let's try ranks from 16 to 35.
    
    for rank in range(16, 36):
        if time.time() > deadline:
            break
        
        U_shape = (n * m, rank)
        V_shape = (m * p, rank)
        W_shape = (p * n, rank)
        
        found = False
        
        # Multiple random restarts
        num_restarts = 20
        for attempt in range(num_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 1000 + rank)
            
            # Initialize with random small values
            U = np.random.randn(*U_shape) * 0.3
            V = np.random.randn(*V_shape) * 0.3
            W = np.random.randn(*W_shape) * 0.3
            
            # Gradient descent
            lr = 0.01
            for it in range(500):
                if time.time() > deadline:
                    break
                
                T = compute_tensor(U, V, W)
                
                # Compute gradients
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for (a, b, g) in valid_indices:
                    err = T[a, b, g] - 1.0
                    for c in range(rank):
                        grad_U[a, c] += 2 * err * V[b, c] * W[g, c]
                        grad_V[b, c] += 2 * err * U[a, c] * W[g, c]
                        grad_W[g, c] += 2 * err * U[a, c] * V[b, c]
                
                for (a, b, g) in invalid_indices:
                    err = T[a, b, g]
                    for c in range(rank):
                        grad_U[a, c] += 2 * err * V[b, c] * W[g, c]
                        grad_V[b, c] += 2 * err * U[a, c] * W[g, c]
                        grad_W[g, c] += 2 * err * U[a, c] * V[b, c]
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Early stop if loss is small
                loss = compute_loss(U, V, W)
                if loss < 1e-8:
                    break
            
            # Try to snap to half-integers
            result = snap_to_halfint(U, V, W)
            if result is not None:
                best_rank = rank
                best_decomp = result
                found = True
                break
        
        if found:
            break
    
    U_final, V_final, W_final = best_decomp
    
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }
