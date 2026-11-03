"""Matrix multiplication 2x3 x 3x6 rank-minimal decomposition."""
import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 18
    pn = p * n  # 12
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 36
    
    # Build the tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Precompute which indices are "on-diagonal" (nonzero in T)
    diag_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                diag_indices.append((a, b, c))
    
    def verify_fast(U, V, W):
        """Faster verification using tensor contraction."""
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        if not np.allclose(T_decomp, T, atol=1e-6):
            return False
        return True
    
    def verify_sparse(U, V, W):
        """Check only the 36 nonzero entries and a subset of zero entries."""
        for (a, b, c) in diag_indices:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val - 1.0) > 1e-6:
                return False
        # Check a random sample of off-diagonal entries
        rng = np.random.RandomState(42)
        for _ in range(200):
            a = rng.randint(nm)
            b = rng.randint(mp)
            c = rng.randint(pn)
            if (a, b, c) in diag_indices:
                continue
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val) > 1e-4:
                return False
        return True
    
    # Create the schoolbook solution as fallback
    U_school = np.zeros((nm, schoolbook_rank))
    V_school = np.zeros((mp, schoolbook_rank))
    W_school = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    def to_lists(U, V, W):
        r = U.shape[1]
        U_list = [[float(U[i,j]) for j in range(r)] for i in range(nm)]
        V_list = [[float(V[i,j]) for j in range(r)] for i in range(mp)]
        W_list = [[float(W[i,j]) for j in range(r)] for i in range(pn)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    best_rank = schoolbook_rank
    best_result = to_lists(U_school, V_school, W_school)
    
    # Strategy: Try structured sparse patterns
    # The tensor has a block structure. For 2x3x6, we can try:
    # 1. Block-wise patterns where each "layer" handles a subset of (i,j,k)
    # 2. Random sparse patterns with controlled density
    
    # Try alternating minimization with better initialization
    # Key insight: Use the structure - for each (i,j) pair in A, the column in U
    # interacts with specific rows in V and W
    
    # Better approach: Use tensor train / matrix product state style search
    # Or: Use the fact that this is equivalent to multiplying 2x3 by 3x6
    
    # Let's try a more systematic approach:
    # For rank r, we need U (6xr), V (18xr), W (12xr)
    # such that sum_c U[a,c]*V[b,c]*W[c,c] = T[a,b,c]
    
    # Try ranks from 20 to 35
    for rank in range(20, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        
        # Multiple random restarts with alternating minimization
        for attempt in range(30):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(rank * 1000 + attempt)
            
            # Initialize with some structure: group by i
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            # Random initialization with scaling
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # Alternating least squares / gradient descent
            # Use Adam-like updates for better convergence
            lr = 0.05
            best_loss = np.inf
            stuck_count = 0
            
            for step in range(1000):
                if time.time() > deadline - 5:
                    break
                
                T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T_decomp - T
                loss = np.sum(diff**2)
                
                if loss < best_loss:
                    best_loss = loss
                    stuck_count = 0
                else:
                    stuck_count += 1
                
                if stuck_count > 50:
                    # Reset with new random init
                    U = np.random.randn(nm, rank) * 0.1
                    V = np.random.randn(mp, rank) * 0.1
                    W = np.random.randn(pn, rank) * 0.1
                    best_loss = np.inf
                    stuck_count = 0
                    continue
                
                if loss < 1e-12:
                    break
                
                # Compute gradients
                grad_U = np.einsum('abc,br,cr->ar', diff, V, W) * 2
                grad_V = np.einsum('abc,ar,cr->br', diff, U, W) * 2
                grad_W = np.einsum('abc,ar,br->cr', diff, U, V) * 2
                
                # Adaptive learning rate
                g_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                if g_norm > 0:
                    lr_eff = lr * min(1.0, 0.1 / g_norm)
                else:
                    lr_eff = lr
                
                U -= lr_eff * grad_U
                V -= lr_eff * grad_V
                W -= lr_eff * grad_W
            
            # Check if we found a valid solution
            if verify_sparse(U, V, W) and verify_fast(U, V, W):
                found = True
                best_rank = rank
                best_result = to_lists(U, V, W)
                break
        
        if found:
            break
    
    # If we found something better, try to polish it
    if best_rank < schoolbook_rank:
        # High-precision polish
        if time.time() < deadline - 20:
            # Extract current solution
            U_polish = np.array(best_result['U'], dtype=np.float64)
            V_polish = np.array(best_result['V'], dtype=np.float64)
            W_polish = np.array(best_result['W'], dtype=np.float64)
            
            for polish_step in range(200):
                if time.time() > deadline - 5:
                    break
                
                T_decomp = np.einsum('ar,br,cr->abc', U_polish, V_polish, W_polish)
                diff = T_decomp - T
                loss = np.sum(diff**2)
                
                if loss < 1e-15:
                    break
                
                grad_U = np.einsum('abc,br,cr->ar', diff, V_polish, W_polish) * 2
                grad_V = np.einsum('abc,ar,cr->br', diff, U_polish, W_polish) * 2
                grad_W = np.einsum('abc,ar,br->cr', diff, U_polish, V_polish) * 2
                
                g_norm = np.sqrt(np.sum(grad_U**2) + np.sum(grad_V**2) + np.sum(grad_W**2))
                lr_eff = 0.01 * min(1.0, 0.01 / (g_norm + 1e-10))
                
                U_polish -= lr_eff * grad_U
                V_polish -= lr_eff * grad_V
                W_polish -= lr_eff * grad_W
            
            if verify_fast(U_polish, V_polish, W_polish):
                best_result = to_lists(U_polish, V_polish, W_polish)
    
    return best_result
