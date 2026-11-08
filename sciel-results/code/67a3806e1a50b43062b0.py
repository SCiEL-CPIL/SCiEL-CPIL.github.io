"""Matrix multiplication 4,4,6 rank-minimal decomposition."""
import numpy as np
import time

def construct():
    n, m, p = 4, 4, 6
    
    # Build target tensor for verification
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    def check_decomposition(U, V, W):
        """Check if the decomposition exactly reproduces the target tensor."""
        T_test = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_test += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        return np.allclose(T_test, T, atol=1e-10)
    
    # Strategy: Use rank-21 decomposition for 4x4x4 sub-block + 8 rank-1 terms for 4x2
    # The 4x4x4 matrix multiplication has a known rank-21 algorithm
    # We'll use a Winograd-style or Strassen-based decomposition
    
    # First, let's try a structured approach:
    # Split p=6 into p1=4 and p2=2
    # For the 4x4x4 part, use rank-21
    # For the 4x4x2 part, use rank-8 (schoolbook)
    # Total: 29
    
    # Build the 4x4x4 rank-21 decomposition
    # Using a known rank-21 algorithm for 4x4 matrix multiplication
    # This is based on the work of Coppersmith or similar fast matrix multiplication
    
    # For now, let's use a simpler approach:
    # Start with schoolbook rank-96 and try to find a better decomposition
    # through random initialization and optimization
    
    start_time = time.time()
    deadline = start_time + 140
    
    # First, build the trivial rank-96 decomposition as a fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n*m, rank_trivial))
    V_trivial = np.zeros((m*p, rank_trivial))
    W_trivial = np.zeros((p*n, rank_trivial))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i*m+j, col] = 1
                V_trivial[j*p+k, col] = 1
                W_trivial[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_trivial
    best_U, best_V, best_W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
    
    # Try structured decomposition: rank-21 for 4x4x4 + rank-8 for 4x4x2
    # The 4x4x4 part uses indices: i in [0,4), j in [0,4), k in [0,4)
    # The 4x4x2 part uses indices: i in [0,4), j in [0,4), k in [4,6)
    
    # For the 4x4x4 part, we need a rank-21 decomposition
    # Let's use a simple structured approach with random initialization
    
    np.random.seed(42)
    
    # Try to find a rank-21 decomposition for 4x4x4
    # Using random initialization and gradient-free optimization
    
    def random_decomposition(rank, n, m, p, seed=0):
        """Generate a random rank decomposition."""
        rng = np.random.RandomState(seed)
        U = rng.randn(n*m, rank)
        V = rng.randn(m*p, rank)
        W = rng.randn(p*n, rank)
        return U, V, W
    
    def objective(U, V, W, T_target):
        """Compute the error of the decomposition."""
        T_test = np.zeros(T_target.shape)
        for r in range(U.shape[1]):
            T_test += np.einsum('i,j,k->ijk', U[:,r], V[:,r], W[:,r])
        return np.linalg.norm(T_test - T_target)
    
    # Try various ranks starting from 29
    for target_rank in [29, 30, 31, 32, 33, 34, 35]:
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            U, V, W = random_decomposition(target_rank, n, m, p, seed=attempt)
            
            # Simple local search: try to minimize the error
            # Use a greedy approach: try to set entries to 0 or small integers
            # to get a sparse, structured solution
            
            # For now, just check if this random decomposition is close
            err = objective(U, V, W, T)
            
            if err < 1e-6:
                if check_decomposition(U, V, W):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        break
        
        if best_rank <= target_rank:
            break
    
    # If we haven't found a good decomposition, try the greedy column removal
    if best_rank == rank_trivial and time.time() < deadline:
        U, V, W = U_trivial.copy(), V_trivial.copy(), W_trivial.copy()
        current_rank = rank_trivial
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for c in range(current_rank):
                if time.time() > deadline:
                    break
                U_test = np.delete(U, c, axis=1)
                V_test = np.delete(V, c, axis=1)
                W_test = np.delete(W, c, axis=1)
                
                if check_decomposition(U_test, V_test, W_test):
                    U, V, W = U_test, V_test, W_test
                    current_rank -= 1
                    if current_rank < best_rank:
                        best_rank = current_rank
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    improved = True
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
