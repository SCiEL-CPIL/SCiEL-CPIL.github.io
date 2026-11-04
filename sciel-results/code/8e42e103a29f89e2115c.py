import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # The trivial decomposition has rank n*m*p = 120
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank))
    V_trivial = np.zeros((m * p, trivial_rank))
    W_trivial = np.zeros((p * n, trivial_rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_trivial[a, idx] = 1
                V_trivial[b, idx] = 1
                W_trivial[c, idx] = 1
                idx += 1
    
    best_rank = trivial_rank
    best_U = U_trivial.copy()
    best_V = V_trivial.copy()
    best_W = W_trivial.copy()

    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W correctly decompose the multiplication tensor using vectorized einsum."""
        rank = U.shape[1]
        # Reshape to 3D: (n, m, rank), (m, p, rank), (p, n, rank)
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # Compute T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Indices: U is (i, j, r), V is (j, k, r), W is (k, i, r)
        # Result T is (i, j, k)
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        return np.allclose(T, np.ones((n, m, p)), atol=tol)

    # Strategy: Leveraging Strassen-like block decomposition ideas.
    # While a full generic Strassen extension is complex, we can attempt
    # random searches with structured initializations that might find
    # lower rank decompositions if they exist.
    
    # We also try a specific structured approach:
    # Partition the matrices into blocks and see if we can share computations.
    # For 4x5 * 5x6, we can view it as a 2x3 block multiplication by 3x2?
    # Or 2x2 blocks?
    # Let's stick to random search with better initialization and checking
    # for ranks slightly below 120, as finding a significantly lower rank
    # (like <100) is extremely difficult without specific algebraic structure.
    
    # However, the prompt asks to combine the strategy with inspirations.
    # Inspiration 3 uses einsum for fast checking.
    # Inspiration 2 uses random integer search.
    # Current program uses random Gaussian search (ineffective).
    
    # Let's try random integer search for ranks 119 down to 100.
    # This is unlikely to find a solution but is a valid search strategy.
    # More importantly, we should check if there's a known lower bound.
    # For 4x4x4, rank is 49 (Strassen).
    # For 4x5x6, the rank is likely close to 120 but might be reducible.
    
    # Let's try a few random integer decompositions for ranks < 120.
    # We will use a time-limited search.
    
    # To make the search more effective, we can try to "merge" terms from the
    # trivial decomposition.
    # But random search is the fallback.
    
    # Let's try ranks 119, 118, ... down to 100 with a few random attempts each.
    # We'll use integer values {-1, 0, 1} or {-2, -1, 0, 1, 2} to keep numbers small.
    
    np.random.seed(42)
    
    for target_rank in range(119, 99, -1):
        if time.time() > deadline:
            break
        
        # Try a few random attempts for this rank
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            # Random integer initialization
            U_rand = np.random.randint(-2, 3, size=(n * m, target_rank))
            V_rand = np.random.randint(-2, 3, size=(m * p, target_rank))
            W_rand = np.random.randint(-2, 3, size=(p * n, target_rank))
            
            if check_decomposition(U_rand, V_rand, W_rand):
                best_rank = target_rank
                best_U = U_rand.copy()
                best_V = V_rand.copy()
                best_W = W_rand.copy()
                break
        
        if best_rank < trivial_rank:
            break

    # If we found a better decomposition, return it
    if best_rank < trivial_rank:
        return {'U': best_U.tolist(), 'V': best_V.tolist(), 'W': best_W.tolist()}
    else:
        # Return the trivial decomposition
        return {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
