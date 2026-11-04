"""Matrix multiplication 3x3x4 rank-minimal tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    DEADLINE = 135.0
    n, m, p = DIMENSIONS

    def build_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    best_U, best_V, best_W = build_schoolbook()
    best_rank = n * m * p

    # Precompute valid (a, b, c) triples and their expected values
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i else 0
    valid_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_triples.append((a, b, c, 1))
    
    # We also need to ensure that all other triples sum to 0.
    # Instead of checking all 1296 triples, we can check the full tensor contraction.
    # Tensor T has shape (9, 12, 12).
    # We construct the target tensor T_target.
    T_target = np.zeros((n*m, m*p, p*n), dtype=int)
    for a, b, c, val in valid_triples:
        T_target[a, b, c] = val

    def check_decomposition(U, V, W):
        """Check if U, V, W exactly decompose T_target."""
        # Compute tensor contraction: sum_r U[a,r] V[b,r] W[c,r]
        # This is equivalent to: for each r, outer product U[:,r] x V[:,r] x W[:,r]
        # We can do this efficiently by iterating over r and adding to a buffer,
        # or using einsum.
        # T_comp[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        return np.array_equal(T_comp, T_target)

    # Search for better rank
    random.seed(42)
    
    # Try ranks from 24 to 35
    # Using half-integers might help, but integers are simpler.
    # Let's stick to {-1, 0, 1} first, then maybe {-2, -1, 0, 1, 2}
    
    for target_rank in range(24, 36):
        if time.time() - start_time > DEADLINE:
            break
        
        max_attempts = 1000
        for _ in range(max_attempts):
            if time.time() - start_time > DEADLINE:
                break
            
            # Generate random sparse matrices
            # Density 0.5 for small ranks?
            density = 0.6
            U = np.zeros((n*m, target_rank), dtype=int)
            V = np.zeros((m*p, target_rank), dtype=int)
            W = np.zeros((p*n, target_rank), dtype=int)
            
            for r in range(target_rank):
                for idx in range(n*m):
                    if random.random() < density:
                        U[idx, r] = random.choice([-1, 1])
                for idx in range(m*p):
                    if random.random() < density:
                        V[idx, r] = random.choice([-1, 1])
                for idx in range(p*n):
                    if random.random() < density:
                        W[idx, r] = random.choice([-1, 1])
            
            if check_decomposition(U, V, W):
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                break # Found a valid decomposition for this rank, try lower next
    
    # Convert numpy arrays to lists of lists for the return format
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
