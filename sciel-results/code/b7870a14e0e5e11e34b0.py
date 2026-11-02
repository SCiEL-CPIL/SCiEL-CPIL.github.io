import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    # Check each (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Compute C[i,k] from the decomposition
                # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # But this is not how tensor decomposition works.
                # The tensor T is: T[(i,j), (j',k'), (k'',i'')] = sum_c U[(i,j),c] * V[(j',k'),c] * W[(k'',i''),c]
                # We need T[(i,j), (j,k), (k,i)] = 1 for all (i,j,k)
                # And T[a,b,c] = 0 for all other (a,b,c)
                pass
    # Better approach: compute the full tensor and check
    # T[a, b, c] = sum_c U[a,c] * V[b,c] * W[c,c]
    # But this is O(n*m * m*p * p*n * rank) which is 15*35*21*rank
    # For rank=105, that's 15*35*21*105 = 1,157,625 operations, which is fine.
    
    # Compute T from U, V, W
    # T[a,b,c] = sum_{c} U[a,c] * V[b,c] * W[c,c]
    # This is a tensor product sum.
    
    # Use einsum: T = np.einsum('ac,bc,dc->abd', U, V, W)
    # But that's not right. Let me think.
    # T[a,b,c] = sum_{r} U[a,r] * V[b,r] * W[c,r]
    # So T = np.einsum('ar,br,cr->abc', U, V, W)
    
    T_computed = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build the target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Check if they match
    if not np.allclose(T_computed, T_target, atol=1e-9):
        return False
    return True

def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 100  # Leave 20s margin
    
    # Trivial decomposition
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = rank_trivial
    
    # Try to find a lower-rank decomposition
    # Use random search with small integer entries
    
    np.random.seed(42)
    
    for trial in range(1000):
        if time.time() > deadline:
            break
        
        # Try a random rank
        # Start with rank = best_rank - 1 and go down
        target_rank = best_rank - 1
        if target_rank < 35:  # Lower bound
            break
        
        # Random initialization
        U = np.random.randint(-1, 2, (n*m, target_rank))
        V = np.random.randint(-1, 2, (m*p, target_rank))
        W = np.random.randint(-1, 2, (p*n, target_rank))
        
        # Verify
        if verify_decomposition(U, V, W, n, m, p):
            best = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
            best_rank = target_rank
            print(f"Found rank {best_rank} at trial {trial}")
    
    return best
