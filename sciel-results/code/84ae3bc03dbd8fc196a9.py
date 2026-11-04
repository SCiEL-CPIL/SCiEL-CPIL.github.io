import numpy as np
import time
import random

def construct():
    n, m, p = 2, 5, 5
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # Schoolbook decomposition of rank n*m*p = 50
    trivial_rank = n * m * p
    U_triv = np.zeros((n * m, trivial_rank))
    V_triv = np.zeros((m * p, trivial_rank))
    W_triv = np.zeros((p * n, trivial_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j, col] = 1.0
                V_triv[j * p + k, col] = 1.0
                W_triv[k * n + i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    best_rank = trivial_rank
    
    def verify(U, V, W, tol=1e-8):
        """Verify exact decomposition. Returns True if valid."""
        rank = U.shape[1]
        # Check all non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val - 1.0) > tol:
                        return False
        # Check zero entries
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i1 = a // m
                    j1 = a % m
                    j2 = b // p
                    k1 = b % p
                    k2 = c // n
                    i2 = c % n
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        continue
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > tol:
                        return False
        return True
    
    # Try structured lower-rank decompositions
    # Strategy: Use the block structure. For each j, the terms involve A[i,j] and B[j,k].
    # We can try to group j values and use shared rank-1 terms.
    
    # Approach 1: Try rank = n*p + (m-1)*n*p/m type constructions
    # Actually, let's try a more direct approach: random search with verification
    
    # For each candidate rank, try random initializations and check
    for rank in range(trivial_rank - 1, 20, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random initializations for this rank
        max_attempts = 50 if rank > 40 else 100
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with sparse structure
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Initialize with random sparse patterns
            for r in range(rank):
                # Each rank-1 term should touch relevant indices
                # Pick random i, j, k and set up the term
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
                
                # Add contributions to make this term contribute to T[(i,j),(j,k),(k,i)]
                # Use random coefficients to allow combination
                scale = random.choice([1.0, -1.0, 0.5, -0.5, 2.0, -2.0])
                
                U[i * m + j, r] = scale
                V[j * p + k, r] = scale
                W[k * n + i, r] = 1.0 / (scale * scale) if scale != 0 else 1.0
            
            # Verify
            if verify(U, V, W):
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = rank
                break  # Found a valid decomposition for this rank, try lower
        
        # If we found a valid one for this rank, continue to try lower
        if best_rank == rank:
            continue
    
    # Try a structured approach: exploit the fact that for n=2, we might
    # be able to use rank = 2*p + something
    # Specifically, try rank = 2*p = 10 as a lower bound attempt
    
    if time.time() < deadline - 20:
        # Try structured decomposition based on separating the two i values
        # For i=0 and i=1, the tensors are somewhat independent in the W component
        # But they share U and V components
        
        # Try rank = m*p + n*p = 25 + 10 = 35? Or other structured ranks
        for candidate_rank in [40, 35, 30, 25, 20]:
            if candidate_rank >= best_rank:
                continue
            if time.time() > deadline - 10:
                break
            
            # Try to build a structured solution
            # Idea: use m*p terms for one part and n*p for another
            # This is heuristic - just try random with this rank
            for attempt in range(30):
                if time.time() > deadline - 10:
                    break
                
                U = np.zeros((n * m, candidate_rank))
                V = np.zeros((m * p, candidate_rank))
                W = np.zeros((p * n, candidate_rank))
                
                # Structured initialization
                # First m*p terms: handle the "main" contributions
                for j in range(m):
                    for k in range(p):
                        idx = j * p + k
                        if idx < candidate_rank:
                            for i in range(n):
                                U[i * m + j, idx] = 1.0
                                V[j * p + k, idx] = 1.0
                                W[k * n + i, idx] = 1.0 / n if n > 0 else 1.0
                
                # Add random perturbation to remaining columns
                for r in range(candidate_rank):
                    if random.random() < 0.3:
                        i = random.randint(0, n - 1)
                        j = random.randint(0, m - 1)
                        k = random.randint(0, p - 1)
                        scale = random.gauss(1.0, 0.5)
                        U[i * m + j, r] += scale
                        V[j * p + k, r] += scale
                        W[k * n + i, r] += scale / (scale * scale + 1e-6)
                
                if verify(U, V, W):
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    best_rank = candidate_rank
                    break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
