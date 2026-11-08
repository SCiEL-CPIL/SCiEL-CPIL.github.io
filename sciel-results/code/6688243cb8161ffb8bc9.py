"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify exact tensor decomposition."""
    n, m, p = int(n), int(m), int(p)
    rank = len(U[0])
    # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Must equal 1 if i,j,k all valid, else 0
    # Actually the tensor for matrix mult: T[i,j,k] = delta_{i,j,k} in the right sense
    # The multiplication tensor: for C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: T[i,j,k] where the "1" entries are when we pick the right (i,j,k)
    # The rank-r decomposition: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # T[i,j,k] should be 1 for all valid (i,j,k) in the multiplication tensor
    # Wait - the multiplication tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No that's not right either.
    
    # The matrix multiplication tensor: T is a 3-tensor of shape (n*m, m*p, p*n)
    # T[(i,j), (j,k), (k,i)] = 1, else 0
    # In the factorization: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where a = i*m+j, b = j*p+k, c = k*n+i
    
    # So T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid i,j,k
    # And T[a,b,c] = 0 otherwise
    
    # Let's verify by computing the full tensor and checking
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Compute T = sum_r U[:,r] outer V[:,r] outer W[:,r]
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is expensive for large tensors. Let's check specific entries.
    
    # For each valid (i,j,k), T[(i*m+j), (j*p+k), (k*n+i)] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # For invalid entries, T should be 0
    # Check a sample of invalid entries
    all_a = set()
    for i in range(n):
        for j in range(m):
            all_a.add(i * m + j)
    all_b = set()
    for j in range(m):
        for k in range(p):
            all_b.add(j * p + k)
    all_c = set()
    for k in range(p):
        for i in range(n):
            all_c.add(k * n + i)
    
    # Check some random invalid entries
    valid_triples = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_triples.add((i*m+j, j*p+k, k*n+i))
    
    rng = random.Random(42)
    for _ in range(200):
        a = rng.randint(0, n*m - 1)
        b = rng.randint(0, m*p - 1)
        c = rng.randint(0, p*n - 1)
        if (a, b, c) in valid_triples:
            continue
        val = 0.0
        for r in range(rank):
            val += U[a, r] * V[b, r] * W[c, r]
        if abs(val) > 1e-9:
            return False
    
    return True

def _round_to_half(x):
    """Round to nearest half-integer."""
    return round(2 * x) / 2.0

def _round_matrix_to_half(M):
    """Round all entries of matrix M to nearest half-integer."""
    return np.array([[_round_to_half(x) for x in row] for row in M])

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Best known: schoolbook
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(nm)]
    best_V = [[0] * best_rank for _ in range(mp)]
    best_W = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    rng = np.random.RandomState(42)
    
    # Try various ranks from low to high
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But realistically, for 3x5x6, the rank is likely between 30 and 90
    # Known bounds: rank >= max(n*m, m*p, p*n), rank <= n*m*p
    
    # Let's try ranks from 30 upward
    for rank in range(30, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts for this rank
        max_restarts = 50 if rank < 50 else 20
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U = rng.randn(nm, rank)
            V = rng.randn(mp, rank)
            W = rng.randn(pn, rank)
            
            # Normalize to avoid overflow
            U = U / np.linalg.norm(U) * np.sqrt(nm)
            V = V / np.linalg.norm(V) * np.sqrt(mp)
            W = W / np.linalg.norm(W) * np.sqrt(pn)
            
            # Iterative optimization: minimize ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
            # Using gradient descent or alternating minimization
            
            # Alternating minimization
            for it in range(30):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each a, U[a,:] should minimize sum_{b,c} T[a,b,c] V[b,:] W[c,:] - U[a,:]
                # This is a linear system
                # Let's use a simpler approach: just do a few iterations of coordinate descent
                
                # Actually, let's use a different approach: 
                # Generate random rank-r decompositions and try to round them
                pass
            
            # Alternative approach: use random projections
            # Create a random rank-r approximation and try to make it exact
            
            # Let's try a different strategy:
            # 1. Start with random U, V, W of shape (nm, rank), (mp, rank), (pn, rank)
            # 2. For each (i,j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # 3. For invalid (a,b,c), we need sum_r U[a,r] * V[b,r] * W[c,r] = 0
            
            # This is a system of equations. Let's set up the equations:
            # There are n*m*p = 90 equations for the "1" entries
            # And many more for the "0" entries
            
            # Actually, let's just try random matrices and check if rounding works
            # This is a shot in the dark but let's try
            
            # Better approach: use the fact that we can choose U, V, W freely
            # Let's parameterize and solve
            
            # For a given rank r, we have nm*r + mp*r + pn*r unknowns
            # And we need to satisfy:
            # 1. For all (i,j,k): sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
            # 2. For all (a,b,c) not in valid set: sum_r U[a,r] V[b,r] W[c,r] = 0
            
            # This is a system of polynomial equations. Hard to solve directly.
            
            # Let's try a heuristic: random init + local search
            U = rng.randn(nm, rank)
            V = rng.randn(mp, rank)
            W = rng.randn(pn, rank)
            
            # Scale to unit norm rows
            for r in range(rank):
                if np.linalg.norm(U[:, r]) > 0:
                    U[:, r] /= np.linalg.norm(U[:, r])
                if np.linalg.norm(V[:, r]) > 0:
                    V[:, r] /= np.linalg.norm(V[:, r])
                if np.linalg.norm(W[:, r]) > 0:
                    W[:, r] /= np.linalg.norm(W[:, r])
            
            # Try to optimize
            for it in range(50):
                if time.time() > deadline - 10:
                    break
                
                # Compute current tensor values
                # For each valid (i,j,k), compute the sum
                errors = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            errors.append(val - 1.0)
                
                # Check invalid entries (sample)
                rng_check = np.random.RandomState(it)
                for _ in range(50):
                    a = rng_check.randint(0, nm)
                    b = rng_check.randint(0, mp)
                    c = rng_check.randint(0, pn)
                    # Check if this is a valid triple
                    valid = False
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                if i*m+j == a and j*p+k == b and k*n+i == c:
                                    valid = True
                    if not valid:
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        errors.append(val)
                
                if not errors:
                    break
                err = np.array(errors)
                loss = np.sum(err ** 2)
                
                if loss < 1e-12:
                    break
                
                # Gradient step
                lr = 0.01
                # Compute gradients
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            grad = val - 1.0
                            dU[a, :] += grad * V[b, :] * W[c, :]
                            dV[b, :] += grad * U[a, :] * W[c, :]
                            dW[c, :] += grad * U[a, :] * V[b, :]
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Round to half-integers
            U_rounded = _round_matrix_to_half(U)
            V_rounded = _round_matrix_to_half(V)
            W_rounded = _round_matrix_to_half(W)
            
            # Verify
            if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                best_rank = rank
                best_U = U_rounded.tolist()
                best_V = V_rounded.tolist()
                best_W = W_rounded.tolist()
                break
        
        if best_rank == rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
