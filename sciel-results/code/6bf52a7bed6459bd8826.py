import numpy as np
import time
from fractions import Fraction
from collections import defaultdict

# Precompute the schoolbook decomposition for 3x5 * 5x7
# n=3, m=5, p=7
N, M, P = 3, 5, 7
RANK = N * M * P  # 105

def _build_schoolbook():
    """Build the 105 rank-1 tensors for schoolbook matrix multiplication."""
    # Each tensor is (u_vec, v_vec, w_vec)
    # u_vec has length N*M = 15
    # v_vec has length M*P = 35
    # w_vec has length P*N = 21
    
    tensors = []
    for i in range(N):
        for j in range(M):
            for k in range(P):
                u = [0.0] * (N * M)
                v = [0.0] * (M * P)
                w = [0.0] * (P * N)
                
                u[i * M + j] = 1.0
                v[j * P + k] = 1.0
                w[k * N + i] = 1.0
                
                tensors.append((u, v, w))
    
    return tensors

def _verify_decomposition(U, V, W, tol=1e-9):
    """
    Verify that the tensor decomposition exactly reconstructs the multiplication tensor.
    T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    where the multiplication tensor has T[a, b, c] = 1 iff a = i*M+j, b = j*P+k, c = k*N+i
    for some i, j, k.
    """
    n, m, p = N, M, P
    rank = len(U[0])
    
    # Build the expected tensor (sparse)
    # T[a, b, c] = 1 when a = i*m + j, b = j*p + k, c = k*n + i
    # For each (i, j, k), there's exactly one nonzero entry
    
    # Check all nonzero entries of the multiplication tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                # Compute T[a, b, c] from the decomposition
                val = 0.0
                for r in range(rank):
                    val += U[a][r] * V[b][r] * W[c][r]
                
                if abs(val - 1.0) > tol:
                    return False
    
    # Check that entries which should be zero are actually zero
    # The multiplication tensor has exactly n*m*p = 105 nonzero entries
    # Any other (a, b, c) triple should give 0
    
    # Build set of valid (a, b, c) triples
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid.add((a, b, c))
    
    # Sample some invalid triples to check
    import random
    rng = random.Random(42)
    for _ in range(200):
        a = rng.randint(0, n * m - 1)
        b = rng.randint(0, m * p - 1)
        c = rng.randint(0, p * n - 1)
        
        if (a, b, c) in valid:
            continue
        
        val = 0.0
        for r in range(rank):
            val += U[a][r] * V[b][r] * W[c][r]
        
        if abs(val) > tol:
            return False
    
    return True

def _try_peeling(tensors, target_rank, deadline):
    """
    Try to find a linear combination of rank-1 tensors that vanishes on the multiplication tensor,
    allowing us to eliminate one term.
    
    We look for coefficients c_1, ..., c_R such that:
    sum_r c_r * (u_r ⊗ v_r ⊗ w_r) = 0 on the support of the multiplication tensor.
    
    This means for each valid (a, b, c): sum_r c_r * U[a,r] * V[b,r] * W[c,r] = 0
    """
    R = len(tensors)
    n, m, p = N, M, P
    
    # For each valid (a,b,c), the constraint is: sum_r c_r * U[a,r] * V[b,r] * W[c,r] = 0
    # This is a linear system in c_r.
    
    # Build the constraint matrix A where A[eq, r] = U[a,r] * V[b,r] * W[c,r]
    # for each valid (a,b,c) = eq
    
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                row = []
                for r in range(R):
                    u_r, v_r, w_r = tensors[r]
                    row.append(u_r[a] * v_r[b] * w_r[c])
                constraints.append(row)
    
    A = np.array(constraints, dtype=float)
    # A has shape (n*m*p, R) = (105, 105)
    
    # We want a nonzero vector c in the null space of A
    # i.e., A @ c = 0
    
    # Compute null space
    try:
        _, S, Vt = np.linalg.svd(A, full_matrices=True)
    except np.linalg.LinAlgError:
        return None
    
    # The null space dimension is R - rank(A)
    rank_A = np.sum(S > 1e-10)
    null_dim = R - rank_A
    
    if null_dim <= 0:
        return None
    
    # The null space vectors are the last null_dim rows of Vt
    null_vecs = Vt[rank_A:]  # shape (null_dim, R)
    
    # Try to use each null space vector to eliminate one term
    # If c_r is the null vector, we can modify the decomposition
    # to eliminate term r (if c_r != 0)
    
    for ns_idx in range(null_dim):
        c = null_vecs[ns_idx]
        
        # Find the largest coefficient to eliminate
        max_idx = np.argmax(np.abs(c))
        max_val = c[max_idx]
        
        if abs(max_val) < 1e-10:
            continue
        
        # We want to eliminate term max_idx
        # New decomposition: for each r != max_idx,
        # new_U[r] = U[r] + (c[r]/c[max_idx]) * U[max_idx]
        # Similarly for V and W
        # And we drop term max_idx
        
        new_tensors = []
        for r in range(R):
            if r == max_idx:
                continue
            
            u_r, v_r, w_r = tensors[r]
            scale = c[r] / max_val
            
            new_u = [u_r[i] + scale * tensors[max_idx][0][i] for i in range(n * m)]
            new_v = [v_r[i] + scale * tensors[max_idx][1][i] for i in range(m * p)]
            new_w = [w_r[i] + scale * tensors[max_idx][2][i] for i in range(p * n)]
            
            new_tensors.append((new_u, new_v, new_w))
        
        # Verify
        U = [t[0] for t in new_tensors]
        V = [t[1] for t in new_tensors]
        W = [t[2] for t in new_tensors]
        
        if _verify_decomposition(U, V, W):
            return new_tensors
        
        # Check deadline
        if time.time() > deadline:
            return None
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook decomposition
    tensors = _build_schoolbook()
    best_rank = len(tensors)
    best_tensors = tensors
    
    # Try to peel off one term
    peeled = _try_peeling(tensors, best_rank - 1, deadline)
    if peeled is not None:
        best_tensors = peeled
        best_rank = len(peeled)
    
    # Convert to the required format
    U = [t[0] for t in best_tensors]
    V = [t[1] for t in best_tensors]
    W = [t[2] for t in best_tensors]
    
    return {"U": U, "V": V, "W": W}
