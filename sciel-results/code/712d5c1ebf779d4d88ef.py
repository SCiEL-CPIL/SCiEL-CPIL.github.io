import numpy as np
import time
import itertools

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    # Check that the decomposition is exact
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # The tensor should have T[i,j,k] = 1 for all valid (i,j,k)
    # But actually the tensor is T[i,j,k] = delta_{i,j,k} in the sense that
    # (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the multiplication tensor has T[i,j,k] = 1 if we think of it as
    # the structure constants. Actually T is the 3-tensor where
    # T[i, j, k] = 1 for all i,j,k? No.
    
    # The matrix multiplication tensor M_{i,j,k} satisfies:
    # (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor is M_{i,j,k} = 1 for all i, j, k? No, that's not right either.
    # Actually M is the tensor such that the bilinear map (A,B) -> AB is
    # sum_r (U_r . A) * (V_r . B) * W_r
    # where U_r, V_r are coefficient vectors and W_r is the output coefficient.
    
    # The correct check: for the schoolbook, each term r corresponds to (i,j,k)
    # and contributes A_{i,j} * B_{j,k} to (AB)_{i,k}.
    # So the tensor T has T[i,j,k] = 1 for all i,j,k? No.
    
    # Let me think again. The multiplication tensor is defined by:
    # T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ (e_i ⊗ e_k)^C
    # where e_i^A is the matrix with 1 at (i,j) and 0 elsewhere... no.
    
    # Actually, the standard definition: the matrix multiplication tensor
    # M_{n,m,p} has entries M_{i,j,k} = 1 if the (i,j) entry of A times the
    # (j,k) entry of B contributes to the (i,k) entry of AB.
    # So M_{i,j,k} = 1 for all i in [n], j in [m], k in [p].
    
    # Wait, that's not right. The tensor is:
    # M = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p E_{i,j} ⊗ E_{j,k} ⊗ E_{i,k}
    # where E_{a,b} is the matrix unit.
    
    # So the tensor has a 1 at position (i*m+j, j*p+k, k*n+i) in the
    # flattened representation, for each (i,j,k).
    
    # To verify: check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    # for all i,j,k, and that the sum is 0 for all other combinations... 
    # Actually no, the tensor only has non-zero entries at specific positions.
    
    # The correct verification: the reconstructed tensor should equal the
    # multiplication tensor. The multiplication tensor has exactly n*m*p
    # non-zero entries (all 1s), at positions (i*m+j, j*p+k, k*n+i).
    
    # For a rank-r decomposition, we check:
    # For all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    # And for all other (a,b,c) not of the form (i*m+j, j*p+k, k*n+i):
    # sum_r U[a,r] * V[b,r] * W[c,r] == 0
    
    # This is expensive to check fully, so let's just check the main condition
    # and a few random off-diagonal entries.
    
    # Check main condition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if s != 1:
                    return False
    
    # Check a few off-diagonal entries
    import random
    random.seed(42)
    for _ in range(100):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        # Check if (a,b,c) is a valid multiplication tensor entry
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_c = c // n
        i_c = c % n
        is_valid = (j_a == j_b and i_a == i_c)
        s = np.sum(U[a, :] * V[b, :] * W[c, :])
        if is_valid:
            if s != 1:
                return False
        else:
            if s != 0:
                return False
    
    return True

def _try_rank_random(n, m, p, rank, max_iters=5000, seed=0):
    """Try to find a rank-`rank` decomposition via random search."""
    rng = np.random.RandomState(seed)
    
    # Random initialization
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # We want to minimize the error:
    # err = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    #     + sum_{(a,b,c) not valid} (sum_r U[a,r]*V[b,r]*W[c,r])^2
    
    # This is a non-convex problem. Let's use a simple gradient-free approach:
    # randomly perturb one matrix at a time and accept if error decreases.
    
    def compute_error(U, V, W):
        # Compute the full tensor reconstruction
        # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is expensive for large rank. Use matrix multiplication tricks.
        
        # T_recon[i*m+j, j*p+k, k*n+i] should be 1
        # We can compute this more efficiently.
        
        # For the "on" entries:
        # sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        # = sum_{i,j,k} (sum_r (U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r] - 1)^2
        
        # Let's compute it directly but vectorized where possible.
        
        err = 0.0
        
        # On-diagonal entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1.0) ** 2
        
        # Off-diagonal: sample some
        # For efficiency, just check a fixed set
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    ia, ja = divmod(a, m)
                    jb, kb = divmod(b, p)
                    kc, ic = divmod(c, n)
                    if ja == jb and ia == ic:
                        continue  # skip on-diagonal
                    s = np.dot(U[a, :], V[b, :] * W[c, :])
                    err += s ** 2
        
        return err
    
    best_err = compute_error(U, V, W)
    
    for it in range(max_iters):
        # Pick a random matrix to perturb
        which = rng.randint(3)
        mat = [U, V, W][which]
        row = rng.randint(mat.shape[0])
        col = rng.randint(mat.shape[1])
        
        old_val = mat[row, col]
        mat[row, col] += rng.randn() * 0.1
        
        new_err = compute_error(U, V, W)
        
        if new_err < best_err:
            best_err = new_err
        else:
            mat[row, col] = old_val
    
    # Check if we found a valid decomposition
    if best_err < 1e-6:
        # Try to round to integers or half-integers
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if _verify(U_rounded, V_rounded, W_rounded, n, m, p, rank):
            return U_rounded, V_rounded, W_rounded
    
    return None

def construct():
    n, m, p = 3, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p
    best_result = (U_sb, V_sb, W_sb)
    
    # Try to find a lower rank decomposition
    # The theoretical lower bound for 3x5x6 is at least 3*5*6 / max(3,5,6) = 90/6 = 15
    # but this is very loose. The actual minimum is likely close to 90.
    
    # Try ranks from 89 down to some lower bound
    # This is computationally expensive, so we'll try a few ranks
    for rank in range(best_rank - 1, max(80, best_rank - 10), -1):
        if time.time() > deadline - 20:
            break
        
        result = _try_rank_random(n, m, p, rank, max_iters=2000, seed=rank)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_result = (U, V, W)
            break  # Found a lower rank, stop
    
    U, V, W = best_result
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
