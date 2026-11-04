"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
import time
import itertools
import random

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """
    Fast verification using tensor contraction.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: sum_r W[k*n+i, r] * U[a*m+b, r] * V[c*p+d, r] == 1 iff a==i and b==c and d==k
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    U_t = U.T  # (rank, n*m)
    V_t = V.T  # (rank, m*p)
    W_t = W.T  # (rank, p*n)
    
    # For each (a, b, c, d), check:
    # sum_r W_t[r, k*n+i] * U_t[r, a*m+b] * V_t[r, c*p+d]
    # where i=a, k=d (the expected non-zero positions)
    # Actually, the full check: for ALL (i,k,a,b,c,d)
    # sum_r W_t[r, k*n+i] * U_t[r, a*m+b] * V_t[r, c*p+d] == delta(a,i)*delta(b,c)*delta(d,k)
    
    # Efficient approach: build the full 6D tensor and compare
    # T[i,k,a,b,c,d] = sum_r W_t[r, k*n+i] * U_t[r, a*m+b] * V_t[r, c*p+d]
    
    # We can compute this as: for each r, outer product of three vectors
    # But that's rank * n * p * n * m * m * p which is too big for large rank.
    
    # Alternative: check only the non-zero positions and a sample of zero positions
    # Non-zero: a==i, b==c, d==k -> n * m * p = 50 positions
    # Zero: the rest -> n*p*n*m*m*p - 50 = 2500 - 50 = 2450 positions
    
    # Let's do the full check efficiently using einsum
    # T = np.einsum('r,ri,rj,rk->ijk', ...) -- no, let me think differently
    
    # For fixed (a,b,c), compute S[i,k] = sum_r W_t[r, k*n+i] * U_t[r, a*m+b] * V_t[r, c*p+d]
    # This should equal delta(a,i) * delta(b,c) * delta(k,d) for all (i,k)
    # Wait, d is fixed in V_t[r, c*p+d], so for fixed (a,b,c,d):
    # S[i,k] = sum_r W_t[r, k*n+i] * (U_t[:, a*m+b] * V_t[:, c*p+d])_r
    # This should equal 1 if a==i and b==c and d==k, else 0
    
    # Vectorized: for each (a,b,c,d), compute the vector v_r = U_t[r,a*m+b]*V_t[r,c*p+d]
    # Then S = W_t @ v (where W_t is (rank, p*n) reshaped to (p, n, rank)? No...)
    # W_t[r, k*n+i] so S[i,k] = sum_r W_t[r, k*n+i] * v_r
    # = sum_r v_r * W_t[r, k*n+i]
    # If we reshape W_t to (rank, p, n), then S = np.einsum('r,rki->ik', v, W_t_reshaped)
    
    # Total: n*m*m*p = 250 combinations of (a,b,c,d)
    # Each requires a rank-length vector multiply and a (p,n) matrix-vector product
    # This is feasible.
    
    W_reshaped = W_t.reshape(rank, p, n)  # W_reshaped[r, k, i] = W_t[r, k*n+i]
    
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    v = U_t[:, a*m+b] * V_t[:, c*p+d]  # (rank,)
                    S = np.einsum('r,rki->ik', v, W_reshaped)  # (p, n) = (k, i)
                    # S[k, i] should be 1 if a==i and b==c and d==k, else 0
                    expected = np.zeros((p, n))
                    if b == c:
                        expected[d, a] = 1.0
                    if not np.allclose(S, expected, atol=tol):
                        return False
    return True

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # Start with schoolbook
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    random.seed(42)
    
    # Strategy 1: Try to find lower rank via structured perturbations of schoolbook
    # For n=2, m=5, p=5: schoolbook is 50.
    # Known lower bounds: max(n*p, m*p, n*m) = max(10, 25, 10) = 25
    # But for 2x5x5, the tensor rank is likely close to 50 or slightly below.
    
    # Try ranks from 49 down to 25
    for target_rank in range(best_rank - 1, 24, -1):
        if not still_going():
            break
        
        # Attempt to find a decomposition of this rank
        # Use random sparse initialization with {-1, 0, 1} entries
        max_attempts = 5 if target_rank > 40 else 10
        found = False
        for attempt in range(max_attempts):
            if not still_going():
                break
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Each column r should have a "pattern" that contributes to the tensor
            # Random sparse with {-1, 1}
            for r in range(target_rank):
                # U column: sparse, ~2-3 nonzeros
                nnz_u = random.randint(2, 4)
                idx_u = random.sample(range(n * m), min(nnz_u, n * m))
                for idx in idx_u:
                    U[idx, r] = random.choice([-1, 1])
                
                # V column: sparse, ~2-3 nonzeros
                nnz_v = random.randint(2, 4)
                idx_v = random.sample(range(m * p), min(nnz_v, m * p))
                for idx in idx_v:
                    V[idx, r] = random.choice([-1, 1])
                
                # W column: sparse, ~1-2 nonzeros
                nnz_w = random.randint(1, 3)
                idx_w = random.sample(range(p * n), min(nnz_w, p * n))
                for idx in idx_w:
                    W[idx, r] = random.choice([-1, 1])
            
            if _verify_fast(U, V, W, n, m, p):
                best_rank = target_rank
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break
        
        if found:
            # Found a lower rank, continue searching for even lower
            continue
    
    # Strategy 2: Try structured "pairing" approaches
    # For n=2, we can try to combine terms across the two rows
    if still_going():
        # Try rank 49: start from schoolbook, merge two terms
        # Schoolbook term (i,j,k) has U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
        # Can we combine (i=0,j,k) and (i=1,j,k')? Not directly.
        # Try: for fixed j, combine k terms
        # This is complex; skip for now unless we have time
        
        # Try a few more random attempts at the best rank found
        for extra_attempt in range(5):
            if not still_going():
                break
            # Try to improve on current best_rank
            target_rank = max(best_rank - 1, 25)
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            for r in range(target_rank):
                nnz_u = random.randint(2, 5)
                idx_u = random.sample(range(n * m), min(nnz_u, n * m))
                for idx in idx_u:
                    U[idx, r] = random.choice([-1, 1])
                
                nnz_v = random.randint(2, 5)
                idx_v = random.sample(range(m * p), min(nnz_v, m * p))
                for idx in idx_v:
                    V[idx, r] = random.choice([-1, 1])
                
                nnz_w = random.randint(1, 3)
                idx_w = random.sample(range(p * n), min(nnz_w, p * n))
                for idx in idx_w:
                    W[idx, r] = random.choice([-1, 1])
            
            if _verify_fast(U, V, W, n, m, p):
                best_rank = target_rank
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best
# EVOLVE-BLOCK-END
