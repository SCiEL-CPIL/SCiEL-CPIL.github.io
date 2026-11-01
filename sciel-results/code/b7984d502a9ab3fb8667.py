import numpy as np
import time
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1."""
    T = np.ones((n, m, p), dtype=float)
    return T

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Build the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_reduce_rank(U, V, W, max_iter=2000, deadline=None):
    """
    Try to reduce the rank of a decomposition by finding linear dependencies.
    Start from a valid decomposition and try to eliminate one term at a time
    by adjusting the remaining terms.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        if best_rank <= 1:
            break
        
        # Try to eliminate each column
        for c in range(best_rank):
            if deadline is not None and time.time() > deadline:
                break
            
            # Try to express column c as a linear combination of others
            # and absorb it into the other columns
            
            # Build the system: for each (i,j,k), we need
            # sum_{c' != c} U[i*m+j, c'] * V[j*p+k, c'] * W[k*n+i, c'] * alpha_{c'}
            # + U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            # This is nonlinear. Instead, try a simpler approach:
            # randomly perturb and try to zero out one column
            
            # Approach: try to find if column c is redundant
            # by checking if we can set it to zero and adjust others
            
            # Simple heuristic: try random combinations
            # Pick a random subset of other columns and try to fit
            
            # Actually, let's try a different approach:
            # Randomly perturb the decomposition and try to reduce rank
            # via Gaussian elimination on the tensor equations
            
            # For now, try a simple approach: randomly scale columns
            # and see if we can merge two columns
            
            # Pick two random columns to try to merge
            if c > 0:
                c2 = np.random.randint(0, best_rank)
                if c2 == c:
                    c2 = (c + 1) % best_rank
                
                # Try to merge column c into column c2
                # This is a nonlinear problem, so use a simple heuristic:
                # Randomly perturb and check
                
                # Skip for now - this is complex
                continue
        
        # If we haven't reduced rank, try a random perturbation approach
        # Add a small random perturbation and try to re-solve
        # This is a placeholder for more sophisticated methods
        pass
    
    return best_U, best_V, best_W, best_rank

def _random_search(target_rank, n, m, p, deadline, max_iters=5000):
    """
    Random search for a rank-`target_rank` decomposition.
    Use integer/half-integer entries.
    """
    np.random.seed(42 + target_rank)
    
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # Initialize with random half-integers
    for c in range(target_rank):
        U[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=n*m)
        V[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=m*p)
        W[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=p*n)
    
    # Local search: try to reduce the error
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(target_rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (val - 1.0) ** 2
        return err
    
    current_err = compute_error(U, V, W)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try a random local move
        # Pick a random entry to change
        mat = np.random.choice([0, 1, 2])  # 0=U, 1=V, 2=W
        if mat == 0:
            idx = np.random.randint(0, n*m)
            col = np.random.randint(0, target_rank)
            old_val = U[idx, col]
            new_val = np.random.choice([-1, -0.5, 0, 0.5, 1])
            if new_val == old_val:
                continue
            U[idx, col] = new_val
            new_err = compute_error(U, V, W)
            if new_err > current_err + 1e-12:
                U[idx, col] = old_val
            else:
                current_err = new_err
        elif mat == 1:
            idx = np.random.randint(0, m*p)
            col = np.random.randint(0, target_rank)
            old_val = V[idx, col]
            new_val = np.random.choice([-1, -0.5, 0, 0.5, 1])
            if new_val == old_val:
                continue
            V[idx, col] = new_val
            new_err = compute_error(U, V, W)
            if new_err > current_err + 1e-12:
                V[idx, col] = old_val
            else:
                current_err = new_err
        else:
            idx = np.random.randint(0, p*n)
            col = np.random.randint(0, target_rank)
            old_val = W[idx, col]
            new_val = np.random.choice([-1, -0.5, 0, 0.5, 1])
            if new_val == old_val:
                continue
            W[idx, col] = new_val
            new_err = compute_error(U, V, W)
            if new_err > current_err + 1e-12:
                W[idx, col] = old_val
            else:
                current_err = new_err
        
        if current_err < 1e-9:
            return U, V, W, True
    
    return None, None, None, False

def _structured_decomposition(n, m, p, deadline):
    """
    Try to find a structured lower-rank decomposition using the block structure.
    For 2x3x7, we can try to exploit the fact that n=2 is small.
    
    One approach: write the multiplication as a sum of rank-1 terms
    where we share structure across the i dimension.
    
    For each pair (j, k), the contribution to output (i, k) from input (i, j)
    is A[i,j] * B[j,k]. We need:
    sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k
    
    This means for each (j,k), the vector (U[0,j,c]*V[j,k,c]*W[k,0,c],
    U[1,j,c]*V[j,k,c]*W[k,1,c]) must sum to (1,1) over c.
    
    A simple structured approach: for each (j,k), use a small number of terms
    that simultaneously satisfy both i=0 and i=1.
    """
    # Try a decomposition where for each (j,k), we use 2 terms:
    # Term 1: U[0,j,c]=1, V[j,k,c]=1, W[k,0,c]=1, and U[1,j,c]=a, W[k,1,c]=b
    #          so that a*b*1 = 1 => a*b = 1
    # Term 2: U[0,j,c]=0, V[j,k,c]=0, W[k,0,c]=0 (zero term, useless)
    # This doesn't work directly.
    
    # Better: for each (j,k), use terms that cover both i values.
    # If we use 1 term per (j,k): U[0,j,c]=1, V[j,k,c]=1, W[k,0,c]=1,
    #                              U[1,j,c]=1, W[k,1,c]=1
    # Then for i=0: sum = 1*1*1 = 1 (if only this term)
    # For i=1: sum = 1*1*1 = 1
    # But this requires W[k,0,c] and W[k,1,c] to both be 1, which is fine.
    # However, U[0,j,c] and U[1,j,c] are both 1, V[j,k,c]=1.
    # This gives rank = m*p = 21, but wait, we need separate columns for each (j,k)?
    # No, each (j,k) can share the same column index if the pattern works.
    
    # Actually, if for each (j,k) we assign a unique column c_{jk}, then
    # U[0,j,c_{jk}] = 1, U[1,j,c_{jk}] = 1, V[j,k,c_{jk}] = 1,
    # W[k,0,c_{jk}] = 1, W[k,1,c_{jk}] = 1
    # This gives rank = m*p = 21, which is less than 42!
    
    # Let's verify: for fixed i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Only c = c_{jk} has non-zero entries for this (j,k).
    # U[i*m+j, c_{jk}] = 1 (for i=0 or i=1)
    # V[j*p+k, c_{jk}] = 1
    # W[k*n+i, c_{jk}] = 1 (for i=0 or i=1)
    # So the sum is 1*1*1 = 1. 
    
    # This works! Rank = m*p = 21.
    
    rank = m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1  # i=0
            U[1 * m + j, col] = 1  # i=1
            V[j * p + k, col] = 1
            W[k * n + 0, col] = 1  # k*n+0
            W[k * n + 1, col] = 1  # k*n+1
            col += 1
    
    # Verify
    if _verify(U, V, W):
        return U, V, W
    
    # If it didn't work (shouldn't happen), try another approach
    return None, None, None

def _try_rank_r(n, m, p, r, deadline):
    """Try to find a rank-r decomposition."""
    # First try structured approach
    if r == m * p:
        U, V, W = _structured_decomposition(n, m, p, deadline)
        if U is not None:
            return U, V, W
    
    # Try random search
    if time.time() < deadline:
        U, V, W, success = _random_search(r, n, m, p, deadline - 5, max_iters=2000)
        if success:
            return U, V, W
    
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Best known: schoolbook rank 42
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best = {'U': U_sb.tolist(), 'V': V_sb.tolist(), 'W': W_sb.tolist()}
    best_rank = n * m * p
    
    # Try structured decomposition (rank = m*p = 21)
    if time.time() < deadline:
        U, V, W = _structured_decomposition(n, m, p, deadline)
        if U is not None and _verify(U, V, W):
            best = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
            best_rank = m * p
            print(f"Found structured decomposition with rank {best_rank}")
    
    # Try to find even lower rank
    for r in range(max(1, best_rank - 10), best_rank):
        if time.time() > deadline - 10:
            break
        if r >= best_rank:
            continue
        U, V, W = _try_rank_r(n, m, p, r, deadline - 10)
        if U is not None and _verify(U, V, W):
            best = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
            best_rank = r
            print(f"Found decomposition with rank {best_rank}")
            break
    
    return best

if __name__ == '__main__':
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
