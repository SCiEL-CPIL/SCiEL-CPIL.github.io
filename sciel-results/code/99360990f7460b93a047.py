"""Find a rank-minimal exact tensor decomposition for 3x3 by 3x6 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
TIME_LIMIT = 135.0  # Safety margin before 150s kill

def _build_schoolbook():
    """Return the schoolbook (rank 54) decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n=3, m=3, p=6, tol=1e-9):
    """
    Verify the decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    if j == j (i.e., always) and the indices match the tensor.
    Actually the matrix multiplication tensor T[i,j,k] = delta_{i,j} for the relevant positions.
    Wait: for multiplying A (n x m) by B (m x p), the output C[i,k] = sum_j A[i,j] B[j,k].
    The tensor is T[i, j, k] which should be 1 when... actually it's the structure tensor.
    The factorization: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    And T[i,j,k] should equal 1 if this represents the identity in the appropriate sense.
    
    Actually, for matrix multiplication, the tensor T[i,j,k] is defined such that
    (A*B)[i,k] = sum_j A[i,j] * B[j,k].
    The bilinear form: sum_{i,j,k} A[i,j] B[j,k] T[i,j,k] = sum_i,k C[i,k] * 1
    But the standard decomposition is: T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    And we need T[i,j,k] = 1 for all (i,j,k) that are "valid" - actually no.
    
    Let me reconsider. The matrix multiplication tensor of size (n,m,p) has:
    T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    and T[i,j,k] = 1 if j is the "middle" index... 
    
    Actually the standard way: the multiplication algorithm computes C[i,k] = sum_j A[i,j] B[j,k].
    The tensor T is such that <A, B, T> = <C, 1, 1> where T[i,j,k] picks out the term.
    Specifically, T[i,j,k] = 1 for all i,j,k? No, that's not right either.
    
    Let me just check: the schoolbook has U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1 for each (i,j,k).
    So sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for each (i,j,k) tuple (since each (i,j,k) has its own column).
    So T[i,j,k] = 1 for all i,j,k? That doesn't seem right for matrix multiplication.
    
    Wait, I think the convention is different. Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]."
    
    The decomposition is: M[A,B] = sum_r U[:,r] outer V[:,r] outer W[:,r]
    where M is the tensor of shape (n*m, m*p, p*n).
    M[i*m+j, j'*p+k, k'*n+i'] = sum_r U[i*m+j, r] V[j'*p+k, r] W[k'*n+i', r]
    
    For this to represent matrix multiplication:
    When we contract A (n*m vector) with the first mode and B (m*p vector) with the second mode,
    we get a (p*n) vector which is C flattened.
    
    C[k',i'] = sum_{j'} A[i',j'] B[j',k']  ... wait no.
    
    Standard: C[i,k] = sum_j A[i,j] B[j,k]
    If we flatten A as A[i*m+j], B as B[j*p+k], C as C[k*n+i] (following the W convention W[k*n+i,r]),
    then the tensor M satisfies:
    sum_{a,b} A[a] B[b] M[a,b,c] = C[c]
    where a = i*m+j, b = j'*p+k, c = k'*n+i'
    
    For this to work: M[i*m+j, j'*p+k, k'*n+i'] should be 1 when i=i' and j=j' and k=k', else 0.
    So M is the "identity-like" tensor: M[a,b,c] = 1 iff a corresponds to (i,j), b to (j,k), c to (k,i) with matching indices.
    
    In the schoolbook: for each (i,j,k), column c has U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1.
    So M[i*m+j, j*p+k, k*n+i] gets contribution 1 from column (i,j,k).
    And M[i*m+j, j'*p+k', k''*n+i''] gets contribution 1 only if i=i'', j=j', k=k'.
    So M[i*m+j, j'*p+k', k''*n+i''] = 1 iff i=i'' AND j=j' AND k=k'.
    
    This is correct for matrix multiplication!
    """
    n, m, p = n, m, p
    rank = len(U[0])
    # Build M and check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > tol:
                    return False
    # Also check that off-diagonal terms are zero
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i == i2 and j == j2 and k == k2:
                                continue
                            # Check M[i*m+j, j2*p+k2, k*n+i] ... no wait
                            # We need to check all entries
                            pass
    # More carefully: check that M[a,b,c] = 1 iff the indices match
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                # Decode
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                expected = 1.0 if (i_a == i_c and j_a == j_b and k_b == k_c) else 0.0
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - expected) > tol:
                    return False
    return True


def _try_search(start_rank, max_rank, deadline):
    """
    Try to find a decomposition with rank between start_rank and max_rank.
    Uses random initialization and Newton-type optimization.
    """
    n, m, p = DIMENSIONS
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_result = None
    t0 = time.time()
    
    for rank in range(start_rank, min(max_rank + 1, 54)):
        if time.time() > deadline:
            break
        # Try multiple random initializations
        max_attempts = 50 if rank <= 40 else 20
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            rng = np.random.RandomState(attempt * 12345 + rank)
            
            # Initialize with small random values
            scale = 1.0 / (rank ** 0.5)
            U = rng.randn(n*m, rank) * scale
            V = rng.randn(m*p, rank) * scale
            W = rng.randn(p*n, rank) * scale
            
            # Add some structure: initialize with a perturbed schoolbook-like pattern
            # Actually, let's try a structured initialization
            # Use a random linear combination of the schoolbook columns
            # Pick 'rank' random combinations of (i,j,k)
            if rank <= 54:
                indices = []
                for _ in range(rank):
                    i = rng.randint(0, n)
                    j = rng.randint(0, m)
                    k = rng.randint(0, p)
                    indices.append((i, j, k))
                U_init = np.zeros((n*m, rank))
                V_init = np.zeros((m*p, rank))
                W_init = np.zeros((p*n, rank))
                for col, (i, j, k) in enumerate(indices):
                    U_init[i*m+j, col] = 1.0
                    V_init[j*p+k, col] = 1.0
                    W_init[k*n+i, col] = 1.0
                # Perturb
                U = U_init + rng.randn(n*m, rank) * 0.1
                V = V_init + rng.randn(m*p, rank) * 0.1
                W = W_init + rng.randn(p*n, rank) * 0.1
            
            # Newton iteration
            success = False
            for it in range(30):
                if time.time() > deadline:
                    break
                
                # Compute current tensor
                M = np.einsum('ar,br,cr->abc', U, V, W)
                err = M - target
                err_flat = err.flatten()
                err_norm = np.linalg.norm(err_flat)
                
                if err_norm < 1e-10:
                    success = True
                    break
                
                # Compute gradients for Newton step
                # dM/du[a,r] = sum_{b,c} V[b,r] W[c,r] * err[a,b,c]
                # dM/dv[b,r] = sum_{a,c} U[a,r] W[c,r] * err[a,b,c]
                # dM/dw[c,r] = sum_{a,b} U[a,r] V[b,r] * err[a,b,c]
                
                # For Newton, we need the Hessian which is expensive.
                # Let's use a simpler gradient-based approach with line search.
                grad_U = np.einsum('abc,br,cr->ar', err, V, W)
                grad_V = np.einsum('abc,ar,cr->br', err, U, W)
                grad_W = np.einsum('abc,ar,br->cr', err, U, V)
                
                # Step size via line search
                lr = 0.1
                for _ in range(10):
                    U_new = U - lr * grad_U
                    V_new = V - lr * grad_V
                    W_new = W - lr * grad_W
                    M_new = np.einsum('ar,br,cr->abc', U_new, V_new, W_new)
                    err_new = M_new - target
                    if np.linalg.norm(err_new.flatten()) < err_norm:
                        break
                    lr *= 0.5
                
                U, V, W = U_new, V_new, W_new
            
            if success or np.linalg.norm((np.einsum('ar,br,cr->abc', U, V, W) - target).flatten()) < 1e-8:
                # Check if entries are close to integer or half-integer
                U_int = np.round(U * 2) / 2
                V_int = np.round(V * 2) / 2
                W_int = np.round(W * 2) / 2
                
                # Verify the rounded version
                U_list = [[float(x) for x in row] for row in U_int]
                V_list = [[float(x) for x in row] for row in V_int]
                W_list = [[float(x) for x in row] for row in W_int]
                
                if _verify(U_list, V_list, W_list):
                    best_result = (rank, U_list, V_list, W_list)
                    return best_result
        
        # Print progress
        print(f"Rank {rank} not found, elapsed: {time.time()-t0:.1f}s")
    
    return best_result


def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT
    
    # Start with schoolbook as fallback
    fallback = _build_schoolbook()
    
    # Try to find lower rank
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # Known: rank(3x3) = 23, so rank(3x3x6) >= 23
    # Try from rank 18 upward
    result = _try_search(18, 40, deadline)
    
    if result is not None:
        rank, U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return fallback


if __name__ == "__main__":
    result = construct()
    print(f"Found decomposition with rank {len(result['U'][0])}")
    print(f"Verification: {_verify(result['U'], result['V'], result['W'])}")
