"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import itertools
import time
from collections import deque

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor."""
    # U shape (n*m, r), V shape (m*p, r), W shape (p*n, r)
    r = U.shape[1]
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 if i,j,k form a valid product term, 0 otherwise
    # Actually the multiplication tensor has T[i,j,k] = 1 for all valid i,j,k
    # because (AB)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor indices are (i, j, k) with value 1 for all i in range(n), j in range(m), k in range(p)
    # Wait, let me re-read. The tensor for matrix multiplication C = A*B where A is n x m, B is m x p
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the trilinear form is sum_{i,j,k} C[i,k] * A[i,j] * B[j,k]
    # The tensor T[i, j, k] = delta(i is in range n, j in range m, k in range p)
    # i.e., T[i,j,k] = 1 for all valid indices.
    
    # Check: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1
    # Actually wait - the tensor decomposition means:
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # And T[i,j,k] = 1 for all i,j,k in valid ranges.
    
    # This is expensive to check for large tensors, so let's do it smarter.
    # For r terms: U @ diag stuff... 
    # Let's just verify on a sample or use the fact that it's the all-ones tensor.
    
    # The all-ones tensor of shape (n*m, m*p, p*n) where index (i*m+j, j*p+k, k*n+i)
    # Hmm, but the tensor is defined on the full space (n*m) x (m*p) x (p*n)
    # T[a, b, c] where a = i*m+j, b = j*p+k, c = k*n+i
    # T is 1 only when the indices are consistent: the j in a matches the j in b, 
    # and the i in a matches the i in c, and the k in b matches the k in c.
    # Otherwise T = 0.
    
    # Actually no. The standard matrix multiplication tensor T is:
    # T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j]B[j,k]
    # But as a tensor in the vector spaces A (dim nm), B (dim mp), C (dim pn):
    # T(a, b, c) = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k, else 0.
    
    # So T is NOT all ones. It's 1 on the "diagonal" pattern and 0 elsewhere.
    
    # Let's build the target tensor and verify.
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Verify: sum_c U[:,c] outer V[:,c] outer W[:,c] == T
    # This is: for each c, outer product U[:,c], V[:,c], W[:,c]
    # Result = sum_c np.einsum('ac,bc,cc->abc', U, V, W) ... no
    # result[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    
    # For efficiency, compute as:
    # temp = U @ V.T  (shape nm x mp) ... no that's not right either.
    # result[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # = sum_r (U[a,r] * V[b,r]) * W[c,r]
    # Let M_ab_r = U[a,r]*V[b,r] ... this is getting complex.
    
    # For small dimensions (2*5=10, 5*6=30, 6*2=12), we can just do it directly.
    # 10 * 30 * 12 = 3600 entries, each requiring sum over r terms.
    # If r is small (say < 60), this is fast.
    
    result = np.zeros((n*m, m*p, p*n))
    for r_idx in range(r):
        result += np.einsum('a,b,c->abc', U[:,r_idx], V[:,r_idx], W[:,r_idx])
    
    return np.allclose(result, T, atol=tol)

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using the structure of the tensor."""
    r = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute sum_c U[:,c] V[:,c] W[:,c] as a 3-tensor
    # Use: for each c, the outer product
    # More efficient: reshape and use matrix operations
    # result[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Let's check only the "active" entries (where T=1) and a sample of "inactive" entries
    # Active entries: n*m*p = 60
    # Inactive: 10*30*12 - 60 = 3540
    
    # Check all active entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    
    # Check a sample of inactive entries
    # An entry (a,b,c) is inactive if there's no (i,j,k) with a=i*m+j, b=j*p+k, c=k*n+i
    # i.e., a%m != b//p or (a//m) != c%n or (b%p) != (c//n)
    checked = 0
    for _ in range(500):  # check 500 random inactive entries
        a = np.random.randint(0, n*m)
        b = np.random.randint(0, m*p)
        c = np.random.randint(0, p*n)
        i_a, j_a = divmod(a, m)
        j_b, k_b = divmod(b, p)
        k_c, i_c = divmod(c, n)
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue  # this is actually active
        val = 0.0
        for cc in range(r):
            val += U[a, cc] * V[b, cc] * W[c, cc]
        if abs(val) > tol:
            return False
        checked += 1
    
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using rational arithmetic or high precision."""
    r = U.shape[1]
    # Check active entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(r):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    # Check all inactive entries (feasible for 3540 entries with small r)
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i_a, j_a = divmod(a, m)
                j_b, k_b = divmod(b, p)
                k_c, i_c = divmod(c, n)
                if j_a == j_b and i_a == i_c and k_b == k_c:
                    continue
                val = 0.0
                for cc in range(r):
                    val += U[a, cc] * V[b, cc] * W[c, cc]
                if abs(val) > 1e-10:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    best_rank = n * m * p
    best_result = None
    
    # Schoolbook fallback
    U_sb = np.zeros((nm, best_rank))
    V_sb = np.zeros((mp, best_rank))
    W_sb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)

    np.random.seed(42)
    
    # Try to find lower rank
    for rank in range(40, best_rank):
        if time.time() > deadline - 10:
            break
            
        found = False
        max_restarts = 5
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize
            scale = 1.0 / np.sqrt(rank)
            U = np.random.randn(nm, rank) * scale
            V = np.random.randn(mp, rank) * scale
            W = np.random.randn(pn, rank) * scale
            
            # Gradient Descent Phase
            lr = 0.05
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            best_loss = np.inf
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                    
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                R = T - recon
                loss = np.sum(R**2)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                # Gradients
                grad_U = -np.einsum('abc,br,cr->ar', R, V, W)
                grad_V = -np.einsum('abc,ar,cr->br', R, U, W)
                grad_W = -np.einsum('abc,ar,br->cr', R, U, V)
                
                # Momentum update
                vU = momentum * vU + (1 - momentum) * grad_U
                vV = momentum * vV + (1 - momentum) * grad_V
                vW = momentum * vW + (1 - momentum) * grad_W
                
                U -= lr * vU
                V -= lr * vV
                W -= lr * vW
                
                # Simple adaptive lr
                if loss > best_loss * 1.2:
                    lr *= 0.8
                    vU = 0
                    vV = 0
                    vW = 0

            # If GD found a reasonably good solution, polish with ALS
            if best_loss < 1e-2:
                U, V, W = _als_polish(best_U, best_V, best_W, T, n, m, p, iterations=10)
                
                # Final verification
                if _verify(U, V, W, n, m, p, tol=1e-8):
                    found = True
                    best_rank = rank
                    best_result = (U, V, W)
                    break
        
        if found:
            break

    # Convert to lists
    U_final, V_final, W_final = best_result
    U_list = [[float(x) for x in row] for row in U_final]
    V_list = [[float(x) for x in row] for row in V_final]
    W_list = [[float(x) for x in row] for row in W_final]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
