import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    
    The convention is: (C)_ik = sum_j A_ij * B_jk
    Output factor follows W[k*n+i, r]
    
    U has shape (n*m, rank) with U[i*m+j, r]
    V has shape (m*p, rank) with V[j*p+k, r]  
    W has shape (p*n, rank) with W[k*n+i, r]
    
    We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{ik}
    Wait, that's not right. Let me think again.
    
    For matrix multiplication: C_ik = sum_j A_ij B_jk
    
    The tensor for this operation is T[i,j,k] = delta_{ik} (the identity in i,k)
    But actually, the tensor is T[i*m+j, j*p+k, k*n+i] ... no.
    
    Let me reconsider. The matrix multiplication tensor M has entries:
    M[(i,j), (j',k), (i',k')] = 1 if i=i' and k=k' and j=j' else 0
    
    So M[a, b, c] = 1 if a encodes (i,j), b encodes (j,k), c encodes (i,k) with matching indices.
    
    U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r should give M[(i,j),(j,k),(i,k)]
    
    Actually the standard form: the tensor T has T[i,j,k,l] = delta_{il} * delta_{jk} for 4-index form.
    But in 3-index form with the given conventions:
    
    T[(i*m+j), (j*p+k), (k*n+i)] ... this doesn't quite work because j and k are shared.
    
    Let me use the evaluator's convention: W[k*n+i, r]
    
    The decomposition is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 if this is the (i,j) x (j,k) -> (i,k) entry, which is always 1 for valid (i,j,k).
    
    Wait no - the tensor for matrix multiplication is such that:
    (A * B)_ik = sum_j A_ij B_jk
    
    The trilinear form is: sum_{i,j,k} T[i*m+j, j*p+k, k*n+i] * A_ij * B_jk = (AB)_ik ... no that's not right either.
    
    The correct statement: the tensor M is defined by M[a,b,c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    and M[a,b,c] = 1 if i=i' and k=k' and j=j', else 0.
    
    So the non-zero entries are at positions (i*m+j, j*p+k, k*n+i) for all valid i,j,k, and each such entry equals 1.
    
    The rank-r decomposition satisfies: sum_{r=1}^r U[a,r] * V[b,r] * W[c,r] = M[a,b,c]
    """
    rank = len(U[0])
    # Check all non-zero positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = 0.0
                for r in range(rank):
                    val += U[a][r] * V[b][r] * W[c][r]
                if abs(val - 1.0) > 1e-10:
                    return False
    # Check some zero positions (sample)
    import random
    random.seed(42)
    for _ in range(min(200, n*m*m*p*p*n)):
        a = random.randint(0, n*m - 1)
        b = random.randint(0, m*p - 1)
        c = random.randint(0, p*n - 1)
        # Check if this is a non-zero position
        i_a = a // m
        j_a = a % m
        j_b = b // p
        k_b = b % p
        k_c = c // n
        i_c = c % n
        is_nonzero = (i_a == i_c and j_a == j_b and k_b == k_c)
        val = 0.0
        for r in range(rank):
            val += U[a][r] * V[b][r] * W[c][r]
        if is_nonzero:
            if abs(val - 1.0) > 1e-10:
                return False
        else:
            if abs(val) > 1e-10:
                return False
    return True

def _try_rank(r, n, m, p, deadline):
    """Try to find a rank-r decomposition. Returns (U, V, W) or None."""
    # For small ranks, use random initialization with gradient-free approach
    # or use the structure of the problem
    
    # Known: schoolbook gives rank = n*m*p = 60
    # The tensor rank of 2x3 * 3x10 is at most 60.
    # Can we do better?
    
    # Try a structured approach: exploit the factorization structure
    # M[(i,j),(j',k),(i',k')] = delta_ii' delta_jj' delta_kk'
    
    # One approach: use the fact that delta_jj' couples the second index
    # We can write: sum_j delta_jj' = sum_j U1[j,j'] ... 
    
    # Let's try random search for smaller ranks
    # But for a 60-dimensional tensor, random search for rank < 60 is hard.
    
    # Alternative: use the known bounds. The rank of matrix multiplication tensor [n,m,p]
    # is at most n*m*p and at least max(n*m, m*p, p*n) = max(6, 30, 20) = 30.
    
    # For [2,3,10], the trivial bound is 60.
    # Strassen-type improvements exist for larger matrices but 2x3*3x10 is small.
    
    # Let me try a more clever construction.
    
    # The tensor is: T(i,j,k) = 1 (with the multi-index convention)
    # T = sum over all (i,j,k) of e_{i,j} * e_{j,k} * e_{k,i}
    
    # Can we group terms? If we fix i, we have sum_{j,k} e_{i,j} * e_{j,k} * e_{k,i}
    # This is a rank-m*p term for each i, giving rank n*m*p total (no improvement).
    
    # Alternatively, fix k: sum_{i,j} e_{i,j} * e_{j,k} * e_{k,i} - rank n*m per k, total p*n*m.
    
    # The key insight: the tensor has a product structure in the j-index.
    # T = sum_j (sum_i e_{i,j} * e_{k,i}) * e_{j,k} ... this doesn't factor cleanly.
    
    # Let me just do a simple random search for rank slightly below 60.
    # Actually, for this small problem, let me use a smarter approach.
    
    # Use the identity: the tensor rank of the identity map I_n (as a bilinear form) is n.
    # But matrix multiplication is more complex.
    
    # For now, return the schoolbook solution as fallback
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a better-than-schoolbook decomposition
    # Start with schoolbook as baseline
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = schoolbook_rank
    
    # Try random search for smaller ranks
    # This is expensive, so only try if we have time
    for target_rank in range(schoolbook_rank - 1, max(29, schoolbook_rank - 5), -1):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        np.random.seed(target_rank * 1000 + 42)
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            U_rand = np.random.randn(n * m, target_rank)
            V_rand = np.random.randn(m * p, target_rank)
            W_rand = np.random.randn(p * n, target_rank)
            
            # Scale to make products of magnitude ~1
            U_rand /= np.sqrt(np.mean(U_rand**2))
            V_rand /= np.sqrt(np.mean(V_rand**2))
            W_rand /= np.sqrt(np.mean(W_rand**2))
            
            # Convert to lists for verification
            U_list = U_rand.tolist()
            V_list = V_rand.tolist()
            W_list = W_rand.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook solution
    return {"U": best_U, "V": best_V, "W": best_W}
