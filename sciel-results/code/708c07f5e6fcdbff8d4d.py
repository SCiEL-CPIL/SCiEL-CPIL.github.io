"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE_SECONDS = 110  # Safety margin within 120s evaluation limit

def _schoolbook_decomposition(n, m, p):
    """Return (U, V, W) for the trivial schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # Target tensor T[k*n+i, j*m+k, c] where c is the output index
    # Actually, the tensor is: T_{(i,j), (j',k), (k',i')} = delta_{j,j'} delta_{i,i'}
    # The decomposition says: T_{abc} = sum_r U_{a,r} V_{b,r} W_{c,r}
    # where a = i*m+j (input A index), b = j'*p+k (input B index), c = k'*n+i' (output index)
    # T_{a,b,c} = 1 if j==j' and i==i' (i.e., a = i*m+j, b = i*p+k, c = k*n+i)
    # Let's verify using numpy
    # T[a,b,c] = delta(j_a, j_b) * delta(i_a, i_c)
    # where a = i_a * m + j_a, b = j_b * p + k_b, c = k_c * n + i_c
    # Sum_r U[a,r] V[b,r] W[c,r] should equal T[a,b,c]
    # Build the tensor from U,V,W
    # For efficiency, compute element-wise
    for a in range(n * m):
        i_a = a // m
        j_a = a % m
        for b in range(m * p):
            j_b = b // p
            k_b = b % p
            expected = 1.0 if (j_a == j_b) else 0.0
            for c in range(p * n):
                i_c = c % n
                k_c = c // n
                expected_val = expected  # delta(i_a, i_c) * delta(j_a, j_b)
                # Wait: T[a,b,c] = delta(j_a, j_b) * delta(i_a, i_c)
                # But i_c comes from c = k*n + i, so i_c = c % n
                # And k_c = c // n
                # The multiplication is (A*B)_{i,k} = sum_j A_{i,j} B_{j,k}
                # So output index is (i,k), input indices are (i,j) and (j,k)
                # T_{(i,j),(j',k),(i',k')} = delta_{j,j'} delta_{i,i'}
                # So for fixed a=(i,j), b=(j',k), c=(i',k'):
                # T = 1 if j==j' and i==i'
                # Note: k must equal k' for consistency? No, k is free.
                # Actually the tensor has indices:
                # a: row of A (i) and col of A (j)
                # b: row of B (j') and col of B (k)
                # c: row of output (i') and col of output (k')
                # T[a,b,c] = delta(j,j') * delta(i,i') * delta(k,k')? 
                # No. The multiplication is (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
                # The bilinear form is: output_{i,k} = sum_{j} A_{i,j} * B_{j,k}
                # So the tensor T is indexed by (i,j) for A, (j',k) for B, (i',k') for output
                # T[(i,j),(j',k),(i',k')] = 1 if j=j' and i=i' and k=k'
                # Wait, that's not right either. Let me think again.
                # The matrix multiplication tensor is a 3-tensor where:
                # T_{ijk, j'k', i'k''} -- no.
                # Standard: M_{n,m,p} has indices (i,j,k) where i in [n], j in [m], k in [p]
                # But here we have 2x2 times 2x13.
                # Let me use the convention from the task:
                # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
                # U[i*m + j, r], V[j*p + k, r], W[k*n + i, r]
                # The tensor is: T[i*m+j, j'*p+k, k'*n+i'] = sum_r U[i*m+j,r] V[j'*p+k,r] W[k'*n+i',r]
                # This should equal the matrix multiplication tensor.
                # For matrix multiplication (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
                # The tensor entry should be 1 when j=j' and i=i' and k=k', 0 otherwise.
                # Let's verify: sum_r U[i*m+j, r] V[j'*p+k, r] W[k'*n+i', r]
                # In schoolbook, U[i*m+j, r]=1 iff r = i*m*p + j*p + k (for some specific k)
                # Actually in schoolbook: column r corresponds to (i,j,k) with i = r//(m*p), j = (r//p)%m, k = r%p
                # U[i*m+j, r] = 1 if i = r//(m*p) and j = (r//p)%m
                # V[j'*p+k, r] = 1 if j' = (r//p)%m and k = r%p
                # W[k'*n+i', r] = 1 if k' = r%p and i' = r//(m*p)
                # So product is 1 iff i = r//(m*p) = i', j = (r//p)%m = j', k = r%p = k'
                # So T = 1 iff i=i', j=j', k=k'. That's correct for matrix mult.
                
                actual = 0.0
                for r in range(rank):
                    actual += U[a, r] * V[b, r] * W[c, r]
                if abs(actual - expected_val) > tol:
                    return False
    return True

def _try_find_decomposition(n, m, p, max_rank, time_limit):
    """Try to find a decomposition of rank <= max_rank using random search."""
    start_time = time.time()
    # This is a placeholder; implementing a full search is complex.
    # For now, just return None to indicate no better decomposition found.
    # In a real implementation, this would use random linear algebra or structured search.
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    rank = U.shape[1]
    
    # Convert to lists as required by the interface
    U_list = [[float(U[i, j]) for j in range(rank)] for i in range(len(U))]
    V_list = [[float(V[i, j]) for j in range(rank)] for i in range(len(V))]
    W_list = [[float(W[i, j]) for j in range(rank)] for i in range(len(W))]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
