import numpy as np

def _verify_decomposition_np(U, V, W, n=4, m=4, p=4):
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # T[i*m+j, j'*p+k, k'*n+i'] = sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r]
    # We want to check if T equals the identity tensor for matmul.
    # The matmul tensor T has T[a,b,c] = 1 if b//p == a%m and c%n == a//m and b%p == c//p? 
    # No, standard matmul: C[i,k] = sum_j A[i,j] B[j,k].
    # Tensor index convention: 
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # Term: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This sums over r.
    # The result should be 1 if the indices correspond to a valid matmul term, i.e., the 'j' in U matches 'j' in V, and 'k' in V matches 'k' in W, and 'i' in W matches 'i' in U.
    # Actually, the tensor for matmul is defined such that T_{ijk} = 1 if j=j' and k=k' and i=i' in the summation?
    # Let's stick to the original check logic but vectorized.
    
    # Construct the full tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is a tensor contraction.
    # We can compute this using np.einsum or explicit loops if size is small.
    # Size is 16 x 16 x 16 = 4096 entries. Rank 49.
    # Using einsum:
    # T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    
    T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
    
    # Check if T matches the expected matmul tensor.
    # Expected: T[a, b, c] = 1 if:
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # So for a given a, b, c:
    # i = a // m, j = a % m
    # j2 = b // p, k = b % p
    # k2 = c // n, i2 = c % n
    # Match if j == j2 and k == k2 and i == i2.
    
    for a in range(n*m):
        i = a // m
        j = a % m
        for b in range(m*p):
            j2 = b // p
            k = b % p
            for c in range(p*n):
                k2 = c // n
                i2 = c % n
                expected = 1 if (j == j2 and k == k2 and i == i2) else 0
                if T[a, b, c] != expected:
                    return False
    return True
