import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def _verify_decomposition(U, V, W, n, m, p, num_tests=5, seed=42):
    """Verify that the tensor decomposition exactly matches matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rng = np.random.RandomState(seed)
    for _ in range(num_tests):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        
        # Efficient verification
        # C[i,k] = sum_{j,r} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can compute this by iterating over r and using outer products, or just loops for small sizes.
        # Given small sizes (n=2, m=4, p=6), loops are fine and less error-prone.
        
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    u_row = U[i*m + j, :]
                    v_row = V[j*p + k, :]
                    w_val = W[k*n + i, :]
                    s += np.dot(u_row * v_row, w_val)
                C_computed[i, k] = s
                
        if not np.allclose(C_computed, C_expected, atol=1e-8, rtol=1e-8):
            return False
    return True

def _build_trivial(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m*p) + j * p + k
                U[i*m + j, r] = 1
                V[j*p + k, r] = 1
                W[k*n + i, r] = 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Fallback to trivial
    U_triv, V_triv, W_triv = _build_trivial(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    
    # Strategy: Construct a rank-24 decomposition based on the insight from other solutions.
    # The insight was: "The new rank 24 construction uses separate terms for each (j,k) pair, 
    # correctly mapping A(i,j) and B(j,k) to C(i,k) without coupling i."
    
    # Let's analyze what "separate terms for each (j,k) pair" means in terms of rank.
    # The trivial decomposition has a term for each (i,j,k). Rank = n*m*p = 48.
    # If we can decouple i, we might be able to share terms across i?
    # Or perhaps the structure allows for a specific linear combination.
    
    # Let's look at the known lower bound.
    # R(2,4,6) >= 24?
    # Actually, R(2,4,4) is known to be 20 (Landsberg et al.).
    # R(2,4,6) could be related.
    # If we split 6 into 4+2, we can use R(2,4,4) + R(2,4,2)?
    # R(2,4,2) is essentially 2 * R(1,4,2) ? No, R(2,4,2) = 2 * R(1,4,2) is not quite right because of the 2x2 structure.
    # However, R(2,4,2) is the rank of multiplying 2x4 by 4x2.
    # This is equivalent to computing 2x2 matrix product where each entry is a 2x2 block? No.
    # It's just a bilinear map.
    # The rank of 2x4 by 4x2 multiplication is at most 2*4*2 = 16.
    # Can we do better?
    # R(2,2,2) = 7.
    # R(2,4,2) can be viewed as 2 independent 1x4x2 multiplications? No, the rows of A are coupled in the output?
    # C[i,k] = sum_j A[i,j] B[j,k].
    # For fixed k, C[:,k] is a linear combination of rows of B? No, columns of B.
    # Let's stick to the explicit rank 24 construction hinted at.
    
    # The hint says: "The seed program's rank 24 attempt was flawed because it forced C(0,k)=C(1,k). 
    # The new rank 24 construction uses separate terms for each (j,k) pair..."
    
    # This suggests that the rank 24 construction is NOT simply R(2,4,4) + R(2,4,2) which would be 20+?
    # If R(2,4,2) is 12, then 20+12=32.
    # If R(2,4,2) is 8, then 20+8=28.
    # But the hint says rank 24.
    
    # Let's try to construct a rank 24 decomposition explicitly.
    # The standard way to get lower rank is Strassen-like recursion or specific combinatorial designs.
    # For 2x4x6, we can view it as 2x(2x2)x(2x3) = (2x2x2) x (2x2x3)?
    # R(2,2,2) = 7.
    # R(2,2,3) ?
    # If R(2,2,3) = 12, then 7*12 = 84? No, tensor product rank is multiplicative.
    # But we are not doing a tensor product of the whole dimensions in a simple way.
    
    # Let's look at the "separate terms for each (j,k) pair" idea.
    # There are m*p = 24 pairs of (j,k).
    # If we can assign 1 rank term to each (j,k) pair, the total rank would be 24.
    # This would imply that the dependence on i is handled by the structure of U, V, W without increasing the rank count beyond the number of (j,k) pairs.
    # This is only possible if the i-dependence is "free" or shared in a specific way.
    # Specifically, if U[i*m+j, r] depends on i in a way that allows the same r to be used for all i?
    # But r is the rank index. If we have 24 rank terms, and 24 (j,k) pairs, maybe each rank term r corresponds to a specific (j,k) pair?
    # Let's try:
    # For each (j,k), assign a rank index r(j,k).
    # We need:
    # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 (if i,j,k match? No, the sum is always 1 for the correct i,k? No, the equation is:
    # For all i,j,k: sum_r ... = 1 is NOT the condition.
    # The condition is:
    # C[i,k] = sum_j A[i,j] B[j,k].
    # This requires that for any i, j, k:
    # The coefficient of A[i,j] B[j,k] in C[i,k] is 1.
    # The coefficient of A[i',j'] B[j',k'] in C[i,k] is 0 if (i',j',k') != (i,j,k) in a way that contributes to C[i,k]?
    # Actually, the bilinear map is defined by the tensor T.
    # T_{i,j,k} = 1 for all i,j,k? No.
    # The tensor T is of shape (n*m, m*p, p*n).
    # T_{(i,j), (j',k), (k',i')} = 1 if i=i', j=j', k=k'?
    # Let's check the contraction:
    # C[i,k] = sum_{j,r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # This must equal sum_{j} A[i,j] B[j,k].
    # This implies that the tensor defined by U, V, W must be equal to the tensor T* where:
    # T*_{i,j,k} (in the sense of the bilinear form) is 1 if the indices match the multiplication path.
    # Specifically, the term A[i,j] B[j,k] contributes to C[i,k].
    # So the "target" tensor has 1s at positions corresponding to valid (i,j,k) triples, and 0s elsewhere?
    # But the indices in U, V, W are coupled.
    # U is indexed by (i,j). V by (j,k). W by (k,i).
    # The rank-r decomposition approximates the tensor T where T_{(i,j),(j,k),(k,i)} = 1 if i,j,k are consistent?
    # Actually, for matrix multiplication, the tensor is just the identity map on the "multiplication" structure.
    # Every triple (i,j,k) contributes 1 to the coefficient of A[i,j]B[j,k] in C[i,k].
    # So there are n*m*p non-zero entries in the "flattened" tensor?
    # Yes, n*m*p = 48.
    # So the trivial rank is 48.
    
    # To get rank 24, we need to "compress" this tensor by a factor of 2.
    # This is possible if there is a symmetry or if we can use Strassen-like tricks.
    # For n=2, we have 2 rows.
    # Maybe we can pair up the i=0 and i=1 terms?
    # If we could show that T_{0,j,k} and T_{1,j,k} can be represented with shared rank terms?
    # But they contribute to different outputs C[0,k] and C[1,k].
    # Unless... the W matrix allows us to route them correctly?
    
    # Let's try a specific construction for rank 24.
    # We have 24 (j,k) pairs.
    # Let's assign rank indices r = 0..23 corresponding to (j,k).
    # Let r = j*p + k.
    # We want to define U, V, W such that:
    # sum_{r=0}^{23} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta_{j, j_r} delta_{k, k_r} ?
    # No, the sum is over r.
    # For a fixed (i,j,k), the term we want is 1.
    # If we set V[j*p+k, r] = 1 if r == j*p+k, else 0. (V is identity-like)
    # Then the sum becomes: U[i*m+j, j*p+k] * W[k*n+i, j*p+k].
    # We want this to be 1 for all i,j,k.
    # So U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1.
    # This requires U and W to be non-zero at specific positions.
    # Let's try U[i*m+j, j*p+k] = 1 and W[k*n+i, j*p+k] = 1.
    # But wait, U has shape (n*m, rank) = (8, 24).
    # V has shape (m*p, rank) = (24, 24).
    # W has shape (p*n, rank) = (12, 24).
    
    # If V is the identity matrix (24x24), then V[j*p+k, r] = 1 if r == j*p+k.
    # Then the condition becomes:
    # U[i*m+j, j*p+k] * W[k*n+i, j*p+k] = 1.
    # We can set U[i*m+j, j*p+k] = 1 and W[k*n+i, j*p+k] = 1.
    # Let's check if this works.
    # For a given (i,j,k), the sum over r is:
    # sum_r U[i*m+j, r] V[r, r] W[k*n+i, r]
    # Since V is identity, only the term where r = j*p+k survives?
    # No, V is indexed by (j*p+k, r).
    # V[j*p+k, r] is 1 if r = j*p+k.
    # So the sum is U[i*m+j, j*p+k] * 1 * W[k*n+i, j*p+k].
    # If we set these to 1, the result is 1.
    
    # Does this satisfy the full matrix multiplication?
    # C[i,k] = sum_j A[i,j] B[j,k] * 1.
    # Yes!
    # So this is a valid rank-24 decomposition!
    
    # Let's verify the dimensions and indices.
    # U: (n*m, rank) -> (8, 24)
    # V: (m*p, rank) -> (24, 24)
    # W: (p*n, rank) -> (12, 24)
    
    # Construction:
    # rank = 24
    # V is the identity matrix of size 24x24.
    # V[row, col] = 1 if row == col.
    # row = j*p + k. col = r.
    # So V[j*p+k, j*p+k] = 1.
    
    # U[i*m+j, j*p+k] = 1.
    # Note: U has rows indexed by i*m+j (0..7).
    # Columns indexed by r (0..23).
    # For a fixed row i*m+j, which columns should be 1?
    # The condition is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for the specific (i,j,k).
    # Wait, the sum is over r.
    # C[i,k] = sum_j A[i,j] B[j,k] (sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r])
    # We want the inner sum to be 1 for the specific j,k?
    # No, the inner sum depends on j and k.
    # For a fixed j and k, let S(j,k) = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
    # We want S(j,k) = 1 for all i?
    # Wait, S(j,k) depends on i because of U[i*m+j, r] and W[k*n+i, r].
    # So S(i,j,k) must be 1 for all i,j,k.
    
    # If V is identity, V[j*p+k, r] is 1 only if r = j*p+k.
    # Then S(i,j,k) = U[i*m+j, j*p+k] * W[k*n+i, j*p+k].
    # We need this product to be 1 for all i,j,k.
    # So we can set U[i*m+j, j*p+k] = 1 and W[k*n+i, j*p+k] = 1.
    
    # Let's check if this causes conflicts.
    # For a fixed U row i*m+j, it has 1s at columns r corresponding to all k?
    # r = j*p + k. As k varies from 0 to p-1, r takes values j*p, j*p+1, ..., j*p+p-1.
    # So for a fixed (i,j), U[i*m+j, :] has 1s in the block of columns corresponding to j.
    # Specifically, U[i*m+j, j*p+k] = 1 for all k.
    
    # For a fixed W row k*n+i, it has 1s at columns r corresponding to all j?
    # r = j*p + k. As j varies from 0 to m-1, r takes values k, k+p, k+2p, ...
    # So W[k*n+i, j*p+k] = 1 for all j.
    
    # This seems consistent.
    # Let's build it.
    
    rank = 24
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = j * p + k
                U[i*m + j, r] = 1
                W[k*n + i, r] = 1
                
    # V is identity
    for r in range(rank):
        V[r, r] = 1
        
    # Verify
    if _verify_decomposition(U, V, W, n, m, p, num_tests=10, seed=123):
        best_rank = rank
        best_U, best_V, best_W = U, V, W
    else:
        # Fallback to trivial if verification fails (should not happen)
        pass
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
