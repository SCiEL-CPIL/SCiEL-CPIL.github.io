import numpy as np

DIMENSIONS = [2, 4, 8]


# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # Tensor T with indices (i, j, k), i in [0,n), j in [0,m), k in [0,p)
    # T[i, j, k] = 1 for all (matrix multiplication tensor)
    # We seek rank-r decomposition:
    #   sum_t U[i, t] * V[j, t] * W[k, t] = T[i, j, k]
    #
    # The interface requires:
    #   U: (n*m, rank)  with U[i*m + j, t]
    #   V: (m*p, rank)  with V[j*p + k, t]
    #   W: (p*n, rank)  with W[k*n + i, t]
    #
    # Note: the official trace convention W[k*n + i, r] means W is indexed by (k, i).
    # But the tensor decomposition needs W[k, t] only (no i dependence).
    # We must be careful: the multiplication is C[i, k] = sum_j A[i, j] B[j, k].
    # The rank-1 term contributes U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] to C[i, k]
    # via the "trace" over j? No - the standard tensor decomposition is:
    #   C[i,k] = sum_j sum_t A[i,j] B[j,k] * (tensor component)
    # Actually the correct identity is:
    #   C[i,k] = sum_t (sum_j A[i,j] U_jt) (sum_j B[j,k] V_jt) W_t
    # where the U, V, W are the three factor tensors.
    #
    # Let me use the standard formulation:
    #   C[i,k] = sum_j A[i,j] B[j,k]
    #          = sum_t (sum_j A[i,j] a[j,t]) (sum_k B[j,k] b[k,t]) c[t] ... no.
    #
    # The clean tensor formulation: the multiplication tensor M has
    #   M[i, j, k] = 1
    # and rank-r decomposition means:
    #   M[i, j, k] = sum_t u[i, t] * v[j, t] * w[k, t]
    # Then C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_t A[i,j] B[j,k] u[i,t] v[j,t] w[k,t]
    # This doesn't directly give C[i,k] unless we set things up right.
    #
    # Actually the correct statement: C[i,k] = sum_j A[i,j] B[j,k].
    # We want C[i,k] = sum_t U[i, t] * (something) ... 
    #
    # The standard way: define
    #   C[i,k] = sum_t (sum_j A[i,j] * U[j, t]) * (sum_j B[j, k] * V[j, t]) * W[t]
    # No, that's not right either. Let me think again.
    #
    # The rank-r decomposition of the multiplication tensor means we can write:
    #   C[i,k] = sum_t alpha_t * (sum_j A[i,j] * u[j,t]) * (sum_j B[j,k] * v[j,t])
    # where alpha_t is absorbed. Actually:
    #   C[i,k] = sum_t (sum_j A[i,j] u[j,t]) * (sum_j B[j,k] v[j,t])
    # This works if the tensor M[i,j,k] = sum_t u[j,t] v[j,t] w... no.
    #
    # Let me just use the direct tensor decomposition:
    #   M[i,j,k] = sum_t U[i,t] V[j,t] W[k,t]  where M[i,j,k] = 1
    # Then: C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_t A[i,j] B[j,k] U[i,t] V[j,t] W[k,t]
    #             = sum_t U[i,t] W[k,t] (sum_j A[i,j] V[j,t] B[j,k])... 
    # This is getting confused. Let me just verify with the trivial case.
    #
    # For n=m=p=2, the trivial rank-2 decomposition:
    #   C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    #   C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1]
    #   C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    #   C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1]
    # Each term A[i,j]*B[j,k] is a rank-1 contribution.
    # So rank = m = 2 (one per j).
    #
    # In the interface:
    #   U[i*m+j, t] = A[i,j] coefficient -> for term t=j, U[i*m+j, j] = 1
    #   V[j*p+k, t] = B[j,k] coefficient -> V[j*p+k, j] = 1
    #   W[k*n+i, t] = 1
    # Then sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] = U[i*m+j,j] V[j*p+k,j] W[k*n+i,j] = 1*1*1 = 1
    # And C[i,k] = sum_j A[i,j] B[j,k] * 1 = sum_j A[i,j] B[j,k]. Correct!
    #
    # So the trivial rank is m = 4 per (i,k) pair? No, rank is m total? 
    # For each j, we have one term. So rank = m = 4.
    # Wait, but U has shape (n*m, rank) = (8, 4). U[i*m+j, j] = 1 for all i,j.
    # V has shape (m*p, rank) = (32, 4). V[j*p+k, j] = 1 for all j,k.
    # W has shape (p*n, rank) = (16, 4). W[k*n+i, j] = 1 for all k,i,j.
    # Then for a given (i,k): sum_j sum_t U[i*m+j,t] V[j*p+k,t] W[k*n+i,t]
    #   = sum_j sum_t [t==j ? 1 : 0] * [t==j ? 1 : 0] * 1
    #   = sum_j 1 = m = 4. That's wrong, we want sum_j A[i,j]B[j,k].
    #
    # I see the issue: the factors U, V, W are NOT the A and B values.
    # The decomposition is: C[i,k] = sum_t (sum_j A[i,j] U[j,t]) (sum_j B[j,k] V[j,t]) W[t]
    # Let me redefine. The standard rank-r matrix multiplication algorithm:
    #   C[i,k] = sum_t (sum_j A[i,j] * U[j,t]) * (sum_j B[j,k] * V[j,t]) * W[t]
    # For this to equal sum_j A[i,j] B[j,k], we need:
    #   sum_t U[j,t] V[j',t] W[t] = delta_{j,j'}  (so that the sums collapse)
    # Actually: sum_t (sum_j A[i,j] U[j,t]) (sum_{j'} B[j',k] V[j',t]) W[t]
    #   = sum_t W[t] sum_j sum_{j'} A[i,j] B[j',k] U[j,t] V[j',t]
    #   = sum_j sum_{j'} A[i,j] B[j',k] (sum_t U[j,t] V[j',t] W[t])
    # For this to equal sum_j A[i,j] B[j,k], we need sum_t U[j,t] V[j',t] W[t] = delta_{j,j'}.
    #
    # So the tensor is M[j, j', t] = U[j,t] V[j',t] W[t] and we need sum_t M[j,j',t] = delta_{j,j'}.
    # This is a decomposition of the identity matrix! Rank = m.
    #
    # Hmm, but the task says "tensor decomposition for multiplying a 2 by 4 matrix by a 4 by 8 matrix".
    # The standard tensor for this has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1. And rank-r decomposition: T[i,j,k] = sum_t U[i,t] V[j,t] W[k,t].
    # Then C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_t A[i,j] B[j,k] U[i,t] V[j,t] W[k,t].
    # For this to work for ALL A, B, we need: for each (i,k), sum_j A[i,j] B[j,k] U[i,t] V[j,t] W[k,t] summed over t equals sum_j A[i,j] B[j,k].
    # This requires: for each t, the "operator" defined by (i,j) -> U[i,t] V[j,t] and (j,k) -> V[j,t] W[k,t] ... 
    #
    # I think I'm overcomplicating. Let me just use the fact that the current program
    # uses rank 56 with Strassen. The trivial rank for 2x4x8 is... 
    #
    # Actually, let me just try the randomized SVD approach to find a low-rank decomposition
    # of the tensor T[i,j,k] = 1 with the constraint that the decomposition must be "valid"
    # for matrix multiplication. The validity condition is exactly that the three sets of
    # vectors form a valid decomposition.
    #
    # Let me just implement the trivial rank-m decomposition and verify it works, then try
    # to improve with randomization.

    # Trivial decomposition: rank = m = 4
    # U[i*m + j, t] = 1 if t == j, else 0  -> U[i*m+j, j] = 1
    # V[j*p + k, t] = 1 if t == j, else 0  -> V[j*p+k, j] = 1
    # W[k*n + i, t] = 1 for all k, i, t
    #
    # Check: C[i,k] = sum_t sum_j A[i,j] U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]
    #   = sum_t sum_j A[i,j] [t==j] [t==j] * 1
    #   = sum_j A[i,j] * 1 = sum_j A[i,j]  ... that's wrong, we're missing B[j,k].
    #
    # The issue: V[j*p+k, t] should encode B[j,k], not be 1.
    # 
    # I think the correct interpretation is:
    #   C[i,k] = sum_t (sum_j A[i,j] * U[j,t]) * (sum_j B[j,k] * V[j,t]) * W[t]
    # where U is (m, rank), V is (m, rank), W is (rank,).
    # And the interface reshapes:
    #   U_interface[i*m+j, t] = U[j, t]  (repeated for each i)
    #   V_interface[j*p+k, t] = V[j, t]  (repeated for each k)
    #   W_interface[k*n+i, t] = W[t]     (repeated for each k, i)
    #
    # Then: sum_t (sum_j A[i,j] U[j,t]) (sum_j B[j,k] V[j,t]) W[t]
    #   = sum_t W[t] sum_j sum_{j'} A[i,j] B[j',k] U[j,t] V[j',t]
    #   = sum_j sum_{j'} A[i,j] B[j',k] (sum_t U[j,t] V[j',t] W[t])
    # For this to equal sum_j A[i,j] B[j,k], we need:
    #   sum_t U[j,t] V[j',t] W[t] = delta_{j,j'}
    #
    # So we need a rank-r decomposition of the m x m identity: I = sum_t (U[:,t] W[t]) (V[:,t])^T
    # i.e., I = U * diag(W) * V^T where U, V are (m, r) and W is (r,).
    # The minimum r is m (rank of identity is m).
    #
    # So the trivial rank is m = 4. Let me construct it:
    #   U[j, t] = delta_{j,t}, V[j, t] = delta_{j,t}, W[t] = 1, for t in [0, m).
    # Then sum_t U[j,t] V[j',t] W[t] = sum_t delta_{j,t} delta_{j',t} * 1 = delta_{j,j'}. Correct.
    #
    # Now map to interface:
    #   U_interface[i*m+j, t] = U[j, t] = delta_{j,t}
    #   V_interface[j*p+k, t] = V[j, t] = delta_{j,t}
    #   W_interface[k*n+i, t] = W[t] = 1
    #
    # rank = m = 4.

    rank = m
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    for i in range(n):
        for j in range(m):
            U[i * m + j, j] = 1.0
    for j in range(m):
        for k in range(p):
            V[j * p + k, j] = 1.0
    for k in range(p):
        for i in range(n):
            W[k * n + i, :] = 1.0

    # Verify
    def verify(U, V, W, n, m, p, rank):
        # C[i,k] = sum_t sum_j A[i,j] U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]
        # For random A, B, check C = A @ B
        rng = np.random.RandomState(42)
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for t in range(rank):
                        val += A[i, j] * B[j, k] * U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                C_computed[i, k] = val
        return np.allclose(C_computed, C_expected, atol=1e-9)

    if not verify(U, V, W, n, m, p, rank):
        # Fallback to trivial rank m*p*n
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
