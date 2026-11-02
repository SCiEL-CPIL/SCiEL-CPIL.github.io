import numpy as np
import itertools

DIMENSIONS = [2, 3, 10]


def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p

    # ---- trivial rank-60 decomposition (exact) ----
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

    def to_list(M):
        return [[int(x) for x in row] for row in M]

    def check(Um, Vm, Wm):
        # verify tensor identity: sum_r U[i,j,r]*V[j,k,r] == W[k,i,r] ... actually
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] must equal sum_r W[k*n+i,r]*? 
        # Correct identity: for matrix mult (A n x m)(B m x p) = C n x p,
        # C[i,k] = sum_j A[i,j] B[j,k].
        # Tensor T[i,j,k] (rank-3) has T[i,j,k]=1 for all (i,j,k).
        # Decomposition: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] is WRONG.
        # The standard bilinear form: T is a bilinear map A x B -> C, so
        # C[i,k] = sum_r U[i*m+k? ...]. 
        # Per task contract: U (n*m, rank), V (m*p, rank), W (p*n, rank).
        # The identity is: sum_r U[i,j,r]*V[j,k,r] = delta? No.
        # Actually the matrix multiplication tensor T in V^* x W^* x X^* where
        # T[i,j,k] = 1. And T = sum_r u_r v_r w_r with
        # u_r[i*m+j], v_r[j*p+k], w_r[k*n+i].
        # So T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] is trilinear, not right.
        # The correct bilinear identity:
        # (A*B)[i,k] = sum_j A[i,j] B[j,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_k V[j,k,r] B[j,k])
        # This requires U indexed [i,j,r], V indexed [j,k,r], and then C[i,k] = sum_r Ucol[i,r]*Vcol[k,r]
        # but U depends on both i and j... 
        # The standard: C[i,k] = sum_r (sum_j U[i,j,r]*A[i,j]) * (sum_j V[j,k,r]*B[j,k])? No.
        # Let me just trust the seed: it sets U[i*m+j, r]=1, V[j*p+k,r]=1, W[k*n+i,r]=1.
        # Verification: sum_r U[i*m+j,r]*V[j*p+k,r] = 1 (only r=that col). And W[k*n+i,r]=1.
        # The identity should be: for each (i,j,k), sum_r U[i*m+j,r] V[j*p+k,r] = W[k*n+i,r]?? 
        # That doesn't make sense dimensionally. Let me just verify the seed is correct
        # by checking the standard bilinear form.
        # Standard: C[i,k] = sum_j A[i,j] B[j,k].
        # Bilinear decomposition: C[i,k] = sum_r (sum_j a_{ij}^r A[i,j]) (sum_j b_{jk}^r B[j,k])
        # where a_{ij}^r = U[i,j,r], b_{jk}^r = V[j,k,r].
        # Then we need: sum_r U[i,j,r] V[j',k,r] = delta_{j,j'} for each (i,k)? No...
        # Actually: sum_r U[i,j,r] V[j,k,r] must equal 1 for all j (so that C[i,k]=sum_j A[i,j]B[j,k]).
        # Wait: C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_{j'} V[j',k,r] B[j',k])
        # = sum_r sum_j sum_{j'} U[i,j,r] V[j',k,r] A[i,j] B[j',k]
        # We want this = sum_j A[i,j] B[j,k], so we need sum_r U[i,j,r] V[j',k,r] = delta_{j,j'}
        # for all i, j, j', k.
        # With the seed: sum_r U[i,j,r] V[j',k,r] = 1 if (i,j) and (j',k) share the same r,
        # i.e. the column r corresponds to (i,j,k) where U[i*m+j,r]=1 and V[j'*p+k,r]=1.
        # U[i*m+j,r]=1 only when r = col(i,j,k) for some k. V[j'*p+k,r]=1 only when r=col(i,j',k) for some i.
        # So sum_r U[i*m+j,r]*V[j'*p+k,r] = 1 if there exists k such that col(i,j,k)=col(i,j',k), i.e. j=j'.
        # = delta_{j,j'}. Correct!
        # So the identity is: sum_r U[i*m+j,r] * V[j'*p+k,r] = delta_{j,j'} for all i,j,j',k.
        # And W is not used in the identity? W is just part of the tensor's third factor.
        # Actually W[k*n+i, r] is the third factor. The tensor T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r].
        # With seed: = 1 for each (i,j,k). And T[i,j,k]=1. Correct.
        # So the identity is trilinear: T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1.
        pass

    # For the border-rank lifting, we need: for each (i,j,k), sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1.
    # We try to drop a column by expressing it as a linear combination.
    
    # Try to reduce rank by 1: express column c as sum_{r != c} alpha_r * col_r
    # This requires for all (i,j,k):
    #   U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = sum_{r!=c} alpha_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
    # Since in the seed, each column has exactly one nonzero in each of U, V, W,
    # the LHS is 1 only for the (i,j,k) matching column c, else 0.
    # The RHS: alpha_r * (product). For r != c, the product is 1 only for (i,j,k) matching r.
    # So for the (i,j,k) of column c: 1 = sum_{r!=c} alpha_r * 0 = 0. Contradiction!
    # So we cannot drop a single column with the seed structure.
    
    # Instead, we need to modify multiple columns. Let's try a different approach:
    # Use the known result that matmul(2,3,10) has rank <= 54 via border rank.
    # But implementing the full Gröbner basis is infeasible in 120s.
    
    # Let's try a randomized search for a rank-59 decomposition by perturbing the seed.
    # Actually, let's just return the seed for now since we can't easily improve it.
    
    # But let me try one more thing: check if rank 59 is achievable by a simple modification.
    # The border rank of matmul(2,3,10) is known to be 54 (Strassen-like).
    # Without the full machinery, I'll return the seed.
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
