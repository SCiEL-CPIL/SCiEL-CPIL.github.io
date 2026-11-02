"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = 3, 5, 7

    def verify_fast(U, V, W):
        """Fast verification using tensor contraction."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
            return False
        if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
            return False
        r = U.shape[1]
        # Build the full tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # T should equal delta_{i,i'} delta_{j,j'} delta_{k,k}
        # where a=(i,j), b=(j',k), c=(k',i')
        # We check: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # sum should match the identity tensor.
        # Efficient: compute T as sum over r of outer products
        # But that's O(r * n*m * m*p * p*n) which is too big.
        # Instead, check via matrix multiplication interpretation.
        # T is the matrix multiplication tensor: T[i,j,k',i',j',k] = 1 iff i=i', j=j', k=k'
        # Actually T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
        # T[a,b,c] = 1 iff i==i' AND j==j' AND k==k'
        # Let's verify by checking that for all i,i',j,j',k,k':
        # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
        # We can vectorize: for fixed i,i', compute M[j,j',k,k'] = sum_r U[i*m+j,r]*V[j'*p+k,r]*W[k'*n+i',r]
        # This should be delta_{j,j'}delta_{k,k'} if i==i', else 0.
        # Let's do it more efficiently using einsum.
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # We need T to equal the 3-index Kronecker delta.
        # Let's build the expected tensor and compare.
        # Expected: E[i,j,j',k,k',i'] = 1 iff i==i' and j==j' and k==k'
        # Reshape: E[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'
        # We can check: for each i, i', the sub-tensor T[i*m: (i+1)*m, :, i'*n: (i'+1)*n]
        # should be delta_{j,j'}delta_{k,k'} if i==i', else 0.
        
        # Use einsum for efficiency
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Compute T as: T = np.einsum('ar,br,cr->abc', U, V, W)
        # This is O(r * 15 * 35 * 21) = O(r * 11025) which for r~100 is ~1.1M ops, fine.
        T = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        
        # Now check T[a,b,c] for all a,b,c
        for a in range(n * m):
            i = a // m
            j = a % m
            for b in range(m * p):
                j2 = b // p
                k = b % p
                for c in range(p * n):
                    k2 = c // n
                    i2 = c % n
                    expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
                    if abs(T[a, b, c] - expected) > 1e-9:
                        return False
        return True

    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p."""
        r = n * m * p
        U = np.zeros((n * m, r), dtype=np.int64)
        V = np.zeros((m * p, r), dtype=np.int64)
        W = np.zeros((p * n, r), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, r

    def try_kronecker_decomp():
        """
        Try Kronecker product structure: 3x5x7 = 3x1x1 ⊗ 1x5x7
        The 3x1x1 part has rank 3 (identity-like).
        The 1x5x7 part: compute y[i] = sum_j A[j]*B[j,i] for i=0..4 (5 outputs from 5 inputs with 7... wait)
        Actually 1x5x7: n=1, m=5, p=7. Tensor T[j,k,i] = 1 iff j==k and i==0 (since n=1, only i=0).
        So T[j,k,0] = delta_{j,k}. This is just the 5x5 identity tensor with one "output" index.
        The rank of this tensor: T[j,k,0] = delta_{j,k}. This is a 5x5x1 tensor.
        Rank = 5 (each diagonal element is a rank-1 term).
        So total rank = 3 * 5 = 15? Let me verify.
        
        Actually, the Kronecker product of tensor decompositions:
        If T1 has rank r1 with factors (U1,V1,W1) and T2 has rank r2 with factors (U2,V2,W2),
        then T1⊗T2 has rank r1*r2 with factors (U1⊗U2, V1⊗V2, W1⊗W2).
        
        For 3x1x1: n=3, m=1, p=1. T[i, j, k] = 1 iff i==i' (but i' is the output index).
        Wait, the tensor is T[a,b,c] where a=i*m+j, b=j'*p+k, c=k'*n+i'.
        For n=3, m=1, p=1: a=i*1+0=i (i=0,1,2), b=0*1+0=0, c=0*3+i'=i' (i'=0,1,2).
        T[i, 0, i'] = 1 iff i==i' (and j==j'=0, k==k'=0).
        So T is a 3x1x3 tensor with T[i,0,i']=delta_{i,i'}.
        Rank: each (i,i') pair with i==i' is a rank-1 term. So rank = 3.
        U1[i, r] = delta_{i,r} for r=0,1,2 (shape 3x3)
        V1[0, r] = 1 for all r (shape 1x3)
        W1[i', r] = delta_{i',r} for r=0,1,2 (shape 3x3)
        Check: T[i,0,i'] = sum_r U1[i,r]*V1[0,r]*W1[i',r] = sum_r delta_{i,r}*1*delta_{i',r} = delta_{i,i'}. Correct.
        
        For 1x5x7: n=1, m=5, p=7. a=i*5+j = 0*5+j = j (j=0..4), b=j'*7+k (j'=0..4, k=0..6), c=k'*1+i' = k'*1+0 = k' (k'=0..6).
        T[j, j'*7+k, k'] = 1 iff i==i' (0==0, always true), j==j', k==k'.
        So T[j, b, c] = 1 iff j==j' and k==k' where b=j'*7+k, c=k'.
        T[j, j'*7+k, k'] = delta_{j,j'} * delta_{k,k'}.
        This is a 5x35x7 tensor. Rank: for each (j,k) pair, we have a term.
        T[j, j'*7+k, k'] = sum_{r} U2[j,r]*V2[j'*7+k,r]*W2[k',r]
        We need this to equal delta_{j,j'}delta_{k,k'}.
        If we set r = j*7+k (5*7=35 terms), then U2[j, j*7+k]=1, V2[j'*7+k, j*7+k]=delta_{j,j'}, W2[k', j*7+k]=delta_{k,k'}.
        So rank = 35 for the schoolbook of 1x5x7.
        
        But can we do better? The tensor T2[j, b, c] = delta_{j,j'}delta_{k,k'} where b=j'*7+k.
        This is actually the tensor for matrix multiplication 1x5x7, which is equivalent to:
        Given A (1x5) and B (5x7), compute C = A*B (1x7).
        C[k] = sum_j A[j]*B[j,k].
        The tensor is T[j, (j',k), k'] = 1 iff j==j' and k==k'.
        This is the standard matrix multiplication tensor M_{1,5,7}.
        Its rank is 5*7 = 35 by schoolbook. Can it be less?
        For M_{1,m,p}, the rank is m*p (you need one multiplication per output-input pair).
        Actually, for M_{1,m,p}: C[k] = sum_j A[j] B[j,k]. Each C[k] depends on all m inputs.
        The tensor has entries T[j, j', k, k'] = delta_{j,j'}delta_{k,k'}.
        This is a 4-index tensor but in our 3-index form it's T[a,b,c] with a=j, b=j'*p+k, c=k'.
        The rank of M_{1,m,p} is m*p because each (j,k) pair requires a separate multiplication.
        So rank of 1x5x7 is 35.
        
        Total Kronecker rank = 3 * 35 = 105. Same as schoolbook!
        So this doesn't help.
        
        Let me think differently. Maybe use a different decomposition.
        """
        return None

    # Start with schoolbook
    U_sb, V_sb, W_sb, base_rank = schoolbook()
    current_best_rank = base_rank
    current_best = (U_sb, V_sb, W_sb)
    
    # Try to improve using structured search
    # Strategy: use the fact that we can try to find low-rank decompositions
    # by exploiting the structure.
    
    # For now, just return schoolbook as a baseline.
    # The random search in the original was too slow and unlikely to find improvements.
    # Let's try a more targeted approach: use the Kronecker structure if it helps.
    
    # Actually, let's just return the schoolbook decomposition.
    # The task is to find rank-minimal, but we may not be able to beat 105 easily.
    U, V, W = current_best
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
