"""Matrix multiplication 4x5*5x6 rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # such that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Schoolbook baseline: rank = n*m*p = 120
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best_decomp = (U_sb, V_sb, W_sb)
    best_rank = rank_sb
    
    def verify_decomp(U, V, W, rank, tol=1e-9):
        """Verify that the decomposition exactly reconstructs T."""
        recon = np.zeros((n, m, p), dtype=np.float64)
        for r in range(rank):
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    uj = u_col[i, j]
                    if uj == 0:
                        continue
                    for k in range(p):
                        recon[i, j, k] += uj * v_col[j, k] * w_col[k, i]
        return np.allclose(recon, np.ones((n, m, p)), atol=tol)
    
    # Block decomposition approach:
    # A (4x5) * B (5x6) = C (4x6)
    # Split A = [A1 | A2] where A1 is 4x4 (cols 0-3), A2 is 4x1 (col 4)
    # Split B columns: B1 = B[:, 0:4] (5x4), B2 = B[:, 4:6] (5x2)
    # C = A1*B1 + A2*B1 + A1*B2 + A2*B2
    # But B1 = [B1a; B1b] where B1a is 4x4, B1b is 1x4
    # B2 = [B2a; B2b] where B2a is 4x2, B2b is 1x2
    # C = A1*B1a + A2*B1b + A1*B2a + A2*B2b
    
    # For A1*B1a (4x4*4x4): can use Strassen-like or schoolbook
    # For A2*B1b (4x1*1x4): rank 4 (outer product)
    # For A1*B2a (4x4*4x2): schoolbook rank 32
    # For A2*B2b (4x1*1x2): rank 4
    
    # Let's try to use a better decomposition for 4x4*4x4.
    # The key insight: we can use the fact that the tensor is all-ones.
    # For the all-ones tensor of shape (a,b,c), the rank is at most min(a*b, b*c, a*c) 
    # if we can find a low-rank factorization.
    
    # Actually, let's think about this differently.
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We want to write 1 = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # One approach: use the fact that we can factor through a smaller space.
    # For example, if we set U[i,j,r] = a[i,r], V[j,k,r] = b[j,r], W[k,i,r] = c[k,r]
    # then sum_r a[i,r]*b[j,r]*c[k,r] needs to equal 1 for all i,j,k.
    # This is possible if the vectors a[i,:], b[j,:], c[k,:] span a space where
    # the trilinear form is constant.
    
    # A simpler approach: use rank-1 terms.
    # For each (i,j,k), we have a term. But we can group them.
    
    # Let's try a structured approach:
    # Partition the (i,j,k) space into groups where we can share factors.
    
    # Approach 1: Use the identity 1 = sum over a partition
    # If we partition {0,...,n-1} x {0,...,m-1} x {0,...,p-1} into groups,
    # and for each group we use one rank-1 term, we get rank = number of groups.
    
    # But that's just schoolbook if each group is a single (i,j,k).
    
    # Approach 2: Use the fact that we can have U[i,j,r] depend only on i and r,
    # V[j,k,r] depend only on j and r, W[k,i,r] depend only on k and r.
    # Then sum_r a[i,r]*b[j,r]*c[k,r] = 1 for all i,j,k.
    # This requires that the matrix M[i,j] = sum_r a[i,r]*b[j,r]*c[k,r] is constant in k.
    # So we need c[k,r] to be such that the weighted sum is independent of k.
    
    # If we set c[k,r] = 1 for all k,r, then we need sum_r a[i,r]*b[j,r] = 1 for all i,j.
    # This is possible with rank at most max(n,m) if we can find such a factorization.
    # But we also need the constraint that the sum is exactly 1.
    
    # Actually, let's try a different structured approach.
    # Use the fact that we can write:
    # 1 = sum_{r=1}^{R} u_r(i,j) * v_r(j,k) * w_r(k,i)
    
    # If we choose u_r(i,j) = alpha_r(i) * beta_r(j), v_r(j,k) = gamma_r(j) * delta_r(k),
    # w_r(k,i) = epsilon_r(k) * zeta_r(i), then:
    # sum_r alpha_r(i) beta_r(j) gamma_r(j) delta_r(k) epsilon_r(k) zeta_r(i) = 1
    # = sum_r [alpha_r(i) zeta_r(i)] * [beta_r(j) gamma_r(j)] * [delta_r(k) epsilon_r(k)] = 1
    
    # This is a product of three sums that must equal 1 for all i,j,k.
    # This is only possible if each sum is constant, which is very restrictive.
    
    # Let's try yet another approach: use a fixed number of "modes" and solve.
    
    # Approach 3: Use a small number of rank-1 terms with carefully chosen values.
    # For R terms, we have R*20 + R*30 + R*24 = 74R variables.
    # We need 4*5*6 = 120 equations (one for each (i,j,k)).
    # So we need at least ceil(120/74) = 2 terms, but that's a lower bound.
    # In practice, we need many more terms.
    
    # Let's try a greedy approach: start with schoolbook and try to merge terms.
    
    # Approach 4: Use the known result that for the all-ones tensor,
    # we can achieve rank = n + m + p - 2 or similar.
    
    # Actually, let's try a concrete construction:
    # Use R = n + m + p - 2 = 4 + 5 + 6 - 2 = 13 terms.
    
    # Construction:
    # For r = 0, ..., n-1: u_r[i,j] = 1 if i=r, else 0. v_r[j,k] = 1 for all j,k. w_r[k,i] = 1 if k=0, else 0.
    # Wait, that doesn't work.
    
    # Let me think more carefully. We want:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try: U[i*m+j, r] = a[i,r] for r < n, and 0 otherwise
    # V[j*p+k, r] = b[j,r] for r < n, and 0 otherwise  
    # W[k*n+i, r] = c[k,r] for r < n, and 0 otherwise
    # Then sum_{r=0}^{n-1} a[i,r]*b[j,r]*c[k,r] = 1 for all i,j,k.
    # This requires that the matrix A[i,j] = sum_r a[i,r]*b[j,r]*c[k,r] is constant in k.
    # So we need c[k,r] to be such that the sum is independent of k.
    # If c[k,r] = 1 for all k, then sum_r a[i,r]*b[j,r] = 1 for all i,j.
    # This is possible: set a[i,r] = 1/sqrt(n) for all i,r, b[j,r] = sqrt(n)/m for all j,r.
    # Then sum_r a[i,r]*b[j,r] = n * (1/sqrt(n)) * (sqrt(n)/m) = n/m.
    # That's not 1 unless n=m.
    
    # Let's try: a[i,r] = 1/n for all i,r, b[j,r] = m/n for all j,r, c[k,r] = 1 for all k,r.
    # Then sum_r a[i,r]*b[j,r]*c[k,r] = n * (1/n) * (m/n) * 1 = m/n.
    # Still not 1.
    
    # Let's try: a[i,r] = 1 for r=i, 0 otherwise (identity-like).
    # b[j,r] = 1 for all j,r. c[k,r] = 1 for all k,r.
    # Then sum_r a[i,r]*b[j,r]*c[k,r] = b[j,i]*c[k,i] = 1*1 = 1. 
    # This works! Rank = n = 4.
    
    # Wait, let me verify:
    # U[i*m+j, r] = 1 if r=i, else 0. So U[i*m+j, i] = 1, U[i*m+j, r] = 0 for r != i.
    # V[j*p+k, r] = 1 for all j,k,r.
    # W[k*n+i, r] = 1 for all k,i,r.
    # Then sum_r U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] = U[i*m+j, i]*V[j*p+k, i]*W[k*n+i, i] = 1*1*1 = 1.
    # Yes! This gives rank = n = 4.
    
    # But wait, can we do even better? Let's check if rank < 4 is possible.
    # The tensor T[i,j,k] = 1 has a certain rank. For the all-ones tensor of shape (n,m,p),
    # the rank is at least max(n,m,p) in some cases.
    
    # Actually, I think the construction above is valid and gives rank = 4.
    # Let's verify it more carefully.
    
    def construct_simple():
        """Simple construction with rank = n = 4."""
        R = n
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        
        for r in range(n):
            # U[i*m+j, r] = 1 if i == r, else 0
            for i in range(n):
                if i == r:
                    for j in range(m):
                        U[i * m + j, r] = 1.0
            # V[j*p+k, r] = 1 for all j,k
            for j in range(m):
                for k in range(p):
                    V[j * p + k, r] = 1.0
            # W[k*n+i, r] = 1 for all k,i
            for k in range(p):
                for i in range(n):
                    W[k * n + i, r] = 1.0
        
        return U, V, W, R
    
    U_s, V_s, W_s, rank_s = construct_simple()
    if verify_decomp(U_s, V_s, W_s, rank_s):
        best_rank = rank_s
        best_decomp = (U_s, V_s, W_s)
        print(f"Simple construction: rank {rank_s}")
    else:
        print("Simple construction failed verification")
    
    # Can we do better than rank 4?
    # Let's try rank 3, 2, 1.
    
    # For rank 1: U[i*m+j, 0]*V[j*p+k, 0]*W[k*n+i, 0] = 1 for all i,j,k.
    # This requires U[i*m+j, 0] = a[i,j], V[j*p+k, 0] = b[j,k], W[k*n+i, 0] = c[k,i].
    # a[i,j]*b[j,k]*c[k,i] = 1 for all i,j,k.
    # This implies a[i,j] = 1/b[j,k]/c[k,i] for all k, so b[j,k]*c[k,i] must be independent of k.
    # Similarly, a[i,j]*b[j,k] must be independent of j, etc.
    # This is only possible if all are constant, which gives 1 = const^3, so const = 1.
    # But then a[i,j] = 1, b[j,k] = 1, c[k,i] = 1, which works!
    # Wait, that's rank 1!
    
    # Let me verify: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1.
    # Then sum_0 U[i*m+j, 0]*V[j*p+k, 0]*W[k*n+i, 0] = 1*1*1 = 1. Yes!
    
    # So rank 1 works! The tensor is rank 1.
    
    def construct_rank1():
        """Rank-1 construction."""
        R = 1
        U = np.ones((n * m, R))
        V = np.ones((m * p, R))
        W = np.ones((p * n, R))
        return U, V, W, R
    
    U_r1, V_r1, W_r1, rank_r1 = construct_rank1()
    if verify_decomp(U_r1, V_r1, W_r1, rank_r1):
        best_rank = rank_r1
        best_decomp = (U_r1, V_r1, W_r1)
        print(f"Rank-1 construction: rank {rank_r1}")
    else:
        print("Rank-1 construction failed verification")
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
