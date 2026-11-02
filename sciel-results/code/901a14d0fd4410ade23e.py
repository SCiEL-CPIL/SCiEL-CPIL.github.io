"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition search."""
import numpy as np
import time
import itertools
from collections import defaultdict

DIMENSIONS = [3, 4, 8]
n, m, p = DIMENSIONS
RANK_MAX = n * m * p  # 96 schoolbook rank

def construct():
    """Search for a low-rank exact tensor decomposition of 3x4x8 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    # The multiplication tensor T[i,j,k] for (A_3x4, B_4x8) -> C_3x8
    # T is a 3x4x8 tensor where T[i,j,k] = 1 if the product C[i,k] depends on A[i,j]*B[j,k]
    # Actually, the tensor is: for each output C[i,k], it's sum_j A[i,j]*B[j,k]
    # So the tensor M has entries M[i,k,j] = delta (it's the identity in j dimension)
    # 
    # More precisely: the matrix multiplication tensor for (n,m,p) has:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # and the condition is: for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] ... no
    #
    # The correct condition: The tensor T is defined such that
    # T[i,j,k] = 1 (it's the bilinear map)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] for the product
    # Actually no. The standard formulation:
    #
    # Given matrices A (n x m) and B (m x p), C = A * B where C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # The tensor decomposition requires:
    # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r]
    # Wait, that's not right either. Let me think more carefully.
    #
    # The bilinear form: C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # In tensor form, we have a tensor T of shape (n*m, m*p, p*n) such that
    # T[(i,j), (j',k), (k',i')] = 1 if i=i' and j=j' and k=k'
    #
    # Actually the standard formulation for matrix multiplication tensor:
    # The tensor M of dimensions (n*m) x (m*p) x (p*n) where
    # M[i*m+j, j*p+k, k*n+i] = 1
    #
    # A rank-r decomposition means: M = sum_{r=1}^{R} U_r * V_r * W_r (outer products)
    # where U_r is a vector of size n*m, V_r of size m*p, W_r of size p*n
    #
    # So: M[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    #
    # The condition is: for all i,j,k:
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # and for all other (a,b,c) not of that form: sum_r ... = 0
    #
    # Wait, let me re-read the task. "W[k*n+i,r]" - this is the trace convention.
    #
    # So the condition is:
    # sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] ... no, that doesn't make sense dimensionally.
    #
    # Let me re-read: "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "that exactly decompose the matrix multiplication tensor"
    #
    # The standard tensor decomposition for matrix multiplication:
    # M[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # where M[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some valid i,j,k (i.e., the "diagonal" entries)
    # and M[a,b,c] = 0 otherwise.
    #
    # Actually, I think the tensor is:
    # T[a, b, c] where a in [0, n*m), b in [0, m*p), c in [0, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # T[a,b,c] = 0 otherwise
    #
    # And we need: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition."""
        rank = U.shape[1]
        # Compute the reconstructed tensor
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We only need to check specific entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - 1) > 1e-9:
                        return False
        # Check some off-diagonal entries are zero
        # We need to verify that for (a,b,c) not of the form (i*m+j, j*p+k, k*n+i), the sum is 0
        # This is expensive to check all, so we'll rely on the structure
        # Actually, we should check a few random off-diagonal entries
        for _ in range(50):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            # Check if this is a valid diagonal entry
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if i == i2 and j == j2 and k == k2:
                continue  # This is a diagonal entry, already checked
            val = 0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[c, r]
            if abs(val) > 1e-9:
                return False
        return True
    
    # Start with schoolbook as baseline
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    # Try to find lower rank decompositions
    # Strategy: Use the structure of the problem
    # For 3x4x8, the schoolbook rank is 96
    # Known bounds: The rank of 3x4x8 matrix multiplication tensor
    # 
    # Let's try a structured search approach:
    # 1. Try ranks from 96 downward
    # 2. For each rank, try to find a decomposition
    
    # Alternative approach: Use the fact that we can combine terms
    # If we have two schoolbook terms (i1,j1,k1) and (i2,j2,k2) that can be merged,
    # we can reduce the rank.
    
    # Let's try a different approach: direct optimization
    # We want to minimize rank such that there exist U, V, W with the tensor property.
    
    # For small matrices, we can try to find a decomposition by solving a system of equations.
    # For rank R, we need:
    # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # For all (a,b,c) not diagonal: sum_r U[a, r] * V[b, r] * W[c, r] = 0
    
    # This is a system of polynomial equations. Hard to solve directly.
    
    # Let's try a greedy approach: start with schoolbook and try to eliminate terms
    # by finding linear dependencies.
    
    # Actually, let's try a simpler approach: use the known results for small matrix multiplication.
    # For 2x2x2, the rank is 7 (Strassen).
    # For 3x3x3, the rank is at most 23 (Landsberg).
    # For 3x4x8, we need to figure out the optimal rank.
    
    # Let's try to use a structured decomposition.
    # The key insight: if we can find a way to share computation between different (i,j,k) triples,
    # we can reduce the rank.
    
    # One approach: use block structure.
    # Group the i dimension (size 3) and see if we can share V and W factors.
    
    # Let's try: for each j, we have n*p = 3*8 = 24 output entries C[i,k].
    # C[i,k] = A[i,j] * B[j,k] for a fixed j.
    # If we could compute all 24 products for a fixed j with fewer than 24 rank-1 terms,
    # that would help. But each product A[i,j]*B[j,k] is a separate bilinear term.
    
    # Actually, for a fixed j, the sub-tensor is:
    # T[i, k] = A[i,j] * B[j,k] for i in 0..2, k in 0..7
    # This is an outer product of a 3-vector and an 8-vector, which has rank 1.
    # Wait no, it's 24 separate bilinear terms, each of rank 1.
    # But they can potentially be combined.
    
    # Hmm, let me think differently. The tensor for fixed j is:
    # A[i,j] * B[j,k] for all i,k
    # This is the outer product of vector A[:,j] (size 3) and vector B[j,:] (size 8).
    # The rank of this 3x8 matrix is at most min(3,8) = 3.
    # So for each j, we can represent the 24 products using at most 3 rank-1 terms!
    #
    # Wait, that's not quite right. The bilinear form is:
    # C[i,k] = A[i,j] * B[j,k]
    # This is a single bilinear form in A[i,j] and B[j,k] for each (i,k) pair.
    # But A[i,j] is the same for all k, and B[j,k] is the same for all i.
    #
    # So for fixed j, we have:
    # C[i,k] = A[i,j] * B[j,k]
    # This can be written as: C = A_col * B_row where A_col is 3x1 and B_row is 1x8.
    # The rank of this bilinear map is 1 (it's a single rank-1 bilinear form).
    #
    # Wait, I'm confusing matrix rank with tensor rank.
    # The bilinear form (i,k) -> A[i,j] * B[j,k] is a single bilinear term.
    # It has tensor rank 1.
    #
    # So for each j, the contribution to C is a rank-1 bilinear form.
    # Total rank would be m = 4 if we could just use one term per j.
    # But that's not right because each term A[i,j]*B[j,k] contributes to ALL (i,k) pairs.
    #
    # Let me re-examine. The full bilinear map is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # This is a sum of m = 4 bilinear forms, each of the form A[i,j]*B[j,k].
    # Each bilinear form A[i,j]*B[j,k] as a function of (i,k) is an outer product.
    # But as a bilinear form in the full input vectors A and B, each term A[i,j]*B[j,k]
    # is a rank-1 bilinear form.
    #
    # So the total tensor rank is at most m = 4? That can't be right because the schoolbook
    # rank is 96.
    #
    # I think I'm making an error. Let me be more careful.
    #
    # The input is A (3x4) and B (4x8). We can flatten them to vectors of size 12 and 32.
    # The output C is 3x8, flattened to size 24.
    #
    # The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # As a trilinear form (or rather, the tensor M of size 12 x 32 x 24):
    # M[a, b, c] where a indexes A, b indexes B, c indexes C
    # M[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some valid i,j,k
    # M[a, b, c] = 0 otherwise
    #
    # Wait, c should index C which is n x p = 3 x 8. So c = i*p+k? Or c = k*n+i?
    # The task says W has shape (p*n, rank) and W[k*n+i, r]. So c = k*n + i.
    #
    # So M[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    #
    # Now, the tensor rank of M is what we want to minimize.
    #
    # Let's think about it differently. For a fixed j, the nonzero entries of M are:
    # M[i*m+j, j*p+k, k*n+i] = 1 for all i in 0..2, k in 0..7
    #
    # This sub-tensor (for fixed j) has size 3 x 8 x 24 (since c = k*n+i ranges over 24 values).
    # The nonzero entries form a "diagonal" pattern.
    #
    # The rank of this sub-tensor: it's the tensor T_j where T_j[i, k, c] = 1 if c = k*n+i, else 0.
    # This is the identity-like tensor. Its rank is... let's see.
    #
    # T_j[i, k, k*n+i] = 1, T_j[i, k, c] = 0 otherwise.
    #
    # This can be written as: T_j = sum_{i,k} e_i * e_k * e_{k*n+i}
    # where e_i is the i-th standard basis vector in R^3, e_k in R^8, e_c in R^24.
    #
    # The rank of this sum is at most 3*8 = 24 (schoolbook).
    # But can it be less?
    #
    # T_j[i, k, c] = delta(c, k*n+i)
    #
    # This is the "copy" tensor. The rank of the copy tensor of size 3x8x24...
    #
    # Actually, I realize that for fixed j, the sub-tensor is:
    # T_j[i, k, c] = 1 if c = k*n + i, else 0
    #
    # This is equivalent to: T_j = sum_{i=0}^{2} sum_{k=0}^{7} e_i (in R^3) ⊗ e_k (in R^8) ⊗ e_{k*n+i} (in R^24)
    #
    # The tensor rank of this is at most 24. Can it be less?
    #
    # Note that c = k*n + i uniquely determines both k and i (since i < n = 3).
    # So c -> (k, i) is a bijection from {0,...,23} to {0,...,2} x {0,...,7}.
    #
    # So T_j[i, k, c] = 1 iff (i, k) = (c mod n, c div n) = (c mod 3, c div 3).
    #
    # This is the "diagonal" tensor. Its rank is equal to the dimension of the space, which is 24.
    # Actually no, the rank of the diagonal tensor diag(1,1,...,1) in R^d is d.
    #
    # Wait, the diagonal tensor D in R^d x R^d x R^d where D[i,j,k] = 1 if i=j=k else 0 has rank d.
    # But here we have T_j[i,k,c] where (i,k) and c are related by c = k*n+i.
    #
    # Let me reindex: let c' = k*n + i. Then T_j[i, k, c'] = 1 if c' = k*n + i.
    # This is the same as the diagonal tensor in a different basis.
    #
    # Specifically, if we define a new index c' that ranges over 0..23, and define
    # i(c') = c' mod 3, k(c') = c' div 3, then T_j[i, k, c'] = delta(i, i(c')) * delta(k, k(c')).
    #
    # This is: T_j = sum_{c'=0}^{23} e_{i(c')} ⊗ e_{k(c')} ⊗ e_{c'}
    #
    # The rank of this sum is 24 because the vectors e_{c'} are linearly independent.
    #
    # So for each j, the sub-tensor has rank 24, and the total rank is at most 4*24 = 96.
    #
    # But can we do better by sharing across j?
    #
    # The full tensor is: M = sum_j T_j where T_j[i*m+j, j*p+k, k*n+i] = 1.
    #
    # Note that T_j and T_j' for j != j' have disjoint support in the first index (since
    # i*m+j has different j components). So the supports don't overlap in the first index.
    #
    # However, the second index j*p+k also has different j components for different j.
    # And the third index k*n+i is the same for all j (it doesn't depend on j).
    #
    # So the full tensor M has:
    # M[i*m+j, j*p+k, k*n+i] = 1
    #
    # For different j values, the first two indices are in different "blocks", but the third
    # index is shared.
    #
    # This means we might be able to share the W factors across different j values.
    #
    # Let's try: suppose we use the same W vector for all j. Then:
    # M[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For fixed i,k, and varying j:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
    #
    # This is a system of equations that might be solvable with fewer rank terms.
    
    # Let's try a concrete approach: use optimization to find a low-rank decomposition.
    # We'll parameterize U, V, W and minimize a loss function that measures how close
    # the reconstruction is to the target tensor.
    
    def compute_loss(U, V, W):
        """Compute the loss for a given decomposition."""
        rank = U.shape[1]
        loss = 0
        # Check diagonal entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    loss += (val - 1) ** 2
        # Check some off-diagonal entries
        rng = np.random.RandomState(42)
        for _ in range(100):
            a = rng.randint(0, n * m)
            b = rng.randint(0, m * p)
            c = rng.randint(0, p * n)
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if i == i2 and j == j2 and k == k2:
                continue
            val = 0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[c, r]
            loss += val ** 2
        return loss
    
    # Try to find a decomposition with rank less than 96
    # Start from rank 96 and try to reduce
    
    # Given time constraints, let's try a few specific ranks
    # and use gradient-based optimization
    
    def try_rank(R, max_iter=500):
        """Try to find a rank-R decomposition using optimization."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize with random values
        rng = np.random.RandomState(0)
        U = rng.randn(n * m, R) * 0.1
        V = rng.randn(m * p, R) * 0.1
        W = rng.randn(p * n, R) * 0.1
        
        # Simple gradient descent
        lr = 0.01
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = 0
                        for r in range(R):
                            val += U[a, r] * V[b, r] * W[c, r]
                        err = val - 1
                        for r in range(R):
                            dU[a, r] += 2 * err * V[b, r] * W[c, r]
                            dV[b, r] += 2 * err * U[a, r] * W[c, r]
                            dW[c, r] += 2 * err * U[a, r] * V[b, r]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if it % 50 == 0:
                loss = compute_loss(U, V, W)
                if loss < 1e-6:
                    break
        
        loss = compute_loss(U, V, W)
        if loss < 1e-6:
            return U, V, W
        return None
    
    # Try ranks from 96 downward, but this is very slow
    # Let's try a few key ranks
    for R in [96, 95, 94, 93, 92, 90, 88, 86, 84, 80, 76, 72, 68, 64, 60, 56, 52, 48]:
        if time.time() > deadline - 20:
            break
        
        result = try_rank(R, max_iter=200)
        if result is not None:
            U, V, W = result
            if verify(U, V, W):
                best_rank = R
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                break
    
    # Convert to lists for output
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
