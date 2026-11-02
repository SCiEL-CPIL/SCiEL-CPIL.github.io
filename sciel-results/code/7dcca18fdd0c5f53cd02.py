import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16

    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank

    # Strassen coefficients for 2x2 * 2x2
    strassen_terms = [
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]

    def strassen_56():
        """8 independent 2x2 x 2x2 Strassen blocks, rank 56."""
        rank = 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        for a in range(2):  # A block column: 2a, 2a+1
            for b in range(4):  # B column block: 2b, 2b+1
                for t in range(7):
                    term = strassen_terms[t]
                    for (i_prime, j_prime), coeff in term['x'].items():
                        u_idx = i_prime * m + (2*a + j_prime)
                        U[u_idx, col] += coeff
                    for (j_prime, k_prime), coeff in term['y'].items():
                        v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                        V[v_idx, col] += coeff
                    for (i_prime, k_prime), coeff in term['z'].items():
                        w_idx = (2*b + k_prime) * n + i_prime
                        W[w_idx, col] += coeff
                    col += 1
        
        return U, V, W, rank

    def verify(U, V, W, rank, atol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :rank] * V[j*p+k, :rank] * W[k*n+i, :rank])
                    if not np.isclose(val, 1.0, atol=atol):
                        return False
        return True

    # Start with Strassen (rank 56)
    U_best, V_best, W_best, rank_best = strassen_56()
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()

    # --- Winograd-style polynomial approach for the middle dimension ---
    # For each row i of A (n=2 rows) and each output column block, we can
    # use fast polynomial multiplication. But the structure here is:
    # C[i, k] = sum_{j=0}^{m-1} A[i,j] * B[j,k]
    # For fixed i and k, this is a dot product of length m=4.
    # The tensor is: T[i*m+j, j*p+k, k*n+i] = 1
    # 
    # Winograd for length-4: multiply two length-4 vectors with 5 multiplications
    # (using the 5-point FFT-like scheme or the known 5-mult algorithm)
    #
    # Actually, for length-4 convolution, the rank is 5 (Winograd).
    # We have n=2 rows of A, and for each row i, we need to compute
    # C[i, :] = A[i, :] * B  (a row vector times a matrix)
    # This is: for each output column k, C[i,k] = dot(A[i,:], B[:,k])
    # This is a bilinear form in A[i,:] and B[:,k].
    # 
    # The bilinear complexity of a length-4 dot product is... well, it's just
    # a sum of 4 products. But we can share computations across k values.
    # 
    # Actually, the key insight: for fixed i, the map B -> C[i,:] is a linear
    # map from R^{m*p} to R^p. And for fixed i, the map A[i,:] -> C[i,:] is
    # a linear map from R^m to R^p (for each column of B).
    # 
    # The total tensor can be viewed as: for each i (0..n-1), we have a
    # bilinear map A[i,:] x B -> C[i,:]. The rank of each such map is at most
    # m*p, but we can do better.
    #
    # Let's think of it differently. The matmul tensor T_{n,m,p} has rank at
    # most n * (rank of length-m dot product with p outputs). 
    #
    # For the "outer" structure: T[i*m+j, j'*p+k, k'*n+i'] = 1 if i=i', j=j', k=k'
    # Wait, no. Let me re-derive.
    # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # We need: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = T[i*m+j, j'*p+k, k'*n+i']
    # where T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' with i=i', j=j', k=k'
    # and 0 otherwise? No...
    #
    # Actually: (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor M_{n,m,p} is defined by M_{i,j,k} = 1 for all i,j,k (it's the
    # "identity" tensor on the contraction index). Wait, that's not right either.
    #
    # The correct statement: we want to compute C = AB where A is n x m, B is m x p.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # In tensor decomposition language: the matmul tensor T has T_{i,j,k} = 1
    # (it's the all-ones tensor? No...)
    #
    # Let me be very careful. The bilinear map is:
    # (A, B) -> C where C_{i,k} = sum_j A_{i,j} B_{j,k}
    # Flattened: A_flat has indices (i,j) -> i*m+j
    # B_flat has indices (j,k) -> j*p+k
    # C_flat has indices (i,k) -> k*n+i (trace convention: W[k*n+i])
    #
    # The tensor T is: T[a, b, c] where a = i*m+j, b = j'*p+k', c = k''*n+i''
    # T[a, b, c] = 1 if i=i'', j=j', k=k', else 0.
    #
    # So T[i*m+j, j'*p+k', k''*n+i''] = delta_{i,i''} * delta_{j,j'} * delta_{k,k'}
    #
    # This is a very structured tensor. For each (i, j, k), there's exactly one
    # non-zero entry. The schoolbook decomposition uses n*m*p = 64 terms.
    #
    # Now, can we do better? The key is that for fixed i, the sub-tensor is:
    # T[i*m+j, j'*p+k, k'*n+i] = delta_{j,j'} * delta_{k,k'}
    # This is just the identity tensor on (j, k) pairs, which has rank m*p = 32.
    # But we have n=2 such independent sub-tensors, so total rank is at most 64.
    #
    # However, we can share structure. The Winograd/FFT approach:
    # For fixed i, C[i, k] = sum_j A[i,j] B[j,k]
    # This is a matrix-vector product: C[i,:] = A[i,:] * B
    # The bilinear complexity of multiplying a 1 x 4 vector by a 4 x 8 matrix
    # is... well, it's 4*8 = 32 scalar multiplications in schoolbook.
    # But we can use fast algorithms.
    #
    # Actually, the key realization: the tensor T_{n,m,p} for matmul has a known
    # structure. For n=2, m=4, p=8, we can use:
    # - Strassen on the 2x2 blocks (giving rank 56 as we have)
    # - Or potentially better decompositions
    #
    # Let's try a different approach: use the fact that for each output row i,
    # we're doing a length-4 dot product for each of the 8 output columns.
    # The Winograd algorithm for length-4 convolution uses 5 multiplications.
    # But here it's not convolution, it's a general dot product.
    #
    # For a general length-m dot product, the rank is m (you can't do better
    # than m multiplications for a generic dot product). So for each i and k,
    # we need m=4 multiplications, giving n*m*p = 64 total.
    #
    # But we can share across k! For fixed i, C[i, :] = A[i, :] * B.
    # This is a linear map from B (4x8) to C[i,:] (1x8).
    # The bilinear complexity is the rank of the tensor for this map.
    #
    # Let's just stick with the Strassen approach and try to reduce rank.

    # Try to find linear dependencies among the 56 terms
    def try_remove_term(col_idx, U, V, W, rank):
        if time.time() > deadline - 20:
            return None
        
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        U_new[:, col_idx] = 0
        V_new[:, col_idx] = 0
        W_new[:, col_idx] = 0
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_new[i*m+j, :rank] * V_new[j*p+k, :rank] * W_new[k*n+i, :rank])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return None
        return U_new, V_new, W_new, rank

    # Try removing each term one by one (greedy)
    removed = []
    for col in range(rank_best):
        if time.time() > deadline - 20:
            break
        result = try_remove_term(col, U_best, V_best, W_best, rank_best)
        if result is not None:
            U_best, V_best, W_best, rank_best = result
            removed.append(col)

    # Try Winograd-style decomposition for the middle dimension
    # For each row i of A, we compute C[i, :] = A[i, :] * B
    # Using Winograd for length-4: 5 multiplications per output
    # But we need to handle all 8 outputs simultaneously.
    
    def winograd_4x8():
        """
        Winograd-style decomposition for 1x4 * 4x8 matrix multiplication.
        Uses the 5-point Winograd scheme for length-4.
        """
        # Winograd for length-4 convolution:
        # Given a = [a0, a1, a2, a3] and b = [b0, b1, b2, b3]
        # Compute c = a * b (dot product)
        # 
        # Standard Winograd F_4: 
        # x0 = a0 + a1, x1 = a0 - a1, x2 = a2 + a3, x3 = a2 - a3
        # y0 = b0 + b1, y1 = b0 - b1, y2 = b2 + b3, y3 = b2 - b3
        # t0 = x0 * y0, t1 = x1 * y1, t2 = x2 * y2, t3 = x3 * y3
        # c = (t0 + t1 + t2 + t3) / 2 ... this isn't quite right for dot product.
        #
        # Actually, for a generic dot product of length 4, the rank is 4.
        # Winograd doesn't help for a single dot product.
        #
        # But for multiple dot products sharing the same B matrix, we can help.
        # C[i, k] = sum_j A[i,j] B[j,k] for k = 0..7
        # 
        # Let's use a different approach: factor B into a product of smaller matrices.
        # B (4x8) = P * Q where P is 4x4 and Q is 4x8.
        # Then C[i, k] = sum_j A[i,j] P[j,l] Q[l,k] = sum_l (sum_j A[i,j] P[j,l]) Q[l,k]
        # = sum_l A'[i,l] Q[l,k]
        # where A' = A * P (2x4 * 4x4 = 2x4)
        # 
        # This doesn't reduce the total rank unless we exploit structure.
        
        # Let's try a direct Winograd-like approach.
        # For the 4x8 B matrix, we can write:
        # B[j, k] for j=0..3, k=0..7
        # 
        # Group the 8 columns into 4 pairs: (k, k+4) for k=0..3
        # B[j, k] and B[j, k+4]
        # 
        # C[i, k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
        # C[i, k+4] = A[i,0]*B[0,k+4] + A[i,1]*B[1,k+4] + A[i,2]*B[2,k+4] + A[i,3]*B[3,k+4]
        
        # This is getting complicated. Let's just return the schoolbook for now
        # and rely on the Strassen + greedy removal.
        return None

    # Try a different Strassen variant: use 2x4 * 4x8 as 2*(2x2) * (4x8)
    # Actually, let's try to use the known result that M_{2,4,8} has rank <= 56
    # and see if we can find a lower rank decomposition.
    
    # Attempt: use the fact that the tensor has a block structure.
    # T[i*m+j, j'*p+k, k'*n+i'] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # 
    # For i=0: T[j, j'*p+k, k'*2+0] = delta_{j,j'} delta_{k,k'}
    # For i=1: T[4+j, j'*p+k, k'*2+1] = delta_{j,j'} delta_{k,k'}
    #
    # These are two independent sub-tensors, each of rank 4*8=32.
    # Total rank <= 64 (schoolbook) or 56 (Strassen).
    
    # Let's try to see if we can merge some terms between the i=0 and i=1 blocks.
    # This would require finding a decomposition that shares terms across i values.
    
    # For now, let's try a randomized search for lower rank.
    def random_search(current_U, current_V, current_W, current_rank, max_iters=100):
        """Try to find a lower rank decomposition by random perturbation."""
        if time.time() > deadline - 30:
            return None
        
        best = (current_U.copy(), current_V.copy(), current_W.copy(), current_rank)
        
        for _ in range(max_iters):
            if time.time() > deadline - 30:
                break
            
            # Try to combine two random columns
            if current_rank < 2:
                break
            
            idx1 = np.random.randint(0, current_rank)
            idx2 = np.random.randint(0, current_rank)
            if idx1 == idx2:
                continue
            
            # Try to see if column idx2 can be expressed as a linear combination
            # of other columns (including idx1)
            # This is a linear algebra problem.
            
            # Build the matrix of all columns except idx2
            U_other = np.delete(current_U, idx2, axis=1)
            V_other = np.delete(current_V, idx2, axis=1)
            W_other = np.delete(current_W, idx2, axis=1)
            
            # Check if the tensor contribution of column idx2 is in the span
            # of the other columns. This is hard to check directly.
            
            # Alternative: try zeroing out column idx2 and see if we can
            # adjust other columns to compensate.
            U_new = current_U.copy()
            V_new = current_V.copy()
            W_new = current_W.copy()
            
            # Zero out idx2
            U_new[:, idx2] = 0
            V_new[:, idx2] = 0
            W_new[:, idx2] = 0
            
            # Check if still valid
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_new[i*m+j, :current_rank] * V_new[j*p+k, :current_rank] * W_new[k*n+i, :current_rank])
                        if not np.isclose(val, 1.0, atol=1e-9):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # We found a valid decomposition with one fewer term
                # Remove the column entirely
                U_new = np.delete(U_new, idx2, axis=1)
                V_new = np.delete(V_new, idx2, axis=1)
                W_new = np.delete(W_new, idx2, axis=1)
                new_rank = current_rank - 1
                best = (U_new, V_new, W_new, new_rank)
                current_U, current_V, current_W, current_rank = best
        
        return best

    if time.time() < deadline - 40:
        result = random_search(U_best, V_best, W_best, rank_best, max_iters=50)
        if result is not None:
            U_best, V_best, W_best, rank_best = result

    # Final verification
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
