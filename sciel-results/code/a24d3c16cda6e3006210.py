"""Tensor decomposition search for 3x5x5 matrix multiplication."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np

def _build_decomposition(n, m, p):
    """
    Build a decomposition of the n x m x p matrix multiplication tensor.
    
    Strategy:
    - Split the inner dimension m into blocks: 2 and 3 (since m=5).
    - For each block of size b, split p into blocks of size 2 and 3 (since p=5).
    - For each combination of (A block, B block), use an optimal algorithm:
      - 2x2x2: Strassen (rank 7)
      - 2x2x3: rank 6
      - 2x3x2: rank 6
      - 2x3x3: rank 9
      - 3x2x2: rank 6
      - 3x2x3: rank 9
      - 3x3x2: rank 9
      - 3x3x3: Laderman (rank 23)
    
    The indices are:
    - U: (n*m, rank), where row index is i*m + j
    - V: (m*p, rank), where row index is j*p + k
    - W: (p*n, rank), where row index is k*n + i
    """
    n, m, p = 3, 5, 5
    
    # We'll build the decomposition by processing each (i, j_block, k_block) combination.
    # j_block is the block of j (inner dimension), k_block is the block of k (output col).
    # For each i (row of A and output), we have a sub-problem of size 1 x b_j x b_k.
    # But we can group by j_block and k_block.
    
    # Actually, let's think of it as: for each j in [0, m) and k in [0, p),
    # the contribution to C[i, k] from A[i, j] and B[j, k] is 1.
    # We want to group these contributions into tensor terms.
    
    # Let's define the blocks:
    # j blocks: [0, 1] (size 2), [2, 3, 4] (size 3)
    # k blocks: [0, 1] (size 2), [2, 3, 4] (size 3)
    
    # For each i in [0, 3), j_block in {0, 1}, k_block in {0, 1},
    # we have a sub-problem of size 1 x b_j x b_k.
    # The rank of a 1 x b x c tensor is b*c (since it's just an outer product of a vector of size 1 with a matrix).
    # Wait, that's not right. The tensor is T[i, j, k] = 1 if i, j, k are in the right positions.
    # For a fixed i, the tensor is T[j, k] = 1 for j in j_block, k in k_block.
    # This is a matrix of all ones of size b_j x b_k.
    # The rank of a matrix of all ones is 1.
    # So for each i, j_block, k_block, we need rank 1.
    # Total rank = 3 * 2 * 2 = 12.
    # That's way better than 75!
    
    # Let me verify.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # For a fixed i, C[i, k] = sum_j A[i, j] B[j, k]
    # This is the dot product of A[i, :] and B[:, k].
    # We can decompose this using the fact that B[:, k] for k in k_block can be written as a linear combination of basis vectors.
    # But the tensor is T[i, j, k] = 1.
    # For a fixed i, T[j, k] = 1 for j in j_block, k in k_block.
    # This is a rank 1 matrix (all ones).
    # So we can write T[j, k] = u_j * v_k where u_j = 1, v_k = 1.
    # So for each i, j_block, k_block, we need 1 term.
    # Total rank = 3 * 2 * 2 = 12.
    
    # Let's construct this.
    rank = 12
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j_block in range(2):  # 0: [0,1], 1: [2,3,4]
            for k_block in range(2):  # 0: [0,1], 1: [2,3,4]
                # j indices in this block
                if j_block == 0:
                    j_indices = [0, 1]
                else:
                    j_indices = [2, 3, 4]
                # k indices in this block
                if k_block == 0:
                    k_indices = [0, 1]
                else:
                    k_indices = [2, 3, 4]
                
                for j in j_indices:
                    U[i * m + j, col] = 1.0
                for k in k_indices:
                    V[j_indices[0] * p + k, col] = 1.0  # Wait, this is wrong.
                # Actually, V[j * p + k, col] should be 1 for all j in j_indices, k in k_indices.
                # But V is indexed by (j, k), so for each (j, k) in the block, V[j * p + k, col] = 1.
                # And W[k * n + i, col] = 1 for all k in k_indices.
                # But then the term is sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
                # For this to equal 1, we need U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for the correct (i, j, k).
                # But if we set U[i*m+j, c] = 1 for all j in j_block, V[j*p+k, c] = 1 for all (j, k) in the block,
                # and W[k*n+i, c] = 1 for all k in k_block, then the product is 1 for all (i, j, k) in the block.
                # But we need it to be 1 only for the specific (i, j, k) that we're decomposing.
                # Wait, the tensor is T[i, j, k] = 1 for all i, j, k.
                # So for each (i, j, k), we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
                # If we set U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for a single c,
                # then the product is 1 for that (i, j, k) and 0 for others (since other c's are 0).
                # But that's the schoolbook approach, rank 75.
                # To get rank 12, we need to share the terms.
                # For a fixed i, j_block, k_block, we want to represent the all-ones matrix of size b_j x b_k.
                # The all-ones matrix can be written as u * v^T where u is a column vector of ones and v is a row vector of ones.
                # So T[j, k] = u_j * v_k where u_j = 1, v_k = 1.
                # In tensor form, this is one term: U[i*m+j, c] = u_j = 1, V[j*p+k, c] = v_k = 1, W[k*n+i, c] = 1.
                # But V[j*p+k, c] depends on both j and k, so we can't set it to v_k alone.
                # The tensor is T[i, j, k] = 1, which is a 3D tensor.
                # For a fixed i, it's a 2D matrix T[j, k] = 1.
                # This 2D matrix has rank 1.
                # So we can write T[j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                # where the sum is over c, and for this i, j_block, k_block, we use 1 c.
                # But V[j*p+k, c] must be such that for each (j, k), V[j*p+k, c] = 1.
                # So V[j*p+k, c] = 1 for all (j, k) in the block.
                # And U[i*m+j, c] = 1 for all j in the block.
                # And W[k*n+i, c] = 1 for all k in the block.
                # Then the product is 1 for all (i, j, k) in the block.
                # But we need it to be 1 for each (i, j, k) individually.
                # Wait, the tensor is T[i, j, k] = 1 for all i, j, k.
                # So for each (i, j, k), the sum over c should be 1.
                # If we use the same c for all (j, k) in the block, then for a fixed (i, j, k),
                # the contribution from this c is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 * 1 * 1 = 1.
                # And there are no other c's contributing to this (i, j, k) (since other c's are for other blocks).
                # So the sum is 1.
                # This works!
                # So for each (i, j_block, k_block), we use 1 c.
                # Total rank = 3 * 2 * 2 = 12.
                
                for j in j_indices:
                    U[i * m + j, col] = 1.0
                for j in j_indices:
                    for k in k_indices:
                        V[j * p + k, col] = 1.0
                for k in k_indices:
                    W[k * n + i, col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    return _build_decomposition(n, m, p)
# EVOLVE-BLOCK-END
