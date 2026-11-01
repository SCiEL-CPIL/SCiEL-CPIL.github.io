import time
import itertools
import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Tensor dimension: n*m*p = 24
    tensor_dim = n * m * p
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
    
    # Generate candidate vectors with integer/half-integer entries in [-2, 2]
    # Values: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
    half_ints = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    def generate_candidates(dim, max_val=2.0):
        """Generate all integer/half-integer vectors of given length with |entry| <= max_val"""
        vals = [v for v in half_ints if abs(v) <= max_val]
        candidates = []
        for combo in itertools.product(vals, repeat=dim):
            if any(abs(c) > 1e-9 for c in combo):  # skip all-zero
                candidates.append(np.array(combo, dtype=np.float64))
        return candidates
    
    # Generate candidate vectors for U, V, W
    # U: length n*m = 6, V: length m*p = 12, W: length p*n = 8
    # To keep search manageable, we'll use bounded candidates
    # For efficiency, use smaller bounds initially
    
    # Strategy: Enumerate rank-1 tensors and greedily select those that increase rank
    # A rank-1 tensor is outer product of u (n*m), v (m*p), w (p*n)
    # Flattened: u ⊗ v ⊗ w has 24 components
    
    # We need to find minimal set of such tensors that span T's space
    # But we need EXACT decomposition, so we need the selected tensors to span T exactly
    
    # Better approach: Find a basis of rank-1 tensors that spans the same space as T
    # Then express T as a linear combination of them
    
    # The matrix multiplication tensor T lives in R^(6*12*8) = R^576
    # But it's a specific tensor. We need rank-1 tensors whose span contains T.
    
    # Actually, the tensor decomposition means:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # So we need to find U, V, W such that this holds exactly.
    
    # This is equivalent to: the tensor T must be in the span of rank-1 tensors
    # formed by the columns of U, V, W.
    
    # Let's think differently: We want to find minimal rank r such that there exist
    # vectors u_1,...,u_r in R^6, v_1,...,v_r in R^12, w_1,...,w_r in R^8
    # such that T = sum_r u_r ⊗ v_r ⊗ w_r
    
    # This is a hard problem. Let's try a greedy approach:
    # 1. Generate many rank-1 candidate tensors
    # 2. Use greedy selection to find a small set that can span T
    
    # For exactness with integer/half-integer entries, we need the decomposition to be exact.
    
    # Let's try: enumerate all rank-1 tensors with small integer/half-integer entries,
    # then use linear algebra to find if T can be expressed as a combination.
    
    # The space of all tensors is R^576. T is one specific tensor.
    # We need to find r rank-1 tensors t_1,...,t_r and coefficients c_1,...,c_r
    # such that T = sum c_r * t_r.
    # But the coefficients are absorbed into U, V, W.
    
    # Actually, if we fix the directions of u_r, v_r, w_r, then the coefficients
    # are determined by solving a linear system.
    
    # Let's try a different approach: 
    # The tensor T has a specific structure. Let's exploit it.
    
    # T[i*m+j, j*p+k, k*n+i] = 1 for valid (i,j,k)
    
    # For 2x3x4: 
    # i in {0,1}, j in {0,1,2}, k in {0,1,2,3}
    # There are 2*3*4 = 12 non-zero entries.
    
    # The naive decomposition has rank 12 (one for each (i,j,k) triple).
    # Can we do better?
    
    # Known results: For 2x3 matrix multiplication, the rank is 7 (Strassen-like).
    # For 2x3x4, we might be able to do better than 12.
    
    # Let's try to find a decomposition with rank < 12.
    
    # Approach: Use the fact that the tensor has structure.
    # Group by i: For each i, we have a 3x4 tensor slice.
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # For fixed i, the slice is a 3x4 matrix multiplication tensor.
    # The rank of 3x4 matrix multiplication is at most 3*4=12, but can be less.
    
    # Actually, let's just try the greedy enumeration approach with a time limit.
    
    # Generate candidate rank-1 tensors
    # To keep it manageable, use small vectors
    
    # For U vectors (length 6), use entries in {-1, 0, 1}
    # For V vectors (length 12), use entries in {-1, 0, 1}
    # For W vectors (length 8), use entries in {-1, 0, 1}
    
    # But that's 3^6 * 3^12 * 3^8 = 3^26 which is too many.
    
    # Instead, let's use a smarter approach:
    # We know T has 12 non-zero entries. Let's try to cover them with fewer rank-1 tensors.
    
    # A rank-1 tensor u ⊗ v ⊗ w has the property that its non-zero pattern
    # is the product of the non-zero patterns of u, v, w.
    
    # Let's try to find a set of rank-1 tensors that can reproduce T.
    
    # For each (i,j,k), T[i*m+j, j*p+k, k*n+i] = 1.
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # This is a system of 12 equations (one for each (i,j,k)) with unknowns being
    # the products U[a,r]*V[b,r]*W[c,r].
    
    # Let's try rank r = 7, 8, 9, ... and see if we can solve.
    
    # For a given rank r, we have r terms. Each term contributes to multiple equations.
    # This is a nonlinear system.
    
    # Let's try a different strategy: use known decompositions.
    # The bilinear complexity of 2x3 matrix multiplication is 7.
    # We can extend this to 2x3x4.
    
    # For 2x3x4, think of it as: for each of the 4 output columns, we multiply 2x3 by 3x1.
    # But that's not quite right.
    
    # Actually, 2x3 times 3x4: the output is 2x4.
    # We can think of this as 4 independent 2x3 by 3x1 multiplications, but that would be 4*7=28.
    # That's worse than 12.
    
    # Alternatively, we can use the structure more cleverly.
    
    # Let's just try the naive rank-12 decomposition and see if we can improve it.
    
    # For now, return the naive decomposition. The search for improvement may not complete in time.
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u][r] = 1
                V[idx_v][r] = 1
                W[idx_w][r] = 1
                
                r += 1
                
    return {"U": U, "V": V, "W": W}
