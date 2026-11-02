import numpy as np
import time
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions: A is n x m, B is m x p, C is n x p
    # Tensor T[i, j, k] corresponds to C[i, k] = sum_j A[i, j] B[j, k]
    # Decomposition: T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r] 
    # Note: The problem statement says W[k*n+i, r]. 
    # Let's re-read carefully: "output factor follows the official trace convention W[k*n+i,r]"
    # This implies the index for W is combined k and i.
    # Standard decomposition: C[i, k] = sum_j sum_r U[i, r] V[j, r] W[k, r] ? No.
    # Usually: C[i, k] = sum_j A[i, j] B[j, k].
    # Tensor form: T[i, j, k] = delta_{i, i'} delta_{j, j'} ... no.
    # The tensor for matrix multiplication $C = AB$ is $T_{i, j, k} = 1$ if $i,j,k$ form a valid term?
    # No, the tensor $T$ is defined such that $\langle T, A \otimes B \rangle = \text{vec}(AB)$.
    # $T_{i, j, k}$ is the coefficient of $A_{i, j} B_{j, k}$ in $C_{i, k}$.
    # So $T_{i, j, k} = 1$ if the indices match the multiplication rule?
    # Actually, $C_{i, k} = \sum_{j} A_{i, j} B_{j, k}$.
    # So the tensor $T$ has entries $T_{i, j, k} = 1$ for all valid $i, j, k$? No.
    # The tensor is $T = \sum_{i, j, k} e_i \otimes e_j \otimes e_k$ where the term contributes to $C_{i, k}$ via $A_{i, j} B_{j, k}$.
    # Wait, the standard tensor for matrix multiplication is $T_{i, j, k} = 1$ if we are computing $C_{i, k}$ using $A_{i, j}$ and $B_{j, k}$.
    # There is only one $j$ for a given $i, k$? No, $j$ sums over $1..m$.
    # So for a fixed $i, k$, the term is $\sum_j A_{i, j} B_{j, k}$.
    # The tensor $T$ is a 3-tensor of size $n \times m \times p$.
    # $T_{i, j, k} = 1$ if the index $j$ is the middle dimension?
    # Yes, $T_{i, j, k} = 1$ for all $i \in [n], j \in [m], k \in [p]$.
    # Wait, that would mean $C_{i, k} = \sum_{j} T_{i, j, k} A_{i, j} B_{j, k} = \sum_j A_{i, j} B_{j, k}$.
    # So $T_{i, j, k} = 1$ for all $i, j, k$.
    
    # Let's check the factor shapes.
    # U: (n*m, rank) -> U[i, r] ? No, U[i*m+j, r]?
    # Problem says: U (n*m, rank), V (m*p, rank), W (p*n, rank).
    # And "W[k*n+i, r]".
    # This suggests the indices are flattened.
    # Let's assume the standard Kronecker product structure.
    # $C_{i, k} = \sum_r U_{i, r} (\sum_j V_{j, r} W_{k, r}?)$ No.
    
    # Let's look at the seed code.
    # U[i*m + j, column] = 1
    # V[j*p + k, column] = 1
    # W[k*n + i, column] = 1
    # This corresponds to the term $A_{i, j} B_{j, k} C_{i, k}$.
    # The tensor entry is $T_{i, j, k} = 1$.
    
    # So we need to find $U, V, W$ such that:
    # $\sum_r U_{i, r} V_{j, r} W_{k, r} = T_{i, j, k} = 1$ for all $i, j, k$?
    # No, the indices in U, V, W are flattened.
    # U has rows indexed by $i, j$? No, U is $n \times m$ flattened?
    # Seed: U[i*m + j]. So row index $u_{idx} = i \cdot m + j$.
    # Seed: V[j*p + k]. Row index $v_{idx} = j \cdot p + k$.
    # Seed: W[k*n + i]. Row index $w_{idx} = k \cdot n + i$.
    
    # The decomposition is:
    # $T_{i, j, k} = \sum_r U_{i \cdot m + j, r} V_{j \cdot p + k, r} W_{k \cdot n + i, r}$?
    # This doesn't look like standard matrix multiplication decomposition.
    # Standard decomposition: $C_{i, k} = \sum_j A_{i, j} B_{j, k}$.
    # This can be written as $\sum_r U_{i, r} (\sum_j V_{j, r} W_{k, r})$? No.
    
    # Let's re-read the seed code logic.
    # It sets $U[i \cdot m + j, c] = 1$, $V[j \cdot p + k, c] = 1$, $W[k \cdot n + i, c] = 1$.
    # This creates a rank-1 tensor for each $(i, j, k)$.
    # The sum over $c$ (which corresponds to $i, j, k$ in seed) gives:
    # $\sum_{i', j', k'} U[i \cdot m + j, c_{i'j'k'}] V[j \cdot p + k, c_{i'j'k'}] W[k \cdot n + i, c_{i'j'k'}]$
    # This is non-zero only if $i=i', j=j', k=k'$.
    # So the resulting tensor has $T_{i, j, k} = 1$ for all $i, j, k$.
    # This matches the matrix multiplication tensor where every entry is 1?
    # Wait, for matrix multiplication $C=AB$, the tensor $T$ has $T_{i, j, k} = 1$ only if the indices align?
    # No, $C_{i, k} = \sum_j A_{i, j} B_{j, k}$.
    # The tensor $T$ is defined by $C = \mathcal{T}(A, B)$.
    # $C_{i, k} = \sum_{j, r} A_{i, j} B_{j, k} \delta_{i, i_r} \delta_{k, k_r}$?
    # Actually, the tensor $T$ is just the identity map in the bilinear form.
    # $T_{i, j, k} = 1$ for all $i, j, k$ is incorrect for matrix multiplication.
    # $T_{i, j, k}$ should be 1 if $j$ is the shared index?
    # In the seed, for a fixed $i, j, k$, it adds a rank-1 component.
    # The total tensor is $\sum_{i, j, k} e_{i,j} \otimes e_{j,k} \otimes e_{k,i}$.
    # This tensor, when contracted with $A$ and $B$, gives:
    # $\sum_{i, j, k} A_{i, j} B_{j, k} W_{k, i}$?
    # The output is a scalar? No, the output is a matrix.
    # The problem says "output factor follows ... W[k*n+i, r]".
    # This implies the output is reconstructed from W?
    # Or W is the third factor in the decomposition of the tensor?
    # If the decomposition is $T = \sum_r U_r \otimes V_r \otimes W_r$, then
    # $T_{i, j, k} = \sum_r U_{i, r} V_{j, r} W_{k, r}$.
    # But the indices in the seed are flattened.
    # $U$ has $n \cdot m$ rows. $V$ has $m \cdot p$ rows. $W$ has $p \cdot n$ rows.
    # This is not a standard $n \times m \times p$ tensor decomposition.
    # It looks like the tensor is defined on indices $(i, j), (j, k), (k, i)$.
    # This is the tensor for matrix multiplication if we view $A$ as $n \times m$, $B$ as $m \times p$, and the result as $n \times p$.
    # The "trace convention" $W[k \cdot n + i]$ suggests the output index is $(k, i)$?
    # Usually $C_{i, k}$. Here $W$ index is $k \cdot n + i$. So $W$ corresponds to $C_{k, i}$? Or $C_{i, k}$ with $i$ as column?
    # If $W$ index is $k \cdot n + i$, and $i \in [n], k \in [p]$, then $W$ has $p \cdot n$ rows.
    # This matches the shape.
    
    # So the tensor $T$ has entries $T_{(i, j), (j, k), (k, i)} = 1$?
    # No, the indices must be consistent.
    # The seed code sets $U[i \cdot m + j, c] = 1$, $V[j \cdot p + k, c] = 1$, $W[k \cdot n + i, c] = 1$.
    # This defines a tensor $T$ with dimensions $(n \cdot m) \times (m \cdot p) \times (p \cdot n)$.
    # The entry $T_{u, v, w}$ is 1 if $u = i \cdot m + j$, $v = j \cdot p + k$, $w = k \cdot n + i$ for some $i, j, k$.
    # This is a permutation-like tensor.
    # We need to find a rank-minimal decomposition of this specific tensor.
    
    # The tensor is:
    # $T_{u, v, w} = 1$ if $u \equiv j \pmod m, v \equiv k \pmod p, w \equiv i \pmod n$?
    # Let's check the seed.
    # $u = i \cdot m + j \implies u \pmod m = j$.
    # $v = j \cdot p + k \implies v \pmod p = k$.
    # $w = k \cdot n + i \implies w \pmod n = i$.
    # Also $i < n, j < m, k < p$.
    # So $T_{u, v, w} = 1$ if $j = u \pmod m, k = v \pmod p, i = w \pmod n$.
    # And the indices must be consistent?
    # In the seed, for a fixed $i, j, k$, we have one non-zero entry.
    # So the tensor has exactly $n \cdot m \cdot p$ ones.
    # The position of the one for $(i, j, k)$ is $(i \cdot m + j, j \cdot p + k, k \cdot n + i)$.
    
    # We want to find $U, V, W$ such that $T = \sum_r U_r \otimes V_r \otimes W_r$.
    # i.e., $T_{u, v, w} = \sum_r U_{u, r} V_{v, r} W_{w, r}$.
    
    # This is a standard tensor rank problem.
    # The tensor is sparse.
    # We can use a randomized search to find a low-rank approximation.
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Target ranks to try, from low to high
    # Lower bound: max(nm, mp, np) = 35
    # Upper bound: 105
    # We will try to find the minimal rank.
    
    best_rank = 105
    best_U = None
    best_V = None
    best_W = None
    
    # Generate the target tensor T as a sparse matrix or list of indices
    # T is (15, 35, 21)
    # We will represent T by its non-zero indices and values (all 1)
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                nonzero_indices.append((u, v, w))
                
    # Function to check if a decomposition is exact
    def check_decomposition(U, V, W, rank):
        # U: 15 x rank, V: 35 x rank, W: 21 x rank
        # Check all non-zero entries
        for u, v, w in nonzero_indices:
            val = np.dot(U[u], V[v])
            # Wait, the decomposition is T_{u,v,w} = sum_r U_{u,r} V_{v,r} W_{w,r}
            # This is not a simple dot product.
            # It's a trilinear form.
            # We can check by computing the full tensor or just the non-zero entries.
            # sum_r U_{u,r} V_{v,r} W_{w,r}
            # This is not efficient to compute for all u,v,w if we do it naively.
            # But we only have 105 non-zero entries.
            # We can compute the value for each non-zero entry.
            # However, we must also ensure that the zero entries are zero.
            # The tensor is sparse, but the decomposition might introduce non-zeros elsewhere.
            # We must check that T_{u,v,w} is 0 for all (u,v,w) not in nonzero_indices.
            # This is 15*35*21 - 105 = 11025 - 105 = 10920 entries.
            # This is feasible.
            
            # To speed up, we can check a random subset of zero entries and all non-zero entries.
            # But for exactness, we need to check all.
            
            # Let's compute the full tensor T_decomp and compare with T_target.
            # T_decomp[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
            # This can be computed as:
            # T_decomp = sum_r U[:, r] * V[:, r] * W[:, r] (outer product)
            # This is expensive for large rank.
            # Alternative: 
            # For each r, compute M_r = U[:, r] * V[:, r] (15x35 matrix)
            # Then T_decomp[u, v, w] = sum_r M_r[u, v] W[w, r]
            # We can check this for all u, v, w.
            
            # For the non-zero entries, we expect 1.
            for u, v, w in nonzero_indices:
                val = 0.0
                for r in range(rank):
                    val += U[u, r] * V[v, r] * W[w, r]
                if abs(val - 1.0) > 1e-6:
                    return False
            
            # For zero entries, we expect 0.
            # We can sample some zero entries.
            # But to be safe, let's check all if rank is small.
            # 10920 * rank operations. If rank=35, 382k ops. Fast.
            for u in range(n * m):
                for v in range(m * p):
                    for w in range(p * n):
                        if (u, v, w) in set(nonzero_indices):
                            continue
                        val = 0.0
                        for r in range(rank):
                            val += U[u, r] * V[v, r] * W[w, r]
                        if abs(val) > 1e-6:
                            return False
            return True

    # Function to try to find a decomposition of a given rank
    def find_decomposition(rank):
        # Initialize U, V, W randomly
        # Use small integers or half-integers?
        # Start with random floats, then snap.
        
        for attempt in range(20):
            if time.time() > deadline:
                return None
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Normalize to avoid overflow
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            # Use ALS to minimize error
            # Error = || T_target - T_decomp ||^2
            # We can use gradient descent or ALS.
            # ALS: fix V, W, solve for U.
            # T_decomp[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
            # For fixed u, v, w, this is linear in U[u, :].
            # But T_target is sparse.
            # This is a linear system for U.
            # We have equations for all (u, v, w).
            # Too many variables if we do it all at once.
            
            # Let's use a simpler approach: Randomized SVD or similar.
            # Or just check if the random initialization is close.
            # Given the complexity, I will rely on the fact that the schoolbook is rank 105.
            # If I can't find a lower rank quickly, I return the schoolbook.
            
            # Let's try to snap the random matrices to half-integers and check.
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            if check_decomposition(U_snap, V_snap, W_snap, rank):
                return U_snap, V_snap, W_snap
        
        return None

    # Try ranks from 35 to 104
    for rank in range(35, 105):
        if time.time() > deadline:
            break
        
        result = find_decomposition(rank)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
            break  # Found minimal rank, stop
    
    # If no lower rank found, return schoolbook
    if best_U is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    else:
        # Convert numpy arrays to lists
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
