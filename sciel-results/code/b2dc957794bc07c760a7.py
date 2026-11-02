"""Matrix multiplication 3,3,4 tensor decomposition."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the factor matrices U, V, W for the schoolbook tensor decomposition
    of 3x3 by 3x4 matrix multiplication.
    
    The decomposition is exact and uses integer coefficients (0s and 1s).
    Rank = 3 * 3 * 4 = 36.
    
    Indexing conventions:
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    The multiplication is C = A * B.
    A is 3x3 (indices i, j)
    B is 3x4 (indices j, k)
    C is 3x4 (indices i, k)
    
    Output factor W follows W[k*n + i, r] ? 
    The task says: "Output factor follows the official trace convention W[k*n+i,r]"
    Wait, the standard W index is usually C's index. 
    Let's check the standard tensor decomposition definition:
    T[i,j,k] = sum_r U[i,j,r] V[k,r] W[r] ? No.
    
    Standard form:
    C_{ik} = sum_{j} A_{ij} B_{jk}
    Tensor T_{ijk} such that T_{ijk} = 1 if ...?
    Usually, the multiplication tensor $T$ is defined such that:
    $A \\times B = \\sum_{r} U_r \\otimes V_r \\otimes W_r$?
    
    Let's look at the interface:
    U: (n*m, rank) -> U[i*m+j, r]
    V: (m*p, rank) -> V[j*p+k, r]
    W: (p*n, rank) -> W[k*n+i, r]
    
    The product of these vectors for a fixed r gives a rank-1 tensor.
    The sum over r of these rank-1 tensors must equal the multiplication tensor.
    The multiplication tensor $T$ has components $T_{i,j,k}$.
    In the standard definition, $T_{i,j,k} = 1$ if the term $A_{ij} B_{jk}$ contributes to $C_{ik}$?
    Actually, the bilinear map is:
    $C_{ik} = \\sum_{j} A_{ij} B_{jk}$.
    
    The tensor $T$ associated with this map has indices $(i, j, k)$?
    Or is it $( (i,j), j, k )$?
    
    The interface defines:
    U index: $i*m + j$ (flattened A index)
    V index: $j*p + k$ (flattened B index)
    W index: $k*n + i$ (flattened C index? No, C is $n \\times p$, so $i, k$. Flattened is usually $i*p+k$ or $k*n+i$.)
    
    The task states: "Output factor follows the official trace convention W[k*n+i,r]".
    This implies the W matrix's row index is $k*n + i$.
    This corresponds to $C_{ik}$ where the row index in the flattened W is determined by $k$ and $i$.
    Usually, if $C$ is $n \\times p$, index $(i,k)$.
    If we flatten row-major: $i*p + k$.
    If we flatten column-major: $k*n + i$.
    The prompt specifies $W[k*n+i, r]$. So the row index of W is $k*n + i$.
    
    The rank-1 term for the schoolbook algorithm corresponds to a specific $(i, j, k)$ triplet.
    For each triplet $(i, j, k)$:
    $A_{ij}$ is at index $i*m + j$ in U.
    $B_{jk}$ is at index $j*p + k$ in V.
    $C_{ik}$ is at index $k*n + i$ in W.
    
    So for the column $r$ corresponding to $(i,j,k)$:
    $U[i*m+j, r] = 1$
    $V[j*p+k, r] = 1$
    $W[k*n+i, r] = 1$
    All other entries in column $r$ are 0.
    
    Summing over all $i, j, k$:
    $\\sum_r U_{u,r} V_{v,r} W_{w,r}$
    This doesn't quite look like the standard contraction.
    Usually, the tensor decomposition is $T_{u,v,w} = \\sum_r U_{u,r} V_{v,r} W_{w,r}$?
    No, the standard definition of tensor rank for bilinear map $A \\times B \\to C$ is:
    We want to compute $C_{ik} = \\sum_j A_{ij} B_{jk}$.
    This can be written as $\\vec{C} = \\sum_r ( \\vec{A}^T U_r ) ( V_r^T \\vec{B} ) W_r^T$?
    
    Let's check the identity:
    If $U, V, W$ are the factors, the computed matrix $\\hat{C}$ has entries:
    $\\hat{C}_{ik} = \\sum_r ( \\sum_{u} A_u U_{u,r} ) ( \\sum_{v} B_v V_{v,r} ) W_{w,r}$ ?
    This depends on how the indices map.
    
    Let's assume the standard convention for these interfaces:
    The algorithm computes $C_{ik}$ as:
    $C_{ik} = \\sum_{r=1}^R ( \\sum_{j} A_{ij} U_{i*m+j, r} ) ( \\sum_{j} V_{j*p+k, r} B_{jk} ) W_{k*n+i, r}$ ?
    This would be weird if $W$ is just a scalar per r.
    
    Actually, usually the decomposition is:
    $C_{ik} = \\sum_r ( U_r^T \\vec{A} ) ( V_r^T \\vec{B} ) W_r$
    where $W_r$ is a scalar? No, $W$ is a matrix.
    
    Let's look at the shapes:
    U: $(nm, R)$
    V: $(mp, R)$
    W: $(pn, R)$
    
    The output is a tensor of size $nm \\times mp \\times pn$.
    The multiplication tensor $T$ has $T_{u,v,w} = 1$ if $u=(i,j), v=(j,k), w=(k,i)$?
    No, $u=(i,j)$, $v=(j,k)$, $w=(k,i)$ is not a tensor in 3 spaces.
    
    The standard decomposition is:
    $T_{u,v,w} = \\sum_r U_{u,r} V_{v,r} W_{w,r}$
    where $T$ is the multiplication tensor.
    The multiplication tensor $T$ has non-zero entries only when the indices are compatible.
    $u = i*m + j$
    $v = j*p + k$
    $w = k*n + i$
    
    So $T_{i*m+j, j*p+k, k*n+i} = 1$.
    All other entries are 0.
    
    If we set $U, V, W$ such that for each $(i,j,k)$, there is a rank-1 component:
    $U_{i*m+j, r} = 1$
    $V_{j*p+k, r} = 1$
    $W_{k*n+i, r} = 1$
    and 0 elsewhere for that $r$.
    
    Then $\\sum_r U_{u,r} V_{v,r} W_{w,r}$
    For a specific $(u,v,w)$:
    The sum is non-zero only if there exists an $r$ such that $U_{u,r}=1, V_{v,r}=1, W_{w,r}=1$.
    This happens if and only if $u, v, w$ come from the same $(i,j,k)$ triplet in the schoolbook assignment.
    i.e., $u = i*m+j$, $v = j*p+k$, $w = k*n+i$.
    In that case, the sum is 1.
    If the indices are not compatible (i.e., they don't form a valid term $A_{ij}B_{jk} \to C_{ik}$), no $r$ will have all three 1s simultaneously (because each $r$ is tied to a specific $i,j,k$).
    
    Thus, this construction exactly reproduces the multiplication tensor.
    Rank = 36.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
