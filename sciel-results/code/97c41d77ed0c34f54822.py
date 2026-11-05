"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a dictionary containing the factor matrices U, V, W for the
    tensor decomposition of the matrix multiplication tensor for
    (n x m) * (m x p) matrices.
    
    The decomposition is of the form:
    C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    (Note: The standard identity is C[i,k] = sum_j A[i,j]B[j,k].
     The tensor T[i,j,k] = 1.
     The factorization T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] is not standard.
     Let's check the contract:
     U: (n*m, rank), index i*m+j
     V: (m*p, rank), index j*p+k
     W: (p*n, rank), index k*n+i
     
     The product sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] should equal the tensor entry.
     For the matrix multiplication tensor, the entry is 1 if we are computing the dot product?
     No, the tensor for matrix multiplication is T_{i,j,k} = 1 if we consider the operation (A,B) -> C where C_{ik} = sum_j A_{ij} B_{jk}.
     Actually, the tensor is usually defined such that <T, A \otimes B> = C.
     The standard decomposition of the matrix multiplication tensor M_{n,m,p} is:
     M_{n,m,p} = sum_{i,j,k} e_i \otimes e_j \otimes e_k^* ?
     
     Let's look at the seed code:
     U[i*m+j][column] = 1
     V[j*p+k][column] = 1
     W[k*n+i][column] = 1
     
     This corresponds to the term A[i,j] * B[j,k] * (something for C).
     Usually, the third factor maps to the output index.
     If the output is C[i,k], the third factor should involve i and k.
     The contract says W is (p*n, rank) with index k*n+i.
     This matches the output indices (k from B, i from A).
     
     So the decomposition is:
     sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta_{something}?
     Actually, for the matrix multiplication tensor, we want:
     C[i,k] = sum_j A[i,j] B[j,k]
     This can be written as sum_{j} (A[i,j] * 1 * B[j,k]).
     This doesn't look like a tensor decomposition of the form U*V*W directly unless we embed the identity.
     
     Let's re-read the standard definition of the matrix multiplication tensor rank.
     The tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes f_k$? No.
     The tensor is $T \in \mathbb{R}^{n \times m} \otimes \mathbb{R}^{m \times p} \otimes \mathbb{R}^{n \times p}$.
     $T(A, B) = A B$.
     $T = \sum_{i,j,k} A_{ij} B_{jk} (AB)_{ik}$.
     So the tensor components are $T_{ij, jk, ik} = 1$? No, the basis vectors are $e_i \otimes e_j$ for A, $e_j \otimes e_k$ for B, and $e_i \otimes e_k$ for C.
     So $T = \sum_{i,j,k} (e_i \otimes e_j) \otimes (e_j \otimes e_k) \otimes (e_i \otimes e_k)$.
     
     The factorization provided in the seed:
     U has indices $(i,j)$. V has indices $(j,k)$. W has indices $(k,i)$.
     Wait, the seed has $W[k*n+i]$. This is index $k$ then $i$.
     The term is $U_{ij} V_{jk} W_{ki}$.
     Summing over $r$: $\sum_r U_{ij,r} V_{jk,r} W_{ki,r}$.
     In the schoolbook case, for each $(i,j,k)$, we have a rank-1 term where $U_{ij}=1, V_{jk}=1, W_{ki}=1$ (and 0 elsewhere for that r).
     Then the sum over $r$ for a fixed $i,j,k$ is 1.
     Does this reproduce the tensor?
     The tensor component for inputs $(i,j)$ in A, $(j,k)$ in B, and output $(i,k)$ in C is 1.
     The factorization gives $\sum_r U_{ij,r} V_{jk,r} W_{ki,r}$.
     If we pick the $r$ corresponding to $(i,j,k)$, the product is $1 \cdot 1 \cdot 1 = 1$.
     Other $r$'s are 0.
     So yes, this is the correct schoolbook decomposition.
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
