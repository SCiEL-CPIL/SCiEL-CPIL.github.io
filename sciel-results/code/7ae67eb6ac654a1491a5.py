"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import sys

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication.
    Dimensions: A (n x m), B (m x p), C (n x p)
    Rank R = n * m * p
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists of lists as per the seed program interface
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    # Quick self-test
    start_time = time.time()
    res = construct()
    end_time = time.time()
    
    # Verify correctness
    n, m, p = DIMENSIONS
    U = res['U']
    V = res['V']
    W = res['W']
    R = len(U[0])
    
    # Check a few random entries
    import random
    random.seed(42)
    for _ in range(10):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        
        # Calculate sum U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        total = 0
        for r in range(R):
            total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
        
        # For schoolbook, this should be 1 if (i,j,k) is a valid index triplet 
        # and 0 otherwise? 
        # Wait, the tensor T_ijk is 1 if the indices correspond to a multiplication term.
        # In the standard tensor definition for matmul:
        # T_ijk = 1 if the term A_ij * B_jk contributes to C_ik.
        # Actually, the tensor T is defined such that C_ik = sum_j T_ijk A_ij B_jk.
        # In the decomposition T_ijk = sum_r U_ir V_jr W_kr.
        # The seed program sets U, V, W such that for each (i,j,k) there is a rank r where
        # U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
        # This implies T_ijk = 1 for all i,j,k?
        # No, the mapping of indices in U, V, W matters.
        
        # Let's re-read the index mapping in the seed:
        # U[i*m+j, r]
        # V[j*p+k, r]
        # W[k*n+i, r]
        
        # The tensor entry T_ijk is the sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
        # In the schoolbook decomposition, for a specific (i,j,k), there is exactly one r (let's call it r_ijk)
        # such that U[i*m+j, r_ijk]=1, V[j*p+k, r_ijk]=1, W[k*n+i, r_ijk]=1.
        # For any other r', at least one of the terms is 0.
        # Thus T_ijk = 1 for all i in [0,n), j in [0,m), k in [0,p).
        
        # Is this the correct tensor for matrix multiplication?
        # C_ik = sum_j A_ij B_jk.
        # This can be written as C_ik = sum_j sum_l sum_k' (A_il B_jk) * delta_{l,j} delta_{k',k} ... no.
        # The tensor T for matmul is usually defined as T_{ijk} = 1 if the indices align for multiplication.
        # Specifically, the multiplication tensor is a sum of basis tensors.
        # T = sum_{i,j,k} e_i \otimes e_j \otimes e_k ? No.
        
        # Let's check the standard definition.
        # The multiplication tensor $M_{n,m,p}$ is the tensor $T \in \mathbb{F}^{n \times m \times p}$
        # such that $T_{ijk} = 1$ if the indices correspond to a valid multiplication term?
        # Actually, $C = A B \implies C_{ik} = \sum_{j} A_{ij} B_{jk}$.
        # This is a multilinear map. The tensor associated with this map is $T$ where
        # $T(e_i \otimes e_j \otimes e_k) = \delta_{?}$?
        
        # Let's look at the factorization form:
        # $C_{ik} = \sum_r (\sum_i U_{ir} A_{?}) ...$
        # The standard form is $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
        # We can rewrite this as $C_{ik} = \sum_{i,j,k} T_{ijk} A_{ij} B_{jk} C_{ik}$? No.
        
        # The tensor decomposition defines the bilinear complexity.
        # $C_{ik} = \sum_{r=1}^R \alpha_r ( \sum_i U_{ir} A_{?} ) ( \sum_j V_{jr} B_{?} )$?
        
        # Let's look at the indices in the seed again.
        # $U$ has size $n \times m$. Index $i \cdot m + j$.
        # $V$ has size $m \times p$. Index $j \cdot p + k$.
        # $W$ has size $p \times n$. Index $k \cdot n + i$.
        
        # The reconstruction formula is typically:
        # $C_{ik} = \sum_{j} \sum_{r} U_{i \cdot m + j, r} V_{j \cdot p + k, r} W_{k \cdot n + i, r} A_{?} B_{?}$?
        
        # Actually, the problem statement says:
        # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) ... that exactly decompose the matrix multiplication tensor"
        
        # The "matrix multiplication tensor" $T_{ijk}$ for $n \times m \times p$ is defined such that
        # $T_{ijk} = 1$ if the term $A_{ij} B_{jk}$ contributes to $C_{ik}$.
        # Wait, $A_{ij}$ and $B_{jk}$ contribute to $C_{ik}$.
        # So for fixed $i, k$, we sum over $j$.
        # The tensor is $T \in \mathbb{R}^{n \times m \times p}$?
        # No, the tensor indices usually correspond to the indices of the matrices involved.
        # $T_{i, j, k}$?
        
        # If $U$ is indexed by $(i,j)$, $V$ by $(j,k)$, $W$ by $(k,i)$,
        # then the term in the sum for rank $r$ is $U_{(i,j), r} V_{(j,k), r} W_{(k,i), r}$.
        # This product is non-zero only if there is a rank $r$ connecting these specific indices.
        
        # In the schoolbook case:
        # For each triplet $(i,j,k)$, we have a rank $r_{ijk}$.
        # $U_{(i,j), r_{ijk}} = 1$.
        # $V_{(j,k), r_{ijk}} = 1$.
        # $W_{(k,i), r_{ijk}} = 1$.
        
        # So the sum over $r$ for a fixed $(i,j,k)$ is 1.
        # For any other combination of indices in the factors, is it 0?
        # Yes, because each rank $r$ is dedicated to exactly one $(i,j,k)$ triplet.
        
        # So the tensor $T$ defined by this decomposition has $T_{ijk} = 1$ for all $i,j,k$.
        # Is this the correct tensor for matrix multiplication?
        
        # Let's verify the reconstruction of $C_{ik}$.
        # $C_{ik} = \sum_{j} A_{ij} B_{jk}$.
        # Using the decomposition:
        # $C_{ik} = \sum_{j} \sum_{r} U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} A_{?} B_{?}$?
        
        # The standard identity is:
        # $C_{ik} = \sum_{r} ( \sum_{i,j} U_{(i,j), r} A_{ij} ) ( \sum_{j,k} V_{(j,k), r} B_{jk} ) W_{(k,i), r}$?
        # No, the indices don't match up for a simple product of sums.
        
        # The correct interpretation of the tensor decomposition for matmul is:
        # $C_{ik} = \sum_{j} A_{ij} B_{jk}$
        # This can be viewed as a trilinear form if we introduce dummy variables, but here we have factors.
        
        # The factors $U, V, W$ allow us to compute $C$ via:
        # $C_{ik} = \sum_{r} W_{(k,i), r} ( \sum_{j} U_{(i,j), r} A_{ij} ) ( \sum_{j} V_{(j,k), r} B_{jk} )$?
        # This would require $U$ and $V$ to sum over $j$, but $U$ depends on $i,j$ and $V$ on $j,k$.
        # If we compute $S^U_{i,r} = \sum_j U_{(i,j), r} A_{ij}$ and $S^V_{k,r} = \sum_j V_{(j,k), r} B_{jk}$,
        # then $C_{ik} = \sum_r W_{(k,i), r} S^U_{i,r} S^V_{k,r}$.
        
        # Let's test this with the schoolbook decomposition.
        # $U_{(i,j), r} = 1$ if $r = r_{ijk}$.
        # $S^U_{i,r} = \sum_j A_{ij} \delta_{r, r_{ijk}}$.
        # For a fixed $r$, $r$ corresponds to a specific $(i_0, j_0, k_0)$.
        # So $S^U_{i,r}$ is non-zero only if $i = i_0$.
        # If $i = i_0$, $S^U_{i_0, r} = A_{i_0, j_0}$.
        # Similarly, $S^V_{k,r}$ is non-zero only if $k = k_0$.
        # If $k = k_0$, $S^V_{k_0, r} = B_{j_0, k_0}$.
        # And $W_{(k,i), r} = 1$ only if $k=k_0$ and $i=i_0$.
        
        # So for the term with rank $r$ (corresponding to $i_0, j_0, k_0$):
        # The contribution to $C_{ik}$ is non-zero only if $i=i_0$ and $k=k_0$.
        # In that case, the value is $1 \cdot A_{i_0, j_0} \cdot B_{j_0, k_0}$.
        
        # Summing over all $r$:
        # $C_{ik} = \sum_{r} W_{(k,i), r} S^U_{i,r} S^V_{k,r}$
        # $= \sum_{i_0, j_0, k_0} \delta_{i, i_0} \delta_{k, k_0} A_{i_0, j_0} B_{j_0, k_0}$
        # $= \sum_{j_0} A_{i, j_0} B_{j_0, k}$
        
        # This matches the definition of matrix multiplication!
        # So the schoolbook decomposition is correct.
        
        assert total == 1, f"Decomposition failed for {i},{j},{k}, got {total}"
        
    print(f"Verification passed. Rank: {R}. Time: {end_time - start_time:.4f}s")
