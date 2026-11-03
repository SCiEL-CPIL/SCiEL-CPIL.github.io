import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Target tensor T[i, j, k] = 1 if j == k else 0
    # Note: The tensor for matrix multiplication A (n x m) * B (m x p) -> C (n x p)
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T maps (A, B) to C.
    # T[i, j, k] is the coefficient of A[i, j] B[j, k] in C[i, k].
    # So T[i, j, k] = 1 if k is the output index? No.
    # The standard tensor definition for matrix multiplication:
    # T_{i, j, k} = \delta_{j, k}? No.
    # Let's check the indices.
    # Input 1: A (n x m). Indices i (0..n-1), j (0..m-1).
    # Input 2: B (m x p). Indices j (0..m-1), k (0..p-1).
    # Output: C (n x p). Indices i (0..n-1), k (0..p-1).
    # C[i, k] = \sum_{j=0}^{m-1} A[i, j] B[j, k].
    # So the tensor T has indices (i, j, k) and value 1 if it contributes to C[i,k].
    # It contributes if the j in A matches the j in B.
    # So T[i, j, k] = 1 for all i, j, k? No.
    # A[i, j] and B[j, k] are multiplied.
    # So the term is A[i, j] B[j, k] 1.
    # So T[i, j, k] = 1?
    # Wait, the tensor is a 3-tensor.
    # T \in R^{n \times m \times p}.
    # T[i, j, k] = 1 if the pair (j, j) is connected?
    # Actually, the matrix multiplication tensor is defined as:
    # T[i, j, k] = 1 if j == j (always true for the summation index)?
    # No, the index j is summed over.
    # The tensor entries are 1 for all i, j, k?
    # Let's check the rank-1 decomposition.
    # Rank 1 term: u_i v_j w_k.
    # Sum: \sum_c u_{i,c} v_{j,c} w_{k,c}.
    # We want this to equal \delta_{j, j'}? No.
    # We want \sum_c u_{i,c} v_{j,c} w_{k,c} = \begin{cases} 1 & \text{if } j \text{ is the shared index?} \end{cases}
    # No, the output is C[i, k].
    # C[i, k] = \sum_{j} A[i, j] B[j, k].
    # So the coefficient of A[i, j] B[j, k] in C[i, k] is 1.
    # So the tensor T has entries T[i, j, k] = 1?
    # If T[i, j, k] = 1 for all i, j, k, then the rank is 1?
    # No, that's not right.
    # The tensor is T[i, j, k] = \delta_{j, j} ? No.
    
    # Let's look at the standard definition.
    # The matrix multiplication tensor $M_{n, m, p}$ has entries $T_{i, j, k} = 1$ if $j=k$? No.
    # The indices are $i \in \{0..n-1\}$, $j \in \{0..m-1\}$, $k \in \{0..p-1\}$.
    # The product is $\sum_{j} A_{i, j} B_{j, k}$.
    # So the term $A_{i, j} B_{j, k}$ appears in $C_{i, k}$.
    # So the tensor entry $T_{i, j, k}$ is 1?
    # If we fix $i, k$, the sum is over $j$.
    # So $T_{i, j, k} = 1$ for all $i, j, k$?
    # If so, the tensor is all ones. Rank 1.
    # That is incorrect. Matrix multiplication rank is not 1.
    
    # Where is the error?
    # The tensor maps $A \otimes B \to C$.
    # $C_{i, k} = \sum_{j} T_{i, j, k} A_{i, j} B_{j, k}$?
    # No, the index $j$ in $A$ must match the index $j$ in $B$.
    # So $A$ has index $j$, $B$ has index $j$.
    # The tensor index $j$ is the *same* index for both inputs.
    # So $T_{i, j, k}$ is non-zero only if the $j$ of A and $j$ of B are the same?
    # Yes, that's inherent in the definition.
    # So $T_{i, j, k} = 1$ for all $i, j, k$?
    # Let's check $2 \times 2$.
    # $C_{00} = A_{00}B_{00} + A_{01}B_{10}$.
    # $C_{01} = A_{00}B_{01} + A_{01}B_{11}$.
    # $C_{10} = A_{10}B_{00} + A_{11}B_{10}$.
    # $C_{11} = A_{10}B_{01} + A_{11}B_{11}$.
    
    # The tensor $T$ has indices $i, j, k$.
    # $T_{0, 0, 0} = 1$ (contributes to $C_{00}$ via $A_{00}B_{00}$)
    # $T_{0, 1, 0} = 1$ (contributes to $C_{00}$ via $A_{01}B_{10}$)
    # $T_{0, 0, 1} = 1$ (contributes to $C_{01}$ via $A_{00}B_{01}$)
    # $T_{0, 1, 1} = 1$ (contributes to $C_{01}$ via $A_{01}B_{11}$)
    # And so on.
    # So $T_{i, j, k} = 1$ for ALL $i, j, k$?
    # If $T$ is all ones, its rank is 1.
    # But matrix multiplication rank is 8 (schoolbook) or 7 (Strassen).
    # Contradiction.
    
    # Ah, the tensor is not $T_{i, j, k}$.
    # The tensor is defined on the space $V_A \otimes V_B \to V_C$.
    # The basis vectors for $V_A$ are $e_{i, j}$.
    # The basis vectors for $V_B$ are $e_{j, k}$.
    # The basis vectors for $V_C$ are $e_{i, k}$.
    # The tensor $T$ is a linear map.
    # $T(e_{i, j} \otimes e_{j', k}) = \delta_{j, j'} e_{i, k}$.
    # So the tensor entries are $T_{(i, j), (j', k), (i', k')} = \delta_{j, j'} \delta_{i, i'} \delta_{k, k'}$.
    # In the flattened index notation:
    # $I = i \cdot m + j$
    # $J = j' \cdot p + k$
    # $K = i' \cdot p + k'$
    # $T[I, J, K] = 1$ if $j = j'$ and $i = i'$ and $k = k'$.
    
    # So, for a fixed term in the sum $\sum_c U_c \otimes V_c \otimes W_c$:
    # $U$ has size $n \cdot m$.
    # $V$ has size $m \cdot p$.
    # $W$ has size $p \cdot n$ (or $n \cdot p$).
    
    # The condition is:
    # $\sum_c U[I, c] V[J, c] W[K, c] = T[I, J, K]$.
    # Where $T[I, J, K] = 1$ if $j(I) = j(J)$ and $i(I) = i(K)$ and $k(J) = k(K)$, else 0.
    # Note: $I = i \cdot m + j$. $j(I) = I \pmod m$. $i(I) = I // m$.
    # $J = j' \cdot p + k$. $j(J) = J // p$. $k(J) = J \pmod p$.
    # $K = i' \cdot p + k'$. $i(K) = K // p$. $k(K) = K \pmod p$.
    
    # So $T[I, J, K] = 1$ if $I \pmod m == J // p$ and $I // m == K // p$ and $J \pmod p == K \pmod p$.
    
    # Let's build the target tensor T.
    T = np.zeros((n * m, m * p, p * n))
    for I in range(n * m):
        i = I // m
        j = I % m
        for J in range(m * p):
            jp = J // p
            k = J % p
            if j != jp:
                continue
            for K in range(p * n):
                ip = K // n
                kp = K % n
                if i != ip or k != kp:
                    continue
                T[I, J, K] = 1.0
                
    def verify(U, V, W):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Compute sum_c U[:, c] V[:, c] W[:, c]
        # Reshape to tensor
        # T_calc[I, J, K] = sum_c U[I, c] V[J, c] W[K, c]
        # This is the outer product sum.
        # We can compute this as:
        # Temp = np.einsum('ic,jc,kc->ijk', U, V, W)
        # Compare Temp with T
        Temp = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(Temp, T, atol=1e-9)

    def als_decompose(rank):
        # Random initialization
        np.random.seed(42)
        U = np.random.randn(n * m, rank)
        V = np.random.randn(m * p, rank)
        W = np.random.randn(p * n, rank)
        
        # Normalize to prevent divergence
        U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-9
        V /= np.linalg.norm(V, axis=0, keepdims=True) + 1e-9
        W /= np.linalg.norm(W, axis=0, keepdims=True) + 1e-9
        
        for _ in range(100):
            if time.time() > deadline:
                return None
            
            # Solve for U: T_(1) = U (W odot V)^T
            # T_(1) is (nm, mp*pn)
            # M_VW is (mp*pn, rank) where M_VW[J*pn+K, c] = V[J, c] W[K, c]
            M_VW = np.einsum('jc,kc->jkc', V, W).reshape(m * p, p * n, rank).transpose(1, 0, 2).reshape(p * n * m * p, rank)
            # Wait, T_(1) indices: I (nm), (J, K) (mp, pn).
            # T[I, J, K] = sum_c U[I, c] (V[J, c] W[K, c])
            # So T_(1) = U @ M_VW.T
            # M_VW has shape (mp*pn, rank).
            # M_VW[idx, c] = V[J, c] W[K, c]
            
            T_1 = T.reshape(n * m, m * p * p * n)
            # Solve U = T_1 @ M_VW.T @ inv(M_VW @ M_VW.T)
            try:
                U_new, _, _, _ = np.linalg.lstsq(M_VW.T, T_1.T, rcond=None)
                U = U_new.T
            except:
                pass
                
            # Solve for V: T_(2) = V (U odot W)^T
            # T[J, I, K] = sum_c V[J, c] (U[I, c] W[K, c])
            M_UW = np.einsum('ic,kc->ikc', U, W).reshape(n * m, p * n, rank).transpose(1, 0, 2).reshape(p * n * n * m, rank)
            T_2 = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
            try:
                V_new, _, _, _ = np.linalg.lstsq(M_UW.T, T_2.T, rcond=None)
                V = V_new.T
            except:
                pass
                
            # Solve for W: T_(3) = W (U odot V)^T
            # T[K, I, J] = sum_c W[K, c] (U[I, c] V[J, c])
            M_UV = np.einsum('ic,jc->ijc', U, V).reshape(n * m, m * p, rank).transpose(2, 0, 1).reshape(rank, n * m * m * p)
            # T_(3) is (pn, nm*mp)
            T_3 = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
            try:
                W_new, _, _, _ = np.linalg.lstsq(M_UV, T_3, rcond=None)
                W = W_new
            except:
                pass
                
        if verify(U, V, W):
            # Try to round to half-integers
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if verify(U_r, V_r, W_r):
                return U_r, V_r, W_r
        return None

    # Try ranks from 50 to 59
    for r in range(50, 60):
        if time.time() > deadline - 10:
            break
        result = als_decompose(r)
        if result:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: Schoolbook
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
# EVOLVE-BLOCK-END
