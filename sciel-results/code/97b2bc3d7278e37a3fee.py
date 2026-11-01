import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    # Target Tensor T with T[i, j, k] = 1 if i==k? No.
    # T is the matrix multiplication tensor.
    # T[i, j, k] = 1 if the operation A[i, :] * B[:, k] contributes to C[i, k].
    # Actually, the tensor is defined such that C = A B implies C[i, k] = sum_j A[i, j] B[j, k].
    # So the tensor T has T[i, j, k] = 1 for all i, j, k? No.
    # The tensor is $T = \sum_{i,j,k} e_{i,0} \otimes e_{0,j} \otimes e_{k,1}$?
    # Standard definition: $T_{ijk} = 1$ if the index $j$ matches?
    # In the decomposition $T = \sum_{l} U_{:,l} \otimes V_{:,l} \otimes W_{:,l}$,
    # The product is $\sum_{l} U_{i,l} V_{j,l} W_{k,l} = \delta_{i,k} \delta_{j,j}$? No.
    # The matrix multiplication tensor $\mathcal{M}_{m,n,p}$ has entries $\mathcal{M}_{ijk} = 1$ if $i \in \{1..m\}, j \in \{1..n\}, k \in \{1..p\}$?
    # No, $\mathcal{M}_{ijk} = 1$ is not correct.
    # The tensor is $\sum_{i,j,k} e_i \otimes e_j \otimes e_k$? No.
    # The tensor is defined by $T(e_i, f_j, g_k) = \delta_{i,k}$? No.
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # This corresponds to the tensor $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$? No.
    # It corresponds to $T = \sum_{j} (\sum_i e_i \otimes e_j) \otimes (\sum_k e_k)$? No.
    # The tensor is $T = \sum_{i,j,k} \delta_{i,k} e_i \otimes e_j \otimes e_k$? No.
    # Let's use the property: $\langle T, A \otimes B \rangle = \text{trace}(A B)$? No, $\text{trace}(A B C^T)$?
    # For matrix multiplication $C = AB$, the tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$?
    # If we contract $T$ with $A$ and $B$, we get $C$.
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # This is equivalent to $T_{ijk} = 1$ for all $i,j,k$? No.
    # $T_{ijk} = 1$ if we consider the basis elements.
    # Actually, the tensor is $\sum_{j} (\sum_i e_i \otimes e_j) \otimes (\sum_k e_k)$?
    # Let's check the Strassen decomposition in the current code.
    # It constructs U, V, W such that $\sum_l U_{i,l} V_{j,l} W_{k,l} = \delta_{i,k}$? No.
    # In the current code, for a 2x2 block:
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # The tensor entries are $T_{ijk} = 1$ if $i,k$ are output indices and $j$ is the summation index?
    # Yes, $T_{ijk} = 1$ for all valid $i,j,k$ is WRONG.
    # $T_{ijk} = 1$ is the identity tensor? No.
    # The matrix multiplication tensor is $\mathcal{M} = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$ is the all-ones tensor.
    # The correct tensor is $\mathcal{M} = \sum_{j} (\sum_i e_i \otimes e_j) \otimes (\sum_k e_k)$? No.
    # $\mathcal{M} = \sum_{i,j,k} \delta_{i,k} e_i \otimes e_j \otimes e_k$? No.
    # Let's derive it.
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # This is a bilinear map.
    # The tensor is $T = \sum_{i,j,k} e_i \otimes e_j \otimes e_k$?
    # If $T_{ijk} = 1$, then $\sum_{i,j,k} A_{ij} B_{jk} T_{ijk} = \sum A B$. This is the trace of AB?
    # No, $\sum_{i,j} A_{ij} B_{ji} = \text{trace}(AB)$.
    # Here we want $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # This is a 3-linear form? No, it's a bilinear map from $A, B$ to $C$.
    # The tensor $T$ is such that $T(A, B) = C$.
    # In index notation: $C_{ik} = \sum_j T_{ijk} A_{ij} B_{jk}$?
    # If $T_{ijk} = 1$ for all $i,j,k$, then $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # Yes! The tensor for matrix multiplication is the all-ones tensor in the appropriate bases?
    # Wait. $A$ has indices $i,j$. $B$ has indices $j,k$.
    # $C$ has indices $i,k$.
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # This corresponds to the tensor $T_{ijk} = 1$ for all $i,j,k$?
    # Let's check dimensions.
    # $A \in \mathbb{R}^{n \times m}$. $B \in \mathbb{R}^{m \times p}$. $C \in \mathbb{R}^{n \times p}$.
    # $T \in \mathbb{R}^{n \times m \times p}$.
    # $T_{i,j,k} = 1$ for all $i,j,k$.
    # Then $\sum_{j} T_{i,j,k} A_{i,j} B_{j,k} = \sum_j A_{i,j} B_{j,k} = C_{i,k}$.
    # Yes, the tensor is the all-ones tensor.
    
    # So we want to decompose the all-ones tensor of size $2 \times 4 \times 6$ into rank $r$.
    # $T_{ijk} = 1 = \sum_{l=1}^r U_{i,l} V_{j,l} W_{k,l}$.
    
    # The current code constructs U, V, W for the all-ones tensor.
    # Let's verify the current code's logic.
    # It uses Strassen.
    # Strassen decomposes the $2 \times 2 \times 2$ all-ones tensor into rank 7.
    # This is correct.

    # So the problem is: Find the minimal rank of the all-ones tensor $1_{2 \times 4 \times 6}$.
    # This is the "rank of the matrix multiplication tensor".
    
    # Known values:
    # $R(1_{2 \times 2 \times 2}) = 7$.
    # $R(1_{2 \times 4 \times 6})$?
    # We have a rank 42 solution.
    # Is it minimal?
    # Lower bound: $R(1_{n \times m \times p}) \ge \max(n,m,p) \times \text{something}$?
    # A simple lower bound is $\frac{mn p}{\max(n,m,p)}$? No.
    # $R(1_{n \times n \times n}) \ge n^2$.
    # For $2 \times 4 \times 6$, $mn p = 48$.
    # Rank 42 is close to 48.
    
    # Let's try to find a rank 40 solution using ALS.
    
    deadline = time.time() + 140
    best_rank = 42
    best_factors = None
    
    # Generate the target tensor
    T = np.ones((n, m, p))
    
    # Try ranks 40, 41
    for r in [40, 41]:
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            np.random.seed(seed)
            
            # Initialize U, V, W
            U = np.random.randn(n, r)
            V = np.random.randn(m, r)
            W = np.random.randn(p, r)
            
            # ALS iterations
            for it in range(20):
                if time.time() > deadline - 5:
                    break
                    
                # Update U
                # For each i, solve for U[i, :]
                # T[i, :, :] = sum_l U[i, l] (V[:, l] o W[:, l])
                # Flatten V o W for each l
                # Let M_U be matrix of size (m*p) x r
                # M_U[:, l] = V[:, l] * W[:, l] (element-wise)
                # T[i, :, :].flatten() = U[i, :] @ M_U
                # Solve least squares
                M_U = np.zeros((m*p, r))
                for l in range(r):
                    M_U[:, l] = (V[:, l] * W[:, l]).flatten()
                T_i = T[i, :, :].flatten()
                # U[i, :] = argmin || T_i - M_U @ U_i ||
                U[i, :], residuals, rank, sv = np.linalg.lstsq(M_U, T_i, rcond=None)
                
                # Update V
                M_V = np.zeros((n*p, r))
                for l in range(r):
                    M_V[:, l] = (U[:, l] * W[:, l]).flatten()
                for j in range(m):
                    T_j = T[:, j, :].flatten()
                    V[j, :], _, _, _ = np.linalg.lstsq(M_V, T_j, rcond=None)
                    
                # Update W
                M_W = np.zeros((n*m, r))
                for l in range(r):
                    M_W[:, l] = (U[:, l] * V[:, l]).flatten()
                for k in range(p):
                    T_k = T[:, :, k].flatten()
                    W[k, :], _, _, _ = np.linalg.lstsq(M_W, T_k, rcond=None)
            
            # Check error
            # Compute reconstructed tensor
            T_recon = np.zeros((n, m, p))
            for l in range(r):
                T_recon += np.outer(U[:, l], V[:, l]) * W[:, l][:, np.newaxis, np.newaxis]
            
            err = np.linalg.norm(T_recon - T)
            # Check if it snaps to half-integers
            # Round U, V, W to nearest 0.5
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            T_recon_round = np.zeros((n, m, p))
            for l in range(r):
                T_recon_round += np.outer(U_round[:, l], V_round[:, l]) * W_round[:, l][:, np.newaxis, np.newaxis]
            
            err_round = np.linalg.norm(T_recon_round - T)
            
            if err_round < 1e-6:
                # Found a valid integer/half-integer solution
                best_rank = r
                # Convert to list of lists for output
                U_out = U_round.tolist()
                V_out = V_round.tolist()
                W_out = W_round.tolist()
                return {"U": U_out, "V": V_out, "W": W_out}
                
    # If no better rank found, return the rank 42 solution
    # Re-use the logic from the current program
    rank_total = 42
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]

    strassen_U = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {0: -1, 2: 1},
        {1: 1, 3: -1}
    ]
    strassen_V = [
        {0: 1, 3: 1},
        {0: 1},
        {1: 1, 3: -1},
        {2: 1, 0: -1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 3: 1}
    ]
    strassen_W = [
        {0: 1, 3: 1},
        {2: 1, 3: -1},
        {1: 1, 3: 1},
        {0: 1, 2: 1},
        {0: -1, 1: 1},
        {3: 1},
        {0: 1}
    ]

    col_offset = 0
    for j_block in range(3):
        for side in ['L', 'R']:
            if side == 'L':
                u_col_offset = 0
                v_row_offset = 0
            else:
                u_col_offset = 2
                v_row_offset = 2

            for prod_idx in range(7):
                col = col_offset + prod_idx
                
                for local_idx, sign in strassen_U[prod_idx].items():
                    if local_idx == 0:
                        u_idx = 0 * 4 + u_col_offset
                    elif local_idx == 1:
                        u_idx = 0 * 4 + u_col_offset + 1
                    elif local_idx == 2:
                        u_idx = 1 * 4 + u_col_offset
                    else:
                        u_idx = 1 * 4 + u_col_offset + 1
                    U[u_idx][col] += sign

                for local_idx, sign in strassen_V[prod_idx].items():
                    if local_idx == 0:
                        v_idx = (v_row_offset) * 6 + 2 * j_block
                    elif local_idx == 1:
                        v_idx = (v_row_offset) * 6 + 2 * j_block + 1
                    elif local_idx == 2:
                        v_idx = (v_row_offset + 1) * 6 + 2 * j_block
                    else:
                        v_idx = (v_row_offset + 1) * 6 + 2 * j_block + 1
                    V[v_idx][col] += sign

                for local_idx, sign in strassen_W[prod_idx].items():
                    if local_idx == 0:
                        w_idx = (2 * j_block) * 2 + 0
                    elif local_idx == 1:
                        w_idx = (2 * j_block + 1) * 2 + 0
                    elif local_idx == 2:
                        w_idx = (2 * j_block) * 2 + 1
                    else:
                        w_idx = (2 * j_block + 1) * 2 + 1
                    W[w_idx][col] += sign

            col_offset += 7

    return {"U": U, "V": V, "W": W}
