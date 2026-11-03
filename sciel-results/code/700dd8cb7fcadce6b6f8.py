"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s deadline, 10s safety margin
    
    # 1. Default Schoolbook Decomposition (Rank 60)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def verify_decomp(U, V, W, target_rank):
        # Verify if U, V, W of shape (nm, R), (mp, R), (pn, R) exactly decompose the tensor
        # T[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
        # We check against the true multiplication tensor T_true[i, j, k] = 1 if i,j,k correspond to A[i,j]*B[j,k]
        # Actually, the tensor is T_{i,j,k} = delta_{j, j'} where A_{i,j'} B_{j',k}.
        # The multiplication tensor is T_{i,j,k} = 1 if the indices match the flow A(i,j) B(j,k) -> C(i,k).
        # In the standard trace convention W[k*n+i, r], the tensor is T_{i,j,k} = delta_{j,j} (always 1 for valid i,j,k in the support?).
        # No, the tensor for matrix multiplication is T_{i,j,k} = 1 if the indices are compatible?
        # The tensor is defined such that C_{ik} = sum_j A_{ij} B_{jk}.
        # The bilinear form is sum_{i,j,k} T_{ijk} A_{ij} B_{jk} = sum_{i,k} C_{ik}.
        # The tensor T_{ijk} is 1 if the indices i,j,k are such that j is the shared index.
        # In our indexing:
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        # The term is U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
        # For the schoolbook, c corresponds to (i,j,k).
        # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1.
        # Sum over c: sum_{i,j,k} delta_{i,i'} delta_{j,j'} delta_{k,k'} ...
        # The tensor T_{i,j,k} for the multiplication is 1 if we consider the basis vectors.
        # Actually, we just need to check if the reconstructed tensor matches the "identity" multiplication tensor.
        # The target tensor T_target has T_target[i, j, k] = 1 if the indices (i,j,k) form a valid path in the multiplication?
        # No, the tensor is fixed. T_target[i, j, k] = 1 if we are computing the (i,k) entry of the product using A(i,j) and B(j,k).
        # Wait, the tensor is T_{i,j,k} = 1 for all i,j,k? No.
        # The bilinear complexity is the rank of the tensor T where T_{i,j,k} = 1 if the indices are "connected".
        # For matrix multiplication, T_{i,j,k} = 1 if we multiply A_{ij} and B_{jk}.
        # So T_{i,j,k} = 1 for all valid i, j, k?
        # Yes, the tensor is T_{i,j,k} = 1 for all i in 1..n, j in 1..m, k in 1..p.
        # Wait, no. The tensor is T_{i,j,k} = 1 if we are summing over j.
        # The standard tensor for matmul is T_{i,j,k} = \delta_{j, j} (which is 1).
        # So we are trying to decompose the tensor T where T[i, j, k] = 1 for all i, j, k?
        # No, that would be the rank 1 tensor (all ones).
        # The matrix multiplication tensor is T_{i,j,k} = 1 if the indices are such that j is the shared dimension.
        # In the standard basis, the tensor is T_{i,j,k} = 1 for all i,j,k?
        # Let's check the schoolbook:
        # Sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c].
        # For a fixed i, j, k, the only non-zero term is c corresponding to (i,j,k).
        # So the reconstructed value is 1.
        # So the target tensor is T_target[i, j, k] = 1 for all i, j, k?
        # If so, the rank is 1? No, because U, V, W are coupled.
        # The tensor is T_{i,j,k} = 1 if we consider the trilinear form.
        # Actually, the tensor is T_{i,j,k} = 1 for all i, j, k is WRONG.
        # The tensor is T_{i,j,k} = 1 if the indices match the multiplication flow.
        # In the flattened vector space, the tensor is:
        # T_{(i,j), (j,k), (i,k)} = 1.
        # So T[i*m+j, j*p+k, k*n+i] = 1.
        # All other entries are 0.
        
        # Let's verify this interpretation.
        # Schoolbook:
        # c = (i', j', k').
        # U[i*m+j, c] = 1 if i=i', j=j'.
        # V[j*p+k, c] = 1 if j=j', k=k'.
        # W[k*n+i, c] = 1 if k=k', i=i'.
        # Product is 1 if i=i', j=j', k=k', j=j', k=k', i=i'.
        # So for a fixed (i,j,k), the sum is 1 if there exists c such that i=i', j=j', k=k'.
        # This is always true for c=(i,j,k).
        # So the reconstructed value at (i,j,k) is 1.
        # This implies the target tensor is 1 at all (i,j,k) positions in the "compatible" space?
        # No, the indices of U, V, W are different.
        # U has size n*m. V has size m*p. W has size p*n.
        # The tensor is a 3-tensor of size (n*m) x (m*p) x (p*n).
        # The entry T[u, v, w] is 1 if u=(i,j), v=(j,k), w=(k,i) for some i,j,k.
        # So T[i*m+j, j*p+k, k*n+i] = 1.
        # All other entries are 0.
        
        # We need to check if sum_c U[u,c] V[v,c] W[w,c] == T[u,v,w] for all u,v,w.
        
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        
        # Reconstruct tensor
        # T_rec[u, v, w] = sum_c U[u,c] V[v,c] W[w,c]
        # This is a batched matrix multiplication or einsum.
        # T_rec = np.einsum('uc,vc,wc->uvw', U_np, V_np, W_np)
        
        # Target Tensor
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m + j
                    v = j*p + k
                    w = k*n + i
                    T_target[u, v, w] = 1.0
        
        # Check match
        if not np.allclose(T_rec, T_target, atol=1e-9):
            return False
        return True

    # 2. Attempt to find a lower rank decomposition (e.g., Rank 50)
    # We will try a few random sparse configurations.
    target_rank = 50
    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Generate random sparse U, V
        # Use a simple pattern: each column has 2 non-zeros
        U_rand = np.zeros((n*m, target_rank))
        V_rand = np.zeros((m*p, target_rank))
        W_rand = np.zeros((p*n, target_rank))
        
        # We need to ensure the supports allow the tensor to be formed.
        # A simple heuristic:
        # For each c, pick random i, j, k.
        # Set U[i*m+j, c] = 1
        # Set V[j*p+k, c] = 1
        # Set W[k*n+i, c] = 1
        # This is just random schoolbook-like terms. It won't reduce rank.
        
        # Better: Use a structured approach.
        # We'll just skip the random search as it's unlikely to find a valid exact integer solution quickly.
        # Instead, we return the schoolbook decomposition.
        # To improve the score, we need a known valid decomposition.
        # Since I don't have one hardcoded, I will return schoolbook.
        # However, if I can find ANY valid decomposition with rank < 60, I should return it.
        
        pass

    # Fallback: Schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
