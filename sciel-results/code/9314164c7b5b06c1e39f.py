import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    U, V, W = build_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

def build_decomposition(n, m, p):
    # Returns U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # Base cases
    if n == 1 and m == 1 and p == 1:
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W
    
    if n == 1 and m == 1:
        # Outer product with vector B (1xp)
        # C = A * B. A is scalar, B is vector.
        # C_k = A * B_k
        # Rank p
        U = [[1] * p]
        V = [[1 if j == k else 0 for k in range(p)] for j in range(p)] # Wait, V is m x p = 1 x p.
        # V[j, k] = delta_{j, k}? No.
        # Term k: u=1, v=e_k, w=e_k?
        # C_k = 1 * V_{0,k} * W_{k,0}.
        # We want C_k = A * B_k.
        # So V_{0,k} = B_k? No, V is the factor for B.
        # In tensor decomposition, U depends on A, V on B, W on C.
        # Term k: u_k = A_0, v_k = B_k, w_k = E_k (basis for C).
        # So U = [A_0, A_0, ...]
        # V = [B_0, B_1, ...] (row vector)
        # W = [E_0, E_1, ...] (columns are basis vectors)
        # Shape U: 1 x p. V: 1 x p. W: p x 1.
        U = [[1] * p]
        V = [[1] * p] # This is wrong. V should extract B_k.
        # If V is 1 x p, and we multiply U (1 x p) and V (1 x p)?
        # No, the sum is over rank.
        # U[i, r], V[j, r], W[k, i].
        # C_{i,k} = sum_r U[i,r] V[j,r] W[k,i].
        # For n=1, m=1, p=p.
        # C_k = sum_r U[0,r] V[0,r] W[k,0].
        # We want C_k = A_0 B_0? No, C_k = A_0 B_k?
        # B is 1 x p. So B_j is just B_k.
        # So we need V[0,r] to pick out B_r?
        # If we set V[0,r] = 1 for r=k, 0 else?
        # Then V is identity?
        # If V is 1 x p, it's a row vector.
        # We need p terms.
        # Term k: U[0,k]=1, V[0,k]=1, W[k,0]=1.
        # Then C_k = 1 * 1 * 1 = 1.
        # This computes C = A * (1, 1, ...).
        # We want C = A * B.
        # So we must encode B in V?
        # No, V is the factor matrix. It multiplies the input vector B.
        # The decomposition is C = sum_r (U[:,r] * A) * (V[:,r] * B) * W[:,r]?
        # No.
        # Standard form: C_{ik} = sum_r U_{ir} V_{jr} W_{kr}.
        # Here indices are i (n), j (m), k (p).
        # U is n x rank. V is m x rank. W is p x n?
        # Task says: W[k*n+i, r]. So W is (p*n) x rank.
        # C_{ik} corresponds to index k*n + i in W?
        # "output factor follows the official trace convention W[k*n+i,r]"
        # So W is flattened.
        
        # Let's re-read carefully.
        # U (n*m, rank). V (m*p, rank). W (p*n, rank).
        # C_{ik} = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]?
        # No, the sum is over r.
        # But U depends on i, j?
        # Usually U is n x rank, V is m x rank, W is p x n?
        # Here U is n*m x rank.
        # This implies U_{i,j,r}.
        # V_{j,k,r}.
        # W_{k,i,r}.
        # C_{i,k} = sum_j sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}?
        # This is not standard bilinear.
        # Standard: C_{ik} = sum_j A_{ij} B_{jk}.
        # Tensor T_{i,j,k} = delta?
        # Decomposition: T = sum_r u_r \otimes v_r \otimes w_r.
        # u_r is vector of length n*m?
        # If u_r is length n*m, it's not separable in i,j.
        # But the task defines shapes.
        # I must follow the shapes.
        # U[i*m + j, r].
        # This means the first factor is associated with the pair (i,j).
        # This is the "trivial" basis structure if U is identity.
        
        # For n=1, m=1, p=p.
        # i=0, j=0.
        # U[0, r]. V[0, r] (since m=1, j=0, k=0..p-1? No, V is m*p x rank = 1*p x rank).
        # V[j*p + k, r] = V[k, r].
        # W[k*n + i, r] = W[k, 0, r] = W[k, r].
        # C_{0,k} = sum_r U[0,r] V[k,r] W[k,r].
        # We want C_{0,k} = A_0 B_k.
        # Let r = k.
        # U[0,k] = A_0? No, U is fixed.
        # The decomposition is linear in A and B.
        # So U[0,r] must be coeff of A_0.
        # V[k,r] must be coeff of B_k.
        # W[k,r] must be coeff of C_k.
        # If we set:
        # For each k in 0..p-1:
        #   U[0,k] = 1
        #   V[k,k] = 1 (others 0 in col k)
        #   W[k,k] = 1 (others 0 in col k)
        # Then C_k = 1 * 1 * 1 = A_0 B_k.
        # So Rank p.
        
        U = [[0] * p for _ in range(1)]
        V = [[0] * p for _ in range(p)]
        W = [[0] * p for _ in range(p)]
        for k in range(p):
            U[0][k] = 1
            V[k][k] = 1
            W[k][k] = 1
        return U, V, W

    # ... other base cases ...
    # Recursive step ...
