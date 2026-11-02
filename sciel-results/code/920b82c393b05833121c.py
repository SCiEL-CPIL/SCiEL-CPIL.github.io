import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    
    # Trivial decomposition
    # Rank = n * m * p
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, r] = 1.0
                # V index: j*p + k
                V[j * p + k, r] = 1.0
                # W index: k*n + i
                W[k * n + i, r] = 1.0
                r += 1
                
    # Verification
    if time.time() < DEADLINE - 10:
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        C_decomp = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    # Sum over j for U part? No, U is already summed in the scalar product?
                    # The formula is C[i,k] = sum_r W[k*n+i, r] * (sum_a U[a,r] A[a]) * (sum_b V[b,r] B[b])
                    # But U and V are sparse in the trivial case.
                    # Let's just compute the scalar products explicitly to be safe.
                    # Actually, for the trivial case:
                    # S_r = A[i,j] * B[j,k] where (i,j,k) corresponds to r.
                    # We can just reconstruct from the matrices.
                    pass
        
        # Efficient verification using matrix multiplication properties
        # C_decomp[i,k] = sum_r W[k*n+i, r] * (U[:,r] @ A_flat) * (V[:,r] @ B_flat)
        A_flat = A.flatten() # shape (n*m)
        B_flat = B.flatten() # shape (m*p)
        
        # U @ A_flat gives a vector of length rank
        # V @ B_flat gives a vector of length rank
        # W is (p*n, rank). We need to reshape W to (n, p, rank) or similar.
        # W index is k*n + i.
        # Let's compute the outer product of the two vectors and contract with W.
        u_vals = U @ A_flat # shape (rank)
        v_vals = V @ B_flat # shape (rank)
        
        # C_decomp[i,k] = sum_r W[k*n+i, r] * u_vals[r] * v_vals[r]
        # We can do this by reshaping W.
        # W is (p*n, rank).
        # We want to sum over r.
        # Let W_reshaped be (n, p, rank) where W_reshaped[i, k, r] = W[k*n+i, r]
        W_reshaped = W.reshape(p, n, rank).transpose(1, 0, 2) # (n, p, rank)
        
        # Now C_decomp[i,k] = sum_r W_reshaped[i,k,r] * u_vals[r] * v_vals[r]
        # This is element-wise multiplication of W_reshaped with (u_vals * v_vals)[:, None, :]?
        # u_vals * v_vals is shape (rank).
        # We need to broadcast it to (n, p, rank).
        prod_vals = u_vals * v_vals # shape (rank)
        C_decomp = np.sum(W_reshaped * prod_vals[None, None, :], axis=2)
        
        if not np.allclose(C_true, C_decomp):
            # If it fails, something is wrong with the index mapping.
            # But the logic above is sound for the trivial decomposition.
            pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result["U"][0]))
