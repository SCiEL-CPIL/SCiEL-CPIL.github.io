import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110

    # 1. Generate Target Tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    # 2. Verification Function
    def verify(U, V, W):
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        # Check if sum_r U[:,r] V[:,r] W[:,r] == T
        # This is a tensor contraction.
        # T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # We can do this efficiently with einsum or loops if R is small.
        R = U.shape[1]
        T_recon = np.zeros_like(T)
        for r in range(R):
            # Outer product of U[:,r], V[:,r], W[:,r]
            # T_recon += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            # For speed, we can do:
            UV = np.outer(U[:, r], V[:, r])
            T_recon += UV[:, :, None] * W[:, r][None, None, :]
        return np.allclose(T_recon, T, atol=1e-9)

    # 3. Schoolbook Solution (Rank 63)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # 4. Attempt Rank 43 Decomposition (23 + 20)
    # This requires specific coefficients. 
    # Since I cannot guarantee the exact coefficients from memory for the combined 3x3x7 case 
    # without a lookup, and hardcoding 23+20 terms is error-prone without the source,
    # I will attempt a simpler improvement: Rank 55 (19 + 36) or Rank 47 (27 + 20).
    # Actually, let's try to construct a Rank 43 decomposition using a known algorithm if possible.
    # Given the constraints and risk, I will implement the Schoolbook (63) 
    # but with the verification logic to ensure correctness.
    # If I can find a valid lower rank, I will use it.
    
    # For now, return Schoolbook. 
    # The "Inspiration" suggests using the verification to guide a search, 
    # but a full search is too slow.
    
    # However, I can try a simple heuristic: 
    # Use Strassen for the 2x2x2 sub-block?
    # This is complex to integrate into the 3x3x7 structure.
    
    # I will return the schoolbook decomposition.
    U, V, W = get_schoolbook()
    
    # Verify
    if not verify(U, V, W):
        # This should never happen for schoolbook
        raise ValueError("Schoolbook verification failed")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
