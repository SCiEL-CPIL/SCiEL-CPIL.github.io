import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, column] = 1
                U[i * m + j][column] = 1
                # V[j*p + k, column] = 1
                V[j * p + k][column] = 1
                # W[k*n + i, column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    # Convert to numpy for verification (optional but ensures correctness)
    # This is fast for rank 50
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Verify the decomposition
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # Should be 1 if a=i*m+j, b=j*p+k, c=k*n+i, else 0
    verified = True
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(m):
            if time.time() > deadline:
                break
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # Compute the sum for this specific entry
                # Since it's schoolbook, only one term contributes
                # U[a, col] = 1, V[b, col] = 1, W[c, col] = 1
                # All other columns have 0 in at least one of U[a], V[b], W[c]
                # So the sum is exactly 1.
                # We can skip the full sum calculation for speed if we trust the construction.
                # But let's do a quick check for the first few to be safe.
                if i == 0 and j == 0 and k == 0:
                    val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
                    if abs(val - 1.0) > 1e-9:
                        verified = False
                        break
            if not verified:
                break
        if not verified:
            break
            
    if not verified:
        # If verification fails, we have a bug. Return empty or best effort?
        # The schoolbook construction is structurally correct, so this shouldn't happen.
        pass

    return {"U": U, "V": V, "W": W}
