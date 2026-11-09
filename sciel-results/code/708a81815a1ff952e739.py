import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = 3, 3, 7

    # Schoolbook tensor
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    # Try to find linear dependencies to reduce rank
    # Check if any column is a linear combination of the previous columns
    # If so, remove it
    cols_to_remove = []
    rank = n * m * p

    for target in range(1, rank):
        if time.time() > deadline - 5:
            break

        # Check if column 'target' is in span of columns [0, target-1]
        A = U[:, :target]
        b = U[:, target]

        try:
            coeffs, residuals, rank_a, sv = np.linalg.lstsq(A, b, rcond=None)
            if residuals and residuals[0] < 1e-10:
                # Check if same coefficients work for V and W
                V_pred = V[:, :target] @ coeffs
                W_pred = W[:, :target] @ coeffs
                if np.allclose(V[:, target], V_pred, atol=1e-10) and \
                   np.allclose(W[:, target], W_pred, atol=1e-10):
                    # We can eliminate column 'target'
                    cols_to_remove.append(target)
        except:
            continue

    # Remove identified columns
    if cols_to_remove:
        mask = [i for i in range(rank) if i not in set(cols_to_remove)]
        U = U[:, mask]
        V = V[:, mask]
        W = W[:, mask]

    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
