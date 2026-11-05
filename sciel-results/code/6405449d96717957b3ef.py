import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try to find a rank-28 solution
    while time.time() < deadline:
        # Initialize U and V with random integers {-1, 0, 1}
        # Use a fixed seed for reproducibility in testing, but random in production?
        # Let's use random seed based on time to explore different solutions
        np.random.seed(int(time.time() * 1000) % 10000)
        
        U = np.random.choice([-1, 0, 1], size=(n * m, rank))
        V = np.random.choice([-1, 0, 1], size=(m * p, rank))
        
        W = np.zeros((p * n, rank))
        
        valid = True
        # Solve for W for each (i, k)
        for i in range(n):
            for k in range(p):
                # Build the 3 x rank matrix A
                # Row j corresponds to equation for j
                A = np.zeros((m, rank))
                for j in range(m):
                    # U index: i*m + j
                    # V index: j*p + k
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                # We need to solve A * w = [1, 1, 1]^T
                # We pick 3 columns that are linearly independent
                # Try all combinations of 3 columns? rank is 28, C(28,3) is 3276.
                # Or just try random permutations until we find one.
                
                found = False
                # Try to find 3 independent columns
                # Use QR decomposition or just check determinants
                # For speed, just pick random triples
                attempts = 0
                while attempts < 1000 and not found:
                    cols = np.random.choice(rank, 3, replace=False)
                    sub = A[:, cols]
                    det = np.linalg.det(sub)
                    if abs(det) > 1e-9:
                        # Solve sub * w_sub = b
                        b = np.ones(m)
                        try:
                            w_sub = np.linalg.solve(sub, b)
                            # Check if w_sub has small denominators
                            # Convert to fractions?
                            # Since A has integer entries, w_sub is rational.
                            # Denominator divides det.
                            # If det is 1 or 2, entries are integer or half-integer.
                            # If det is larger, check if numerator is divisible.
                            
                            # Simple check: multiply by det and see if integer
                            w_sub_scaled = w_sub * det
                            if np.allclose(w_sub_scaled, np.round(w_sub_scaled)):
                                # Valid
                                for idx, col in enumerate(cols):
                                    W[k * n + i, col] = w_sub[idx]
                                found = True
                            else:
                                # Maybe scale U, V later?
                                # For now, reject this triple
                                pass
                        except np.linalg.LinAlgError:
                            pass
                    attempts += 1
                
                if not found:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            # Verify the solution
            # Check all i, j, k
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(rank):
                            s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        if abs(s - 1) > 1e-6:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            
            if ok:
                # Check if entries are integer or half-integer
                def is_half_int(x):
                    return abs(x * 2 - round(x * 2)) < 1e-6
                
                if all(is_half_int(x) for x in U.flatten()) and \
                   all(is_half_int(x) for x in V.flatten()) and \
                   all(is_half_int(x) for x in W.flatten()):
                    # Convert to list of lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
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
