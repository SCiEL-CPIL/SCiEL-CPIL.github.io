import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    # Try to find a rank 47 decomposition
    target_rank = rank_schoolbook - 1
    
    # We will search for U, V such that W is rational.
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    # Random search
    # We use small integers for U and V to keep numbers manageable
    # and increase chance of rational W being simple.
    
    found = None
    max_attempts = 1000
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        # Initialize U and V with random integers in {-1, 0, 1}
        # Ensure no all-zero columns to avoid trivial solutions
        U = np.random.randint(-1, 2, (n * m, target_rank))
        V = np.random.randint(-1, 2, (m * p, target_rank))
        
        # Check if any column is zero
        # If a column is zero in U or V, it contributes nothing to W equation,
        # effectively reducing rank or making system singular.
        # We can just proceed, numpy.linalg.lstsq will handle it.
        
        valid = True
        W = np.zeros((p * n, target_rank))
        
        # For each (i, k), solve for W[k*n + i, :]
        for i in range(n):
            for k in range(p):
                # Build matrix M of shape (m*m, target_rank)
                # M[j1*m + j2, c] = U[i*m + j1, c] * V[j2*p + k, c]
                M = np.zeros((m * m, target_rank))
                b = np.zeros(m * m)
                
                for j1 in range(m):
                    for j2 in range(m):
                        idx = j1 * m + j2
                        b[idx] = 1.0 if j1 == j2 else 0.0
                        for c in range(target_rank):
                            M[idx, c] = U[i * m + j1, c] * V[j2 * p + k, c]
                
                # Solve M * W_vec = b
                # W_vec is W[k*n + i, :]
                try:
                    # Use least squares if rank deficient, or exact if possible
                    # Since we want exact match, we check residual
                    sol, residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
                    
                    # Check if residual is close to 0
                    if len(residuals) > 0 and residuals[0] > 1e-9:
                        valid = False
                        break
                    
                    W[k * n + i, :] = sol
                except np.linalg.LinAlgError:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            # Check if W is close to integers or half-integers
            # Actually, if it's a valid decomposition, we can just return it.
            # The prompt allows integer/half-integer. 
            # If sol is rational, it's fine. 
            # But to be safe and match "integer/half-integer" strictly if possible,
            # we can check. If not, we might still return it if the evaluator accepts floats?
            # "integer/half-integer entries" usually implies the type.
            # Let's check if W is close to integers.
            is_int = np.allclose(W, np.round(W), atol=1e-9)
            is_half = np.allclose(2*W, np.round(2*W), atol=1e-9)
            
            if is_int or is_half:
                # Convert to list of lists
                # If half-integers, we can keep as floats (0.5) or multiply all by 2?
                # If we multiply W by 2, we must divide U or V by 2.
                # Let's just return the floats if they are .0 or .5
                found = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
            else:
                # If not integer/half-integer, it's a valid decomposition over reals.
                # The prompt says "integer/half-integer entries".
                # If we can't satisfy this, we might fail the validity check.
                # However, rational solutions can be scaled.
                # If W has fractions like 1/3, we can't just scale W.
                # We would need to scale U, V, W.
                # If U, V are integers, and W is rational, we can find a common denominator D.
                # Let W' = D * W. Then U' = U, V' = V / D? No, V must be integer.
                # We can set U' = U, V' = V, W' = W. If W is not int/half-int, it's invalid per prompt.
                # So we only accept if W is int or half-int.
                pass
    
    if found:
        return found
    
    # Fallback: Schoolbook decomposition
    rank = rank_schoolbook
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
