import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 63

    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W, tol=1e-9):
        """Check if the decomposition is exact (within tolerance)."""
        n, m, p = DIMENSIONS
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # For matrix mult, T[i,j,k] = 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(U.shape[1]):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def to_half_integer_grid(U, V, W, tol=1e-6):
        """Try to snap to nearest half-integers and verify."""
        def snap(A):
            return np.round(np.array(A) * 2) / 2.0
        Us = snap(U)
        Vs = snap(V)
        Ws = snap(W)
        if verify(Us.tolist(), Vs.tolist(), Ws.tolist(), tol=1e-6):
            return Us.tolist(), Vs.tolist(), Ws.tolist()
        return None

    # Strategy: try to find a lower-rank decomposition using structured search
    # The key insight: for 3x3x7, we can try to share structure across the 7 output columns
    # by using a rank-r decomposition where the U factor is shared
    
    # Try ranks from 50 down to 40 with randomized search
    best_result = None
    
    for target_rank in range(55, 39, -1):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(attempt * 137 + target_rank)
            
            # Random initialization with small values
            U = np.random.uniform(-1, 1, size=(n * m, target_rank))
            V = np.random.uniform(-1, 1, size=(m * p, target_rank))
            W = np.random.uniform(-1, 1, size=(p * n, target_rank))
            
            # Use a simple iterative refinement: alternate between fixing two factors
            # and solving for the third via least squares
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each row i*m+j, solve for U[i*m+j,:] given V, W
                # The equation is: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # This is a linear system in U[i*m+j,:]
                for i in range(n):
                    for j in range(m):
                        # Build the system: for each k, sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1
                        A_mat = np.zeros((p, target_rank))
                        b_vec = np.ones(p)
                        for k in range(p):
                            for c in range(target_rank):
                                A_mat[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        # Solve A_mat @ U[i*m+j,:] = b_vec
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        except:
                            pass
                
                # Update V: for each row j*p+k, solve for V[j*p+k,:] given U, W
                for j in range(m):
                    for k in range(p):
                        A_mat = np.zeros((n, target_rank))
                        b_vec = np.ones(n)
                        for i in range(n):
                            for c in range(target_rank):
                                A_mat[i, c] = U[i * m + j, c] * W[k * n + i, c]
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        except:
                            pass
                
                # Update W: for each row k*n+i, solve for W[k*n+i,:] given U, V
                for k in range(p):
                    for i in range(n):
                        A_mat = np.zeros((m, target_rank))
                        b_vec = np.ones(m)
                        for j in range(m):
                            for c in range(target_rank):
                                A_mat[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            max_err = max(max_err, abs(s - 1.0))
                
                if max_err < 1e-8:
                    found = True
                    break
            
            if found:
                # Try to snap to half-integers
                snapped = to_half_integer_grid(U, V, W)
                if snapped:
                    best_result = {"U": snapped[0], "V": snapped[1], "W": snapped[2]}
                    return best_result
                else:
                    # Return the floating point solution if it's exact
                    if verify(U.tolist(), V.tolist(), W.tolist(), tol=1e-8):
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        return best_result
    
    # Fallback to schoolbook
    return make_schoolbook()
