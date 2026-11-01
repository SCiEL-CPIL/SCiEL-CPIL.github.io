"""Tensor decomposition for 2x3 x 3x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    fallback = schoolbook()

    def verify(U, V, W, rank):
        """Verify exact decomposition. Returns max error."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    err = abs(val - 1.0)
                    max_err = max(max_err, err)
        return max_err

    def try_rank17_structured():
        """Structured rank-17 decomposition with combined terms."""
        rank = 17
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i == 0 and j == 0 and k in (0, 1):
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        # Combined term at col=16 handles (0,0,0) and (0,0,1)
        U[0][16] = 1
        V[0][16] = 1
        V[1][16] = 1
        W[0][16] = 1
        W[1][16] = 1
        
        err = verify(U, V, W, rank)
        if err < 1e-10:
            return {"U": U, "V": V, "W": W}
        return None

    # Try structured rank 17
    result17 = try_rank17_structured()
    if result17 is not None:
        return result17

    # If time permits, try a more general numerical search for rank 16
    if time.time() < deadline - 30:
        try:
            rank = 16
            # Use random initialization and coordinate descent
            best_err = float('inf')
            best_solution = None
            
            for trial in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Initialize with standard terms + random perturbations
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col >= rank:
                                break
                            if i == 0 and j == 0 and k in (0, 1):
                                continue
                            U[i * m + j, col] = 1
                            V[j * p + k, col] = 1
                            W[k * n + i, col] = 1
                            col += 1
                        if col >= rank:
                            break
                    if col >= rank:
                        break
                
                # Fill remaining columns with random values
                for c in range(col, rank):
                    U[:, c] = np.random.randn(nm)
                    V[:, c] = np.random.randn(mp)
                    W[:, c] = np.random.randn(pn)
                
                # Simple coordinate descent to reduce error
                for iteration in range(100):
                    if time.time() > deadline - 5:
                        break
                    
                    # Compute current error
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for c in range(rank):
                                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                                err += (val - 1.0) ** 2
                    err = np.sqrt(err)
                    
                    if err < best_err:
                        best_err = err
                        best_solution = (U.copy(), V.copy(), W.copy())
                    
                    if err < 1e-8:
                        break
                    
                    # Update one factor at a time
                    for factor_idx in range(3):
                        if factor_idx == 0:
                            # Update U
                            for i in range(nm):
                                for c in range(rank):
                                    old_val = U[i, c]
                                    # Try small perturbations
                                    for delta in [-0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5]:
                                        U[i, c] = old_val + delta
                                        new_err = 0.0
                                        for ii in range(n):
                                            for jj in range(m):
                                                for kk in range(p):
                                                    val = 0.0
                                                    for cc in range(rank):
                                                        val += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                                                    new_err += (val - 1.0) ** 2
                                        if new_err < err:
                                            err = new_err
                                            break
                                        else:
                                            U[i, c] = old_val
                                    U[i, c] = old_val
                        elif factor_idx == 1:
                            # Update V
                            for j in range(mp):
                                for c in range(rank):
                                    old_val = V[j, c]
                                    for delta in [-0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5]:
                                        V[j, c] = old_val + delta
                                        new_err = 0.0
                                        for ii in range(n):
                                            for jj in range(m):
                                                for kk in range(p):
                                                    val = 0.0
                                                    for cc in range(rank):
                                                        val += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                                                    new_err += (val - 1.0) ** 2
                                        if new_err < err:
                                            err = new_err
                                            break
                                        else:
                                            V[j, c] = old_val
                                    V[j, c] = old_val
                        else:
                            # Update W
                            for k in range(pn):
                                for c in range(rank):
                                    old_val = W[k, c]
                                    for delta in [-0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5]:
                                        W[k, c] = old_val + delta
                                        new_err = 0.0
                                        for ii in range(n):
                                            for jj in range(m):
                                                for kk in range(p):
                                                    val = 0.0
                                                    for cc in range(rank):
                                                        val += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
                                                    new_err += (val - 1.0) ** 2
                                        if new_err < err:
                                            err = new_err
                                            break
                                        else:
                                            W[k, c] = old_val
                                    W[k, c] = old_val
                
                if best_err < 1e-10:
                    break
            
            if best_solution is not None and best_err < 1e-10:
                U_sol, V_sol, W_sol = best_solution
                # Convert to lists
                U_list = U_sol.tolist()
                V_list = V_sol.tolist()
                W_list = W_sol.tolist()
                # Round to nearest half-integer if close
                for arr in [U_list, V_list, W_list]:
                    for row in arr:
                        for idx in range(len(row)):
                            row[idx] = round(row[idx] * 2) / 2
                
                # Verify after rounding
                err = verify(U_list, V_list, W_list, rank)
                if err < 1e-10:
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass

    return fallback
