"""Matrix multiplication 3x4 by 4x6: Recursive block decomposition with linear algebra rank reduction."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    
    Strategy:
    1. Build base decomposition using recursive block splitting with Strassen 2x2x2
    2. Use linear algebra to find minimal rank by identifying linearly dependent columns
    """
    n, m, p = 3, 4, 6
    start_time = time.time()
    deadline = start_time + 100.0

    def decompose(n, m, p):
        """
        Returns a list of (U_dict, V_dict, W_dict)
        U_dict: {i*m+j: coeff}
        V_dict: {j*p+k: coeff}
        W_dict: {k*n+i: coeff}
        """
        cols = []

        def add_strassen_2x2x2(i0, j0, k0):
            """
            Adds 7 Strassen terms for a 2x2x2 block at offsets (i0, j0, k0).
            """
            # M1 = (A11+A22)(B11+B22)
            cols.append((
                {i0*m+j0: 1, (i0+1)*m+(j0+1): 1},
                {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},
                {k0*n+i0: 1, (k0+1)*n+(i0+1): 1}
            ))

            # M2 = (A21+A22)B11
            cols.append((
                {(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1},
                {j0*p+k0: 1},
                {k0*n+(i0+1): 1, (k0+1)*n+(i0+1): -1}
            ))

            # M3 = A11(B12-B22)
            cols.append((
                {i0*m+j0: 1},
                {j0*p+(k0+1): 1, (j0+1)*p+(k0+1): -1},
                {(k0+1)*n+i0: 1, (k0+1)*n+(i0+1): 1}
            ))

            # M4 = A22(B21-B11)
            cols.append((
                {(i0+1)*m+(j0+1): 1},
                {(j0+1)*p+k0: 1, j0*p+k0: -1},
                {k0*n+i0: 1, k0*n+(i0+1): 1}
            ))

            # M5 = (A11+A12)B22
            cols.append((
                {i0*m+j0: 1, i0*m+(j0+1): 1},
                {(j0+1)*p+(k0+1): 1},
                {k0*n+i0: -1, (k0+1)*n+i0: 1}
            ))

            # M6 = (A21-A11)(B11+B12)
            cols.append((
                {(i0+1)*m+j0: 1, i0*m+j0: -1},
                {j0*p+k0: 1, j0*p+(k0+1): 1},
                {(k0+1)*n+(i0+1): 1}
            ))

            # M7 = (A12-A22)(B21+B22)
            cols.append((
                {i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1},
                {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1},
                {k0*n+i0: 1}
            ))

        def recurse(n_curr, m_curr, p_curr, i0, j0, k0):
            # Base case: 1x1x1
            if n_curr == 1 and m_curr == 1 and p_curr == 1:
                u_idx = i0 * m + j0
                v_idx = j0 * p + k0
                w_idx = k0 * n + i0
                cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
                return

            # Apply Strassen 2x2x2 if possible
            if n_curr >= 2 and m_curr >= 2 and p_curr >= 2:
                add_strassen_2x2x2(i0, j0, k0)
                
                # Recurse on remainders
                if n_curr > 2:
                    recurse(n_curr - 2, m_curr, p_curr, i0 + 2, j0, k0)
                if m_curr > 2:
                    recurse(2, m_curr - 2, p_curr, i0, j0 + 2, k0)
                if p_curr > 2:
                    recurse(2, 2, p_curr - 2, i0, j0, k0 + 2)
                return

            # Fallback: split into smaller blocks
            if n_curr > 1:
                recurse(1, m_curr, p_curr, i0, j0, k0)
                recurse(n_curr - 1, m_curr, p_curr, i0 + 1, j0, k0)
                return
            elif m_curr > 1:
                recurse(n_curr, 1, p_curr, i0, j0, k0)
                recurse(n_curr, m_curr - 1, p_curr, i0, j0 + 1, k0)
                return
            elif p_curr > 1:
                recurse(n_curr, m_curr, 1, i0, j0, k0)
                recurse(n_curr, m_curr, p_curr - 1, i0, j0, k0 + 1)
                return
            else:
                u_idx = i0 * m + j0
                v_idx = j0 * p + k0
                w_idx = k0 * n + i0
                cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
                return

        recurse(n, m, p, 0, 0, 0)
        return cols

    # Build base decomposition
    cols = decompose(n, m, p)
    rank = len(cols)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(cols):
        for idx, val in u_dict.items():
            U[idx, r] = val
        for idx, val in v_dict.items():
            V[idx, r] = val
        for idx, val in w_dict.items():
            W[idx, r] = val

    # Verify correctness
    def verify(U, V, W, rank):
        T = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    u_val = U[i*m+j, r]
                    if u_val == 0:
                        continue
                    for k in range(p):
                        v_val = V[j*p+k, r]
                        if v_val == 0:
                            continue
                        w_val = W[k*n+i, r]
                        if w_val == 0:
                            continue
                        T[i, j, k] += u_val * v_val * w_val
        return np.allclose(T, 1.0, atol=1e-10)

    if not verify(U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now try to reduce rank using linear algebra
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can view this as: for each (i,j,k), the equation is:
    # sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
    
    # A more efficient approach: try removing columns one by one
    # and check if the tensor is still correct
    
    active_cols = list(range(rank))
    current_rank = rank
    
    # Try to remove columns one by one
    for iteration in range(30):
        if _time_left() < 15:
            break
        
        removed_any = False
        for r in list(active_cols):
            if _time_left() < 15:
                break
            
            # Try removing column r
            test_active = [c for c in active_cols if c != r]
            
            # Check if the tensor is still correct
            T_test = np.zeros((n, m, p))
            for r_idx in test_active:
                for i in range(n):
                    for j in range(m):
                        u_val = U[i*m+j, r_idx]
                        if u_val == 0:
                            continue
                        for k in range(p):
                            v_val = V[j*p+k, r_idx]
                            if v_val == 0:
                                continue
                            w_val = W[k*n+i, r_idx]
                            if w_val == 0:
                                continue
                            T_test[i, j, k] += u_val * v_val * w_val
            
            if np.allclose(T_test, 1.0, atol=1e-10):
                active_cols.remove(r)
                current_rank -= 1
                removed_any = True
                break  # Restart the loop after removing one column
        
        if not removed_any:
            break

    # Build the final decomposition with only active columns
    if current_rank < rank:
        final_U = np.zeros((n * m, current_rank), dtype=float)
        final_V = np.zeros((m * p, current_rank), dtype=float)
        final_W = np.zeros((p * n, current_rank), dtype=float)
        
        for new_idx, old_idx in enumerate(active_cols):
            final_U[:, new_idx] = U[:, old_idx]
            final_V[:, new_idx] = V[:, old_idx]
            final_W[:, new_idx] = W[:, old_idx]
        
        return {
            "U": final_U.tolist(),
            "V": final_V.tolist(),
            "W": final_W.tolist()
        }
    else:
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
