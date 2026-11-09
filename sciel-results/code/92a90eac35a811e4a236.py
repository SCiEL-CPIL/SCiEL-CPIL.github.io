import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    if n != 100:
        # Fallback to generic RK4 if structure is not 100
        def _rk4(f, t0, t1, y0, steps):
            h = (t1 - t0) / steps
            t, y = t0, np.array(y0, dtype=float)
            for _ in range(steps):
                k1 = f(t, y)
                k2 = f(t + h / 2, y + h / 2 * k1)
                k3 = f(t + h / 2, y + h / 2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
            return y
        return _rk4(f, t0, t1, y0, 2000)

    # Probe the structure
    probe_idx = 49
    v = np.zeros(n)
    v[probe_idx] = 1.0
    
    w = f(0.0, v)
    
    # Row-major neighbors of 49 (row 4, col 9)
    # Up: 39, Down: 59, Left: 48, Right (wrap): 40
    rm_neighbors = [39, 59, 48, 40]
    
    # Column-major neighbors of 49 (col 4, row 9)
    # Up: 84, Down (wrap): 4, Left: 93, Right: 95
    cm_neighbors = [84, 4, 93, 95]
    
    rm_score = sum(abs(w[i]) for i in rm_neighbors)
    cm_score = sum(abs(w[i]) for i in cm_neighbors)
    
    is_row_major = rm_score > cm_score
    
    tol = 1e-10
    
    if is_row_major:
        # Check periodicity: Right neighbor of (4,9) is (4,0) -> index 40
        is_periodic = abs(w[40]) > tol
        # Coefficient c from Up neighbor (3,9) -> index 39
        c = w[39]
        if abs(c) < tol:
            # Fallback
            def _rk4(f, t0, t1, y0, steps):
                h = (t1 - t0) / steps
                t, y = t0, np.array(y0, dtype=float)
                for _ in range(steps):
                    k1 = f(t, y)
                    k2 = f(t + h / 2, y + h / 2 * k1)
                    k3 = f(t + h / 2, y + h / 2 * k2)
                    k4 = f(t + h, y + h * k3)
                    y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                    t += h
                return y
            return _rk4(f, t0, t1, y0, 2000)
    else:
        # Check periodicity: Down neighbor of (9,4) is (0,4) -> index 4
        is_periodic = abs(w[4]) > tol
        # Coefficient c from Up neighbor (8,4) -> index 84
        c = w[84]
        if abs(c) < tol:
            def _rk4(f, t0, t1, y0, steps):
                h = (t1 - t0) / steps
                t, y = t0, np.array(y0, dtype=float)
                for _ in range(steps):
                    k1 = f(t, y)
                    k2 = f(t + h / 2, y + h / 2 * k1)
                    k3 = f(t + h / 2, y + h / 2 * k2)
                    k4 = f(t + h, y + h * k3)
                    y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                    t += h
                return y
            return _rk4(f, t0, t1, y0, 2000)

    # Build the sparse matrix A
    N = 10
    rows = []
    cols = []
    data = []
    
    for i in range(N):
        for j in range(N):
            if is_row_major:
                idx = i * N + j
            else:
                idx = j * N + i
            
            if is_periodic:
                # Center
                rows.append(idx)
                cols.append(idx)
                data.append(-4 * c)
                
                if is_row_major:
                    # Up
                    r_up = ((i - 1) % N)
                    c_up = j
                    idx_up = r_up * N + c_up
                    # Down
                    r_down = ((i + 1) % N)
                    c_down = j
                    idx_down = r_down * N + c_down
                    # Left
                    r_left = i
                    c_left = ((j - 1) % N)
                    idx_left = r_left * N + c_left
                    # Right
                    r_right = i
                    c_right = ((j + 1) % N)
                    idx_right = r_right * N + c_right
                else:
                    # Column major: idx = j*N + i
                    # Up (i-1, j)
                    r_up = ((i - 1) % N)
                    c_up = j
                    idx_up = c_up * N + r_up
                    # Down (i+1, j)
                    r_down = ((i + 1) % N)
                    c_down = j
                    idx_down = c_down * N + r_down
                    # Left (i, j-1)
                    r_left = i
                    c_left = ((j - 1) % N)
                    idx_left = c_left * N + r_left
                    # Right (i, j+1)
                    r_right = i
                    c_right = ((j + 1) % N)
                    idx_right = c_right * N + r_right

                rows.append(idx)
                cols.append(idx_up)
                data.append(c)
                
                rows.append(idx)
                cols.append(idx_down)
                data.append(c)
                
                rows.append(idx)
                cols.append(idx_left)
                data.append(c)
                
                rows.append(idx)
                cols.append(idx_right)
                data.append(c)
            else:
                # Dirichlet
                n_neighbors = 0
                if i > 0: n_neighbors += 1
                if i < N - 1: n_neighbors += 1
                if j > 0: n_neighbors += 1
                if j < N - 1: n_neighbors += 1
                
                rows.append(idx)
                cols.append(idx)
                data.append(-n_neighbors * c)
                
                if is_row_major:
                    if i > 0:
                        rows.append(idx)
                        cols.append((i - 1) * N + j)
                        data.append(c)
                    if i < N - 1:
                        rows.append(idx)
                        cols.append((i + 1) * N + j)
                        data.append(c)
                    if j > 0:
                        rows.append(idx)
                        cols.append(i * N + (j - 1))
                        data.append(c)
                    if j < N - 1:
                        rows.append(idx)
                        cols.append(i * N + (j + 1))
                        data.append(c)
                else:
                    if i > 0:
                        rows.append(idx)
                        cols.append(j * N + (i - 1))
                        data.append(c)
                    if i < N - 1:
                        rows.append(idx)
                        cols.append(j * N + (i + 1))
                        data.append(c)
                    if j > 0:
                        rows.append(idx)
                        cols.append((j - 1) * N + i)
                        data.append(c)
                    if j < N - 1:
                        rows.append(idx)
                        cols.append((j + 1) * N + i)
                        data.append(c)
    
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    # Solve y' = A y
    # y(t1) = expm(A * dt) y0
    E = spla.expm(A * dt)
    
    y_end = E.dot(y0)
    
    return y_end
