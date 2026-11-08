import numpy as np
import time

def construct():
    n, m, p = 3, 5, 5
    start_time = time.time()
    DEADLINE = 145
    
    def check_deadline():
        return (time.time() - start_time) > DEADLINE
    
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        T = np.zeros((n, m, p), dtype=float)
        for c in range(rank):
            U_col = U[:, c].reshape(n, m)
            V_col = V[:, c].reshape(m, p)
            W_col = W[:, c].reshape(p, n)
            T += U_col[:, :, None] * V_col[:, :, None].T * W_col[:, :, None]
        if np.max(np.abs(T - 1.0)) > 1e-10:
            return False
        return True
    
    def add_strassen_block(i0, j0, k0, U, V, W, col):
        c = col + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c = col + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        c = col + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c = col + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        c = col + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        c = col + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        c = col + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
    
    # First, try to find the best Strassen block configuration (gives rank 71)
    from itertools import combinations
    
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_blocks.append((i0, j0, k0))
    
    best_rank = n * m * p
    best_solution = None
    
    for num_blocks in range(4, -1, -1):
        if check_deadline():
            break
        found = False
        for combo in combinations(all_blocks, num_blocks):
            if check_deadline():
                break
            valid = True
            covered = [[[False]*p for _ in range(m)] for _ in range(n)]
            for i0, j0, k0 in combo:
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            if covered[i0+di][j0+dj][k0+dk]:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid: break
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered[i0+di][j0+dj][k0+dk] = True
            if not valid:
                continue
            rank = num_blocks * 7 + (n*m*p - num_blocks*8)
            if rank >= best_rank:
                continue
            U = [[0.0]*rank for _ in range(n*m)]
            V = [[0.0]*rank for _ in range(m*p)]
            W = [[0.0]*rank for _ in range(p*n)]
            col = 0
            for i0, j0, k0 in combo:
                add_strassen_block(i0, j0, k0, U, V, W, col)
                col += 7
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if not covered[i][j][k]:
                            U[i*m+j][col] = 1.0
                            V[j*p+k][col] = 1.0
                            W[k*n+i][col] = 1.0
                            col += 1
            actual_rank = col
            U = [row[:actual_rank] for row in U]
            V = [row[:actual_rank] for row in V]
            W = [row[:actual_rank] for row in W]
            if verify(U, V, W):
                best_rank = actual_rank
                best_solution = {"U": U, "V": V, "W": W}
                found = True
                break
        if found:
            break
    
    if best_solution is None:
        rank = n * m * p
        U = [[0.0]*rank for _ in range(n*m)]
        V = [[0.0]*rank for _ in range(m*p)]
        W = [[0.0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1.0
                    V[j*p+k][c] = 1.0
                    W[k*n+i][c] = 1.0
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    # Now try to reduce the rank further using local search
    # Convert to numpy for easier manipulation
    U_np = np.array(best_solution["U"], dtype=float)
    V_np = np.array(best_solution["V"], dtype=float)
    W_np = np.array(best_solution["W"], dtype=float)
    current_rank = U_np.shape[1]
    
    # Try to eliminate columns one by one
    # For each column c, check if we can set it to zero and still satisfy the equations
    # This is equivalent to checking if the remaining columns span the same tensor space
    
    def can_eliminate_column(U, V, W, c):
        """Check if column c can be eliminated."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        U[:, c] = 0
        V[:, c] = 0
        W[:, c] = 0
        # Verify
        rank = U.shape[1]
        T = np.zeros((n, m, p), dtype=float)
        for cc in range(rank):
            U_col = U[:, cc].reshape(n, m)
            V_col = V[:, cc].reshape(m, p)
            W_col = W[:, cc].reshape(p, n)
            T += U_col[:, :, None] * V_col[:, :, None].T * W_col[:, :, None]
        return np.max(np.abs(T - 1.0)) < 1e-10
    
    # Try eliminating each column
    improved = True
    while improved and not check_deadline():
        improved = False
        for c in range(U_np.shape[1]):
            if check_deadline():
                break
            if can_eliminate_column(U_np, V_np, W_np, c):
                # Remove column c
                mask = [i for i in range(U_np.shape[1]) if i != c]
                U_np = U_np[:, mask]
                V_np = V_np[:, mask]
                W_np = W_np[:, mask]
                improved = True
                break
    
    if U_np.shape[1] < best_rank:
        return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
    
    return best_solution

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = 3, 5, 5
    print(f"Rank: {U.shape[1]}")
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(U.shape[1]):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-10:
                    ok = False
                    break
            if not ok: break
        if not ok: break
    print(f"Valid: {ok}")
