"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall-clock limit
    
    n, m, p = DIMENSIONS
    
    def verify(U, V, W):
        """Verify the decomposition using numpy for speed."""
        rank = len(U[0])
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(np.dot(U[u_idx], V[v_idx]), W[w_idx])
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    def build_rank33():
        """Build the rank-33 Strassen-based decomposition."""
        rank = 33
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        def add_strassen(i0, j0, k0):
            nonlocal col
            u_00 = i0 * m + j0
            u_01 = i0 * m + j0 + 1
            u_10 = (i0 + 1) * m + j0
            u_11 = (i0 + 1) * m + j0 + 1
            
            v_00 = j0 * p + k0
            v_01 = j0 * p + k0 + 1
            v_10 = (j0 + 1) * p + k0
            v_11 = (j0 + 1) * p + k0 + 1
            
            w_00 = k0 * n + i0
            w_10 = k0 * n + i0 + 1
            w_01 = (k0 + 1) * n + i0
            w_11 = (k0 + 1) * n + i0 + 1
            
            # M1 = (A00 + A11)(B00 + B11)
            U[u_00][col] += 1
            U[u_11][col] += 1
            V[v_00][col] += 1
            V[v_11][col] += 1
            W[w_00][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M2 = (A10 + A11)B00
            U[u_10][col] += 1
            U[u_11][col] += 1
            V[v_00][col] += 1
            W[w_10][col] += 1
            W[w_11][col] -= 1
            col += 1
            
            # M3 = A00(B01 - B11)
            U[u_00][col] += 1
            V[v_01][col] += 1
            V[v_11][col] -= 1
            W[w_01][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M4 = A11(B10 - B00)
            U[u_11][col] += 1
            V[v_10][col] += 1
            V[v_00][col] -= 1
            W[w_00][col] += 1
            W[w_10][col] += 1
            col += 1
            
            # M5 = (A00 + A01)B11
            U[u_00][col] += 1
            U[u_01][col] += 1
            V[v_11][col] += 1
            W[w_00][col] -= 1
            W[w_01][col] += 1
            col += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            U[u_10][col] += 1
            U[u_00][col] -= 1
            V[v_00][col] += 1
            V[v_01][col] += 1
            W[w_11][col] += 1
            col += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            U[u_01][col] += 1
            U[u_11][col] -= 1
            V[v_10][col] += 1
            V[v_11][col] += 1
            W[w_00][col] += 1
            col += 1
        
        # Apply Strassen to three 2x2x2 blocks
        add_strassen(0, 0, 0)
        add_strassen(0, 0, 2)
        add_strassen(0, 0, 4)
        
        # Handle the third column of A (j=2) with schoolbook
        for k in range(6):
            u_idx = 0 * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + 0
            U[u_idx][col] += 1
            V[v_idx][col] += 1
            W[w_idx][col] += 1
            col += 1
            
            u_idx = 1 * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + 1
            U[u_idx][col] += 1
            V[v_idx][col] += 1
            W[w_idx][col] += 1
            col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def try_search(target_rank, max_iters=20000):
        """Try to find a decomposition of the given rank using random search."""
        if time.time() > deadline - 30:
            return None
        
        U = np.zeros((n * m, target_rank), dtype=np.float64)
        V = np.zeros((m * p, target_rank), dtype=np.float64)
        W = np.zeros((p * n, target_rank), dtype=np.float64)
        
        # Initialize with random values
        for r in range(target_rank):
            U[:, r] = np.random.randn(n * m)
            V[:, r] = np.random.randn(m * p)
            W[:, r] = np.random.randn(p * n)
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 30:
                break
            
            # Compute error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        s = np.dot(np.dot(U[u_idx], V[v_idx]), W[w_idx])
                        err += (s - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-15:
                # Found a solution, verify and return
                result = {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                if verify(result["U"], result["V"], result["W"]):
                    return result
                break
            
            # Perturb a random entry
            r = np.random.randint(target_rank)
            choice = np.random.randint(3)
            if choice == 0:
                row = np.random.randint(n * m)
                U[row, r] += np.random.randn() * 0.1
            elif choice == 1:
                row = np.random.randint(m * p)
                V[row, r] += np.random.randn() * 0.1
            else:
                row = np.random.randint(p * n)
                W[row, r] += np.random.randn() * 0.1
        
        return None
    
    # First, try the known rank-33 solution
    result = build_rank33()
    
    if verify(result["U"], result["V"], result["W"]):
        best_rank = 33
        best_result = result
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        best_rank = rank
        best_result = {"U": U, "V": V, "W": W}
    
    # Try to find better solutions
    for target_rank in [32, 31, 30, 29, 28]:
        if time.time() > deadline - 30:
            break
        
        result = try_search(target_rank, max_iters=10000)
        if result is not None:
            if verify(result["U"], result["V"], result["W"]):
                best_rank = target_rank
                best_result = result
                break
    
    return best_result

# EVOLVE-BLOCK-END
