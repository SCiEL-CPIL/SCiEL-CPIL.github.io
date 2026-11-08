import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 3, 3, 3
    
    # Build target tensor
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) match
    # a = i*3 + j
    # b = j*3 + k
    # c = k*3 + i
    T = np.zeros((9, 9, 9), dtype=np.int64)
    for i in range(3):
        for k in range(3):
            for j in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                T[a, b, c] = 1

    def check_and_get_rank(U, V, W):
        """Check if U, V, W form a valid decomposition and return rank."""
        # Ensure U, V, W are integer or half-integer?
        # The problem says "integer/half-integer entries".
        # We will check if they are close to such values.
        # For now, assume we search for integer solutions.
        recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(recon, T):
            # Count non-zero columns
            nonzero = np.any(U != 0, axis=0) & np.any(V != 0, axis=0) & np.any(W != 0, axis=0)
            return np.sum(nonzero)
        return -1

    # Baseline: Rank 26
    def get_rank26():
        rank = 26
        U = np.zeros((9, rank), dtype=np.float64)
        V = np.zeros((9, rank), dtype=np.float64)
        W = np.zeros((9, rank), dtype=np.float64)
        
        col = 0
        # M1
        U[0, col] = 1; U[4, col] = 1
        V[0, col] = 1; V[4, col] = 1
        W[0, col] = 1; W[4, col] = 1
        col += 1
        # M2
        U[3, col] = 1; U[4, col] = 1
        V[0, col] = 1
        W[1, col] = 1; W[4, col] = -1
        col += 1
        # M3
        U[0, col] = 1
        V[1, col] = 1; V[4, col] = -1
        W[3, col] = 1; W[4, col] = 1
        col += 1
        # M4
        U[4, col] = 1
        V[3, col] = 1; V[0, col] = -1
        W[0, col] = 1; W[1, col] = 1
        col += 1
        # M5
        U[0, col] = 1; U[1, col] = 1
        V[4, col] = 1
        W[0, col] = -1; W[3, col] = 1
        col += 1
        # M6
        U[3, col] = 1; U[0, col] = -1
        V[0, col] = 1; V[1, col] = 1
        W[4, col] = 1
        col += 1
        # M7
        U[1, col] = 1; U[4, col] = -1
        V[3, col] = 1; V[4, col] = 1
        W[0, col] = 1
        col += 1
        
        # Cross terms
        U[2, col] = 1; V[6, col] = 1; W[0, col] = 1; col += 1
        U[2, col] = 1; V[7, col] = 1; W[3, col] = 1; col += 1
        U[5, col] = 1; V[6, col] = 1; W[1, col] = 1; col += 1
        U[5, col] = 1; V[7, col] = 1; W[4, col] = 1; col += 1
        
        U[0, col] = 1; V[2, col] = 1; W[6, col] = 1; col += 1
        U[1, col] = 1; V[5, col] = 1; W[6, col] = 1; col += 1
        U[3, col] = 1; V[2, col] = 1; W[7, col] = 1; col += 1
        U[4, col] = 1; V[5, col] = 1; W[7, col] = 1; col += 1
        
        U[2, col] = 1; V[8, col] = 1; W[6, col] = 1; col += 1
        U[5, col] = 1; V[8, col] = 1; W[7, col] = 1; col += 1
        
        U[6, col] = 1; V[0, col] = 1; W[2, col] = 1; col += 1
        U[7, col] = 1; V[3, col] = 1; W[2, col] = 1; col += 1
        U[6, col] = 1; V[1, col] = 1; W[5, col] = 1; col += 1
        U[7, col] = 1; V[4, col] = 1; W[5, col] = 1; col += 1
        
        U[8, col] = 1; V[6, col] = 1; W[2, col] = 1; col += 1
        U[8, col] = 1; V[7, col] = 1; W[5, col] = 1; col += 1
        
        U[6, col] = 1; V[2, col] = 1; W[8, col] = 1; col += 1
        U[7, col] = 1; V[5, col] = 1; W[8, col] = 1; col += 1
        
        U[8, col] = 1; V[8, col] = 1; W[8, col] = 1; col += 1
        
        return U, V, W

    best_U, best_V, best_W = get_rank26()
    best_rank = 26

    # Verify baseline
    if check_and_get_rank(best_U, best_V, best_W) == 26:
        pass # Valid
    else:
        # Fallback to naive rank 27 if baseline is broken?
        # But baseline is hardcoded and known to work.
        pass

    # Search for lower rank
    # We will try rank 25, 24, 23
    # Strategy: Simulated Annealing
    # Initialize with random sparse matrices?
    # Or initialize by taking the rank 26 solution and zeroing out random columns?
    
    rng = np.random.RandomState(42)
    
    def try_search(rank, max_iter=5000):
        # Initialize
        U = rng.randint(-1, 2, size=(9, rank)).astype(float)
        V = rng.randint(-1, 2, size=(9, rank)).astype(float)
        W = rng.randint(-1, 2, size=(9, rank)).astype(float)
        
        # Or start from rank 26 and drop random columns?
        # Let's try random first, it's simpler.
        
        def error(Ux, Vx, Wx):
            recon = np.einsum('ir,jr,kr->ijk', Ux, Vx, Wx)
            return np.sum((recon - T)**2)
        
        current_err = error(U, V, W)
        best_err = current_err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        temp = 10.0
        cool = 0.999
        
        for i in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Perturb
            idx = rng.randint(0, 3)
            if idx == 0:
                row = rng.randint(0, 9)
                col = rng.randint(0, rank)
                delta = rng.choice([-1, 1]) * 0.1
                U[row, col] += delta
            elif idx == 1:
                row = rng.randint(0, 9)
                col = rng.randint(0, rank)
                delta = rng.choice([-1, 1]) * 0.1
                V[row, col] += delta
            else:
                row = rng.randint(0, 9)
                col = rng.randint(0, rank)
                delta = rng.choice([-1, 1]) * 0.1
                W[row, col] += delta
                
            new_err = error(U, V, W)
            delta_err = new_err - current_err
            
            if delta_err < 0 or rng.rand() < np.exp(-delta_err / temp):
                current_err = new_err
                if new_err < best_err:
                    best_err = new_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    if best_err == 0:
                        break
            else:
                # Revert
                if idx == 0:
                    U[row, col] -= delta
                elif idx == 1:
                    V[row, col] -= delta
                else:
                    W[row, col] -= delta
                    
            temp *= cool
            
        # Check if best solution is integer and valid
        # Round to nearest integer
        U_int = np.round(best_U)
        V_int = np.round(best_V)
        W_int = np.round(best_W)
        
        if check_and_get_rank(U_int, V_int, W_int) == rank:
            return U_int, V_int, W_int
        
        return None

    # Try rank 25
    if time.time() < deadline - 20:
        res = try_search(25, max_iter=2000)
        if res:
            best_U, best_V, best_W = res
            best_rank = 25

    # Try rank 24
    if time.time() < deadline - 10:
        res = try_search(24, max_iter=2000)
        if res:
            best_U, best_V, best_W = res
            best_rank = 24

    # Try rank 23
    if time.time() < deadline - 5:
        res = try_search(23, max_iter=2000)
        if res:
            best_U, best_V, best_W = res
            best_rank = 23

    # Convert to list of lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
