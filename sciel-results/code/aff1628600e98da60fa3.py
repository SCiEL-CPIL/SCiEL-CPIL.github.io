"""Matrix multiplication 4,5,5 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Schoolbook decomposition: rank = n * m * p = 100
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_school

    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_full(U_arr, V_arr, W_arr, R):
        """Full verification using numpy."""
        result = np.zeros((n*m, m*p, p*n), dtype=np.float64)
        for r in range(R):
            result += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
        return np.allclose(result, T, atol=1e-9)

    def verify_quick(U_arr, V_arr, W_arr, R):
        """Quick check: verify non-zero entries and sample zeros."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_arr[a] * V_arr[b], W_arr[c])
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check 100 random zero entries
        for _ in range(100):
            a = random.randint(0, n*m - 1)
            b = random.randint(0, m*p - 1)
            c = random.randint(0, p*n - 1)
            i, j = divmod(a, m)
            j2, k = divmod(b, p)
            k2, i2 = divmod(c, n)
            if j == j2 and i == i2 and k == k2:
                continue
            val = np.dot(U_arr[a] * V_arr[b], W_arr[c])
            if abs(val) > 1e-9:
                return False
        return True

    def try_solve_for_W(U_arr, V_arr, R):
        """Given U and V, solve for W and check if solution is valid."""
        W_arr = np.zeros((p * n, R), dtype=np.float64)
        
        for i in range(n):
            for k in range(p):
                c = k * n + i
                # Build system: for each j, sum_r U[i*m+j,r]*V[j*p+k,r]*W[c,r] = 1
                A_sys = np.zeros((m, R), dtype=np.float64)
                for j in range(m):
                    A_sys[j, :] = U_arr[i * m + j, :] * V_arr[j * p + k, :]
                
                try:
                    w_sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)
                except:
                    return None
                
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    return None
                
                W_arr[c, :] = w_sol
        
        return W_arr

    def try_solve_for_U(V_arr, W_arr, R):
        """Given V and W, solve for U."""
        U_arr = np.zeros((n * m, R), dtype=np.float64)
        
        for i in range(n):
            for j in range(m):
                a = i * m + j
                # Build system: for each k, sum_r U[a,r]*V[j*p+k,r]*W[k*n+i,r] = 1
                A_sys = np.zeros((p, R), dtype=np.float64)
                for k in range(p):
                    A_sys[k, :] = V_arr[j * p + k, :] * W_arr[k * n + i, :]
                
                try:
                    u_sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, np.ones(p), rcond=None)
                except:
                    return None
                
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    return None
                
                U_arr[a, :] = u_sol
        
        return U_arr

    def try_solve_for_V(U_arr, W_arr, R):
        """Given U and W, solve for V."""
        V_arr = np.zeros((m * p, R), dtype=np.float64)
        
        for j in range(m):
            for k in range(p):
                b = j * p + k
                # Build system: for each i, sum_r U[i*m+j,r]*V[b,r]*W[k*n+i,r] = 1
                A_sys = np.zeros((n, R), dtype=np.float64)
                for i in range(n):
                    A_sys[i, :] = U_arr[i * m + j, :] * W_arr[k * n + i, :]
                
                try:
                    v_sol, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)
                except:
                    return None
                
                if len(residuals) > 0 and residuals[0] > 1e-8:
                    return None
                
                V_arr[b, :] = v_sol
        
        return V_arr

    # Strategy 1: Structured circulant patterns
    # Use circulant matrices based on roots of unity patterns
    def generate_circulant_pattern(R, sparsity=3):
        """Generate a circulant-like pattern matrix."""
        U_arr = np.zeros((n * m, R), dtype=np.float64)
        
        # Create base pattern of length R
        base = np.zeros(R)
        nz_cols = random.sample(range(R), min(sparsity, R))
        for c in nz_cols:
            base[c] = random.choice([-1, 1])
        
        # Apply circulant structure to each row group
        for a in range(n * m):
            shift = a % R
            U_arr[a, :] = np.roll(base, shift)
        
        return U_arr

    # Strategy 2: Block-structured patterns
    def generate_block_pattern(R, block_size):
        """Generate block-structured pattern."""
        U_arr = np.zeros((n * m, R), dtype=np.float64)
        n_blocks = R // block_size + 1
        
        for a in range(n * m):
            block_idx = a % n_blocks
            start = block_idx * block_size
            end = min(start + block_size, R)
            for c in range(start, end):
                if random.random() < 0.5:
                    U_arr[a, c] = random.choice([-1, 1])
        
        return U_arr

    # Strategy 3: Fourier-based patterns
    def generate_fourier_pattern(R):
        """Generate pattern based on discrete Fourier transform structure."""
        U_arr = np.zeros((n * m, R), dtype=np.float64)
        
        # Use DFT matrix structure
        for a in range(n * m):
            for c in range(R):
                val = np.cos(2 * np.pi * a * c / R)
                if abs(val) > 0.5:
                    U_arr[a, c] = 1 if val > 0 else -1
        
        return U_arr

    # Strategy 4: Kronecker product structure
    def generate_kronecker_pattern(R):
        """Generate pattern using Kronecker product structure."""
        U_arr = np.zeros((n * m, R), dtype=np.float64)
        
        # Factor R into components
        factors = []
        temp = R
        for f in [2, 3, 4, 5]:
            while temp % f == 0:
                factors.append(f)
                temp //= f
        if temp > 1:
            factors.append(temp)
        
        if not factors:
            return generate_circulant_pattern(R, 3)
        
        # Build Kronecker pattern
        pattern = np.array([[1]])
        for f in factors:
            block = np.zeros((f, f))
            for i in range(f):
                block[i, i] = 1
                if i + 1 < f:
                    block[i, i+1] = -1
            pattern = np.kron(pattern, block)
        
        # Tile pattern to fit dimensions
        for a in range(n * m):
            for c in range(R):
                i = a % pattern.shape[0]
                j = c % pattern.shape[1]
                U_arr[a, c] = pattern[i, j]
        
        return U_arr

    # Main search loop
    pattern_generators = [
        generate_circulant_pattern,
        generate_block_pattern,
        generate_fourier_pattern,
        generate_kronecker_pattern
    ]

    for R in range(95, 24, -1):
        if time.time() > deadline - 5:
            break
        
        for trial in range(50):
            if time.time() > deadline - 2:
                break
            
            # Choose pattern generator
            gen_idx = trial % len(pattern_generators)
            
            if gen_idx == 0:
                U_arr = pattern_generators[0](R, random.randint(2, 5))
            elif gen_idx == 1:
                U_arr = pattern_generators[1](R, random.randint(2, 4))
            elif gen_idx == 2:
                U_arr = pattern_generators[2](R)
            else:
                U_arr = pattern_generators[3](R)
            
            # Generate V with different pattern
            v_gen_idx = (trial + 1) % len(pattern_generators)
            if v_gen_idx == 0:
                V_arr = pattern_generators[0](R, random.randint(2, 5))
            elif v_gen_idx == 1:
                V_arr = pattern_generators[1](R, random.randint(2, 4))
            elif v_gen_idx == 2:
                V_arr = pattern_generators[2](R)
            else:
                V_arr = pattern_generators[3](R)
            
            # Try to solve for W
            W_arr = try_solve_for_W(U_arr, V_arr, R)
            if W_arr is None:
                continue
            
            # Snap to half-integers
            W_arr = np.round(W_arr * 2) / 2
            
            # Verify
            if verify_quick(U_arr, V_arr, W_arr, R):
                if verify_full(U_arr, V_arr, W_arr, R):
                    best_U = U_arr.tolist()
                    best_V = V_arr.tolist()
                    best_W = W_arr.tolist()
                    best_rank = R
                    print(f"Found rank {R} decomposition with pattern {gen_idx}!")
                    return {"U": best_U, "V": best_V, "W": best_W}
            
            # Also try solving for U given V, W
            U_sol = try_solve_for_U(V_arr, W_arr, R)
            if U_sol is not None:
                U_sol = np.round(U_sol * 2) / 2
                if verify_quick(U_sol, V_arr, W_arr, R):
                    if verify_full(U_sol, V_arr, W_arr, R):
                        best_U = U_sol.tolist()
                        best_V = V_arr.tolist()
                        best_W = W_arr.tolist()
                        best_rank = R
                        print(f"Found rank {R} decomposition (solved for U)!")
                        return {"U": best_U, "V": best_V, "W": best_W}
            
            # Try solving for V given U, W
            V_sol = try_solve_for_V(U_arr, W_arr, R)
            if V_sol is not None:
                V_sol = np.round(V_sol * 2) / 2
                if verify_quick(U_arr, V_sol, W_arr, R):
                    if verify_full(U_arr, V_sol, W_arr, R):
                        best_U = U_arr.tolist()
                        best_V = V_sol.tolist()
                        best_W = W_arr.tolist()
                        best_rank = R
                        print(f"Found rank {R} decomposition (solved for V)!")
                        return {"U": best_U, "V": best_V, "W": best_W}
        
        # If we've tried many ranks without success, move on
        if time.time() > deadline - 10:
            break
    
    # Final fallback: try a few more random attempts at lower ranks
    for R in range(50, 30, -1):
        if time.time() > deadline - 3:
            break
        
        for trial in range(20):
            if time.time() > deadline - 1:
                break
            
            # Pure random sparse
            U_arr = np.zeros((n * m, R), dtype=np.float64)
            V_arr = np.zeros((m * p, R), dtype=np.float64)
            
            for a in range(n * m):
                num_nz = random.randint(2, 4)
                cols = random.sample(range(R), num_nz)
                for c in cols:
                    U_arr[a, c] = random.choice([-1, 1])
            
            for b in range(m * p):
                num_nz = random.randint(2, 4)
                cols = random.sample(range(R), num_nz)
                for c in cols:
                    V_arr[b, c] = random.choice([-1, 1])
            
            W_arr = try_solve_for_W(U_arr, V_arr, R)
            if W_arr is None:
                continue
            
            W_arr = np.round(W_arr * 2) / 2
            
            if verify_quick(U_arr, V_arr, W_arr, R):
                if verify_full(U_arr, V_arr, W_arr, R):
                    best_U = U_arr.tolist()
                    best_V = V_arr.tolist()
                    best_W = W_arr.tolist()
                    best_rank = R
                    print(f"Found rank {R} decomposition (random fallback)!")
                    return {"U": best_U, "V": best_V, "W": best_W}
    
    # Return best found
    if best_rank < rank_school:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U, "V": V, "W": W}
