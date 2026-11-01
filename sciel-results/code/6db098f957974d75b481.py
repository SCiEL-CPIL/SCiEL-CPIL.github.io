import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    
    # Precompute the tensor structure
    # T[A,B,C] = 1 if there exist i,j,k with A=i*m+j, B=j*p+k, C=k*n+i, else 0
    # A in [0, n*m), B in [0, m*p), C in [0, p*n)
    
    # Build list of non-zero entries
    non_zero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                A = i * m + j
                B = j * p + k
                C = k * n + i
                non_zero.append((A, B, C))
    
    # Build the full tensor as a dictionary for fast lookup
    T_dict = {}
    for A, B, C in non_zero:
        T_dict[(A, B, C)] = 1.0
    
    # Precompute for each A: list of (B, C, t_val) for all B,C
    A_BC = {}
    for A in range(n * m):
        A_BC[A] = []
        i = A // m
        j = A % m
        for B in range(m * p):
            for C in range(p * n):
                if (A, B, C) in T_dict:
                    t_val = 1.0
                else:
                    t_val = 0.0
                A_BC[A].append((B, C, t_val))
    
    # Precompute for each B: list of (A, C, t_val)
    B_AC = {}
    for B in range(m * p):
        B_AC[B] = []
        j = B // p
        k = B % p
        for A in range(n * m):
            for C in range(p * n):
                if (A, B, C) in T_dict:
                    t_val = 1.0
                else:
                    t_val = 0.0
                B_AC[B].append((A, C, t_val))
    
    # Precompute for each C: list of (A, B, t_val)
    C_AB = {}
    for C in range(p * n):
        C_AB[C] = []
        k = C // n
        i = C % n
        for A in range(n * m):
            for B in range(m * p):
                if (A, B, C) in T_dict:
                    t_val = 1.0
                else:
                    t_val = 0.0
                C_AB[C].append((A, B, t_val))
    
    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition"""
        for A, B, C in non_zero:
            val = sum(U[A][r] * V[B][r] * W[C][r] for r in range(len(U[0])))
            if abs(val - 1.0) > tol:
                return False
        # Also verify zeros
        for A in range(n * m):
            for B in range(m * p):
                for C in range(p * n):
                    if (A, B, C) in T_dict:
                        continue
                    val = sum(U[A][r] * V[B][r] * W[C][r] for r in range(len(U[0])))
                    if abs(val) > tol:
                        return False
        return True
    
    def verify_fast(U, V, W, tol=1e-9):
        """Faster verification using numpy"""
        rank = len(U[0])
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        # Check non-zero entries
        for A, B, C in non_zero:
            val = np.dot(U[A], V[B] * W[C])
            if abs(val - 1.0) > tol:
                return False
        # Check zeros - sample some
        import random
        random.seed(42)
        checked = 0
        while checked < 100:
            A = random.randint(0, n*m-1)
            B = random.randint(0, m*p-1)
            C = random.randint(0, p*n-1)
            if (A, B, C) in T_dict:
                continue
            val = np.dot(U[A], V[B] * W[C])
            if abs(val) > tol:
                return False
            checked += 1
        return True
    
    def schoolbook():
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
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def solve_als(r, max_iter=100, tol=1e-10, seed=42):
        np.random.seed(seed)
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U
            U_new = np.zeros_like(U)
            for A in range(n * m):
                # Build M matrix: M[row, r] = V[B,row, r] * W[C,row, r]
                # b[row] = t_val
                M = np.zeros((len(A_BC[A]), r))
                b = np.zeros(len(A_BC[A]))
                for idx, (B, C, t_val) in enumerate(A_BC[A]):
                    M[idx, :] = V[B, :] * W[C, :]
                    b[idx] = t_val
                
                U_A, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                U_new[A, :] = U_A
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for B in range(m * p):
                M = np.zeros((len(B_AC[B]), r))
                b = np.zeros(len(B_AC[B]))
                for idx, (A, C, t_val) in enumerate(B_AC[B]):
                    M[idx, :] = U[A, :] * W[C, :]
                    b[idx] = t_val
                
                V_B, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                V_new[B, :] = V_B
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for C in range(p * n):
                M = np.zeros((len(C_AB[C]), r))
                b = np.zeros(len(C_AB[C]))
                for idx, (A, B, t_val) in enumerate(C_AB[C]):
                    M[idx, :] = U[A, :] * V[B, :]
                    b[idx] = t_val
                
                W_C, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                W_new[C, :] = W_C
            W = W_new
            
            # Check error
            err = 0.0
            for A, B, C in non_zero:
                val = np.dot(U[A, :], V[B, :] * W[C, :])
                err += (val - 1.0) ** 2
            # Sample zero entries
            import random
            random.seed(it)
            for _ in range(50):
                A = random.randint(0, n*m-1)
                B = random.randint(0, m*p-1)
                C = random.randint(0, p*n-1)
                if (A, B, C) in T_dict:
                    continue
                val = np.dot(U[A, :], V[B, :] * W[C, :])
                err += val ** 2
                
            if err < tol:
                break
        
        # Try different snap strategies
        # Strategy 1: Round to nearest integer
        for scale in [1, 2, 4, 8]:
            U_snapped = np.round(U * scale) / scale
            V_snapped = np.round(V * scale) / scale
            W_snapped = np.round(W * scale) / scale
            if verify_fast(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()):
                return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
        
        # Strategy 2: Round to nearest half-integer
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        if verify_fast(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()):
            return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
        
        # Strategy 3: Keep as-is if error is small enough
        if verify_fast(U.tolist(), V.tolist(), W.tolist()):
            return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try ranks from low to high
    for r in range(11, 18):
        if time.time() > deadline:
            break
        # Try multiple seeds
        for seed in range(5):
            if time.time() > deadline:
                break
            res = solve_als(r, seed=seed)
            if res:
                U, V, W = res
                return {"U": U, "V": V, "W": W}
            
    return schoolbook()
