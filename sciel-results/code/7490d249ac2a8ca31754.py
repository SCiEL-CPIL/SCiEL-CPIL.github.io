# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    schoolbook_rank = n * m * p  # 54
    
    # Non-zero entries of the tensor
    non_zeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                non_zeros.append((a, b, c))
    
    nz_a = np.array([x[0] for x in non_zeros])
    nz_b = np.array([x[1] for x in non_zeros])
    nz_c = np.array([x[2] for x in non_zeros])
    
    # Precompute index structures for efficient updates
    # For each a, list of (b,c) pairs where tensor has non-zero
    a_to_bc = {}
    b_to_ac = {}
    c_to_ab = {}
    for a, b, c in non_zeros:
        if a not in a_to_bc:
            a_to_bc[a] = []
        a_to_bc[a].append((b, c))
        if b not in b_to_ac:
            b_to_ac[b] = []
        b_to_ac[b].append((a, c))
        if c not in c_to_ab:
            c_to_ab[c] = []
        c_to_ab[c].append((a, b))
    
    def try_decomposition(r, num_restarts=20, max_iter=300, tol=1e-10):
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        start_time = time.time()
        time_limit = 10  # seconds per rank
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Use structured initialization based on schoolbook
            np.random.seed(restart * 7919 + r * 104729)
            
            # Initialize with small random perturbations around a structured basis
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # For the first min(r, 54) columns, use schoolbook-like structure
            num_struct = min(r, 54)
            for col in range(num_struct):
                # Map column to (i,j,k)
                i = col // (m * p)
                j = (col // p) % m
                k = col % p
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
            
            # Add random components for remaining columns
            if r > num_struct:
                U[:, num_struct:] = np.random.randn(n * m, r - num_struct) * 0.1
                V[:, num_struct:] = np.random.randn(m * p, r - num_struct) * 0.1
                W[:, num_struct:] = np.random.randn(p * n, r - num_struct) * 0.1
            
            # Also add small random noise to structured part
            if num_struct > 0:
                U[:, :num_struct] += np.random.randn(n * m, num_struct) * 0.05
                V[:, :num_struct] += np.random.randn(m * p, num_struct) * 0.05
                W[:, :num_struct] += np.random.randn(p * n, num_struct) * 0.05
            
            for it in range(max_iter):
                if time.time() - start_time > time_limit - 0.5:
                    break
                
                # Update U: solve for U given V, W
                D_U = np.einsum('bd, cd -> bcd', V, W)  # (m*p, p*n, r)
                
                U_new = np.zeros((n * m, r))
                for a in range(n * m):
                    if a in a_to_bc:
                        rows = []
                        rhs = []
                        for (b, c) in a_to_bc[a]:
                            rows.append(D_U[b, c, :])
                            rhs.append(1.0)
                        A = np.array(rows)
                        b_vec = np.array(rhs)
                        try:
                            U_new[a] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
                        except:
                            U_new[a] = U[a]
                    else:
                        U_new[a] = U[a]
                U = U_new
                
                # Update V
                D_V = np.einsum('ad, cd -> acd', U, W)  # (n*m, p*n, r)
                
                V_new = np.zeros((m * p, r))
                for b in range(m * p):
                    if b in b_to_ac:
                        rows = []
                        rhs = []
                        for (a, c) in b_to_ac[b]:
                            rows.append(D_V[a, c, :])
                            rhs.append(1.0)
                        A = np.array(rows)
                        b_vec = np.array(rhs)
                        try:
                            V_new[b] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
                        except:
                            V_new[b] = V[b]
                    else:
                        V_new[b] = V[b]
                V = V_new
                
                # Update W
                D_W = np.einsum('ad, bd -> abd', U, V)  # (n*m, m*p, r)
                
                W_new = np.zeros((p * n, r))
                for c in range(p * n):
                    if c in c_to_ab:
                        rows = []
                        rhs = []
                        for (a, b) in c_to_ab[c]:
                            rows.append(D_W[a, b, :])
                            rhs.append(1.0)
                        A = np.array(rows)
                        b_vec = np.array(rhs)
                        try:
                            W_new[c] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
                        except:
                            W_new[c] = W[c]
                    else:
                        W_new[c] = W[c]
                W = W_new
                
                # Check error
                U_A = U[nz_a, :]
                V_B = V[nz_b, :]
                W_C = W[nz_c, :]
                Prods = U_A * V_B * W_C
                Sums = Prods.sum(axis=1)
                err = np.sum((Sums - 1.0) ** 2)
                
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if err < tol:
                    break
                
                # Early stop if not improving
                if it > 50 and err > best_err * 1.01:
                    break
        
        return best_err, best_U, best_V, best_W
    
    start_time = time.time()
    time_limit = 140  # Safety margin before 150s kill
    
    # Try ranks from 53 down to 50
    for r in range(schoolbook_rank - 1, schoolbook_rank - 5, -1):
        if time.time() - start_time > time_limit:
            break
        
        remaining = time_limit - (time.time() - start_time)
        if remaining < 5:
            break
        
        # Allocate time based on rank
        time_per_rank = min(25, remaining - 2)
        if time_per_rank < 3:
            break
        
        err, U, V, W = try_decomposition(r, num_restarts=30, max_iter=200, tol=1e-10)
        
        if err < 1e-8:
            # Round to nearest 0.5
            U_rounded = [[round(U[a, d] * 2) / 2 for d in range(r)] for a in range(n * m)]
            V_rounded = [[round(V[b, d] * 2) / 2 for d in range(r)] for b in range(m * p)]
            W_rounded = [[round(W[c, d] * 2) / 2 for d in range(r)] for c in range(p * n)]
            
            # Verify
            verified = True
            for a, b, c in non_zeros:
                val = sum(U_rounded[a][d] * V_rounded[b][d] * W_rounded[c][d] for d in range(r))
                if abs(val - 1.0) > 1e-6:
                    verified = False
                    break
            
            if verified:
                return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
    
    # Return schoolbook
    rank = schoolbook_rank
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
# EVOLVE-BLOCK-END
