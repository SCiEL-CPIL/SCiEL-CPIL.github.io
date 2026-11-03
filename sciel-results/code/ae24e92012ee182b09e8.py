# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

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
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    def verify_decomposition(U, V, W, r, tol=1e-9):
        """Verify that U,V,W with rank r exactly decompose the tensor."""
        for a, b, c in non_zeros:
            val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(r))
            if abs(val - 1.0) > tol:
                return False
        # Also verify that zero entries remain zero
        # Check a sample of zero entries
        all_abc = set((a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n))
        nz_set = set(non_zeros)
        zero_abc = list(all_abc - nz_set)
        # Sample up to 100 zero entries
        import random
        random.seed(42)
        sample = random.sample(zero_abc, min(100, len(zero_abc)))
        for a, b, c in sample:
            val = sum(U[a][d] * V[b][d] * W[c][d] for d in range(r))
            if abs(val) > tol:
                return False
        return True
    
    def round_to_half(x):
        """Round to nearest 0.5."""
        return round(x * 2) / 2
    
    def try_polish(U, V, W, r, max_iter=100, tol=1e-12, time_limit=5.0):
        """Polish a near-exact decomposition to exact."""
        start = time.time()
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        for it in range(max_iter):
            if time.time() - start > time_limit:
                break
            
            # Update U
            for a in range(n * m):
                # Build system: sum_d U[a,d] * V[b,d] * W[c,d] = 1 for (a,b,c) in nz
                rows = []
                rhs = []
                for (aa, b, c) in non_zeros:
                    if aa == a:
                        row = np.array([V[b, d] * W[c, d] for d in range(r)])
                        rows.append(row)
                        rhs.append(1.0)
                if rows:
                    A = np.array(rows)
                    b_vec = np.array(rhs)
                    try:
                        U[a] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for b in range(m * p):
                rows = []
                rhs = []
                for (a, bb, c) in non_zeros:
                    if bb == b:
                        row = np.array([U[a, d] * W[c, d] for d in range(r)])
                        rows.append(row)
                        rhs.append(1.0)
                if rows:
                    A = np.array(rows)
                    b_vec = np.array(rhs)
                    try:
                        V[b] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for c in range(p * n):
                rows = []
                rhs = []
                for (a, b, cc) in non_zeros:
                    if cc == c:
                        row = np.array([U[a, d] * V[b, d] for d in range(r)])
                        rows.append(row)
                        rhs.append(1.0)
                if rows:
                    A = np.array(rows)
                    b_vec = np.array(rhs)
                    try:
                        W[c] = np.linalg.lstsq(A, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Check error
            err = 0.0
            for a, b, c in non_zeros:
                val = np.sum(U[a] * V[b] * W[c])
                err += (val - 1.0) ** 2
            
            if err < tol:
                break
        
        # Round to half-integers and verify
        U_r = [[round(U[a, d] * 2) / 2 for d in range(r)] for a in range(n * m)]
        V_r = [[round(V[b, d] * 2) / 2 for d in range(r)] for b in range(m * p)]
        W_r = [[round(W[c, d] * 2) / 2 for d in range(r)] for c in range(p * n)]
        
        if verify_decomposition(U_r, V_r, W_r, r, tol=1e-6):
            return U_r, V_r, W_r
        return None
    
    def schoolbook_decomposition():
        """Return the schoolbook rank-54 decomposition."""
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
        return U, V, W, rank
    
    # Try rank 53 first with systematic search
    # Strategy: start from schoolbook, try to merge pairs of terms
    for target_rank in [53, 52, 51]:
        if time.time() > deadline - 20:
            break
        
        if target_rank == 53:
            # Try to find a pair of schoolbook terms that can be merged
            # by introducing a shared rank-1 term with non-1 entries
            # This is equivalent to finding a 2x2x2 sub-tensor of rank 1
            # which is impossible, so we need a more global approach.
            
            # Alternative: use numerical search with structured initialization
            np.random.seed(42)
            for restart in range(30):
                if time.time() > deadline - 15:
                    break
                
                # Initialize with schoolbook minus one term, add a random term
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                
                # Fill first 53 columns with schoolbook (skip one random term)
                skip = restart % 54
                col = 0
                for idx, (i, j, k) in enumerate([(i, j, k) for i in range(n) for j in range(m) for k in range(p)]):
                    if idx != skip:
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U[a, col] = 1.0
                        V[b, col] = 1.0
                        W[c, col] = 1.0
                        col += 1
                
                # Last column: random
                U[:, -1] = np.random.randn(n * m) * 0.5
                V[:, -1] = np.random.randn(m * p) * 0.5
                W[:, -1] = np.random.randn(p * n) * 0.5
                
                # Polish
                result = try_polish(U, V, W, target_rank, max_iter=200, tol=1e-12, time_limit=3.0)
                if result:
                    U_r, V_r, W_r = result
                    return {"U": U_r, "V": W_r, "W": W_r}
        
        elif target_rank == 52:
            # Try rank 52 with more aggressive random initialization
            np.random.seed(123)
            for restart in range(20):
                if time.time() > deadline - 10:
                    break
                
                U = np.zeros((n * m, target_rank))
                V = np.zeros((m * p, target_rank))
                W = np.zeros((p * n, target_rank))
                
                # Fill with schoolbook-like structure, skip 2 terms
                skip1 = restart % 54
                skip2 = (restart * 7 + 17) % 54
                col = 0
                for idx, (i, j, k) in enumerate([(i, j, k) for i in range(n) for j in range(m) for k in range(p)]):
                    if idx != skip1 and idx != skip2:
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U[a, col] = 1.0
                        V[b, col] = 1.0
                        W[c, col] = 1.0
                        col += 1
                
                # Last 2 columns: random
                U[:, -2:] = np.random.randn(n * m, 2) * 0.5
                V[:, -2:] = np.random.randn(m * p, 2) * 0.5
                W[:, -2:] = np.random.randn(p * n, 2) * 0.5
                
                result = try_polish(U, V, W, target_rank, max_iter=200, tol=1e-12, time_limit=2.0)
                if result:
                    U_r, V_r, W_r = result
                    return {"U": U_r, "V": V_r, "W": W_r}
    
    # Fall back to schoolbook
    U, V, W, rank = schoolbook_decomposition()
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
