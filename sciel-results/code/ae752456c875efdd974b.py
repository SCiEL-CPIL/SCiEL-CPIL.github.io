"""Matrix multiplication 3x4x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n=3, m=4, p=8):
    """Check if the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check all consistent triples sum to 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > 1e-8:
                    return False
    
    # Check inconsistent j
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                if j == j2: continue
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j2*p+k, :] * W[k*n+i, :])
                    if abs(s) > 1e-8:
                        return False
    
    # Check inconsistent k
    for j in range(m):
        for k in range(p):
            for k2 in range(p):
                if k == k2: continue
                for i in range(n):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k2*n+i, :])
                    if abs(s) > 1e-8:
                        return False
    
    # Check inconsistent i
    for i in range(n):
        for i2 in range(n):
            if i == i2: continue
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i2, :])
                    if abs(s) > 1e-8:
                        return False
    
    return True

def _build_strassen_3x3():
    """Build Strassen decomposition for 3x3 matrix multiplication.
    Returns U, V, W of shape (9, r), (9, r), (9, r) where r is the rank.
    For 3x3, we can use a rank-7 or higher decomposition.
    Actually, let's use the standard 3x3 = 3x3*3x3 with Strassen-like approach.
    For simplicity, let's build a 3x3 decomposition using block structure.
    """
    n = 3
    # Schoolbook for 3x3: rank 27
    # We can do better. Let's use a known rank-15 or similar.
    # Actually, for 3x3 the best known is 15 (Strassen-like).
    # Let me construct one systematically.
    
    # For 3x3 matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # 
    # Let's use the following approach: decompose into 2x2 Strassen + 1x1
    # Split into blocks: top-left 2x2, top-right 1x2, bottom-left 2x1, bottom-right 1x1
    # Actually this gets complex. Let me just use schoolbook for now and try to optimize.
    
    # Schoolbook 3x3: rank 27
    rank = 27
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U[i*3+j, col] = 1.0
                V[j*3+k, col] = 1.0
                W[k*3+i, col] = 1.0
                col += 1
    
    return U, V, W

def _build_decomp_3x4x8():
    """Build a decomposition for 3x4x8 matrix multiplication.
    
    Strategy: 
    - The 4 dimension in the middle can be split into 2+2
    - The 8 dimension can be split into 4+4 or 2+2+2+2
    - Use block structure to reduce rank
    
    Let me think about this more carefully.
    C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    
    Split j into j1 (0,1) and j2 (2,3):
    C[i,k] = sum_{j1=0}^{1} A[i,j1]*B[j1,k] + sum_{j2=2}^{3} A[i,j2]*B[j2,k]
    
    For the first part: 3x2 * 2x8 -> 3x8, schoolbook rank = 3*2*8 = 48
    For the second part: 3x2 * 2x8 -> 3x8, schoolbook rank = 3*2*8 = 48
    Total: 96 (same as schoolbook)
    
    Better: use Strassen-like for the 3x3 subproblems.
    Actually, let me try a different approach.
    
    Split the 8 columns into 4+4:
    C[:, 0:4] = A * B[:, 0:4]  (3x4 * 4x4)
    C[:, 4:8] = A * B[:, 4:8]  (3x4 * 4x4)
    
    For each 3x4*4x4: split the 4 into 2+2
    3x4*4x4 = (3x2*2x4 + 3x2*2x4) for each half
    Each 3x2*2x4: schoolbook = 24
    Total per 3x4*4x4: 48
    Total: 96
    
    Hmm, still 96. Let me try to use Strassen for 3x3 parts.
    
    Actually, let me try a completely different approach:
    Use the fact that 3x4x8 can be decomposed using tensor rank methods.
    
    Let me try: rank = 3*4*8 - savings from structure.
    
    Alternative: use the Winograd/Strassen approach for the 4x4 blocks.
    
    Let me try constructing explicitly with a lower rank.
    """
    n, m, p = 3, 4, 8
    
    # Start with schoolbook
    rank0 = n * m * p  # 96
    U = np.zeros((n*m, rank0))
    V = np.zeros((m*p, rank0))
    W = np.zeros((p*n, rank0))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    
    return U, V, W

def _try_optimize(U, V, W, target_rank, n=3, m=4, p=8, max_iters=50, tol=1e-6):
    """Try to optimize a decomposition to achieve target_rank.
    Uses alternating least squares with constraint satisfaction.
    """
    U = U.copy().astype(float)
    V = V.copy().astype(float)
    W = W.copy().astype(float)
    
    for it in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                rows = []
                rhs = []
                # Consistent
                for k in range(p):
                    rows.append(V[j*p+k, :] * W[k*n+i, :])
                    rhs.append(1.0)
                # Inconsistent j
                for j2 in range(m):
                    if j2 == j: continue
                    for k in range(p):
                        rows.append(V[j2*p+k, :] * W[k*n+i, :])
                        rhs.append(0.0)
                # Inconsistent i
                for i2 in range(n):
                    if i2 == i: continue
                    for k in range(p):
                        rows.append(V[j*p+k, :] * W[k*n+i2, :])
                        rhs.append(0.0)
                
                A_mat = np.array(rows)
                b_vec = np.array(rhs)
                try:
                    U[i*m+j, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                rows = []
                rhs = []
                for i in range(n):
                    rows.append(U[i*m+j, :] * W[k*n+i, :])
                    rhs.append(1.0)
                for j2 in range(m):
                    if j2 == j: continue
                    for i in range(n):
                        rows.append(U[i*m+j2, :] * W[k*n+i, :])
                        rhs.append(0.0)
                for k2 in range(p):
                    if k2 == k: continue
                    for i in range(n):
                        rows.append(U[i*m+j, :] * W[k2*n+i, :])
                        rhs.append(0.0)
                
                A_mat = np.array(rows)
                b_vec = np.array(rhs)
                try:
                    V[j*p+k, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                rows = []
                rhs = []
                for j in range(m):
                    rows.append(U[i*m+j, :] * V[j*p+k, :])
                    rhs.append(1.0)
                for k2 in range(p):
                    if k2 == k: continue
                    for j in range(m):
                        rows.append(U[i*m+j, :] * V[j*p+k2, :])
                        rhs.append(0.0)
                for i2 in range(n):
                    if i2 == i: continue
                    for j in range(m):
                        rows.append(U[i2*m+j, :] * V[j*p+k, :])
                        rhs.append(0.0)
                
                A_mat = np.array(rows)
                b_vec = np.array(rhs)
                try:
                    W[k*n+i, :], _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                except:
                    pass
        
        # Check error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1.0)**2
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    if j == j2: continue
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j2*p+k, :] * W[k*n+i, :])
                        err += s**2
        for j in range(m):
            for k in range(p):
                for k2 in range(p):
                    if k == k2: continue
                    for i in range(n):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k2*n+i, :])
                        err += s**2
        for i in range(n):
            for i2 in range(n):
                if i == i2: continue
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i2, :])
                        err += s**2
        
        if err < tol:
            break
    
    return U, V, W, err

def construct():
    n, m, p = 3, 4, 8
    t_start = time.time()
    time_limit = 130  # Safety margin before 150s kill
    
    # Start with schoolbook
    U0, V0, W0 = _build_decomp_3x4x8()
    best_U = U0.copy()
    best_V = V0.copy()
    best_W = W0.copy()
    best_rank = n * m * p  # 96
    
    # Try to find lower rank using structured approaches
    # Approach 1: Try ranks from 90 down to 60
    rng = np.random.RandomState(12345)
    
    target_ranks = list(range(90, 97)) + list(range(80, 90)) + list(range(70, 80))
    
    for target_rank in target_ranks:
        if time.time() - t_start > time_limit:
            break
        if best_rank <= target_rank:
            continue
        
        # Try multiple random initializations
        n_attempts = 5
        for attempt in range(n_attempts):
            if time.time() - t_start > time_limit:
                break
            
            scale = 1.0 / np.sqrt(target_rank)
            U = rng.randn(n * m, target_rank) * scale
            V = rng.randn(m * p, target_rank) * scale
            W = rng.randn(p * n, target_rank) * scale
            
            U, V, W, err = _try_optimize(U, V, W, target_rank, n, m, p, max_iters=20)
            
            if _verify(U, V, W, n, m, p):
                # Round to half-integers
                Ur = np.round(U * 2) / 2
                Vr = np.round(V * 2) / 2
                Wr = np.round(W * 2) / 2
                if _verify(Ur, Vr, Wr, n, m, p):
                    U, V, W = Ur, Vr, Wr
                
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    print(f"Found rank {target_rank}")
                    break  # Move to next lower target
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    # Convert to int where possible
    def _convert(v):
        if abs(v - round(v)) < 1e-9:
            return int(round(v))
        elif abs(v * 2 - round(v * 2)) < 1e-9:
            return (2 * int(round(v * 2))) / 2
        return v
    
    U_out = [[_convert(v) for v in row] for row in U_out]
    V_out = [[_convert(v) for v in row] for row in V_out]
    W_out = [[_convert(v) for v in row] for row in W_out]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
