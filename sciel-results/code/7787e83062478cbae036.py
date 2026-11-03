import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = 130  # Safety margin before 150s limit
    
    # Build the tensor T[i,j,k] = 1 if (i,j) is in A, (j,k) is in B, and contributes to C[i,k]
    # Actually, for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is defined such that T[(i,j), (j,k), (i,k)] = 1
    
    # Let's use a different indexing:
    # A has indices (i, j) where i in [0,n), j in [0,m)
    # B has indices (j, k) where j in [0,m), k in [0,p)
    # C has indices (i, k) where i in [0,n), k in [0,p)
    
    # The multiplication tensor has entries T[a_idx, b_idx, c_idx]
    # where a_idx = i*m + j, b_idx = j*p + k, c_idx = i*p + k
    # T[a_idx, b_idx, c_idx] = 1 if the indices are consistent
    
    def a_to_ij(a_idx):
        return a_idx // m, a_idx % m
    
    def b_to_jk(b_idx):
        return b_idx // p, b_idx % p
    
    def c_to_ik(c_idx):
        return c_idx // p, c_idx % p
    
    # Build the tensor as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = i * p + k
                T[a_idx, b_idx, c_idx] = 1
    
    # Trivial rank is n*m*p = 32
    # Strassen gives 28
    
    # Try to find a rank-27 decomposition using randomized search
    # The idea: if we have a rank-28 decomposition, we can try to find a linear combination
    # of the rank-28 terms that gives a valid rank-27 decomposition
    
    # First, get the rank-28 Strassen decomposition
    rank_28 = _strassen_28()
    
    # Try to reduce to rank 27
    for trial in range(10):
        if time.time() > deadline:
            break
        result = _try_rank_27(T, rank_28, trial)
        if result is not None:
            return result
    
    # Try rank 26
    for trial in range(5):
        if time.time() > deadline:
            break
        result = _try_rank_26(T, rank_28, trial)
        if result is not None:
            return result
    
    # Return the best we found (rank 28)
    return rank_28

def _strassen_28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    strassen = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),   # M1
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),        # M2
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),        # M3
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),        # M4
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),        # M5
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),        # M6
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),        # M7
    ]
    
    def emit_block(a_cs, b_rs, b_cs, c_cs, off):
        for s in range(7):
            col = off + s
            uc, vc, wc = strassen[s]
            for li, c in uc.items():
                r, cc = li // 2, li % 2
                U[r * m + a_cs + cc][col] += c
            for li, c in vc.items():
                r, cc = li // 2, li % 2
                V[(b_rs + r) * p + b_cs + cc][col] += c
            for li, c in wc.items():
                r, cc = li // 2, li % 2
                W[(c_cs + cc) * n + r][col] += c
    
    emit_block(0, 0, 0, 0, 0)
    emit_block(0, 0, 2, 2, 7)
    emit_block(2, 2, 0, 0, 14)
    emit_block(2, 2, 2, 2, 21)
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    
    # Build the tensor from the decomposition
    # T_decomp[a_idx, b_idx, c_idx] = sum_r U[a_idx,r] * V[b_idx,r] * W[c_idx,r]
    T_decomp = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    
    for r in range(rank):
        u_col = np.array([U[i][r] for i in range(n*m)])
        v_col = np.array([V[j][r] for j in range(m*p)])
        w_col = np.array([W[k][r] for k in range(p*n)])
        
        # Outer product
        for i in range(n*m):
            if u_col[i] != 0:
                for j in range(m*p):
                    if v_col[j] != 0:
                        for k in range(p*n):
                            T_decomp[i, j, k] += u_col[i] * v_col[j] * w_col[k]
    
    # Check against the true tensor
    T_true = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = i * p + k
                T_true[a_idx, b_idx, c_idx] = 1
    
    # Check if they match (allowing for scaling)
    # The decomposition should give T_true exactly
    if not np.allclose(T_decomp, T_true):
        return False
    
    # Check that entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if val * 2 != int(val * 2):
                    return False
    
    return True

def _try_rank_27(T, rank_28, trial):
    """Try to find a rank-27 decomposition by perturbing the rank-28 one."""
    n, m, p = DIMENSIONS
    np.random.seed(42 + trial)
    
    # Start from rank-28 and try to eliminate one term
    U28 = np.array(rank_28["U"])
    V28 = np.array(rank_28["V"])
    W28 = np.array(rank_28["W"])
    
    # Try each column as a candidate to eliminate
    for col_to_elim in range(28):
        if time.time() > 130:
            return None
        
        # Create rank-27 by removing this column
        mask = [i for i in range(28) if i != col_to_elim]
        U27 = U28[:, mask]
        V27 = V28[:, mask]
        W27 = W28[:, mask]
        
        # Add a small perturbation and try to fix
        # This is a simplified approach - in practice we'd need to solve a system
        
        # For now, just check if removing any single column still works (unlikely)
        # This is just a placeholder for a more sophisticated search
        
        # Try a randomized local search
        U27 = U27.astype(np.float64)
        V27 = V27.astype(np.float64)
        W27 = W27.astype(np.float64)
        
        # Add small random perturbations
        for _ in range(100):
            if time.time() > 130:
                return None
            
            # Random perturbation
            U27 += np.random.normal(0, 0.01, U27.shape)
            V27 += np.random.normal(0, 0.01, V27.shape)
            W27 += np.random.normal(0, 0.01, W27.shape)
            
            # Check if valid
            if _verify_decomposition(U27.tolist(), V27.tolist(), W27.tolist()):
                # Convert to fractions if needed
                return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    return None

def _try_rank_26(T, rank_28, trial):
    """Try to find a rank-26 decomposition."""
    # Similar approach but more aggressive
    return None
# EVOLVE-BLOCK-END
