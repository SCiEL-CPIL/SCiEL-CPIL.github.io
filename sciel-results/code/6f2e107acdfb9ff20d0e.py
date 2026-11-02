"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n=2, m=2, p=4):
    """Verify that the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    nm, r = U.shape
    mp, _ = V.shape
    pn, _ = W.shape
    # Build tensor T[i,j,k] where i in [0,nm), j in [0,mp), k in [0,pn)
    # T = sum_r U[i,r] * V[j,r] * W[k,r]
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    # Expected: for matrix multiply A(n,m) @ B(m,p) -> C(n,p)
    # A index: i = a*m + b  (a in [0,n), b in [0,m))
    # B index: j = b*p + c  (b in [0,m), c in [0,p))
    # C index: k = c*n + a  (c in [0,p), a in [0,n))
    # C[a,c] = sum_b A[a,b] * B[b,c]
    # So T[i,j,k] should be 1 if i=a*m+b, j=b*p+c, k=c*n+a, else 0
    expected = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                expected[i, j, k] = 1.0
    return np.allclose(T, expected, atol=1e-10)

def _strassen_rank14():
    """Return the known rank-14 Strassen-based decomposition."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Left half
    c = 0; U[0][c]=1; U[3][c]=1; V[0][c]=1; V[5][c]=1; W[0][c]=1; W[3][c]=1
    c = 1; U[2][c]=1; U[3][c]=1; V[0][c]=1; W[1][c]=1; W[3][c]=-1
    c = 2; U[0][c]=1; V[1][c]=1; V[5][c]=-1; W[2][c]=1; W[3][c]=1
    c = 3; U[3][c]=1; V[4][c]=1; V[0][c]=-1; W[0][c]=1; W[1][c]=1
    c = 4; U[0][c]=1; U[1][c]=1; V[5][c]=1; W[0][c]=-1; W[2][c]=1
    c = 5; U[2][c]=1; U[0][c]=-1; V[0][c]=1; V[1][c]=1; W[3][c]=1
    c = 6; U[1][c]=1; U[3][c]=-1; V[4][c]=1; V[5][c]=1; W[0][c]=1
    
    # Right half
    c = 7; U[0][c]=1; U[3][c]=1; V[2][c]=1; V[7][c]=1; W[4][c]=1; W[7][c]=1
    c = 8; U[2][c]=1; U[3][c]=1; V[2][c]=1; W[5][c]=1; W[7][c]=-1
    c = 9; U[0][c]=1; V[3][c]=1; V[7][c]=-1; W[6][c]=1; W[7][c]=1
    c = 10; U[3][c]=1; V[6][c]=1; V[2][c]=-1; W[4][c]=1; W[5][c]=1
    c = 11; U[0][c]=1; U[1][c]=1; V[7][c]=1; W[4][c]=-1; W[6][c]=1
    c = 12; U[2][c]=1; U[0][c]=-1; V[2][c]=1; V[3][c]=1; W[7][c]=1
    c = 13; U[1][c]=1; U[3][c]=-1; V[6][c]=1; V[7][c]=1; W[4][c]=1
    
    return U, V, W

def _try_rank_reduction(U, V, W, target_rank, deadline):
    """Try to reduce rank from current to target_rank using random perturbation and rounding."""
    n, m, p = 2, 2, 4
    cur_rank = len(U[0])
    if target_rank >= cur_rank:
        return None
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try: drop one column at a time, see if we can compensate
    # More systematically: try all pairs of columns to merge
    for drop in range(cur_rank - target_rank + 1):
        if time.time() > deadline:
            return None
        # Try dropping column 'drop' and redistributing
        # This is a heuristic - try small perturbations
        for trial in range(50):
            if time.time() > deadline:
                return None
            U_try = U_arr.copy()
            V_try = V_arr.copy()
            W_try = W_arr.copy()
            # Zero out column 'drop'
            U_try[:, drop] = 0
            V_try[:, drop] = 0
            W_try[:, drop] = 0
            # Check if still valid (it won't be, but we use it as a starting point)
            # Instead, let's try a different approach:
            # Start with random half-integer matrices of target rank and verify
            pass
    
    return None

def _random_search_rank(r, deadline, seed=42):
    """Random search for a decomposition of rank r with half-integer entries."""
    n, m, p = 2, 2, 4
    nm, mp, pn = n*m, m*p, p*n
    np.random.seed(seed)
    
    best_err = float('inf')
    best = None
    
    for trial in range(2000):
        if time.time() > deadline:
            break
        
        # Random half-integer entries: values in {-1, -0.5, 0, 0.5, 1}
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(nm, r))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(mp, r))
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(pn, r))
        
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        
        # Expected tensor
        expected = np.zeros((nm, mp, pn))
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    i = a * m + b
                j = b * p + c
                k = c * n + a
                expected[i, j, k] = 1.0
        
        err = np.abs(T - expected).max()
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        
        if err < 1e-10:
            return U, V, W
    
    return None

def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Start with known rank-14 solution
    U14, V14, W14 = _strassen_rank14()
    
    # Try to find rank-13 or lower
    for target_rank in [13, 12, 11]:
        if time.time() > deadline:
            break
        
        # Method 1: Random search with half-integer entries
        result = _random_search_rank(target_rank, deadline, seed=42 + target_rank)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Method 2: Start from rank-14, try to merge columns
        # Try removing one column and adjusting others
        cur_rank = 14
        if target_rank == cur_rank - 1:
            U_arr = np.array(U14, dtype=float)
            V_arr = np.array(V14, dtype=float)
            W_arr = np.array(W14, dtype=float)
            
            # Try each column as candidate to remove
            for drop_col in range(cur_rank):
                if time.time() > deadline:
                    break
                U_try = U_arr.copy()
                V_try = V_arr.copy()
                W_try = W_arr.copy()
                
                # Zero out the dropped column
                U_try[:, drop_col] = 0
                V_try[:, drop_col] = 0
                W_try[:, drop_col] = 0
                
                # Try to compensate by slightly modifying other columns
                # This is a local search - try small perturbations
                for pert_trial in range(100):
                    if time.time() > deadline:
                        break
                    
                    U_p = U_try.copy()
                    V_p = V_try.copy()
                    W_p = W_try.copy()
                    
                    # Add small random half-integer perturbations to non-zero columns
                    for col in range(cur_rank):
                        if col == drop_col:
                            continue
                        if np.random.random() < 0.3:
                            U_p[:, col] += np.random.choice([-0.5, 0.5])
                        if np.random.random() < 0.3:
                            V_p[:, col] += np.random.choice([-0.5, 0.5])
                        if np.random.random() < 0.3:
                            W_p[:, col] += np.random.choice([-0.5, 0.5])
                    
                    # Round to nearest half-integer
                    U_p = np.round(U_p * 2) / 2
                    V_p = np.round(V_p * 2) / 2
                    W_p = np.round(W_p * 2) / 2
                    
                    T = np.einsum('ir,jr,kr->ijk', U_p, V_p, W_p)
                    expected = np.zeros((4, 8, 8))
                    for a in range(2):
                        for b in range(2):
                            for c in range(4):
                                expected[a*2+b, b*4+c, c*2+a] = 1.0
                    
                    if np.allclose(T, expected, atol=1e-10):
                        return {"U": U_p.tolist(), "V": V_p.tolist(), "W": W_p.tolist()}
    
    # Fallback: return rank-14
    return {"U": U14, "V": V14, "W": W14}
# EVOLVE-BLOCK-END
