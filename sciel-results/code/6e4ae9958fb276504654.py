import numpy as np
import time
import scipy.optimize
from fractions import Fraction

DIMENSIONS = [3, 3, 4]
n, m, p = DIMENSIONS

def _verify_decomposition(U, V, W, n, m, p, tol=1e-12):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    R = U.shape[1]
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    U_reshaped = U.T.reshape(R, n, m)
    V_reshaped = V.T.reshape(R, m, p)
    W_reshaped = W.T.reshape(R, p, n)
    
    T_computed = np.einsum('rij,rjk,rki->ikj', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, p, m))
    
    return np.allclose(T_computed, T_target, atol=tol)

def _verify_exact_fraction(U, V, W, n, m, p):
    """Verify using exact fractions."""
    R = U.shape[1]
    U_frac = [[Fraction(U[i, r]) for r in range(R)] for i in range(n * m)]
    V_frac = [[Fraction(V[i, r]) for r in range(R)] for i in range(m * p)]
    W_frac = [[Fraction(W[i, r]) for r in range(R)] for i in range(p * n)]
    
    # T[i,k,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 if i==j else 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                total = Fraction(0)
                for r in range(R):
                    total += U_frac[i*m+j][r] * V_frac[j*p+k][r] * W_frac[k*n+i][r]
                expected = Fraction(1) if i == j else Fraction(0)
                if total != expected:
                    return False
    return True

def _schoolbook():
    """Return schoolbook decomposition with rank n*m*p = 36."""
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return U, V, W

def _strassen_2x2x2():
    """Strassen decomposition for 2x2 x 2x2 matrix multiplication. Rank 7."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _construct_rank34():
    """
    Construct a rank-34 decomposition using block decomposition:
    - A[0:2,:2] * B[:2,:] -> C[0:2,:] using Strassen (rank 7)
    - A[0:2,2:3] * B[2:3,:] -> C[0:2,:] (rank 8)
    - A[2:3,:3] * B[:3,:] -> C[2:3,:] (rank 12)
    Total: 7 + 8 + 12 = 27? No wait, let me recalculate.
    
    Actually: 
    - Block (i=0,1; j=0,1; k=0,1,2,3): 2x2x4 -> use Strassen for 2x2x2 twice = 14
    - Block (i=0,1; j=2; k=0,1,2,3): 2x1x4 = 8
    - Block (i=2; j=0,1,2; k=0,1,2,3): 1x3x4 = 12
    Total = 14 + 8 + 12 = 34
    """
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Part 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2] using Strassen
    S_U, S_V, S_W = _strassen_2x2x2()
    
    U1 = np.zeros((n * m, 7))
    V1 = np.zeros((m * p, 7))
    W1 = np.zeros((p * n, 7))
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U1[i * m + j, r] = S_U[i * 2 + j, r]
                V1[j * p + i, r] = S_V[i * 2 + j, r]  # Note: V index is j*p+k where j is col of B = row of A
                W1[i * n + j, r] = S_W[i * 2 + j, r]  # W index is k*n+i where k is row of B = col of C
    
    # Wait, I need to be more careful with the indexing.
    # U has shape (n*m, R), indexed by (i, j) where i is row of A, j is col of A
    # V has shape (m*p, R), indexed by (j, k) where j is row of B (= col of A), k is col of B
    # W has shape (p*n, R), indexed by (k, i) where k is row of B (= col of C), i is col of C (= row of A)
    
    # For Strassen 2x2x2: 
    # A is 2x2, B is 2x2, C is 2x2
    # U_2x2[row, col] corresponds to A[row, col]
    # V_2x2[row, col] corresponds to B[row, col]
    # W_2x2[row, col] corresponds to C[row, col]
    
    # In the full problem:
    # A[0:2, 0:2] -> U indices: (0,0), (0,1), (1,0), (1,1) -> U[0], U[1], U[3], U[4]
    # B[0:2, 0:2] -> V indices: (0,0), (0,1), (1,0), (1,1) -> V[0], V[1], V[4], V[5]
    # C[0:2, 0:2] -> W indices: (0,0), (0,1), (1,0), (1,1) -> W[0], W[n], W[n+0], W[n+1]
    
    # Let me redo this more carefully
    U1 = np.zeros((n * m, 7))
    V1 = np.zeros((m * p, 7))
    W1 = np.zeros((p * n, 7))
    
    for r in range(7):
        for i in range(2):  # row of A / col of C
            for j in range(2):  # col of A / row of B
                # U[i*m + j, r] = S_U[i*2 + j, r]
                U1[i * m + j, r] = S_U[i * 2 + j, r]
                # V[j*p + i, r] = S_V[i*2 + j, r]  -- NO! V is indexed by (j, k) where j=row of B, k=col of B
                # In Strassen, B[row, col] where row corresponds to j (col of A), col corresponds to k (col of C)
                # So V[j*p + k, r] = S_B[j*2 + k, r]
                # But S_V is indexed as S_V[row, col] = B[row, col]
                # So for j in 0..1, k in 0..1: V[j*p + k, r] = S_V[j*2 + k, r]
                pass
    
    # Let me just do it element by element
    U1 = np.zeros((n * m, 7))
    V1 = np.zeros((m * p, 7))
    W1 = np.zeros((p * n, 7))
    
    for r in range(7):
        # U: A[i,j] for i=0,1; j=0,1
        for i in range(2):
            for j in range(2):
                U1[i * m + j, r] = S_U[i * 2 + j, r]
        # V: B[j,k] for j=0,1; k=0,1
        for j in range(2):
            for k in range(2):
                V1[j * p + k, r] = S_V[j * 2 + k, r]
        # W: C[k,i] for k=0,1; i=0,1
        for k in range(2):
            for i in range(2):
                W1[k * n + i, r] = S_W[k * 2 + i, r]
    
    U_parts.append(U1)
    V_parts.append(V1)
    W_parts.append(W1)
    
    # Part 2: A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4] using Strassen
    S_U2, S_V2, S_W2 = _strassen_2x2x2()
    
    U2 = np.zeros((n * m, 7))
    V2 = np.zeros((m * p, 7))
    W2 = np.zeros((p * n, 7))
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U2[i * m + j, r] = S_U2[i * 2 + j, r]
        for j in range(2):
            for k in range(2):
                V2[j * p + (k + 2), r] = S_V2[j * 2 + k, r]
        for k in range(2):
            for i in range(2):
                W2[(k + 2) * n + i, r] = S_W2[k * 2 + i, r]
    
    U_parts.append(U2)
    V_parts.append(V2)
    W_parts.append(W2)
    
    # Part 3: A[0:2,2:3] * B[2:3,0:4] -> C[0:2,0:4]
    U3 = np.zeros((n * m, 8))
    V3 = np.zeros((m * p, 8))
    W3 = np.zeros((p * n, 8))
    
    col = 0
    for i in range(2):
        for k in range(4):
            U3[i * m + 2, col] = 1
            V3[2 * p + k, col] = 1
            W3[k * n + i, col] = 1
            col += 1
    
    U_parts.append(U3)
    V_parts.append(V3)
    W_parts.append(W3)
    
    # Part 4: A[2:3,0:3] * B[0:3,0:4] -> C[2:3,0:4]
    U4 = np.zeros((n * m, 12))
    V4 = np.zeros((m * p, 12))
    W4 = np.zeros((p * n, 12))
    
    col = 0
    for j in range(3):
        for k in range(4):
            U4[2 * m + j, col] = 1
            V4[j * p + k, col] = 1
            W4[k * n + 2, col] = 1
            col += 1
    
    U_parts.append(U4)
    V_parts.append(V4)
    W_parts.append(W4)
    
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    return U, V, W

def _residual(x, n, m, p, R):
    """Compute residual for optimization."""
    size_U = n * m * R
    size_V = m * p * R
    U = x[:size_U].reshape(n * m, R)
    V = x[size_U:size_U + size_V].reshape(m * p, R)
    W = x[size_U + size_V:].reshape(p * n, R)
    
    U_reshaped = U.T.reshape(R, n, m)
    V_reshaped = V.T.reshape(R, m, p)
    W_reshaped = W.T.reshape(R, p, n)
    
    T_computed = np.einsum('rij,rjk,rki->ikj', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, p, m))
    
    return (T_computed - T_target).flatten()

def _try_rank_optimize(R, seed=0, max_iter=200):
    """Try to find a rank-R decomposition using optimization."""
    np.random.seed(seed)
    
    # Initialize with random values
    U = np.random.randn(n * m, R) * 0.5
    V = np.random.randn(m * p, R) * 0.5
    W = np.random.randn(p * n, R) * 0.5
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = scipy.optimize.least_squares(
            _residual, x0, args=(n, m, p, R),
            max_nfev=max_iter,
            xtol=1e-14, ftol=1e-14, gtol=1e-14
        )
        
        if np.linalg.norm(result.fun) < 1e-10:
            size_U = n * m * R
            size_V = m * p * R
            U_opt = result.x[:size_U].reshape(n * m, R)
            V_opt = result.x[size_U:size_U + size_V].reshape(m * p, R)
            W_opt = result.x[size_U + size_V:].reshape(p * n, R)
            
            # Round to half-integers
            U_round = np.round(U_opt * 2) / 2
            V_round = np.round(V_opt * 2) / 2
            W_round = np.round(W_opt * 2) / 2
            
            if _verify_decomposition(U_round, V_round, W_round, n, m, p, tol=1e-10):
                return U_round, V_round, W_round
    except Exception:
        pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Get the rank-34 solution as baseline
    try:
        U34, V34, W34 = _construct_rank34()
        if _verify_decomposition(U34, V34, W34, n, m, p, tol=1e-10):
            best_U, best_V, best_W = U34, V34, W34
            best_rank = 34
        else:
            U34, V34, W34 = _schoolbook()
            best_U, best_V, best_W = U34, V34, W34
            best_rank = 36
    except Exception:
        U34, V34, W34 = _schoolbook()
        best_U, best_V, best_W = U34, V34, W34
        best_rank = 36
    
    # Try to find lower rank solutions
    # Try rank 33, 32, 31, 30 with multiple seeds
    for target_rank in [33, 32, 31, 30]:
        if time.time() > deadline - 20:
            break
        
        found = False
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            res = _try_rank_optimize(target_rank, seed=seed, max_iter=150)
            if res is not None:
                U_opt, V_opt, W_opt = res
                best_U, best_V, best_W = U_opt, V_opt, W_opt
                best_rank = target_rank
                found = True
                break
        
        if found:
            break  # Found a lower rank, stop searching
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p, tol=1e-10):
        # Fallback to schoolbook
        U_sb, V_sb, W_sb = _schoolbook()
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = 36
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
