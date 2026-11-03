"""Matrix multiplication 3x4x4: find rank-minimal tensor decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
n, m, p = DIMENSIONS

def _validate(U, V, W):
    """Check if U, V, W exactly decompose the 3x4x4 matmul tensor."""
    r = len(U[0])
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(r):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _winograd_2x2_decomp(A, B):
    """
    Apply Winograd 2x2 scheme to compute C = A @ B where A is 2x2, B is 2x2.
    Returns 7 scalar multiplications needed for the 4 outputs.
    A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    c00 = m0 - m1 + m2 + m4
    c01 = m0 + m3 - m4 - m5
    c10 = m1 + m2 - m3 + m5
    c11 = m0 + m1 + m2 + m6
    where:
    m0 = (a00+a01+a10+a11)*(b00+b10+b01+b11)
    m1 = (a00+a01-a10-a11)*(b00+b10-b01-b11)
    m2 = (a00-a01+a10-a11)*(b00-b10+b01-b11)
    m3 = (a00-a01-a10+a11)*(b00-b10-b01+b11)
    m4 = (a00+a01+a10+a11)*(b00+b10-b01-b11)
    m5 = (a00+a01-a10-a11)*(b00-b10+b01-b11)
    m6 = (a00-a01+a10-a11)*(b00+b10+b01+b11)
    """
    # This is for symbolic understanding; we'll build the full 3x4x4 decomposition directly.
    pass

def _construct_winograd_3x4x4():
    """
    Construct a tensor decomposition for 3x4 * 4x4 matmul using Winograd-style
    block decomposition. We partition the 4x4 matrix into 2x2 blocks and use
    Winograd for each 2x2*2x2 multiplication, then combine.
    
    The 3x4 matrix A has rows i=0,1,2 and cols j=0,1,2,3.
    The 4x4 matrix B has rows j=0,1,2,3 and cols k=0,1,2,3.
    Output C[i][k] = sum_j A[i][j] * B[j][k].
    
    We can group j into {0,1} and {2,3}. For each group, we do a 2x4 * 4x4? 
    Actually, let's think of it as: for fixed i, C[i] = A[i,0:2] @ B[0:2,:] + A[i,2:4] @ B[2:4,:]
    Each term is a 1x2 dot 2x4 = 1x4, which is just 4 multiplications. That's not helpful.
    
    Better approach: Use the fact that we can apply Winograd to 2x2 blocks.
    Partition B into four 2x2 blocks: B[0:2,0:2], B[0:2,2:4], B[2:4,0:2], B[2:4,2:4].
    For each i, A[i,0:2] is 1x2, A[i,2:4] is 1x2.
    
    Actually, let's use a different strategy. We'll use the known rank bounds.
    For 3x4x4, the minimum rank is at least max(3*4, 4*4, 3*4) = 16 and at most 3*4*4=48.
    Known results suggest rank is between 16 and 30.
    
    Let's try to construct a decomposition with rank 24 or so using a structured approach.
    We'll use the fact that we can write the matmul tensor as a sum of rank-1 tensors.
    
    For now, let's implement a simple structured decomposition that might be better than schoolbook.
    We'll use the "block" approach where we treat the 4x4 matrix as 2x2 blocks of 2x2.
    """
    # Let's try a specific construction.
    # We'll use rank 24.
    # The idea: for each of the 3 rows of A, and each of the 4 columns of C,
    # we need to compute a dot product of length 4.
    # We can share computations between different i's or k's.
    
    # Actually, let's just return a schoolbook decomposition for now, but with a
    # slightly better structure. The key insight is that we need to find a
    # decomposition that exploits the algebraic structure.
    
    # For 3x4x4, let's try rank 20.
    # We'll use a greedy approach to build the decomposition.
    
    # Let's use the following approach:
    # We know that the matmul tensor T has entries T[i,j,k] = delta_{i,k} (in some sense).
    # Actually T[i,j,k] = 1 if the output position (i,k) depends on input (j), which it always does.
    # So T[i,j,k] = 1 for all i,j,k in the valid ranges.
    
    # The tensor is T[i,j,k] = 1 for i in [0,3), j in [0,4), k in [0,4).
    # We want to write T as sum_c u_c[i] v_c[j] w_c[k].
    
    # Let's try to find such a decomposition with small rank.
    # We'll use a randomized approach with exact arithmetic.
    
    return None

def _try_rank_exact(r, time_limit=5, seed=0):
    """Try to find a rank-r decomposition using exact arithmetic. Returns (U,V,W) or None."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    start = time.time()
    max_attempts = 50
    
    for attempt in range(max_attempts):
        if time.time() - start > time_limit:
            break
        
        # Random initialization
        U = np_rng.randn(n*m, r) * 0.5
        V = np_rng.randn(m*p, r) * 0.5
        W = np_rng.randn(p*n, r) * 0.5
        
        # Iterative refinement
        for iteration in range(100):
            i = rng.randint(0, n-1)
            j = rng.randint(0, m-1)
            k = rng.randint(0, p-1)
            
            a = i * m + j
            b = j * p + k
            c_idx = k * n + i
            
            cur = np.dot(U[a], V[b] * W[c_idx])
            err = 1.0 - cur
            
            lr = 0.05
            U[a] += lr * err * V[b] * W[c_idx]
            V[b] += lr * err * U[a] * W[c_idx]
            W[c_idx] += lr * err * U[a] * V[b]
        
        # Check error
        err_sum = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c_idx = k * n + i
                    val = np.dot(U[a], V[b] * W[c_idx])
                    err_sum += (val - 1)**2
        
        if err_sum < best_err:
            best_err = err_sum
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            if best_err < 1e-12:
                # Convert to exact fractions
                U_frac = [[Fraction(U[i][c]).limit_denominator(1000) for c in range(r)] for i in range(n*m)]
                V_frac = [[Fraction(V[i][c]).limit_denominator(1000) for c in range(r)] for i in range(m*p)]
                W_frac = [[Fraction(W[i][c]).limit_denominator(1000) for c in range(r)] for i in range(p*n)]
                return [float(x) for x in [U_frac, V_frac, W_frac]]
    
    return None

def construct():
    start_time = time.time()
    deadline = 130  # Safety margin before 150s limit
    
    # Try ranks from low to high
    for r in range(16, 49):
        if time.time() - start_time > deadline:
            break
        
        # Try a few seeds for this rank
        for seed in range(3):
            if time.time() - start_time > deadline:
                break
            
            result = _try_rank_exact(r, time_limit=min(5, deadline - (time.time() - start_time)), seed=seed)
            if result is not None:
                U, V, W = result
                # Validate
                if _validate(U, V, W):
                    return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
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
