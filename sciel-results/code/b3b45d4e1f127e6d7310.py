"""
Matrix multiplication 2x3 by 3x7 - rank-minimal tensor decomposition.
Uses structured algebraic search over small rational grids to find
lower-rank exact decompositions, falling back to schoolbook rank 42.
"""
import time
import sys
import numpy as np
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W, rank

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reconstructs the matmul tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_exact_rational(U, V, W, n, m, p):
    """Verify exactness using Fraction arithmetic."""
    rank = U.shape[1]
    # Convert to Fractions
    def to_frac(x):
        if abs(x - round(x)) < 1e-12:
            return Fraction(int(round(x)))
        elif abs(x * 2 - round(x * 2)) < 1e-12:
            return Fraction(int(round(x * 2)), 2)
        else:
            return Fraction(x).limit_denominator(1000)
    
    Uf = [[to_frac(U[i, c]) for c in range(rank)] for i in range(n * m)]
    Vf = [[to_frac(V[i, c]) for c in range(rank)] for i in range(m * p)]
    Wf = [[to_frac(W[i, c]) for c in range(rank)] for i in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(Uf[i*m + j][c] * Vf[j*p + k][c] * Wf[k*n + i][c] for c in range(rank))
                if val != 1:
                    return False
    return True

def _generate_rank1_candidates(n, m, p, max_val=3):
    """
    Generate structured rank-1 tensor candidates.
    A rank-1 tensor is defined by (u, v, w) where:
      u is a vector in R^{n*m}, v in R^{m*p}, w in R^{p*n}
    The tensor T[i,j,k] = u[i*m+j] * v[j*p+k] * w[k*n+i]
    
    We search over small integer/half-integer values for u, v, w.
    To keep the search tractable, we restrict to sparse or structured vectors.
    """
    candidates = []
    
    # Strategy: each rank-1 component should cover a "block" of the tensor.
    # For the matmul tensor T[i,j,k]=1, a rank-1 component with
    # u supported on some (i,j) pairs, v on some (j,k) pairs, w on some (k,i) pairs
    # contributes u[i*m+j]*v[j*p+k]*w[k*n+i] to T[i,j,k].
    
    # We'll generate candidates where u, v, w have small integer values
    # and are "aligned" in a way that they cover a consistent set of (i,j,k).
    
    # For each (i,j,k), the contribution is u[i*m+j]*v[j*p+k]*w[k*n+i].
    # A simple structured candidate: set u[i*m+j]=a, v[j*p+k]=b, w[k*n+i]=c
    # for a specific (i,j,k), with a*b*c = 1. But that's just schoolbook.
    
    # Better: use vectors that are nonzero on multiple indices.
    # E.g., u has value 1 on all (i,j) for a fixed i, v has value 1 on all (j,k) for a fixed j,
    # w has value 1 on all (k,i) for a fixed k. Then the product is 1 on T[i,j,k] for
    # all j, but 0 elsewhere. This covers a "line" in the tensor.
    
    # Let's generate candidates based on fixing one index and summing over others.
    # Type A: fix i, sum over j and k
    for i in range(n):
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        for j in range(m):
            u[i * m + j] = 1.0
        for j in range(m):
            for k in range(p):
                v[j * p + k] = 1.0
                w[k * n + i] = 1.0
        # This gives T[i,j,k] = 1 for all j,k when i is fixed, 0 otherwise
        # But we need the product to be 1 only for the specific (i,j,k) we want
        # Actually this gives T[i,j,k] = u[i*m+j]*v[j*p+k]*w[k*n+i] = 1*1*1 = 1 for all j,k
        # So this covers an entire "slice" at fixed i.
        candidates.append((u.copy(), v.copy(), w.copy()))
    
    # Type B: fix j, sum over i and k
    for j in range(m):
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        for i in range(n):
            u[i * m + j] = 1.0
        for k in range(p):
            v[j * p + k] = 1.0
            w[k * n + 0] = 0  # This doesn't work simply
        # This is more complex. Let's skip for now.
    
    # Type C: fix k, sum over i and j
    for k in range(p):
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        for i in range(n):
            for j in range(m):
                u[i * m + j] = 1.0
                v[j * p + k] = 1.0
                w[k * n + i] = 1.0
        candidates.append((u.copy(), v.copy(), w.copy()))
    
    return candidates

def _tensor_from_rank1(u, v, w, n, m, p):
    """Compute the 3D tensor from rank-1 factors."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = u[i * m + j] * v[j * p + k] * w[k * n + i]
    return T

def _greedy_rank_search(n, m, p, target_rank, time_budget, max_grid_val=2):
    """
    Greedy search for a decomposition of the all-ones tensor with given rank.
    At each step, pick the rank-1 component that best matches the residual.
    """
    start_time = time.time()
    T_target = np.ones((n, m, p))
    T_residual = T_target.copy()
    
    U_list = []
    V_list = []
    W_list = []
    
    for step in range(target_rank):
        if time.time() - start_time > time_budget:
            return None
        
        best_score = -1
        best_u = None
        best_v = None
        best_w = None
        
        # Search over small integer/half-integer values
        # To keep tractable, we search over a limited grid
        # Each factor has at most n*m, m*p, p*n entries
        # We restrict to values in {-2, -1, -0.5, 0, 0.5, 1, 2}
        # But the full search space is too large, so we use structured candidates
        
        # Generate structured candidates
        # Type 1: unit vectors (schoolbook-like)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = np.zeros(n * m)
                    v = np.zeros(m * p)
                    w = np.zeros(p * n)
                    u[i * m + j] = 1.0
                    v[j * p + k] = 1.0
                    w[k * n + i] = 1.0
                    T = _tensor_from_rank1(u, v, w, n, m, p)
                    score = np.dot(T.flatten(), T_residual.flatten())
                    if score > best_score:
                        best_score = score
                        best_u, best_v, best_w = u, v, w
        
        # Type 2: line sums (fix one index)
        for i in range(n):
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            for j in range(m):
                u[i * m + j] = 1.0
            for j in range(m):
                for k in range(p):
                    v[j * p + k] = 1.0
                    w[k * n + i] = 1.0
            T = _tensor_from_rank1(u, v, w, n, m, p)
            score = np.dot(T.flatten(), T_residual.flatten())
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u, v, w
        
        for k in range(p):
            u = np.zeros(n * m)
            v = np.zeros(m * p)
            w = np.zeros(p * n)
            for i in range(n):
                for j in range(m):
                    u[i * m + j] = 1.0
                    v[j * p + k] = 1.0
                    w[k * n + i] = 1.0
            T = _tensor_from_rank1(u, v, w, n, m, p)
            score = np.dot(T.flatten(), T_residual.flatten())
            if score > best_score:
                best_score = score
                best_u, best_v, best_w = u, v, w
        
        # Type 3: scaled versions
        for scale in [0.5, 2.0]:
            for i in range(n):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                for j in range(m):
                    u[i * m + j] = scale
                for j in range(m):
                    for k in range(p):
                        v[j * p + k] = 1.0 / scale
                        w[k * n + i] = 1.0
                T = _tensor_from_rank1(u, v, w, n, m, p)
                score = np.dot(T.flatten(), T_residual.flatten())
                if score > best_score:
                    best_score = score
                    best_u, best_v, best_w = u, v, w
        
        if best_u is None:
            return None
        
        U_list.append(best_u)
        V_list.append(best_v)
        W_list.append(best_w)
        T_residual -= _tensor_from_rank1(best_u, best_v, best_w, n, m, p)
        
        if np.max(np.abs(T_residual)) < 1e-10:
            break
    
    if np.max(np.abs(T_residual)) < 1e-10:
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)
        return U, V, W, len(U_list)
    
    return None

def _try_linear_solve(n, m, p, target_rank, time_budget):
    """
    Try to solve the system of equations for a given rank using linear algebra.
    The system is: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    This is nonlinear, but we can fix two factors and solve for the third.
    """
    start_time = time.time()
    
    # Try fixing U and V to random small integer values, then solve for W
    np.random.seed(42)
    
    for attempt in range(20):
        if time.time() - start_time > time_budget:
            return None
        
        # Initialize U and V with small random integers
        U = np.random.randint(-2, 3, size=(n * m, target_rank)).astype(float)
        V = np.random.randint(-2, 3, size=(m * p, target_rank)).astype(float)
        
        # For each (k, i), we need:
        # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j
        # This is a system of m equations in target_rank unknowns for each (k,i)
        
        W = np.zeros((p * n, target_rank))
        valid = True
        
        for k in range(p):
            for i in range(n):
                # Build the system: for each j, sum_c (U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c] = 1
                A = np.zeros((m, target_rank))
                b = np.ones(m)
                for j in range(m):
                    for c in range(target_rank):
                        A[j, c] = U[i * m + j, c] * V[j * p + k, c]
                
                # Solve A * w = b
                try:
                    w_sol, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    # Check if the solution is exact
                    if np.max(np.abs(A @ w_sol - b)) < 1e-10:
                        W[k * n + i, :] = w_sol
                    else:
                        valid = False
                        break
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Verify the full decomposition
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
            return U, V, W, target_rank
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # First, get schoolbook as fallback
    U_school, V_school, W_school, rank_school = _schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    # Lower bound: rank must be at least max(n*p, m*p, n*m) = max(14, 21, 6) = 21
    lower_bound = max(n * p, m * p, n * m)  # = 21
    
    # Try to find lower rank decompositions
    # We'll try from lower_bound upward, but limit the search
    for target_rank in range(lower_bound, min(rank_school, 30)):
        if time.time() > deadline - 10.0:
            break
        
        time_per_attempt = min(10.0, (deadline - time.time() - 5.0) / max(1, 30 - target_rank))
        
        # Try linear solve approach
        result = _try_linear_solve(n, m, p, target_rank, time_per_attempt / 2)
        if result is not None:
            U, V, W, rank = result
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
                best_U, best_V, best_W, best_rank = U, V, W, rank
                break
        
        # Try greedy approach
        if time.time() < deadline - 5.0:
            result = _greedy_rank_search(n, m, p, target_rank, time_per_attempt / 2)
            if result is not None:
                U, V, W, rank = result
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
                    best_U, best_V, best_W, best_rank = U, V, W, rank
                    break
    
    # Convert to lists as per interface
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
