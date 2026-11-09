"""Matrix multiplication 2x3x5 - search for rank-minimal decomposition."""
import numpy as np
from fractions import Fraction
import time
import random

def _validate(U, V, W, n=2, m=3, p=5):
    """Validate that U, V, W form a valid decomposition."""
    rank = len(U[0])
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(rank):
                    s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n=2, m=3, p=5):
    """Return schoolbook decomposition with rank = n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][r] = 1
                V[v_idx][r] = 1
                W[w_idx][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}

def _try_structured_search(n, m, p, target_rank, seed=42, max_attempts=50):
    """
    Try to find a valid decomposition with given rank using structured search.
    
    Strategy: Start from schoolbook and try to express it as a sum of fewer rank-1 terms
    by finding linear combinations.
    """
    rng = random.Random(seed)
    
    # The tensor T has T[i,j,k] = 1 for all valid i,j,k
    # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k
    
    # Approach: Use the fact that the all-ones tensor can be decomposed in various ways.
    # For 2x3x5, we can try to find patterns.
    
    # Let's try a parametric family of decompositions.
    # For each rank r, we can try to find U, V, W such that the tensor equation holds.
    
    # One approach: assume U and V have a specific structure and solve for W.
    
    # Let's try: U[i*m+j, r] depends only on (i, j) and r in a structured way.
    # Similarly for V and W.
    
    # For now, let's try random search with better initialization.
    # Use values from a larger set and try to satisfy the equations.
    
    vals = [-2, -1, -0.5, 0, 0.5, 1, 2, 3, -3]
    
    for attempt in range(max_attempts):
        U = [[0] * target_rank for _ in range(n * m)]
        V = [[0] * target_rank for _ in range(m * p)]
        W = [[0] * target_rank for _ in range(p * n)]
        
        # Initialize with random values, but ensure some structure
        # Make U sparse with a few non-zero entries per row
        for i in range(n * m):
            num_nonzero = rng.randint(1, min(3, target_rank))
            indices = rng.sample(range(target_rank), num_nonzero)
            for idx in indices:
                U[i][idx] = rng.choice(vals)
        
        for j in range(m * p):
            num_nonzero = rng.randint(1, min(4, target_rank))
            indices = rng.sample(range(target_rank), num_nonzero)
            for idx in indices:
                V[j][idx] = rng.choice(vals)
        
        for k in range(p * n):
            num_nonzero = rng.randint(1, min(3, target_rank))
            indices = rng.sample(range(target_rank), num_nonzero)
            for idx in indices:
                W[k][idx] = rng.choice(vals)
        
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_lagrangian_relaxation(n, m, p, target_rank, seed=42):
    """
    Try to find a decomposition using a Lagrangian relaxation approach.
    
    We want to minimize rank subject to the tensor equation constraints.
    This is a non-convex problem, but we can try to relax it.
    """
    rng = random.Random(seed)
    
    # The tensor equation is: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a system of n*m*p = 30 equations in rank*(n*m + m*p + p*n) variables.
    
    # Let's try to solve it using a iterative approach.
    # Initialize U, V, W randomly, then iterate to satisfy the equations.
    
    vals = [-1, 0, 1, 2, -2, 0.5, -0.5]
    
    U = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(n * m)]
    V = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(m * p)]
    W = [[rng.choice(vals) for _ in range(target_rank)] for _ in range(p * n)]
    
    # Iterate: for each (i,j,k), adjust the terms to make the sum equal to 1
    for iteration in range(100):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    
                    # Compute current sum
                    s = 0
                    for r in range(target_rank):
                        s += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                    
                    # If s != 1, try to adjust
                    if s != 1:
                        # Find a term that can be adjusted
                        best_r = None
                        best_improvement = 0
                        for r in range(target_rank):
                            u_val = U[u_idx][r]
                            v_val = V[v_idx][r]
                            w_val = W[w_idx][r]
                            
                            # Try changing W[w_idx][r] to fix the sum
                            if u_val != 0 and v_val != 0:
                                new_w_val = (1 - (s - u_val * v_val * w_val)) / (u_val * v_val)
                                # Check if this is a valid value (integer or half-integer)
                                if abs(new_w_val * 2 - round(new_w_val * 2)) < 1e-9:
                                    improvement = abs(1 - s)
                                    if improvement > best_improvement:
                                        best_improvement = improvement
                                        best_r = r
                                        best_new_w_val = new_w_val
                        
                        if best_r is not None:
                            W[w_idx][best_r] = best_new_w_val
        
        # Check if we have a valid solution
        if _validate(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_block_structure(n, m, p, target_rank, seed=42):
    """
    Try to exploit the block structure of the 2x3x5 multiplication.
    
    The 2x3 matrix A has 2 rows and 3 columns.
    The 3x5 matrix B has 3 rows and 5 columns.
    The output C has 2 rows and 5 columns.
    
    We can try to find a decomposition that respects this structure.
    """
    rng = random.Random(seed)
    
    # Let's try to find a decomposition where U, V, W have a block structure.
    # For example, U could be block diagonal or have a specific pattern.
    
    # One approach: use the fact that the all-ones tensor can be written as:
    # T[i,j,k] = sum_r a_r[i] * b_r[j] * c_r[k]
    # where a_r[i] is 0/1, b_r[j] is 0/1, c_r[k] is 0/1.
    # This is the schoolbook decomposition.
    
    # To reduce rank, we need to find a more efficient representation.
    # This is related to the Strassen-like algorithms.
    
    # For 2x3x5, let's try to find a decomposition with rank < 30.
    
    # Let's try a parametric search.
    # Assume U[i*m+j, r] = u_i[j, r] where u_i is a 3 x target_rank matrix.
    # Similarly for V and W.
    
    # Let's try to find u_i, v_j, w_k such that:
    # sum_r u_0[j, r] * v_j[k, r] * w_k[0, r] = 1 for all j, k
    # sum_r u_1[j, r] * v_j[k, r] * w_k[1, r] = 1 for all j, k
    
    # This is a system of 30 equations.
    
    # Let's try to solve it using a greedy approach.
    
    u = [[[0] * target_rank for _ in range(m)] for _ in range(n)]  # u[i][j][r]
    v = [[[0] * target_rank for _ in range(p)] for _ in range(m)]  # v[j][k][r]
    w = [[[0] * target_rank for _ in range(n)] for _ in range(p)]  # w[k][i][r]
    
    # Initialize with random values
    vals = [-1, 0, 1]
    
    for i in range(n):
        for j in range(m):
            for r in range(target_rank):
                u[i][j][r] = rng.choice(vals)
    
    for j in range(m):
        for k in range(p):
            for r in range(target_rank):
                v[j][k][r] = rng.choice(vals)
    
    for k in range(p):
        for i in range(n):
            for r in range(target_rank):
                w[k][i][r] = rng.choice(vals)
    
    # Iterate to satisfy the equations
    for iteration in range(50):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute current sum
                    s = 0
                    for r in range(target_rank):
                        s += u[i][j][r] * v[j][k][r] * w[k][i][r]
                    
                    if s != 1:
                        # Try to adjust w[k][i][r] for some r
                        for r in range(target_rank):
                            uv = u[i][j][r] * v[j][k][r]
                            if uv != 0:
                                new_w = (1 - (s - u[i][j][r] * v[j][k][r] * w[k][i][r])) / uv
                                if abs(new_w * 2 - round(new_w * 2)) < 1e-9:
                                    w[k][i][r] = new_w
                                    break
        
        # Check if we have a valid solution
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(target_rank):
                        s += u[i][j][r] * v[j][k][r] * w[k][i][r]
                    if s != 1:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert back to the required format
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            for i in range(n):
                for j in range(m):
                    for r in range(target_rank):
                        U[i * m + j][r] = u[i][j][r]
            
            for j in range(m):
                for k in range(p):
                    for r in range(target_rank):
                        V[j * p + k][r] = v[j][k][r]
            
            for k in range(p):
                for i in range(n):
                    for r in range(target_rank):
                        W[k * n + i][r] = w[k][i][r]
            
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    """
    Return the matrix multiplication tensor for 2x3 * 3x5 as a rank decomposition.
    U: (n*m, rank) = (6, rank)
    V: (m*p, rank) = (15, rank)
    W: (p*n, rank) = (10, rank)
    
    Indexing conventions:
    U[i*m + j, r] corresponds to A[i,j]
    V[j*p + k, r] corresponds to B[j,k]
    W[k*n + i, r] corresponds to C[i,k]
    """
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 30
    
    # Try to find lower rank decompositions
    # Lower bound is at least max(n*p, m*n, m*p) = max(10, 6, 15) = 15
    # Try ranks from 15 to 29
    
    for rank in range(15, best_rank):
        if time.time() > deadline:
            break
        
        # Try different search strategies for this rank
        result = None
        
        # Try structured search
        if time.time() < deadline - 5:
            result = _try_structured_search(n, m, p, rank, seed=rank * 1000, max_attempts=20)
        
        # Try Lagrangian relaxation
        if result is None and time.time() < deadline - 5:
            result = _try_lagrangian_relaxation(n, m, p, rank, seed=rank * 1000 + 1)
        
        # Try block structure
        if result is None and time.time() < deadline - 5:
            result = _try_block_structure(n, m, p, rank, seed=rank * 1000 + 2)
        
        if result is not None:
            best = result
            best_rank = rank
            break
    
    return best
