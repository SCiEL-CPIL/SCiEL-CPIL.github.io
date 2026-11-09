"""Matrix multiplication 3x3x5 - rank-minimal exact tensor decomposition."""
import time
import itertools
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 5]
N, M, P = DIMENSIONS
NM = N * M  # 9
MP = M * P  # 15
PN = P * N  # 15


def _verify_exact(U, V, W):
    """Check if U,V,W exactly decompose the matmul tensor T[i,j,k]=1.
    U: (NM, rank), V: (MP, rank), W: (PN, rank)
    Need: sum_c U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c] == 1 for all i,j,k
    """
    rank = len(U[0])
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0
                for c in range(rank):
                    s += U[i * M + j][c] * V[j * P + k][c] * W[k * N + i][c]
                if s != 1:
                    return False
    return True


def _schoolbook():
    """Return the trivial rank-45 decomposition."""
    rank = N * M * P  # 45
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_structured_rank(rank, deadline):
    """Try to find a rank-r decomposition using structured search.
    
    Strategy: The tensor T[i,j,k]=1 has the property that for fixed (i,j),
    the k-profile is all-ones. We can try to exploit this by having the
    W-factor depend on k in a way that allows sharing.
    
    Key idea: For each column c, define:
      u_c[i,j] = U[i*M+j, c]
      v_c[j,k] = V[j*P+k, c]  
      w_c[k,i] = W[k*N+i, c]
    
    We need: sum_c u_c[i,j] * v_c[j,k] * w_c[k,i] = 1 for all i,j,k.
    
    This is a 3-way tensor rank problem. We use random restarts with
    structured initialization and local refinement.
    """
    start_time = time.time()
    
    # Try various random seeds and structured initializations
    best_result = None
    best_err = float('inf')
    
    for seed in range(200):
        if time.time() > deadline:
            break
        
        rng = random.Random(seed)
        
        # Initialize with small integer/half-integer values
        # Use values from {-1, -0.5, 0, 0.5, 1}
        vals = [-1, -0.5, 0, 0.5, 1]
        
        U = [[rng.choice(vals) for _ in range(rank)] for _ in range(NM)]
        V = [[rng.choice(vals) for _ in range(rank)] for _ in range(MP)]
        W = [[rng.choice(vals) for _ in range(rank)] for _ in range(PN)]
        
        # Check if this works
        if _verify_exact(U, V, W):
            return {"U": U, "V": V, "W": W}
        
        # Try local refinement: flip individual entries
        for flip_iter in range(50):
            if time.time() > deadline:
                break
            
            # Pick a random entry to flip
            mat = rng.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = rng.randint(0, NM - 1)
                col = rng.randint(0, rank - 1)
                old = U[row][col]
                U[row][col] = -old if old != 0 else rng.choice([0.5, -0.5, 1, -1])
                if _verify_exact(U, V, W):
                    return {"U": U, "V": V, "W": W}
                U[row][col] = old
            elif mat == 'V':
                row = rng.randint(0, MP - 1)
                col = rng.randint(0, rank - 1)
                old = V[row][col]
                V[row][col] = -old if old != 0 else rng.choice([0.5, -0.5, 1, -1])
                if _verify_exact(U, V, W):
                    return {"U": U, "V": V, "W": W}
                V[row][col] = old
            else:
                row = rng.randint(0, PN - 1)
                col = rng.randint(0, rank - 1)
                old = W[row][col]
                W[row][col] = -old if old != 0 else rng.choice([0.5, -0.5, 1, -1])
                if _verify_exact(U, V, W):
                    return {"U": U, "V": V, "W": W}
                W[row][col] = old
    
    return None


def _try_systematic_rank(rank, deadline):
    """More systematic search: fix some structure and search the rest.
    
    For rank r, we can try to construct the decomposition by:
    1. Choosing U to have a specific sparsity pattern
    2. Solving for V and W given U
    3. Verifying exactness
    """
    start_time = time.time()
    
    # Try different sparsity patterns for U
    # Each row of U should have a few non-zero entries
    
    for pattern_seed in range(50):
        if time.time() > deadline:
            break
        
        rng = random.Random(1000 + pattern_seed)
        
        # Create sparse U: each row has 2-3 non-zero entries
        U = [[0] * rank for _ in range(NM)]
        for i in range(NM):
            num_nonzero = rng.randint(2, min(3, rank))
            cols = rng.sample(range(rank), num_nonzero)
            for c in cols:
                U[i][c] = rng.choice([-1, 1])
        
        # For each (i,j), we need: sum_c U[i*M+j,c] * V[j*P+k,c] * W[k*N+i,c] = 1
        # This is a system of equations in V and W entries
        
        # Try to solve by fixing V and solving for W, or vice versa
        # Use random V and check if W can be determined
        
        for v_seed in range(20):
            if time.time() > deadline:
                break
            
            V = [[rng.choice([-1, -0.5, 0, 0.5, 1]) for _ in range(rank)] for _ in range(MP)]
            
            # Now try to find W such that the equation holds
            # For each (k,i): sum_c W[k*N+i,c] * (sum over j of U[i*M+j,c]*V[j*P+k,c]) = 1
            # Wait, that's not quite right. Let me re-derive.
            
            # T[i,j,k] = sum_c U[i*M+j,c] * V[j*P+k,c] * W[k*N+i,c]
            # For fixed (i,k): sum_j sum_c U[i*M+j,c] * V[j*P+k,c] * W[k*N+i,c] = sum_j 1 = M = 3
            # No wait, T[i,j,k]=1 for each specific (i,j,k), not summed over j.
            
            # So for each (i,j,k): sum_c U[i*M+j,c] * V[j*P+k,c] * W[k*N+i,c] = 1
            
            # This is a bilinear system in V and W given U.
            # Let's just do random search for V and W together.
            
            pass  # This approach is complex; fall back to random search
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = N * M * P  # 45
    
    # Try to find lower rank decompositions
    # Known lower bounds: rank >= max(N*M, M*P, P*N) = max(9, 15, 15) = 15
    # For 3x3x3, min rank is 23 (Landsberg). For 3x3x5, likely higher.
    # Let's try ranks from 15 upward.
    
    # Prioritize trying ranks that are more likely to work
    # Based on known results and heuristics
    ranks_to_try = list(range(15, best_rank))
    
    # Sort by likelihood: try ranks near known good values first
    # For rectangular matrix multiplication, rank is often around n*m*p / log factor
    # Let's just try in order from low to high
    
    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Allocate time budget based on rank
        # Lower ranks get more time (harder to find), higher ranks get less
        time_budget = min(10, max(2, (deadline - time.time()) / max(1, best_rank - target_rank)))
        
        result = _try_structured_rank(target_rank, start_time + time_budget)
        if result is not None:
            return result
    
    # Fallback: return schoolbook decomposition
    return best
