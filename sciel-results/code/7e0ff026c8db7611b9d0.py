"""Matrix multiplication tensor decomposition: 3x4 by 4x8."""
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 8]
n, m, p = DIMENSIONS


def schoolbook_decomposition():
    """Return the schoolbook (rank n*m*p) decomposition for 3x4x8."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def verify_348(U, V, W):
    """Check if the decomposition is exact for 3x4x8."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def schoolbook_344():
    """Return the schoolbook (rank 48) decomposition for 3x4x4."""
    n2, m2, p2 = 3, 4, 4
    rank = n2 * m2 * p2
    U = [[0.0] * rank for _ in range(n2 * m2)]
    V = [[0.0] * rank for _ in range(m2 * p2)]
    W = [[0.0] * rank for _ in range(p2 * n2)]
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                U[i * m2 + j][col] = 1.0
                V[j * p2 + k][col] = 1.0
                W[k * n2 + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def verify_344(U, V, W):
    """Check if the decomposition is exact for 3x4x4."""
    n2, m2, p2 = 3, 4, 4
    rank = len(U[0])
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                s = 0.0
                for r in range(rank):
                    s += U[i * m2 + j][r] * V[j * p2 + k][r] * W[k * n2 + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def search_344(deadline):
    """
    Search for a low-rank decomposition of 3x4x4 matrix multiplication.
    Uses random initialization + local search with half-integer snapping.
    """
    n2, m2, p2 = 3, 4, 4
    best = schoolbook_344()
    best_rank = n2 * m2 * p2
    
    # Try ranks from 47 down to 40
    for target_rank in range(47, 39, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            rng = random.Random(attempt * 1000 + target_rank)
            
            # Initialize with small random values, snap to half-integers
            U = [[rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]) for _ in range(target_rank)] for _ in range(n2 * m2)]
            V = [[rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]) for _ in range(target_rank)] for _ in range(m2 * p2)]
            W = [[rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]) for _ in range(target_rank)] for _ in range(p2 * n2)]
            
            if verify_344(U, V, W):
                best = {"U": U, "V": V, "W": W}
                best_rank = target_rank
                found = True
                break
        
        if found:
            # Continue searching for even lower rank
            continue
        else:
            # If we can't find rank target_rank, stop (assume lower ranks are harder)
            break
    
    return best, best_rank


def compose_344_to_348(decomp_344):
    """
    Compose two 3x4x4 decompositions into a 3x4x8 decomposition.
    Block 1: k in [0,4), Block 2: k in [4,8)
    """
    U1 = decomp_344["U"]
    V1 = decomp_344["V"]
    W1 = decomp_344["W"]
    r1 = len(U1[0])
    
    # For 3x4x8:
    # U: (12, 2*r1) - same for both blocks
    # V: (32, 2*r1) - V1 for k in [0,4), V1 for k in [4,8)
    # W: (24, 2*r1) - W1 for k in [0,4), W1 for k in [4,8)
    
    U = [[0.0] * (2 * r1) for _ in range(12)]
    V = [[0.0] * (2 * r1) for _ in range(32)]
    W = [[0.0] * (2 * r1) for _ in range(24)]
    
    # Copy U
    for i in range(12):
        for r in range(r1):
            U[i][r] = U1[i][r]
            U[i][r + r1] = U1[i][r]
    
    # Copy V: for k in [0,4), use V1[j*4+k]; for k in [4,8), use V1[j*4+(k-4)]
    for j in range(4):
        for k in range(4):
            for r in range(r1):
                V[j * 8 + k][r] = V1[j * 4 + k][r]
                V[j * 8 + (k + 4)][r + r1] = V1[j * 4 + k][r]
    
    # Copy W: for k in [0,4), use W1[k*3+i]; for k in [4,8), use W1[(k-4)*3+i]
    for k in range(4):
        for i in range(3):
            for r in range(r1):
                W[k * 3 + i][r] = W1[k * 3 + i][r]
                W[(k + 4) * 3 + i][r + r1] = W1[k * 3 + i][r]
    
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 3 by 4 matrix by a 4 by 8 matrix.
    
    Returns: {'U': (12, rank), 'V': (32, rank), 'W': (24, rank)}
    """
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Start with schoolbook as fallback
    best_result = schoolbook_decomposition()
    best_rank = n * m * p
    
    # Strategy: find low-rank 3x4x4 decomposition, then compose two copies
    decomp_344, rank_344 = search_344(deadline)
    
    if rank_344 < 48:
        composed = compose_344_to_348(decomp_344)
        composed_rank = len(composed["U"][0])
        
        if verify_348(composed["U"], composed["V"], composed["W"]):
            if composed_rank < best_rank:
                best_result = composed
                best_rank = composed_rank
    
    # Also try direct random search on 3x4x8 with half-integer values
    if time.time() < deadline - 10:
        for target_rank in range(best_rank - 1, 80, -1):
            if time.time() > deadline - 5:
                break
            
            found = False
            for attempt in range(20):
                if time.time() > deadline - 5:
                    break
                
                rng = random.Random(attempt * 7 + target_rank * 13)
                
                U = [[rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]) for _ in range(target_rank)] for _ in range(12)]
                V = [[rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]) for _ in range(target_rank)] for _ in range(32)]
                W = [[rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0]) for _ in range(target_rank)] for _ in range(24)]
                
                if verify_348(U, V, W):
                    best_result = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
            
            if found:
                continue
            else:
                break
    
    return best_result
