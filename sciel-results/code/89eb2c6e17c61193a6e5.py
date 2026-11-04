"""Matrix multiplication 4x4 @ 4x5: rank-minimal tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]
N, M, P = 4, 4, 5
NM = N * M
MP = M * P
PN = P * N


def _schoolbook():
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-9):
    rank = len(U[0]) if U else 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                total = 0.0
                for r in range(rank):
                    total += U[i * M + j][r] * V[j * P + k][r] * W[k * N + i][r]
                if abs(total - 1.0) > tol:
                    return False
    return True


def _rank_of(decomp):
    if not decomp["U"]:
        return 0
    return len(decomp["U"][0])


def _compute_error(U, V, W):
    """Compute the maximum absolute error of the decomposition."""
    rank = len(U[0]) if U else 0
    max_err = 0.0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                total = 0.0
                for r in range(rank):
                    total += U[i * M + j][r] * V[j * P + k][r] * W[k * N + i][r]
                err = abs(total - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _random_local_search(target_rank, deadline, seed=42):
    """
    Attempt to find a decomposition of target_rank using local search.
    Values are restricted to half-integers {-1, -0.5, 0, 0.5, 1}.
    """
    rng = random.Random(seed)
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    # Initialize with random values
    U = [[rng.choice(values) for _ in range(target_rank)] for _ in range(NM)]
    V = [[rng.choice(values) for _ in range(target_rank)] for _ in range(MP)]
    W = [[rng.choice(values) for _ in range(target_rank)] for _ in range(PN)]
    
    current_err = _compute_error(U, V, W)
    
    if current_err < 1e-9:
        return {"U": U, "V": V, "W": W}
    
    best_err = current_err
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Local search loop
    while time.time() < deadline:
        # Pick a random matrix and a random element
        mat_type = rng.randint(0, 2)
        if mat_type == 0:
            mat = U
            rows = NM
        elif mat_type == 1:
            mat = V
            rows = MP
        else:
            mat = W
            rows = PN
            
        r_idx = rng.randint(0, rows - 1)
        c_idx = rng.randint(0, target_rank - 1)
        
        old_val = mat[r_idx][c_idx]
        
        # Try a new value
        new_val = rng.choice(values)
        if new_val == old_val:
            continue
            
        mat[r_idx][c_idx] = new_val
        new_err = _compute_error(U, V, W)
        
        if new_err < current_err:
            current_err = new_err
            if current_err < best_err:
                best_err = current_err
                best_U = [row[:] for row in U]
                best_V = [row[:] for row in V]
                best_W = [row[:] for row in W]
                if best_err < 1e-9:
                    return {"U": best_U, "V": best_V, "W": best_W}
        else:
            # Revert
            mat[r_idx][c_idx] = old_val
            
    return None


def construct():
    deadline = time.time() + 110
    best = _schoolbook()
    best_rank = _rank_of(best)
    
    # Try to find a lower rank decomposition using local search
    # We know rank 80 is possible.
    # Theoretical lower bound is roughly 4*4*5 / something? 
    # Let's try ranks from 70 to 79.
    
    # To save time, we prioritize higher ranks (easier to find) first?
    # Or lower ranks (better score)?
    # Let's try 79, 78, 77... down to 60.
    
    # We will allocate time proportional to the difficulty?
    # Let's just try a few random seeds for each rank.
    
    ranks_to_try = list(range(79, 59, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        # Try a few seeds for this rank
        for seed in range(5):
            if time.time() > deadline - 10:
                break
                
            result = _random_local_search(r, deadline - 10, seed=seed + r)
            if result is not None:
                if _verify(result["U"], result["V"], result["W"]):
                    new_rank = _rank_of(result)
                    if new_rank < best_rank:
                        best = result
                        best_rank = new_rank
                        # If we found a very good rank, we can stop early or continue
                        if best_rank <= 65:
                            return best
    
    return best
