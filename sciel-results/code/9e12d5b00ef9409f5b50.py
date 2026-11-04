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
    rank = len(U[0])
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
    return len(decomp["U"][0])


def _try_merge_columns(U, V, W, a, b):
    """Try to merge columns a and b if they are proportional in a way that preserves the product."""
    rank = len(U[0])
    if a >= rank or b >= rank or a == b:
        return None
    
    # Check if U[:,a] = s * U[:,b] for some s
    s = None
    for r in range(NM):
        if U[r][a] != 0 and U[r][b] != 0:
            s = U[r][a] / U[r][b]
            break
        elif U[r][a] != 0 or U[r][b] != 0:
            return None
    
    if s is None:
        if all(U[r][a] == 0 for r in range(NM)) and all(U[r][b] == 0 for r in range(NM)):
            s = 1.0
        else:
            return None
    
    # Verify U proportionality
    for r in range(NM):
        if abs(U[r][a] - s * U[r][b]) > 1e-9:
            return None
    
    # Similarly for V and W
    sv = None
    for r in range(MP):
        if V[r][a] != 0 and V[r][b] != 0:
            sv = V[r][a] / V[r][b]
            break
        elif V[r][a] != 0 or V[r][b] != 0:
            return None
    
    if sv is None:
        if all(V[r][a] == 0 for r in range(MP)) and all(V[r][b] == 0 for r in range(MP)):
            sv = 1.0
        else:
            return None
    
    for r in range(MP):
        if abs(V[r][a] - sv * V[r][b]) > 1e-9:
            return None
    
    sw = None
    for r in range(PN):
        if W[r][a] != 0 and W[r][b] != 0:
            sw = W[r][a] / W[r][b]
            break
        elif W[r][a] != 0 or W[r][b] != 0:
            return None
    
    if sw is None:
        if all(W[r][a] == 0 for r in range(PN)) and all(W[r][b] == 0 for r in range(PN)):
            sw = 1.0
        else:
            return None
    
    for r in range(PN):
        if abs(W[r][a] - sw * W[r][b]) > 1e-9:
            return None
    
    # Product scale: s * sv * sw should be 1 for the terms to be identical
    if abs(s * sv * sw - 1.0) > 1e-9:
        return None
    
    # They are identical in product, so we can drop one
    new_U = [row[:b] + row[b+1:] for row in U]
    new_V = [row[:b] + row[b+1:] for row in V]
    new_W = [row[:b] + row[b+1:] for row in W]
    
    return {"U": new_U, "V": new_V, "W": new_W}


def _linear_algebra_search(U, V, W, deadline):
    """Use linear algebra to check if rank reduction is possible.
    
    For a given rank r, the decomposition is valid iff for all (i,j,k):
    sum_{r'=0}^{r-1} U[i*M+j, r'] * V[j*P+k, r'] * W[k*N+i, r'] = 1
    
    This is a system of 80 equations. We can try to find if there's a linear
    dependence among the columns that allows us to reduce rank.
    """
    rank = len(U[0])
    if rank <= 1:
        return None
    
    # Build the 80-dimensional target tensor
    target = np.zeros((N, M, P))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                target[i, j, k] = 1.0
    
    # For each column r, compute its contribution pattern
    # A column r contributes to entry (i,j,k) by U[i*M+j,r] * V[j*P+k,r] * W[k*N+i,r]
    
    # Try to find if any column is redundant
    for col_to_remove in range(rank):
        if time.time() > deadline:
            break
        
        # Remove column col_to_remove and check if remaining columns can still satisfy
        # This is expensive, so only try a few
        if col_to_remove % 10 != 0:
            continue
        
        new_rank = rank - 1
        new_U = [row[:col_to_remove] + row[col_to_remove+1:] for row in U]
        new_V = [row[:col_to_remove] + row[col_to_remove+1:] for row in V]
        new_W = [row[:col_to_remove] + row[col_to_remove+1:] for row in W]
        
        if _verify(new_U, new_V, new_W):
            return {"U": new_U, "V": new_V, "W": new_W}
    
    return None


def _random_perturbation_search(U, V, W, deadline, max_trials=100):
    """Try random small perturbations to see if we can find a lower-rank decomposition."""
    rank = len(U[0])
    rng = random.Random(42)
    
    for trial in range(max_trials):
        if time.time() > deadline:
            break
        
        # Try to find two columns that are nearly proportional
        a = rng.randint(0, rank - 1)
        b = rng.randint(0, rank - 1)
        if a == b:
            continue
        
        merged = _try_merge_columns(U, V, W, a, b)
        if merged is not None:
            if _verify(merged["U"], merged["V"], merged["W"]):
                return merged
    
    return None


def construct():
    deadline = time.time() + 110
    best = _schoolbook()
    best_rank = _rank_of(best)
    
    U, V, W = best["U"], best["V"], best["W"]
    rank = _rank_of(best)
    
    # Phase 1: Try to reduce rank by merging proportional columns
    changed = True
    while changed and time.time() < deadline - 30:
        changed = False
        for a in range(rank):
            if time.time() > deadline - 30:
                break
            for b in range(a + 1, rank):
                if time.time() > deadline - 30:
                    break
                merged = _try_merge_columns(U, V, W, a, b)
                if merged is not None:
                    if _verify(merged["U"], merged["V"], merged["W"]):
                        U, V, W = merged["U"], merged["V"], merged["W"]
                        rank = _rank_of({"U": U, "V": V, "W": W})
                        changed = True
                        best = {"U": U, "V": V, "W": W}
                        best_rank = rank
                        break
            if changed:
                break
    
    # Phase 2: Linear algebra search for redundant columns
    if time.time() < deadline - 20:
        result = _linear_algebra_search(U, V, W, deadline - 20)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"]):
                new_rank = _rank_of(result)
                if new_rank < best_rank:
                    best = result
                    best_rank = new_rank
    
    # Phase 3: Random perturbation search
    if time.time() < deadline - 10:
        result = _random_perturbation_search(U, V, W, deadline - 10)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"]):
                new_rank = _rank_of(result)
                if new_rank < best_rank:
                    best = result
                    best_rank = new_rank
    
    return best
