"""Matrix multiplication (2,3,9) rank search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify(U, V, W, n, m, p, r):
    # U:(n*m,r) V:(m*p,r) W:(p*n,r)
    Ur = U.reshape(n, m, r)
    Vr = V.reshape(m, p, r)
    Wr = W.reshape(p, n, r)
    T = np.einsum('imc,jpc,knc->ijk', Ur, Vr, Wr)
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _search_233(deadline):
    """Search for a rank decomposition of 2x3x3."""
    n, m, p = 2, 3, 3
    best_rank = n * m * p
    best_solution = None
    
    val_sets = [
        [0, 1, -1],
        [0, 1, -1, 0.5, -0.5],
        [0, 1, -1, 2, -2],
        [0, 1, -1, 0.5, -0.5, 2, -2, 1.5, -1.5],
    ]
    
    # Lower bound for 2x3x3 is max(6, 9, 6) = 9
    for r in range(9, best_rank):
        if time.time() > deadline:
            break
        # Try to find a solution of rank r
        # Use a more targeted search: fix U and W to be sparse/basis-like and solve for V?
        # Or just random search with many trials.
        trials = 20000
        for _ in range(trials):
            if time.time() > deadline:
                break
            vs = random.choice(val_sets)
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            # Generate sparse random matrices
            for c in range(r):
                # U: 6 entries
                for idx in range(n * m):
                    if random.random() < 0.4:
                        U[idx, c] = random.choice(vs)
                # V: 9 entries
                for idx in range(m * p):
                    if random.random() < 0.4:
                        V[idx, c] = random.choice(vs)
                # W: 6 entries
                for idx in range(p * n):
                    if random.random() < 0.4:
                        W[idx, c] = random.choice(vs)
                        
            if _verify(U, V, W, n, m, p, r):
                best_rank = r
                best_solution = (U, V, W)
                # Found a valid one. Can we go lower? Yes, continue loop for lower r?
                # We are iterating r ascending, so this is the minimum found so far.
                # We can break out of this r loop and return, or continue to see if we find even lower?
                # We are ascending, so if we find r=15, we should check if r=14 is possible.
                # So we should NOT break the outer loop immediately, but update best.
                # Actually, if we find one at r, we can return it as a valid candidate, 
                # but maybe a lower r exists. 
                # However, finding one at r is hard. 
                # Let's return the first one found for this r, and let the caller decide if it's good enough.
                # Or better: return the best found so far.
                pass # Continue searching for lower r? No, we are at r. If we find one, it's valid.
                # But maybe r-1 is possible. So we should not return immediately.
                # We just record it.
                pass

    if best_solution:
        return best_solution
    return None

def _construct_from_233(U233, V233, W233, r233):
    """Construct 2x3x9 decomposition from 2x3x3."""
    n, m, p = 2, 3, 9
    R = 3 * r233
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Map indices
    # U233: (2, 3, r233) -> U: (2, 3, R)
    # V233: (3, 3, r233) -> V: (3, 9, R)
    # W233: (3, 2, r233) -> W: (9, 2, R)
    
    for g in range(3):
        for c in range(r233):
            col = 3 * c + g
            # U: i in 0..1, j in 0..2
            for i in range(2):
                for j in range(3):
                    U[i * 3 + j, col] = U233[i, j, c]
            
            # V: j in 0..2, k in 0..8 (k = 3g + l)
            for j in range(3):
                for l in range(3):
                    k = 3 * g + l
                    V[j * 9 + k, col] = V233[j, l, c]
            
            # W: k in 0..8 (k = 3g + l), i in 0..1
            for l in range(3):
                k = 3 * g + l
                for i in range(2):
                    W[k * 2 + i, col] = W233[l, i, c]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110.0
    best = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try to find a better solution using the 2x3x3 block structure
    sol_233 = _search_233(deadline)
    if sol_233:
        U233, V233, W233 = sol_233
        r233 = U233.shape[1]
        candidate = _construct_from_233(U233, V233, W233, r233)
        r_candidate = 3 * r233
        if r_candidate < best_rank:
            best = candidate
            best_rank = r_candidate

    # Fallback: direct search for 2x3x9 if time permits
    if time.time() < deadline - 10:
        # Simple random search for direct 2x3x9 with rank < best_rank
        val_sets = [
            [0, 1, -1],
            [0, 1, -1, 0.5, -0.5],
        ]
        # Only search a few ranks below best_rank
        for r in range(max(27, best_rank - 5), best_rank):
            if time.time() > deadline:
                break
            trials = 1000
            for _ in range(trials):
                if time.time() > deadline:
                    break
                vs = random.choice(val_sets)
                U = np.zeros((n * m, r))
                V = np.zeros((m * p, r))
                W = np.zeros((p * n, r))
                for c in range(r):
                    for idx in range(n * m):
                        if random.random() < 0.5:
                            U[idx, c] = random.choice(vs)
                    for idx in range(m * p):
                        if random.random() < 0.5:
                            V[idx, c] = random.choice(vs)
                    for idx in range(p * n):
                        if random.random() < 0.5:
                            W[idx, c] = random.choice(vs)
                if _verify(U, V, W, n, m, p, r):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = r
                    return best

    return best
# EVOLVE-BLOCK-END
