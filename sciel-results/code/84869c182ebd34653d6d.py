"""Matrix multiplication 4x4 @ 4x5: rank-minimal tensor decomposition."""
import time
import random

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


def _verify(U, V, W):
    rank = len(U[0])
    for i in range(N):
        for j in range(M):
            for k in range(P):
                total = 0
                for r in range(rank):
                    total += U[i * M + j][r] * V[j * P + k][r] * W[k * N + i][r]
                if abs(total - 1.0) > 1e-9:
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
        # Check if both are zero
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
    new_U = [row[:a] + row[a+1:] if i != b else row[:a] + row[a+1:] for i, row in enumerate(U)]
    # Actually, just remove column b
    new_U = [row[:b] + row[b+1:] for row in U]
    new_V = [row[:b] + row[b+1:] for row in V]
    new_W = [row[:b] + row[b+1:] for row in W]
    
    return {"U": new_U, "V": new_V, "W": new_W}


def construct():
    deadline = time.time() + 110
    best = _schoolbook()
    best_rank = _rank_of(best)
    
    # Try to reduce rank by merging proportional columns
    U, V, W = best["U"], best["V"], best["W"]
    rank = _rank_of(best)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for a in range(rank):
            if time.time() > deadline - 10:
                break
            for b in range(a + 1, rank):
                if time.time() > deadline - 10:
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
    
    return best
