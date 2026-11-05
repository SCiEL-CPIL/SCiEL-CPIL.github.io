"""
Rank-minimal tensor decomposition for 3x5x6 matrix multiplication.

Uses a structured random search approach with local refinement:
1. Initialize with random rational entries at various ranks
2. Apply iterative projection to minimize tensor norm
3. Enforce exact integer/half-integer constraints via lattice rounding
4. Aim to discover non-trivial low-rank decompositions

The goal is to find a decomposition with rank < 90 (schoolbook).
"""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

_START_TIME = time.time()
TIME_LIMIT = 130.0  # Safety margin before 150s limit

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _schoolbook(n, m, p):
    """Return schoolbook decomposition (rank n*m*p)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                if s != 1:
                    return False
    return True

def _tensor_residual(U, V, W, n, m, p):
    """Compute the maximum residual of the tensor decomposition."""
    rank = len(U[0]) if U else 0
    max_res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                res = abs(s - 1.0)
                if res > max_res:
                    max_res = res
    return max_res

def _try_rank(rank, n, m, p, seed=None, max_iters=5000, tol=1e-10):
    """
    Try to find a rank-`rank` decomposition using iterative refinement.
    Uses random initialization followed by multiplicative weight updates.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if _time_remaining() < 10.0:
        return None
    
    # Initialize randomly
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    
    # Convert to lists for computation
    U_list = [[float(U[i][r]) for r in range(rank)] for i in range(n * m)]
    V_list = [[float(V[i][r]) for r in range(rank)] for i in range(m * p)]
    W_list = [[float(W[i][r]) for r in range(rank)] for i in range(p * n)]
    
    best_U, best_V, best_W = U_list, V_list, W_list
    best_res = _tensor_residual(U_list, V_list, W_list, n, m, p)
    
    # Iterative refinement: adjust entries to reduce residual
    learning_rate = 0.1
    for it in range(max_iters):
        if _time_remaining() < 10.0:
            break
        
        if it % 100 == 0 and learning_rate > 0.001:
            learning_rate *= 0.95
        
        # Compute residuals for each (i,j,k)
        residuals = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_list[i * m + j][r] * V_list[j * p + k][r] * W_list[k * n + i][r]
                    residuals[(i, j, k)] = s - 1.0
        
        # Compute gradients for U, V, W
        dU = np.zeros((n * m, rank))
        dV = np.zeros((m * p, rank))
        dW = np.zeros((p * n, rank))
        
        for (i, j, k), res in residuals.items():
            ui = U_list[i * m + j]
            vj = V_list[j * p + k]
            wk = W_list[k * n + i]
            for r in range(rank):
                # d(U[i,j,r])/d(res) = V[j,p+k,r] * W[k,n+i,r]
                dU[i * m + j][r] += res * vj[r] * wk[r]
                dV[j * p + k][r] += res * ui[r] * wk[r]
                dW[k * n + i][r] += res * ui[r] * vj[r]
        
        # Update
        for i in range(n * m):
            for r in range(rank):
                U_list[i][r] -= learning_rate * dU[i][r]
        for i in range(m * p):
            for r in range(rank):
                V_list[i][r] -= learning_rate * dV[i][r]
        for i in range(p * n):
            for r in range(rank):
                W_list[i][r] -= learning_rate * dW[i][r]
        
        # Check convergence
        res = _tensor_residual(U_list, V_list, W_list, n, m, p)
        if res < best_res:
            best_res = res
            best_U = [row[:] for row in U_list]
            best_V = [row[:] for row in V_list]
            best_W = [row[:] for row in W_list]
        
        if res < tol:
            break
    
    # Try to snap to half-integers if close
    if best_res < 0.5:
        snapped_U, snapped_V, snapped_W = _snap_to_half_integers(best_U, best_V, best_W)
        snap_res = _tensor_residual(snapped_U, snapped_V, snapped_W, n, m, p)
        if snap_res < 1e-6:
            return {"U": snapped_U, "V": snapped_V, "W": snapped_W}
    
    return None

def _snap_to_half_integers(U, V, W):
    """Snap values to nearest half-integer."""
    U_s = [[int(round(2 * U[i][r])) / 2 for r in range(len(U[0]))] for i in range(len(U))]
    V_s = [[int(round(2 * V[i][r])) / 2 for r in range(len(V[0]))] for i in range(len(V))]
    W_s = [[int(round(2 * W[i][r])) / 2 for r in range(len(W[0]))] for i in range(len(W))]
    return U_s, V_s, W_s

def _strassen_2x2x2():
    """
    Return the Strassen decomposition for 2x2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # m1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2 = (a21+a22)*b11
    U[1][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # m3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # m4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # m5 = (a11+a12)*b22
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # m6 = (a21-a11)*(b11+b12)
    U[1][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # m7 = (a12-a22)*(b21+b22)
    U[2][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[2][6] = 1
    
    return U, V, W

def _block_strassen_decomposition(n, m, p):
    """
    Build a block Strassen decomposition for nxmxp matrix multiplication.
    Uses 2x2x2 Strassen blocks placed to cover the multiplication.
    """
    # Start with schoolbook as baseline
    schoolbook = _schoolbook(n, m, p)
    
    # Define blocks: each block is a 2x2x2 sub-problem at offset (i0, j0, k0)
    blocks = []
    for i0 in range(max(0, n - 2) + 1):
        for j0 in range(max(0, m - 2) + 1):
            for k0 in range(max(0, p - 2) + 1):
                if i0 + 2 <= n and j0 + 2 <= m and k0 + 2 <= p:
                    blocks.append((i0, j0, k0))
    
    if not blocks:
        return schoolbook
    
    # Find minimal set of blocks that cover all (i,j,k)
    needed = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                needed.add((i, j, k))
    
    block_covers = {}
    for idx, (i0, j0, k0) in enumerate(blocks):
        covered = set()
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    i, j, k = i0 + di, j0 + dj, k0 + dk
                    if 0 <= i < n and 0 <= j < m and 0 <= k < p:
                        covered.add((i, j, k))
        block_covers[idx] = covered
    
    # Greedy set cover
    remaining = set(needed)
    selected = []
    while remaining and _time_remaining() > 5.0:
        best_idx = None
        best_count = 0
        for idx in range(len(blocks)):
            if idx in selected:
                continue
            count = len(remaining & block_covers[idx])
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining -= block_covers[best_idx]
    
    if remaining:
        return schoolbook
    
    # Build decomposition using Strassen on each selected block
    U_dict = {}
    V_dict = {}
    W_dict = {}
    
    rank = 0
    for idx in selected:
        i0, j0, k0 = blocks[idx]
        U_s, V_s, W_s = _strassen_2x2x2()
        for r in range(7):
            for i in range(2):
                for j in range(2):
                    ui = U_s[i * 2 + j][r]
                    if ui != 0:
                        key = (i0 + i, j0 + j)
                        if key not in U_dict:
                            U_dict[key] = {}
                        U_dict[key][rank] = U_dict[key].get(rank, 0) + ui
            for j in range(2):
                for k in range(2):
                    vj = V_s[j * 2 + k][r]
                    if vj != 0:
                        key = (j0 + j, k0 + k)
                        if key not in V_dict:
                            V_dict[key] = {}
                        V_dict[key][rank] = V_dict[key].get(rank, 0) + vj
            for k in range(2):
                for i in range(2):
                    wk = W_s[k * 2 + i][r]
                    if wk != 0:
                        key = (k0 + k, i0 + i)
                        if key not in W_dict:
                            W_dict[key] = {}
                        W_dict[key][rank] = W_dict[key].get(rank, 0) + wk
            rank += 1
    
    # Convert to matrix form
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for (i, j), coeffs in U_dict.items():
        for r, val in coeffs.items():
            U_mat[i * m + j][r] = val
    for (j, k), coeffs in V_dict.items():
        for r, val in coeffs.items():
            V_mat[j * p + k][r] = val
    for (k, i), coeffs in W_dict.items():
        for r, val in coeffs.items():
            W_mat[k * n + i][r] = val
    
    # Verify
    if _verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        return {"U": U_mat, "V": V_mat, "W": W_mat}
    else:
        return schoolbook

def construct():
    n, m, p = DIMENSIONS
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # First try block Strassen
    try:
        strassen_result = _block_strassen_decomposition(n, m, p)
        strassen_rank = len(strassen_result["U"][0])
        if strassen_rank < best_rank and _verify_decomposition(
            strassen_result["U"], strassen_result["V"], strassen_result["W"], n, m, p
        ):
            best_result = strassen_result
            best_rank = strassen_rank
    except Exception:
        pass
    
    # Try random search at various ranks
    if _time_remaining() > 30.0:
        for target_rank in [best_rank - 10, best_rank - 20, best_rank - 30]:
            if target_rank < 10 or _time_remaining() < 20.0:
                break
            for seed in range(3):
                if _time_remaining() < 15.0:
                    break
                try:
                    result = _try_rank(target_rank, n, m, p, seed=seed, max_iters=2000)
                    if result is not None:
                        res_rank = len(result["U"][0])
                        if res_rank < best_rank and _verify_decomposition(
                            result["U"], result["V"], result["W"], n, m, p
                        ):
                            best_result = result
                            best_rank = res_rank
                except Exception:
                    pass
    
    # Final verification
    if _verify_decomposition(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        return best_result
    else:
        return _schoolbook(n, m, p)
