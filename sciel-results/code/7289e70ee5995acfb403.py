"""
Rank-minimal exact tensor decomposition for 2x2 * 2x4 matrix multiplication.
Uses a "middle product" style decomposition: two Strassen 2x2x2 blocks that
share A-side structure, plus a cross-term to exploit the width-4 B structure.
Target: rank 13 (below the independent-block rank 14).
"""
import time
import random
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n=2, m=2, p=4, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(t_val - 1.0) > tol:
                    return False
    for i_idx in range(n * m):
        for j_idx in range(m * p):
            for k_idx in range(p * n):
                a = i_idx // m
                b = i_idx % m
                b_prime = j_idx // p
                c = j_idx % p
                c_prime = k_idx // n
                a_prime = k_idx % n
                if a == a_prime and b == b_prime:
                    continue
                t_val = 0.0
                for r in range(rank):
                    t_val += U[i_idx, r] * V[j_idx, r] * W[k_idx, r]
                if abs(t_val) > tol:
                    return False
    return True


def _build_rank14():
    """Two independent Strassen 2x2x2 blocks, rank 14."""
    n, m, p = 2, 2, 4
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    # Left block (cols 0,1)
    r = 0; U[0][r]=1; U[3][r]=1; V[0][r]=1; V[5][r]=1; W[0][r]=1; W[3][r]=1
    r = 1; U[2][r]=1; U[3][r]=1; V[0][r]=1; W[1][r]=1; W[3][r]=-1
    r = 2; U[0][r]=1; V[1][r]=1; V[5][r]=-1; W[2][r]=1; W[3][r]=1
    r = 3; U[3][r]=1; V[4][r]=1; V[0][r]=-1; W[0][r]=1; W[1][r]=1
    r = 4; U[0][r]=1; U[1][r]=1; V[5][r]=1; W[0][r]=-1; W[2][r]=1
    r = 5; U[2][r]=1; U[0][r]=-1; V[0][r]=1; V[1][r]=1; W[3][r]=1
    r = 6; U[1][r]=1; U[3][r]=-1; V[4][r]=1; V[5][r]=1; W[0][r]=1
    # Right block (cols 2,3)
    r = 7; U[0][r]=1; U[3][r]=1; V[2][r]=1; V[7][r]=1; W[4][r]=1; W[7][r]=1
    r = 8; U[2][r]=1; U[3][r]=1; V[2][r]=1; W[5][r]=1; W[7][r]=-1
    r = 9; U[0][r]=1; V[3][r]=1; V[7][r]=-1; W[6][r]=1; W[7][r]=1
    r = 10; U[3][r]=1; V[6][r]=1; V[2][r]=-1; W[4][r]=1; W[5][r]=1
    r = 11; U[0][r]=1; U[1][r]=1; V[7][r]=1; W[4][r]=-1; W[6][r]=1
    r = 12; U[2][r]=1; U[0][r]=-1; V[2][r]=1; V[3][r]=1; W[7][r]=1
    r = 13; U[1][r]=1; U[3][r]=-1; V[6][r]=1; V[7][r]=1; W[4][r]=1
    return {"U": U, "V": V, "W": W}


def _build_system_for_W(U, V, n=2, m=2, p=4):
    """Build linear system A*W_flat=b for solving W given U,V."""
    rank = U.shape[1]
    num_unknowns = p * n * rank
    num_eqs = n * m * m * p * p * n
    A = np.zeros((num_eqs, num_unknowns))
    b = np.zeros(num_eqs)
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                w_idx = k * n + i
                b[eq_idx] = 1.0
                for r in range(rank):
                    col = w_idx * rank + r
                    A[eq_idx, col] = U[i*m+j, r] * V[j*p+k, r]
                eq_idx += 1
    return A, b


def _build_system_for_U(V, W, n=2, m=2, p=4):
    """Build linear system for U given V, W."""
    rank = V.shape[1]
    num_unknowns = n * m * rank
    num_eqs = n * m * m * p * p * n
    A = np.zeros((num_eqs, num_unknowns))
    b = np.zeros(num_eqs)
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                w_idx = k * n + i
                b[eq_idx] = 1.0
                for r in range(rank):
                    col = (i*m+j) * rank + r
                    A[eq_idx, col] = V[j*p+k, r] * W[w_idx, r]
                eq_idx += 1
    return A, b


def _build_system_for_V(U, W, n=2, m=2, p=4):
    """Build linear system for V given U, W."""
    rank = U.shape[1]
    num_unknowns = m * p * rank
    num_eqs = n * m * m * p * p * n
    A = np.zeros((num_eqs, num_unknowns))
    b = np.zeros(num_eqs)
    eq_idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                w_idx = k * n + i
                b[eq_idx] = 1.0
                for r in range(rank):
                    col = (j*p+k) * rank + r
                    A[eq_idx, col] = U[i*m+j, r] * W[w_idx, r]
                eq_idx += 1
    return A, b


def _verify_fast(U, V, W, n=2, m=2, p=4, tol=1e-8):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                t_val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(t_val - 1.0) > tol:
                    return False
    for i_idx in range(n * m):
        for j_idx in range(m * p):
            for k_idx in range(p * n):
                a = i_idx // m; b = i_idx % m
                b_prime = j_idx // p; c = j_idx % p
                c_prime = k_idx // n; a_prime = k_idx % n
                if a == a_prime and b == b_prime:
                    continue
                t_val = sum(U[i_idx, r] * V[j_idx, r] * W[k_idx, r] for r in range(rank))
                if abs(t_val) > tol:
                    return False
    return True


def _try_rank13_middle_product(deadline):
    """
    Middle-product decomposition: share A-side Strassen factors between the
    two 2x2x2 blocks and use cross-terms in V to exploit width-4 structure.
    Strategy:
      - Start with rank-14 Strassen.
      - Try to merge pairs of columns (from left and right blocks) by solving
        a combined system, effectively reducing rank by 1.
      - Also try structured random search with alternating minimization.
    """
    n, m, p = 2, 2, 4
    target_rank = 13
    sol14 = _build_rank14()
    U14 = np.array(sol14["U"], dtype=float)
    V14 = np.array(sol14["V"], dtype=float)
    W14 = np.array(sol14["W"], dtype=float)

    # Strategy A: Merge pairs of columns from left and right blocks
    # Try all pairs (i from left 0-6, j from right 7-13)
    for i in range(7):
        for j in range(7, 14):
            if time.time() > deadline - 20:
                return None
            # Keep all columns except j, but try to absorb column i's info into j
            # Actually: remove column i, keep j, then solve for W
            keep = [c for c in range(14) if c != i]
            U_red = U14[:, keep]
            V_red = V14[:, keep]
            # Solve for W
            A, b = _build_system_for_W(U_red, V_red, n, m, p)
            W_flat, _ = nnls(A, b)
            W_new = W_flat.reshape(p * n, target_rank)
            if _verify_fast(U_red, V_red, W_new, n, m, p, tol=1e-7):
                return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_new.tolist()}

    # Strategy B: Remove a right-block column, solve W
    for j in range(7, 14):
        if time.time() > deadline - 20:
            return None
        keep = [c for c in range(14) if c != j]
        U_red = U14[:, keep]
        V_red = V14[:, keep]
        A, b = _build_system_for_W(U_red, V_red, n, m, p)
        W_flat, _ = nnls(A, b)
        W_new = W_flat.reshape(p * n, target_rank)
        if _verify_fast(U_red, V_red, W_new, n, m, p, tol=1e-7):
            return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_new.tolist()}

    # Strategy C: Structured alternating minimization from perturbed rank-14
    for attempt in range(30):
        if time.time() > deadline - 15:
            return None
        # Perturb rank-14 solution slightly, then project to rank 13
        noise_scale = 0.1 * (attempt + 1)
        U_pert = U14.copy()
        V_pert = V14.copy()
        W_pert = W14.copy()
        # Add small noise
        U_pert = U_pert + noise_scale * np.random.randn(*U_pert.shape)
        V_pert = V_pert + noise_scale * np.random.randn(*V_pert.shape)
        W_pert = W_pert + noise_scale * np.random.randn(*W_pert.shape)
        # Truncate to rank 13 (drop last column)
        U_cur = U_pert[:, :target_rank]
        V_cur = V_pert[:, :target_rank]
        W_cur = W_pert[:, :target_rank]

        # Alternating minimization
        for it in range(80):
            if time.time() > deadline - 15:
                break
            # Update W
            A, b = _build_system_for_W(U_cur, V_cur, n, m, p)
            W_flat, _ = nnls(A, b)
            W_cur = W_flat.reshape(p * n, target_rank)
            if _verify_fast(U_cur, V_cur, W_cur, n, m, p, tol=1e-7):
                return {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}
            # Update U
            A, b = _build_system_for_U(V_cur, W_cur, n, m, p)
            U_flat, _ = nnls(A, b)
            U_cur = U_flat.reshape(n * m, target_rank)
            if _verify_fast(U_cur, V_cur, W_cur, n, m, p, tol=1e-7):
                return {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}
            # Update V
            A, b = _build_system_for_V(U_cur, W_cur, n, m, p)
            V_flat, _ = nnls(A, b)
            V_cur = V_flat.reshape(m * p, target_rank)
            if _verify_fast(U_cur, V_cur, W_cur, n, m, p, tol=1e-7):
                return {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}

    # Strategy D: Random sparse initialization with full alternating minimization
    for attempt in range(40):
        if time.time() > deadline - 10:
            return None
        U_rand = np.zeros((n * m, target_rank))
        V_rand = np.zeros((m * p, target_rank))
        for r in range(target_rank):
            u_nz = random.randint(1, 3)
            v_nz = random.randint(1, 4)
            u_idx = random.sample(range(n * m), min(u_nz, n * m))
            v_idx = random.sample(range(m * p), min(v_nz, m * p))
            for idx in u_idx:
                U_rand[idx, r] = random.choice([-1.0, 1.0])
            for idx in v_idx:
                V_rand[idx, r] = random.choice([-1.0, 1.0])

        # Solve for W
        A, b = _build_system_for_W(U_rand, V_rand, n, m, p)
        W_flat, _ = nnls(A, b)
        W_cur = W_flat.reshape(p * n, target_rank)

        U_cur = U_rand.copy()
        V_cur = V_rand.copy()

        for it in range(60):
            if time.time() > deadline - 10:
                break
            A, b = _build_system_for_W(U_cur, V_cur, n, m, p)
            W_flat, _ = nnls(A, b)
            W_cur = W_flat.reshape(p * n, target_rank)
            if _verify_fast(U_cur, V_cur, W_cur, n, m, p, tol=1e-7):
                return {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}
            A, b = _build_system_for_U(V_cur, W_cur, n, m, p)
            U_flat, _ = nnls(A, b)
            U_cur = U_flat.reshape(n * m, target_rank)
            if _verify_fast(U_cur, V_cur, W_cur, n, m, p, tol=1e-7):
                return {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}
            A, b = _build_system_for_V(U_cur, W_cur, n, m, p)
            V_flat, _ = nnls(A, b)
            V_cur = V_flat.reshape(m * p, target_rank)
            if _verify_fast(U_cur, V_cur, W_cur, n, m, p, tol=1e-7):
                return {"U": U_cur.tolist(), "V": V_cur.tolist(), "W": W_cur.tolist()}

    return None


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    # Build the known rank-14 solution
    result_14 = _build_rank14()

    # Verify the rank-14 solution
    if _verify(result_14["U"], result_14["V"], result_14["W"], n, m, p):
        # Try to find a rank-13 solution using middle-product structured search
        result_13 = _try_rank13_middle_product(deadline)
        if result_13 is not None:
            return result_13
        return result_14

    # Fallback to schoolbook rank 16
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if time.time() > deadline:
                    return {"U": U, "V": V, "W": W}
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
