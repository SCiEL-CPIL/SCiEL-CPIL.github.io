import time
import sys
import random
import numpy as np

DIMENSIONS = [2, 4, 4]

def _build_strassen_28():
    """Build the baseline rank-28 Strassen decomposition."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0, start_r):
        a11 = a_r0 * m + a_c0
        a12 = a_r0 * m + a_c0 + 1
        a21 = (a_r0 + 1) * m + a_c0
        a22 = (a_r0 + 1) * m + a_c0 + 1
        b11 = b_r0 * p + b_c0
        b12 = b_r0 * p + b_c0 + 1
        b21 = (b_r0 + 1) * p + b_c0
        b22 = (b_r0 + 1) * p + b_c0 + 1
        c11_idx = c_c0 * n + c_r0
        c12_idx = (c_c0 + 1) * n + c_r0
        c21_idx = c_c0 * n + (c_r0 + 1)
        c22_idx = (c_c0 + 1) * n + (c_r0 + 1)

        r = start_r
        U[a11][r] += 1; U[a22][r] += 1
        V[b11][r] += 1; V[b22][r] += 1
        W[c11_idx][r] += 1; W[c22_idx][r] += 1
        r += 1
        U[a21][r] += 1; U[a22][r] += 1
        V[b11][r] += 1
        W[c21_idx][r] += 1; W[c22_idx][r] += -1
        r += 1
        U[a11][r] += 1
        V[b12][r] += 1; V[b22][r] += -1
        W[c12_idx][r] += 1; W[c22_idx][r] += 1
        r += 1
        U[a22][r] += 1
        V[b21][r] += 1; V[b11][r] += -1
        W[c11_idx][r] += 1; W[c21_idx][r] += 1
        r += 1
        U[a11][r] += 1; U[a12][r] += 1
        V[b22][r] += 1
        W[c11_idx][r] += -1; W[c12_idx][r] += 1
        r += 1
        U[a21][r] += 1; U[a11][r] += -1
        V[b11][r] += 1; V[b12][r] += 1
        W[c22_idx][r] += 1
        r += 1
        U[a12][r] += 1; U[a22][r] += -1
        V[b21][r] += 1; V[b22][r] += 1
        W[c11_idx][r] += 1
        r += 1
        return r

    r = add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    r = add_strassen_block(0, 2, 2, 0, 0, 0, r)
    r = add_strassen_block(0, 0, 0, 2, 0, 2, r)
    r = add_strassen_block(0, 2, 2, 2, 0, 2, r)
    assert r == 28
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    for trial in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_decomposed = np.zeros((n, p))
        for r in range(rank):
            a_val = 0
            for i in range(n):
                for j in range(m):
                    a_val += U[i * m + j][r] * A[i, j]
            b_val = 0
            for j in range(m):
                for k in range(p):
                    b_val += V[j * p + k][r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_decomposed[i, k] += W[k * n + i][r] * a_val * b_val
        
        if not np.allclose(C_decomposed, C_expected, atol=1e-10):
            return False
    return True


def _verify_fast(U, V, W, n, m, p):
    """Faster verification using matrix operations."""
    rank = len(U[0])
    U_mat = np.array(U).T  # (rank, n*m)
    V_mat = np.array(V).T  # (rank, m*p)
    W_mat = np.array(W).T  # (rank, p*n)
    
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # For each r: a_r = U_mat[r] @ vec(A), b_r = V_mat[r] @ vec(B)
        A_vec = A.reshape(-1)  # (n*m,)
        B_vec = B.reshape(-1)  # (m*p,)
        
        a_vals = U_mat @ A_vec  # (rank,)
        b_vals = V_mat @ B_vec  # (rank,)
        
        # C_decomposed[i,k] = sum_r W_mat[r, k*n+i] * a_vals[r] * b_vals[r]
        prod = a_vals * b_vals  # (rank,)
        C_flat = W_mat.T @ prod  # (p*n,)
        C_decomposed = C_flat.reshape(p, n).T  # (n, p)
        
        if not np.allclose(C_decomposed, C_expected, atol=1e-10):
            return False
    return True


def _try_eliminate_term(U, V, W, n, m, p, term_idx, deadline):
    """Try to eliminate term_idx by absorbing it into other terms."""
    rank = len(U[0])
    if rank <= 1:
        return None
    
    # Create copies with term_idx removed
    new_rank = rank - 1
    U_new = [row[:] for row in U]
    V_new = [row[:] for row in V]
    W_new = [row[:] for row in W]
    
    # Remove the term by shifting columns
    for row in U_new:
        del row[term_idx]
    for row in V_new:
        del row[term_idx]
    for row in W_new:
        del row[term_idx]
    
    # Check if the remaining terms still work
    if _verify_fast(U_new, V_new, W_new, n, m, p):
        return {"U": U_new, "V": V_new, "W": W_new}
    return None


def _try_random_reduction(U, V, W, n, m, p, target_rank, deadline):
    """Try to find a rank-target_rank decomposition via random search."""
    # This is a placeholder for more sophisticated search
    # For now, just try eliminating terms one by one
    return None


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Start with the rank-28 Strassen decomposition
    result = _build_strassen_28()
    U = result["U"]
    V = result["V"]
    W = result["W"]
    
    # Verify baseline
    assert _verify_fast(U, V, W, n, m, p), "Baseline decomposition failed verification"
    
    best_rank = 28
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to eliminate terms one by one
    current_U, current_V, current_W = U, V, W
    current_rank = 28
    
    # Try each term for elimination
    for term_idx in range(current_rank):
        if time.time() > deadline:
            break
        
        result = _try_eliminate_term(current_U, current_V, current_W, n, m, p, term_idx, deadline)
        if result is not None:
            current_U = result["U"]
            current_V = result["V"]
            current_W = result["W"]
            current_rank -= 1
            best_rank = current_rank
            best_result = result
            # Restart the loop since indices changed
            term_idx = -1
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    n, m, p = DIMENSIONS
    print(f"U shape: ({len(result['U'])}, {len(result['U'][0])})")
    print(f"V shape: ({len(result['V'])}, {len(result['V'][0])})")
    print(f"W shape: ({len(result['W'])}, {len(result['W'][0])})")
    print("Verification:", _verify_fast(result['U'], result['V'], result['W'], n, m, p))
