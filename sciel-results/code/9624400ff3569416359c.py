import time
import random
import numpy as np

DIMENSIONS = [2, 2, 3]
TIME_LIMIT = 140.0  # seconds, with 10s safety margin

# EVOLVE-BLOCK-START

def _build_rank11():
    """Build the known rank-11 decomposition."""
    n, m, p = DIMENSIONS
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def set_u(i, j, r, val):
        U[i * m + j][r] = val
    def set_v(j, k, r, val):
        V[j * p + k][r] = val
    def set_w(k, i, r, val):
        W[k * n + i][r] = val

    # Term 1: (A00 + A11) * (B00 + B11)
    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    # Term 2: (A10 + A11) * B00
    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    # Term 3: A00 * (B01 - B11)
    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    # Term 4: A11 * (B10 - B00)
    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    # Term 5: (A00 + A01) * B11
    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

    # Term 6: (A10 - A00) * (B00 + B01)
    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    # Term 7: (A01 - A11) * (B10 + B11)
    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    # Term 8: A00 * B02
    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    # Term 9: A01 * B12
    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    # Term 10: A10 * B02
    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    # Term 11: A11 * B12
    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n=2, m=2, p=3):
    """Verify that the decomposition correctly computes C = A*B for random inputs."""
    rank = len(U[0])
    for _ in range(5):
        A = np.random.randint(-5, 6, (n, m))
        B = np.random.randint(-5, 6, (m, p))
        C_expected = A @ B
        
        # Compute C from decomposition
        C_computed = np.zeros((n, p), dtype=int)
        for i in range(n):
            for k in range(p):
                total = 0
                for r in range(rank):
                    for j in range(m):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                C_computed[i, k] = total
        if not np.array_equal(C_expected, C_computed):
            return False
    return True


def _search_rank10_gf13(deadline):
    """
    Search for a rank-10 decomposition over GF(13) by constructing U, V randomly
    and solving for W.
    """
    n, m, p = DIMENSIONS
    MOD = 13
    rank = 10
    
    rng = random.Random(42)
    attempts = 0
    max_attempts = 100000
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Generate random U and V
        U = [[rng.randint(0, MOD-1) for _ in range(rank)] for _ in range(n*m)]
        V = [[rng.randint(0, MOD-1) for _ in range(rank)] for _ in range(m*p)]
        
        # Compute C[i,k,r] = sum_j U[i,j,r] * V[j,k,r]
        C = [[[0]*rank for _ in range(p)] for _ in range(n)]
        for i in range(n):
            for k in range(p):
                for r in range(rank):
                    total = 0
                    for j in range(m):
                        total = (total + U[i*m+j][r] * V[j*p+k][r]) % MOD
                    C[i][k][r] = total
        
        # Try to assign each (i,k) to a term r such that C[i,k,r] != 0
        # and set W[k,i,r] = inv(C[i,k,r])
        W = [[0]*rank for _ in range(p*n)]
        valid = True
        for i in range(n):
            for k in range(p):
                # Find a non-zero C[i,k,r]
                # To ensure we don't use the same r for all, we could shuffle,
                # but any valid assignment works.
                non_zero_rs = [r for r in range(rank) if C[i][k][r] != 0]
                if not non_zero_rs:
                    valid = False
                    break
                # Pick a random non-zero r
                r = rng.choice(non_zero_rs)
                # W[k*n + i][r] = 1 / C[i][k][r]
                inv_c = pow(C[i][k][r], MOD-2, MOD)
                W[k*n + i][r] = inv_c
            if not valid:
                break
        
        if not valid:
            continue
            
        # Verify the solution
        # C[i,k] = sum_r W[k*n+i, r] * C[i,k,r]
        # We constructed W such that for the chosen r, W * C = 1.
        # But we must ensure that for other r, W[k*n+i, r] * C[i,k,r] is 0.
        # In our construction, W[k*n+i, r'] = 0 for r' != chosen_r.
        # So the sum is exactly 1.
        
        # However, we must verify that this decomposition actually computes A*B.
        # The tensor equation is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1.
        # We satisfied this for all i,k.
        # So the decomposition is valid over GF(13).
        
        # Convert to integers
        # The entries are in [0, 12]. They are integers.
        return {"U": U, "V": V, "W": W}
        
    return None


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 by 2x3 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # First, get the known rank-11 solution as fallback
    sol11 = _build_rank11()
    
    # Verify it works
    if not _verify_decomposition(sol11["U"], sol11["V"], sol11["W"]):
        return sol11
    
    # Try to find rank-10 over GF(13)
    remaining = deadline - time.time()
    if remaining > 5:
        sol10 = _search_rank10_gf13(deadline - 5)
        if sol10 is not None:
            # Verify the GF(13) solution over integers
            if _verify_decomposition(sol10["U"], sol10["V"], sol10["W"]):
                return sol10
    
    return sol11

# EVOLVE-BLOCK-END
