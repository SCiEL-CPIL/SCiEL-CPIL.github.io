"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify the tensor decomposition"""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    total = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total[i,j,k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
    expected = np.ones((n, m, p))
    if not np.allclose(total, expected, atol=1e-6):
        return False, np.max(np.abs(total - expected))
    return True, 0.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    def _build_rank54():
        """Rank 54 solution using Strassen on 2x2 blocks for j-pairs and k-pairs."""
        rank = 54
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
                U[0*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                # P2 = (A_1,j0 + A_1,j1) B_j0,k0
                U[1*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                W[k0*n + 1][col] += 1
                W[k1*n + 1][col] -= 1
                col += 1
                # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
                U[0*m + j0][col] += 1
                V[j0*p + k1][col] += 1
                V[j1*p + k1][col] -= 1
                W[k1*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
                U[1*m + j1][col] += 1
                V[j1*p + k0][col] += 1
                V[j0*p + k0][col] -= 1
                W[k0*n + 0][col] += 1
                W[k0*n + 1][col] += 1
                col += 1
                # P5 = (A_0,j0 + A_0,j1) B_j1,k1
                U[0*m + j0][col] += 1
                U[0*m + j1][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] -= 1
                W[k1*n + 0][col] += 1
                col += 1
                # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
                U[1*m + j0][col] += 1
                U[0*m + j0][col] -= 1
                V[j0*p + k0][col] += 1
                V[j0*p + k1][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
                U[0*m + j1][col] += 1
                U[1*m + j1][col] -= 1
                V[j1*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                col += 1
        
        # Remaining j=4: direct multiplication
        for k in range(p):
            U[0*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            U[1*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
        return {"U": U, "V": V, "W": W}

    def _build_rank48():
        """
        Rank 48 solution: Use Strassen on 2x2x2 blocks for j={0,1}x all k-pairs,
        and for j={2,3,4} use a 2x3x6 sub-decomposition.
        
        For j={0,1}: 2 rows, 6 cols in B. Use 3 k-pairs with Strassen: 3*7=21
        For j={2,3,4}: 3 rows, 6 cols. 
          - j={2,3}: 2x2x6 -> use 3 k-pairs with Strassen: 3*7=21
          - j={4}: 1x6 direct: 6 terms
        Total: 21 + 21 + 6 = 48
        """
        rank = 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        # j={0,1} with each k_pair: Strassen 2x2x2
        for k0, k1 in k_pairs:
            j0, j1 = 0, 1
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            # P2 = (A_1,j0 + A_1,j1) B_j0,k0
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            # P5 = (A_0,j0 + A_0,j1) B_j1,k1
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
        
        # j={2,3} with each k_pair: Strassen 2x2x2
        for k0, k1 in k_pairs:
            j0, j1 = 2, 3
            # P1 = (A_0,j0 + A_1,j1)(B_j0,k0 + B_j1,k1)
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            # P2 = (A_1,j0 + A_1,j1) B_j0,k0
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            # P3 = A_0,j0 (B_j0,k1 - B_j1,k1)
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            # P4 = A_1,j1 (B_j1,k0 - B_j0,k0)
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            # P5 = (A_0,j0 + A_0,j1) B_j1,k1
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            # P6 = (A_1,j0 - A_0,j0)(B_j0,k0 + B_j0,k1)
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            # P7 = (A_0,j1 - A_1,j1)(B_j1,k0 + B_j1,k1)
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
        
        # j=4: direct
        for k in range(p):
            U[0*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            U[1*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
        return {"U": U, "V": V, "W": W}

    def _build_rank42():
        """
        Attempt rank 42: Use a 2x3x3 block decomposition.
        Split j into {0,1,2} and {3,4}, k into {0,1,2} and {3,4,5}.
        For 2x3x3 blocks, use a known rank-11 decomposition if available,
        otherwise fall back to direct.
        
        Actually, let's try: 2x3x6 as 2x3x3 + 2x3x3 = 2 * rank(2x3x3)
        If rank(2x3x3) = 11, total = 22 for j={0,1,2}
        Then j={3,4} as 2x2x6 = 2 * rank(2x2x3) = 2 * 7 = 14? No, 2x2x6.
        
        Let me just try direct construction and verify.
        """
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        
        # For j={0,1,2}, k={0,1,2}: use a 2x3x3 decomposition
        # Try a structured approach: treat as 2x2x3 + 2x1x3
        # 2x2x3: use Strassen-like, rank = 7 (for 2x2x2) + 2 (for 2x2x1) = 9? 
        # Actually 2x2x3 can be done in 7+2=9 or maybe less.
        # Let's just use direct for now and see if we can verify.
        
        # Direct for j={0,1,2}, k={0,1,2}: 2*3*3 = 18 terms
        for j in range(3):
            for k in range(3):
                for i in range(2):
                    U[i*m + j][col] += 1
                    V[j*p + k][col] += 1
                    W[k*n + i][col] += 1
                    col += 1
        
        # Direct for j={0,1,2}, k={3,4,5}: 2*3*3 = 18 terms
        for j in range(3):
            for k in range(3, 6):
                for i in range(2):
                    U[i*m + j][col] += 1
                    V[j*p + k][col] += 1
                    W[k*n + i][col] += 1
                    col += 1
        
        # Direct for j={3,4}, k={0,1,2}: 2*2*3 = 12 terms
        for j in range(3, 5):
            for k in range(3):
                for i in range(2):
                    U[i*m + j][col] += 1
                    V[j*p + k][col] += 1
                    W[k*n + i][col] += 1
                    col += 1
        
        # Direct for j={3,4}, k={3,4,5}: 2*2*3 = 12 terms
        for j in range(3, 5):
            for k in range(3, 6):
                for i in range(2):
                    U[i*m + j][col] += 1
                    V[j*p + k][col] += 1
                    W[k*n + i][col] += 1
                    col += 1
        
        # Total: 18+18+12+12 = 60, way too many. This won't work.
        # Return None to indicate failure
        return None

    # Build and verify rank-54 baseline
    baseline = _build_rank54()
    valid, diff = _verify(baseline["U"], baseline["V"], baseline["W"], n, m, p)
    
    if not valid:
        return baseline

    best = baseline
    best_rank = 54

    # Try rank-48
    if time.time() < deadline:
        r48 = _build_rank48()
        valid48, diff48 = _verify(r48["U"], r48["V"], r48["W"], n, m, p)
        if valid48:
            best = r48
            best_rank = 48

    # Try rank-42
    if time.time() < deadline:
        r42 = _build_rank42()
        if r42 is not None:
            valid42, diff42 = _verify(r42["U"], r42["V"], r42["W"], n, m, p)
            if valid42:
                best = r42
                best_rank = 42

    return best
# EVOLVE-BLOCK-END
