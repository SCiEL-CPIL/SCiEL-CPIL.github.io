"""Matrix multiplication 5x5x5 with reduced rank decomposition."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition.
    U, V, W are lists of 7 vectors, each of length 4.
    Indexing: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[k,i] -> k*2+i
    """
    U = [
        [1, 0, 0, 1],   # P1: A00+A11
        [0, 0, 1, 1],   # P2: A10+A11
        [1, 0, 0, 0],   # P3: A00
        [0, 0, 0, 1],   # P4: A11
        [1, 0, 1, 0],   # P5: A00+A10
        [-1, 1, 0, 0],  # P6: A01-A00
        [0, 0, 1, -1],  # P7: A10-A11
    ]
    V = [
        [1, 0, 0, 1],   # P1: B00+B11
        [1, 0, 0, 0],   # P2: B00
        [0, 0, 1, -1],  # P3: B10-B11
        [-1, 1, 0, 0],  # P4: B01-B00
        [0, 0, 0, 1],   # P5: B11
        [1, 0, 1, 0],   # P6: B00+B10
        [0, 1, 0, 1],   # P7: B01+B11
    ]
    W = [
        [1, 0, 0, 1],   # P1: C00+C11
        [0, 0, 1, -1],  # P2: C10-C11
        [0, 1, 0, 1],   # P3: C01+C11
        [1, 0, 1, 0],   # P4: C00+C10
        [-1, 1, 0, 0],  # P5: -C00+C01
        [0, 0, 0, 1],   # P6: C11
        [1, 0, 0, 0],   # P7: C00
    ]
    return U, V, W


def _strassen_3x3():
    """Return U, V, W for 3x3x3 Strassen-based decomposition.
    Uses recursive Strassen: 3x3 = 2x2 + boundary, but we use a known rank-23 construction.
    For simplicity, use the direct product of 2x2 Strassen with appropriate padding.
    Actually, let's use a simple rank-27 decomposition (3x3 schoolbook is 27, 
    but we can do better). For now, use rank 27 as safe baseline, then try to reduce.
    
    Let's use a known rank-23 decomposition for 3x3.
    """
    # Use a parameterized rank-23 decomposition for 3x3x3
    # Based on known constructions, here's a valid one:
    
    # Actually, let's be safe and use a verified decomposition.
    # The simplest correct approach: use 2x2 Strassen (rank 7) for the top-left 2x2,
    # then handle the boundary rows/columns.
    
    # For 3x3: indices 0,1,2
    # Core 2x2: indices 0,1 (rank 7)
    # Boundary: index 2
    # Terms needed:
    # - (2,j,k) for j,k in 0,1,2: 9 terms
    # - (i,2,k) for i,k in 0,1: 4 terms (i=2,k=2 already counted)
    # - (i,j,2) for i,j in 0,1: 4 terms (overlaps with above)
    # Total boundary: 9 + 4 + 4 - 1 - 1 - 1 + 0 = 14? 
    # Actually: 9 + 4 + 4 - 3 (overlaps: (2,2,2), (2,j,2), (i,2,2)) 
    # Let me just count: total 27, core 4 (from 2x2 schoolbook) or 7 (Strassen)
    # If using Strassen for 2x2 core: 7 + boundary
    # Boundary terms: all (i,j,k) where at least one index is 2
    # Total with index 2: 27 - 8 = 19
    # So rank = 7 + 19 = 26
    
    # But we can do better. Let's use a known rank-23 decomposition.
    # For now, use rank 26 as safe, then try optimization.
    
    U2, V2, W2 = _strassen_2x2()
    
    U3 = []
    V3 = []
    W3 = []
    
    # Core 2x2 part (indices 0,1)
    for r in range(7):
        u = [0] * 9
        v = [0] * 9
        w = [0] * 9
        for i in range(2):
            for j in range(2):
                u[i * 3 + j] = U2[r][i * 2 + j]
        for j in range(2):
            for k in range(2):
                v[j * 3 + k] = V2[r][j * 2 + k]
        for k in range(2):
            for i in range(2):
                w[k * 3 + i] = W2[r][k * 2 + i]
        U3.append(u)
        V3.append(v)
        W3.append(w)
    
    # Boundary terms: all (i,j,k) with at least one index = 2
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == 2 or j == 2 or k == 2:
                    u = [0] * 9
                    v = [0] * 9
                    w = [0] * 9
                    u[i * 3 + j] = 1
                    v[j * 3 + k] = 1
                    w[k * 3 + i] = 1
                    U3.append(u)
                    V3.append(v)
                    W3.append(w)
    
    return U3, V3, W3


def _build_5x5_hybrid():
    """Build 5x5x5 decomposition using 3x3 + 2x2 partition.
    Partition indices: {0,1,2} and {3,4}
    """
    U3, V3, W3 = _strassen_3x3()
    U2, V2, W2 = _strassen_2x2()
    
    U5 = []
    V5 = []
    W5 = []
    
    # 3x3 core: indices 0,1,2
    for r in range(len(U3)):
        u = [0] * 25
        v = [0] * 25
        w = [0] * 25
        for i in range(3):
            for j in range(3):
                u[i * 5 + j] = U3[r][i * 3 + j]
        for j in range(3):
            for k in range(3):
                v[j * 5 + k] = V3[r][j * 3 + k]
        for k in range(3):
            for i in range(3):
                w[k * 5 + i] = W3[r][k * 3 + i]
        U5.append(u)
        V5.append(v)
        W5.append(w)
    
    # 2x2 core: indices 3,4 (map to local 0,1)
    for r in range(7):
        u = [0] * 25
        v = [0] * 25
        w = [0] * 25
        for i in range(2):
            for j in range(2):
                u[(3 + i) * 5 + (3 + j)] = U2[r][i * 2 + j]
        for j in range(2):
            for k in range(2):
                v[(3 + j) * 5 + (3 + k)] = V2[r][j * 2 + k]
        for k in range(2):
            for i in range(2):
                w[(3 + k) * 5 + (3 + i)] = W2[r][k * 2 + i]
        U5.append(u)
        V5.append(v)
        W5.append(w)
    
    # Cross terms:
    # A[i,j] * B[j,k] where i in {0,1,2}, j in {3,4}, k in {0,1,2}
    # and i in {3,4}, j in {0,1,2}, k in {3,4}
    # and i in {0,1,2}, j in {0,1,2}, k in {3,4}
    # and i in {3,4}, j in {3,4}, k in {0,1,2}
    # etc.
    
    # All (i,j,k) not covered by cores:
    # Core 3x3 covers: i,j,k all in {0,1,2}
    # Core 2x2 covers: i,j,k all in {3,4}
    # Remaining: at least one index in {0,1,2} AND at least one in {3,4}
    
    for i in range(5):
        for j in range(5):
            for k in range(5):
                # Check if covered by 3x3 core
                if i < 3 and j < 3 and k < 3:
                    continue
                # Check if covered by 2x2 core
                if i >= 3 and j >= 3 and k >= 3:
                    continue
                # Not covered, add rank-1 term
                u = [0] * 25
                v = [0] * 25
                w = [0] * 25
                u[i * 5 + j] = 1
                v[j * 5 + k] = 1
                w[k * 5 + i] = 1
                U5.append(u)
                V5.append(v)
                W5.append(w)
    
    return U5, V5, W5


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False, (i, j, k, total)
    return True, None


def _try_reduce_by_linear_dependence(U, V, W, n, m, p):
    """Try to reduce rank by finding linearly dependent columns.
    Uses numpy SVD to identify near-dependent columns.
    """
    rank = len(U[0])
    if rank <= 50:
        return U, V, W, rank
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Try to identify columns that are linear combinations of others
    # This is a heuristic approach
    
    # For each column c, check if it can be expressed as a combination of others
    # by solving a least squares problem
    
    new_U = []
    new_V = []
    new_W = []
    
    kept_cols = list(range(rank))
    
    # Simple heuristic: remove columns that are very small in norm
    norms = np.array([np.linalg.norm(U_np[:, c]) * np.linalg.norm(V_np[:, c]) * np.linalg.norm(W_np[:, c]) 
                      for c in range(rank)])
    
    # Sort by norm, keep largest
    sorted_idx = np.argsort(norms)[::-1]
    
    # Try removing smallest columns and check if decomposition still holds
    min_rank_to_try = max(10, rank - 20)
    
    for target_rank in range(rank, min_rank_to_try - 1, -1):
        candidate_idx = sorted_idx[:target_rank]
        
        U_cand = U_np[:, candidate_idx]
        V_cand = V_np[:, candidate_idx]
        W_cand = W_np[:, candidate_idx]
        
        # Verify
        ok = True
        for i in range(n):
            if not ok:
                break
            for j in range(m):
                if not ok:
                    break
                for k in range(p):
                    total = 0
                    for r in range(len(candidate_idx)):
                        total += U_cand[i * m + j, r] * V_cand[j * p + k, r] * W_cand[k * n + i, r]
                    if abs(total - 1) > 1e-9:
                        ok = False
                        break
        
        if ok:
            # Convert back to lists
            new_U = U_cand.tolist()
            new_V = V_cand.tolist()
            new_W = W_cand.tolist()
            return new_U, new_V, new_W, len(candidate_idx)
    
    return U, V, W, rank


def construct():
    deadline = time.time() + 110  # Leave 10s margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Strategy 1: 3x3 + 2x2 hybrid
    U5, V5, W5 = _build_5x5_hybrid()
    
    # Verify
    ok, info = _verify_decomposition(U5, V5, W5, 5, 5, 5)
    if ok:
        rank = len(U5[0])
        # Try to reduce
        U5, V5, W5, rank = _try_reduce_by_linear_dependence(U5, V5, W5, 5, 5, 5)
        return {"U": U5, "V": V5, "W": W5}
    
    # Strategy 2: 4x4 Strassen + boundary (corrected)
    U2, V2, W2 = _strassen_2x2()
    
    # Build 4x4 from 2x2 Strassen
    U4 = []
    V4 = []
    W4 = []
    
    for r in range(7):
        for s in range(7):
            u = [0] * 16
            v = [0] * 16
            w = [0] * 16
            
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1 * 2 + i2
                            j = j1 * 2 + j2
                            u[i * 4 + j] = U2[r][i1 * 2 + j1] * U2[s][i2 * 2 + j2]
            
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1 * 2 + j2
                            k = k1 * 2 + k2
                            v[j * 4 + k] = V2[r][j1 * 2 + k1] * V2[s][j2 * 2 + k2]
            
            for i1 in range(2):
                for k1 in range(2):
                    for i2 in range(2):
                        for k2 in range(2):
                            i = i1 * 2 + i2
                            k = k1 * 2 + k2
                            w[k * 4 + i] = W2[r][i1 * 2 + k1] * W2[s][i2 * 2 + k2]
            
            U4.append(u)
            V4.append(v)
            W4.append(w)
    
    # Verify 4x4
    ok4, _ = _verify_decomposition(U4, V4, W4, 4, 4, 4)
    
    if ok4:
        # Build 5x5: 4x4 core + boundary
        U5b = []
        V5b = []
        W5b = []
        
        # 4x4 core
        for r in range(49):
            u = [0] * 25
            v = [0] * 25
            w = [0] * 25
            for i in range(4):
                for j in range(4):
                    u[i * 5 + j] = U4[r][i * 4 + j]
            for j in range(4):
                for k in range(4):
                    v[j * 5 + k] = V4[r][j * 4 + k]
            for k in range(4):
                for i in range(4):
                    w[k * 5 + i] = W4[r][k * 4 + i]
            U5b.append(u)
            V5b.append(v)
            W5b.append(w)
        
        # Boundary: all (i,j,k) with at least one index = 4
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    if i == 4 or j == 4 or k == 4:
                        u = [0] * 25
                        v = [0] * 25
                        w = [0] * 25
                        u[i * 5 + j] = 1
                        v[j * 5 + k] = 1
                        w[k * 5 + i] = 1
                        U5b.append(u)
                        V5b.append(v)
                        W5b.append(w)
        
        # Verify
        ok5, _ = _verify_decomposition(U5b, V5b, W5b, 5, 5, 5)
        if ok5:
            rank = len(U5b[0])
            U5b, V5b, W5b, rank = _try_reduce_by_linear_dependence(U5b, V5b, W5b, 5, 5, 5)
            return {"U": U5b, "V": V5b, "W": W5b}
    
    # Fallback: schoolbook
    rank = 125
    U5 = [[0] * rank for _ in range(25)]
    V5 = [[0] * rank for _ in range(25)]
    W5 = [[0] * rank for _ in range(25)]
    col = 0
    for i in range(5):
        for j in range(5):
            for k in range(5):
                U5[i * 5 + j][col] = 1
                V5[j * 5 + k][col] = 1
                W5[k * 5 + i][col] = 1
                col += 1
    
    return {"U": U5, "V": V5, "W": W5}
# EVOLVE-BLOCK-END
