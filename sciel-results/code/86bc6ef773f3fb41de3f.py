"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    def add_strassen_block(U, V, W, i0, j0, k0, col_offset):
        """Add a 2x2x2 Strassen decomposition starting at (i0, j0, k0)."""
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
    
    def add_laderman_block(U, V, W, i0, j0, k0, col_offset):
        """Add a 3x3x3 Laderman decomposition starting at (i0, j0, k0). Rank 23."""
        # Laderman's algorithm for 3x3x3 matrix multiplication
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m+j0][c] += 1; U[(i0+1)*m+(j0+1)][c] += 1
        V[j0*p+k0][c] += 1; V[(j0+1)*p+(k0+1)][c] += 1
        W[k0*n+i0][c] += 1; W[(k0+1)*n+(i0+1)][c] += 1
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m+j0][c] += 1; U[(i0+1)*m+(j0+1)][c] += 1
        V[j0*p+k0][c] += 1
        W[k0*n+(i0+1)][c] += 1; W[(k0+1)*n+(i0+1)][c] += -1
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m+j0][c] += 1
        V[j0*p+(k0+1)][c] += 1; V[(j0+1)*p+(k0+1)][c] += -1
        W[(k0+1)*n+i0][c] += 1; W[(k0+1)*n+(i0+1)][c] += 1
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m+(j0+1)][c] += 1
        V[(j0+1)*p+k0][c] += 1; V[j0*p+k0][c] += -1
        W[k0*n+i0][c] += 1; W[k0*n+(i0+1)][c] += 1
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m+j0][c] += 1; U[i0*m+(j0+1)][c] += 1
        V[(j0+1)*p+(k0+1)][c] += 1
        W[k0*n+i0][c] += -1; W[(k0+1)*n+i0][c] += 1
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m+j0][c] += 1; U[i0*m+j0][c] += -1
        V[j0*p+k0][c] += 1; V[j0*p+(k0+1)][c] += 1
        W[(k0+1)*n+(i0+1)][c] += 1
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m+(j0+1)][c] += 1; U[(i0+1)*m+(j0+1)][c] += -1
        V[(j0+1)*p+k0][c] += 1; V[(j0+1)*p+(k0+1)][c] += 1
        W[k0*n+i0][c] += 1
        # M8 = (A00+A01+A10+A11)(B00+B01+B10+B11) - simplified
        # Actually Laderman has 23 terms. Let me use the standard formulation.
        # For simplicity, let's use a known rank-23 decomposition for 3x3x3.
        # M8 = A22(B00+B01+B10+B11)
        c = col_offset + 7
        U[(i0+2)*m+(j0+2)][c] += 1
        V[j0*p+k0][c] += 1; V[j0*p+(k0+1)][c] += 1
        V[(j0+1)*p+k0][c] += 1; V[(j0+1)*p+(k0+1)][c] += 1
        W[k0*n+(i0+2)][c] += 1; W[(k0+1)*n+(i0+2)][c] += 1
        W[k0*n+i0][c] += 1; W[k0*n+(i0+1)][c] += 1
        W[(k0+1)*n+i0][c] += 1; W[(k0+1)*n+(i0+1)][c] += 1
        # M9 = (A02+A12)B00
        c = col_offset + 8
        U[i0*m+(j0+2)][c] += 1; U[(i0+1)*m+(j0+2)][c] += 1
        V[j0*p+k0][c] += 1
        W[k0*n+i0][c] += 1; W[k0*n+(i0+1)][c] += 1
        # M10 = A02(B01-B11)
        c = col_offset + 9
        U[i0*m+(j0+2)][c] += 1
        V[j0*p+(k0+1)][c] += 1; V[(j0+1)*p+(k0+1)][c] += -1
        W[(k0+1)*n+i0][c] += 1; W[(k0+1)*n+(i0+1)][c] += 1
        # M11 = A12(B10-B00)
        c = col_offset + 10
        U[(i0+1)*m+(j0+2)][c] += 1
        V[(j0+1)*p+k0][c] += 1; V[j0*p+k0][c] += -1
        W[k0*n+i0][c] += 1; W[k0*n+(i0+1)][c] += 1
        # M12 = (A02+A12)B22
        c = col_offset + 11
        U[i0*m+(j0+2)][c] += 1; U[(i0+1)*m+(j0+2)][c] += 1
        V[(j0+2)*p+(k0+2)][c] += 1
        W[(k0+2)*n+i0][c] += 1; W[(k0+2)*n+(i0+1)][c] += 1
        # M13 = (A20+A21)B00
        c = col_offset + 12
        U[(i0+2)*m+j0][c] += 1; U[(i0+2)*m+(j0+1)][c] += 1
        V[j0*p+k0][c] += 1
        W[k0*n+(i0+2)][c] += 1; W[k0*n+(i0+2)+1][c] += 1
        # M14 = A20(B01-B11)
        c = col_offset + 13
        U[(i0+2)*m+j0][c] += 1
        V[j0*p+(k0+1)][c] += 1; V[(j0+1)*p+(k0+1)][c] += -1
        W[(k0+1)*n+(i0+2)][c] += 1; W[(k0+1)*n+(i0+2)+1][c] += 1
        # M15 = A21(B10-B00)
        c = col_offset + 14
        U[(i0+2)*m+(j0+1)][c] += 1
        V[(j0+1)*p+k0][c] += 1; V[j0*p+k0][c] += -1
        W[k0*n+(i0+2)][c] += 1; W[k0*n+(i0+2)+1][c] += 1
        # M16 = (A20+A21)B22
        c = col_offset + 15
        U[(i0+2)*m+j0][c] += 1; U[(i0+2)*m+(j0+1)][c] += 1
        V[(j0+2)*p+(k0+2)][c] += 1
        W[(k0+2)*n+(i0+2)][c] += 1; W[(k0+2)*n+(i0+2)+1][c] += 1
        # M17 = A22(B20-B00)
        c = col_offset + 16
        U[(i0+2)*m+(j0+2)][c] += 1
        V[(j0+2)*p+k0][c] += 1; V[j0*p+k0][c] += -1
        W[k0*n+(i0+2)][c] += 1; W[k0*n+(i0+2)+1][c] += 1
        # M18 = A22(B21-B01)
        c = col_offset + 17
        U[(i0+2)*m+(j0+2)][c] += 1
        V[(j0+2)*p+(k0+1)][c] += 1; V[j0*p+(k0+1)][c] += -1
        W[(k0+1)*n+(i0+2)][c] += 1; W[(k0+1)*n+(i0+2)+1][c] += 1
        # M19 = A22(B22)
        c = col_offset + 18
        U[(i0+2)*m+(j0+2)][c] += 1
        V[(j0+2)*p+(k0+2)][c] += 1
        W[(k0+2)*n+(i0+2)][c] += 1
        # M20 = (A00+A22)(B00+B22)
        c = col_offset + 19
        U[i0*m+j0][c] += 1; U[(i0+2)*m+(j0+2)][c] += 1
        V[j0*p+k0][c] += 1; V[(j0+2)*p+(k0+2)][c] += 1
        W[k0*n+i0][c] += 1; W[(k0+2)*n+(i0+2)][c] += 1
        # M21 = (A01+A10+A22)(B10+B01+B22)
        c = col_offset + 20
        U[i0*m+(j0+1)][c] += 1; U[(i0+1)*m+j0][c] += 1; U[(i0+2)*m+(j0+2)][c] += 1
        V[(j0+1)*p+k0][c] += 1; V[j0*p+(k0+1)][c] += 1; V[(j0+2)*p+(k0+2)][c] += 1
        W[k0*n+(i0+1)][c] += 1; W[(k0+1)*n+i0][c] += 1; W[(k0+2)*n+(i0+2)][c] += 1
        # M22 = (A02+A12+A20+A21)(B02+B12+B20+B21)
        c = col_offset + 21
        U[i0*m+(j0+2)][c] += 1; U[(i0+1)*m+(j0+2)][c] += 1
        U[(i0+2)*m+j0][c] += 1; U[(i0+2)*m+(j0+1)][c] += 1
        V[j0*p+(k0+2)][c] += 1; V[(j0+1)*p+(k0+2)][c] += 1
        V[(j0+2)*p+k0][c] += 1; V[(j0+2)*p+(k0+1)][c] += 1
        W[(k0+2)*n+i0][c] += 1; W[(k0+2)*n+(i0+1)][c] += 1
        W[k0*n+(i0+2)][c] += 1; W[k0*n+(i0+2)+1][c] += 1
        # M23 = A00A11A22 cross terms handled implicitly
    
    def build_decomposition(blocks_strassen, blocks_laderman):
        """Build decomposition from given block placements."""
        # Calculate total rank
        total_rank = len(blocks_strassen) * 7 + len(blocks_laderman) * 23
        
        # Mark covered triples
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for i0, j0, k0 in blocks_strassen:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if i0+di < n and j0+dj < m and k0+dk < p:
                            covered[i0+di][j0+dj][k0+dk] = True
        for i0, j0, k0 in blocks_laderman:
            for di in range(3):
                for dj in range(3):
                    for dk in range(3):
                        if i0+di < n and j0+dj < m and k0+dk < p:
                            covered[i0+di][j0+dj][k0+dk] = True
        
        # Count remaining
        remaining = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        remaining += 1
        
        total_rank += remaining
        
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col = 0
        for i0, j0, k0 in blocks_strassen:
            add_strassen_block(U, V, W, i0, j0, k0, col)
            col += 7
        for i0, j0, k0 in blocks_laderman:
            # For now, just use schoolbook for laderman blocks to keep it simple
            # We'll add proper Laderman later if time permits
            for di in range(3):
                for dj in range(3):
                    for dk in range(3):
                        if i0+di < n and j0+dj < m and k0+dk < p:
                            U[(i0+di)*m+(j0+dj)][col] += 1
                            V[(j0+dj)*p+(k0+dk)][col] += 1
                            W[(k0+dk)*n+(i0+di)][col] += 1
                            col += 1
            # This is wrong - we need exactly 23 terms for Laderman
            # Let's just skip Laderman for now and focus on Strassen optimization
        
        # Schoolbook for remaining
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not covered[i][j][k]:
                        U[i*m+j][col] += 1
                        V[j*p+k][col] += 1
                        W[k*n+i][col] += 1
                        col += 1
        
        return U, V, W, total_rank
    
    # Strategy: Try different Strassen block placements
    # Current: 4 blocks at (0,0,0), (0,2,0), (0,0,2), (0,2,2) covering i∈{0,1}
    # Alternative: Try to cover more triples or find better arrangements
    
    best_rank = n * m * p
    best_result = None
    
    # Try the current configuration
    strassen_blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
    ]
    U, V, W, rank = build_decomposition(strassen_blocks, [])
    
    # Verify
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target[i*m+j, j*p+k, k*n+i] = 1
        T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        if np.allclose(T_decomp, target, atol=1e-10):
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Try alternative placements
    alt_configs = [
        # Cover i∈{0,1} fully in j,k
        [(0,0,0), (0,2,0), (0,0,2), (0,2,2)],
        # Try to include i=2
        [(0,0,0), (0,2,0), (0,0,2), (0,2,2), (2,0,0)],
        [(0,0,0), (0,2,0), (0,0,2), (0,2,2), (2,0,2)],
        [(0,0,0), (0,2,0), (0,0,2), (0,2,2), (2,3,0)],
        [(0,0,0), (0,2,0), (0,0,2), (0,2,2), (2,3,2)],
    ]
    
    for config in alt_configs:
        if time.time() > deadline - 10:
            break
        # Check for overlaps
        covered_check = [[[False]*p for _ in range(m)] for _ in range(n)]
        valid = True
        for i0, j0, k0 in config:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if i0+di >= n or j0+dj >= m or k0+dk >= p:
                            valid = False
                            break
                        if covered_check[i0+di][j0+dj][k0+dk]:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered_check[i0+di][j0+dj][k0+dk] = True
            if not valid:
                break
        if not valid:
            continue
        
        U, V, W, rank = build_decomposition(config, [])
        if rank < best_rank:
            try:
                U_np = np.array(U, dtype=float)
                V_np = np.array(V, dtype=float)
                W_np = np.array(W, dtype=float)
                target = np.zeros((n*m, m*p, p*n), dtype=float)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            target[i*m+j, j*p+k, k*n+i] = 1
                T_decomp = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
                if np.allclose(T_decomp, target, atol=1e-10):
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
            except Exception:
                pass
    
    # Try greedy rank reduction on the best result
    if best_result is not None and time.time() < deadline - 20:
        U = [row[:] for row in best_result["U"]]
        V = [row[:] for row in best_result["V"]]
        W = [row[:] for row in best_result["W"]]
        current_rank = len(U[0])
        
        # Try to merge pairs of rank-1 terms
        # For each pair of columns, check if their sum can be represented with fewer terms
        # This is complex; skip for now
        
    if best_result is None:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        c2 = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][c2] = 1
                    V_s[j*p+k][c2] = 1
                    W_s[k*n+i][c2] = 1
                    c2 += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return best_result
# EVOLVE-BLOCK-END
