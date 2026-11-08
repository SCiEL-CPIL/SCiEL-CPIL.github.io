import numpy as np
import time
import sys

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill

    # Greedy placement of 2x2x2 Strassen blocks to maximize coverage
    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    blocks = []
    
    # All possible block positions
    all_positions = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                all_positions.append((i0, j0, k0))
    
    # Greedily add blocks that maximize new coverage
    # Each block costs 7 rank and covers up to 8 entries
    # Only add if it covers at least 3 new entries (net gain: 3-7 = -4, but we save 3 schoolbook terms)
    # Actually, a block covers 8 entries with rank 7, saving 1 rank per 8 entries
    # With overlap, if it covers g new entries, net rank change = 7 - g
    # So we want g > 7 to be net positive, but that's impossible (max 8)
    # g = 8: net -1 (save 1 rank)
    # g = 7: net 0
    # g = 6: net +1 (worse)
    # So we only add blocks that cover >= 7 new entries
    
    while time.time() < deadline - 30:
        best_block = None
        best_gain = 0
        for pos in all_positions:
            i0, j0, k0 = pos
            gain = 0
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        if not covered[i0+di][j0+dj][k0+dk]:
                            gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = pos
        
        if best_gain < 7:  # Only add if it saves at least 1 rank
            break
        
        blocks.append(best_block)
        i0, j0, k0 = best_block
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered[i0+di][j0+dj][k0+dk] = True
    
    # Count schoolbook terms
    schoolbook_count = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    schoolbook_count += 1
    
    total_rank = len(blocks) * 7 + schoolbook_count
    
    # Build factor matrices
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Verify correctness
    if time.time() < deadline - 5:
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0
                    actual = sum(U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c] for c in range(total_rank))
                    if abs(actual - expected) > 1e-9:
                        # Fallback to full schoolbook
                        U = [[0.0] * (n*m*p) for _ in range(n * m)]
                        V = [[0.0] * (n*m*p) for _ in range(m * p)]
                        W = [[0.0] * (n*m*p) for _ in range(p * n)]
                        col = 0
                        for ii in range(n):
                            for jj in range(m):
                                for kk in range(p):
                                    U[ii*m+jj][col] = 1.0
                                    V[jj*p+kk][col] = 1.0
                                    W[kk*n+ii][col] = 1.0
                                    col += 1
                        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
