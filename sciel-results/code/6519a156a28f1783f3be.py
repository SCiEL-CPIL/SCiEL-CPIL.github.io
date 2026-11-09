import time
import random
import itertools
import numpy as np
from scipy.linalg import qr, solve
from fractions import Fraction

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    def make_strassen_2x2():
        """
        Return U, V, W for 2x2x2 Strassen with shapes (4,7), (4,7), (4,7).
        U indexed as U[i*2+j], V as V[j*2+k], W as W[k*2+i] for i,j,k in {0,1}.
        Must satisfy: for all (i,j,k) in {0,1}^3, sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] = 1
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A00+A11)*(B00+B11)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A10+A11)*B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3 = A00*(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4 = A11*(B10-B00)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A00+A01)*B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6 = (A10-A00)*(B00+B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def verify_decomposition(U, V, W, rank, n, m, p, tol=1e-10):
        """Check: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += W[k * n + i, r] * U[i * m + j, r] * V[j * p + k, r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_decomposition_exact(U, V, W, rank, n, m, p):
        """Exact verification using Fraction arithmetic."""
        # Convert to fractions for exact check
        Uf = [[Fraction(int(round(U[i, r] * 1000000)), 1000000) for r in range(rank)] for i in range(n * m)]
        Vf = [[Fraction(int(round(V[i, r] * 1000000)), 1000000) for r in range(rank)] for i in range(m * p)]
        Wf = [[Fraction(int(round(W[i, r] * 1000000)), 1000000) for r in range(rank)] for i in range(p * n)]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Wf[k * n + i][r] * Uf[i * m + j][r] * Vf[j * p + k][r]
                    if s != 1:
                        return False
        return True
    
    def build_blocked_decomposition(a_row_blocks, a_col_blocks, b_col_blocks):
        """
        Build a blocked decomposition.
        a_row_blocks: list of (start, end) for A rows
        a_col_blocks: list of (start, end) for A cols = B rows
        b_col_blocks: list of (start, end) for B cols = C cols
        Returns (U, V, W, rank) or None if invalid.
        """
        n, m, p = 3, 4, 6
        sU, sV, sW = make_strassen_2x2()
        
        # Calculate total rank
        total_rank = 0
        for (ar_s, ar_e) in a_row_blocks:
            ar_size = ar_e - ar_s
            for (ac_s, ac_e) in a_col_blocks:
                ac_size = ac_e - ac_s
                for (bc_s, bc_e) in b_col_blocks:
                    bc_size = bc_e - bc_s
                    if ar_size == 2 and ac_size == 2 and bc_size == 2:
                        total_rank += 7  # Strassen
                    else:
                        total_rank += ar_size * ac_size * bc_size  # schoolbook
        
        U = np.zeros((n * m, total_rank), dtype=float)
        V = np.zeros((m * p, total_rank), dtype=float)
        W = np.zeros((p * n, total_rank), dtype=float)
        
        col = 0
        for (ar_s, ar_e) in a_row_blocks:
            ar_size = ar_e - ar_s
            for (ac_s, ac_e) in a_col_blocks:
                ac_size = ac_e - ac_s
                b_row_s = ac_s
                b_row_e = ac_e
                for (bc_s, bc_e) in b_col_blocks:
                    bc_size = bc_e - bc_s
                    
                    if ar_size == 2 and ac_size == 2 and bc_size == 2:
                        # Use Strassen for (2,2,2) block
                        for r in range(7):
                            for ii in range(2):
                                for jj in range(2):
                                    U[(ar_s + ii) * m + (ac_s + jj), col] = sU[ii * 2 + jj, r]
                            for jj in range(2):
                                for kk in range(2):
                                    V[(b_row_s + jj) * p + (bc_s + kk), col] = sV[jj * 2 + kk, r]
                            for kk in range(2):
                                for ii in range(2):
                                    W[(bc_s + kk) * n + (ar_s + ii), col] = sW[kk * 2 + ii, r]
                            col += 1
                    else:
                        # Use schoolbook for this block
                        for ii in range(ar_size):
                            for jj in range(ac_size):
                                for kk in range(bc_size):
                                    U[(ar_s + ii) * m + (ac_s + jj), col] = 1.0
                                    V[(b_row_s + jj) * p + (bc_s + kk), col] = 1.0
                                    W[(bc_s + kk) * n + (ar_s + ii), col] = 1.0
                                    col += 1
        
        assert col == total_rank, f"Column count {col} != expected {total_rank}"
        
        # Verify
        if not verify_decomposition(U, V, W, total_rank, n, m, p):
            return None
        
        return U, V, W, total_rank
    
    # Start with schoolbook
    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    if time.time() - start_time > deadline - 15:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try various blocked decompositions
    # A has 3 rows, 4 cols; B has 4 rows, 6 cols; C has 3 rows, 6 cols
    
    # Possible row partitions of A (3 rows):
    # [(0,3)] - one block of size 3
    # [(0,2),(2,3)] - sizes 2,1
    # [(0,1),(1,3)] - sizes 1,2
    # [(0,1),(1,2),(2,3)] - sizes 1,1,1
    
    # Possible col partitions of A / row partitions of B (4 rows):
    # [(0,4)] - one block of size 4
    # [(0,2),(2,4)] - sizes 2,2
    # [(0,3),(3,4)] - sizes 3,1
    # [(0,1),(1,4)] - sizes 1,3
    # [(0,1),(1,3),(3,4)] - sizes 1,2,1
    # [(0,1),(1,2),(2,4)] - sizes 1,1,2
    # [(0,2),(2,3),(3,4)] - sizes 2,1,1
    # [(0,1),(1,2),(2,3),(3,4)] - sizes 1,1,1,1
    
    # Possible col partitions of B / col partitions of C (6 cols):
    # [(0,6)] - one block of size 6
    # [(0,2),(2,6)] - sizes 2,4
    # [(0,4),(4,6)] - sizes 4,2
    # [(0,3),(3,6)] - sizes 3,3
    # [(0,2),(2,4),(4,6)] - sizes 2,2,2
    # [(0,1),(1,5)] - sizes 1,5
    # [(0,5),(5,6)] - sizes 5,1
    # [(0,1),(1,3),(3,6)] - sizes 1,2,3
    # [(0,1),(1,5),(5,6)] - sizes 1,4,1
    # [(0,2),(2,3),(3,6)] - sizes 2,1,3
    # [(0,3),(3,4),(4,6)] - sizes 3,1,2
    # [(0,2),(2,4),(4,5),(5,6)] - sizes 2,2,1,1
    # [(0,1),(1,2),(2,6)] - sizes 1,1,4
    # [(0,4),(4,5),(5,6)] - sizes 4,1,1
    # [(0,1),(1,3),(3,5),(5,6)] - sizes 1,2,2,1
    # [(0,2),(2,3),(3,5),(5,6)] - sizes 2,1,2,1
    # [(0,1),(1,2),(2,4),(4,6)] - sizes 1,1,2,2
    # [(0,2),(2,3),(3,4),(4,6)] - sizes 2,1,1,2
    # [(0,1),(1,3),(3,4),(4,6)] - sizes 1,2,1,2
    # [(0,2),(2,4),(4,5),(5,6)] already listed
    # [(0,1),(1,2),(2,3),(3,6)] - sizes 1,1,1,3
    # [(0,3),(3,4),(4,5),(5,6)] - sizes 3,1,1,1
    # [(0,1),(1,2),(2,4),(4,5),(5,6)] - sizes 1,1,2,1,1
    # [(0,2),(2,3),(3,4),(4,5),(5,6)] - sizes 2,1,1,1,1
    # [(0,1),(1,3),(3,4),(4,5),(5,6)] - sizes 1,2,1,1,1
    # [(0,1),(1,2),(2,3),(3,4),(4,6)] - sizes 1,1,1,1,2
    # [(0,2),(2,3),(3,4),(4,5),(5,6)] already listed
    # [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)] - sizes 1,1,1,1,1,1
    
    a_row_partitions = [
        [(0, 3)],
        [(0, 2), (2, 3)],
        [(0, 1), (1, 3)],
        [(0, 1), (1, 2), (2, 3)],
    ]
    
    a_col_partitions = [
        [(0, 4)],
        [(0, 2), (2, 4)],
        [(0, 3), (3, 4)],
        [(0, 1), (1, 4)],
        [(0, 1), (1, 3), (3, 4)],
        [(0, 1), (1, 2), (2, 4)],
        [(0, 2), (2, 3), (3, 4)],
        [(0, 1), (1, 2), (2, 3), (3, 4)],
    ]
    
    b_col_partitions = [
        [(0, 6)],
        [(0, 2), (2, 6)],
        [(0, 4), (4, 6)],
        [(0, 3), (3, 6)],
        [(0, 2), (2, 4), (4, 6)],
        [(0, 1), (1, 5)],
        [(0, 5), (5, 6)],
        [(0, 1), (1, 3), (3, 6)],
        [(0, 1), (1, 5), (5, 6)],
        [(0, 2), (2, 3), (3, 6)],
        [(0, 3), (3, 4), (4, 6)],
        [(0, 2), (2, 4), (4, 5), (5, 6)],
        [(0, 1), (1, 2), (2, 6)],
        [(0, 4), (4, 5), (5, 6)],
        [(0, 1), (1, 3), (3, 5), (5, 6)],
        [(0, 2), (2, 3), (3, 5), (5, 6)],
        [(0, 1), (1, 2), (2, 4), (4, 6)],
        [(0, 2), (2, 3), (3, 4), (4, 6)],
        [(0, 1), (1, 3), (3, 4), (4, 6)],
        [(0, 1), (1, 2), (2, 3), (3, 6)],
        [(0, 3), (3, 4), (4, 5), (5, 6)],
        [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6)],
        [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
        [(0, 1), (1, 3), (3, 4), (4, 5), (5, 6)],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6)],
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
    ]
    
    # Try all combinations, prioritizing those with more 2x2x2 blocks
    candidates = []
    for ar_part in a_row_partitions:
        for ac_part in a_col_partitions:
            for bc_part in b_col_partitions:
                # Estimate rank
                est_rank = 0
                for (ar_s, ar_e) in ar_part:
                    ar_size = ar_e - ar_s
                    for (ac_s, ac_e) in ac_part:
                        ac_size = ac_e - ac_s
                        for (bc_s, bc_e) in bc_part:
                            bc_size = bc_e - bc_s
                            if ar_size == 2 and ac_size == 2 and bc_size == 2:
                                est_rank += 7
                            else:
                                est_rank += ar_size * ac_size * bc_size
                
                # Only try if potentially better than current best
                if est_rank < best_rank:
                    candidates.append((est_rank, ar_part, ac_part, bc_part))
    
    # Sort by estimated rank (ascending)
    candidates.sort(key=lambda x: x[0])
    
    # Try the best candidates
    for est_rank, ar_part, ac_part, bc_part in candidates:
        if time.time() - start_time > deadline - 10:
            break
        
        result = build_blocked_decomposition(ar_part, ac_part, bc_part)
        if result is not None:
            bU, bV, bW, brank = result
            if brank < best_rank:
                best_U, best_V, best_W, best_rank = bU, bV, bW, brank
                print(f"Found better decomposition with rank {brank}")
    
    # Also try the specific 66-rank decomposition
    result = build_blocked_decomposition(
        [(0, 2), (2, 3)],
        [(0, 2), (2, 4)],
        [(0, 2), (2, 4), (4, 6)]
    )
    if result is not None:
        bU, bV, bW, brank = result
        if brank < best_rank:
            best_U, best_V, best_W, best_rank = bU, bV, bW, brank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
