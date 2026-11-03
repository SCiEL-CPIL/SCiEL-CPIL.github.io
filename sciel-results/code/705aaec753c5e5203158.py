import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check all entries
    # T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Fallback: Schoolbook (Rank 72)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U = U_sb.tolist()
    best_V = V_sb.tolist()
    best_W = W_sb.tolist()
    best_rank = rank_schoolbook

    # --- STRATEGY: COMBINATORIAL BLOCK ASSEMBLY ---
    # We model the problem as tiling the index space (i in 0..2, j in 0..3, k in 0..5)
    # with blocks that cover disjoint subsets of indices.
    # Each block corresponds to a known tensor decomposition (e.g., 2x2x2 Strassen, 1x2x2, etc.)
    
    # Define available block types:
    # (size_i, size_j, size_k, rank, U_block, V_block, W_block)
    # Indices in U are relative to the block's top-left corner (0..size_i-1, 0..size_j-1)
    # Indices in V are relative (0..size_j-1, 0..size_k-1)
    # Indices in W are relative (0..size_k-1, 0..size_i-1) -- Note: W index is k*n+i, so k is first, i is second.
    
    def get_block_2x2x2_strassen():
        # Rank 7
        U = np.zeros((2, 2, 7))
        V = np.zeros((2, 2, 7))
        W = np.zeros((2, 2, 7)) # W[k, i, c]
        
        # Strassen 2x2x2
        # C11 = A11*(B11+B22) - (A21-A11)*(B11-B21) ... standard formulation
        # Let's use the standard 7 multiplications:
        # M1 = A11 * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B21 + B22)
        # M4 = A22 * (B12 + B22)
        # M5 = (A11 + A12) * (B22 - B12)
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 - M3 + M7
        # C12 = M5 + M3
        # C21 = M2 + M4
        # C22 = M1 - M2 + M4 - M5
        
        # We need U[i,j,c], V[j,k,c], W[k,i,c] such that sum U*V*W = 1 for (i,j,k) in block.
        
        # c=0: M1
        U[0,0,0]=1; V[0,0,0]=1; W[0,0,0]=1
        U[0,0,0]=1; V[1,1,0]=1; W[1,0,0]=1
        
        # c=1: M2
        U[1,0,1]=1; U[1,1,1]=1; V[0,0,1]=1; W[0,1,1]=1
        
        # c=2: M3
        U[0,0,2]=1; V[1,0,2]=1; V[1,1,2]=1; W[0,0,2]=-1; W[1,0,2]=-1
        
        # c=3: M4
        U[1,1,3]=1; V[0,1,3]=1; V[1,1,3]=1; W[0,1,3]=1; W[1,1,3]=1
        
        # c=4: M5
        U[0,0,4]=1; U[0,1,4]=1; V[1,1,4]=1; V[0,1,4]=-1; W[1,0,4]=1; W[1,1,4]=-1
        
        # c=5: M6
        U[1,0,5]=1; U[0,0,5]=-1; V[0,0,5]=1; V[0,1,5]=1; W[0,0,5]=1; W[0,1,5]=1
        
        # c=6: M7
        U[0,1,6]=1; U[1,1,6]=-1; V[1,0,6]=1; V[1,1,6]=1; W[0,0,6]=1; W[1,0,6]=1
        
        return U, V, W, 7

    def get_block_1x2x2():
        # Rank 4 (Schoolbook for 1x2*2x2? No, 1x2*2x2 is 4 mults. Rank 4.)
        # Actually, 1x2 * 2x2:
        # A is 1x2, B is 2x2. C is 1x2.
        # C1 = A1*B11 + A2*B21
        # C2 = A1*B12 + A2*B22
        # This is just 4 multiplications.
        U = np.zeros((1, 2, 4))
        V = np.zeros((2, 2, 4))
        W = np.zeros((2, 1, 4)) # W[k, i, c]
        
        # c=0: A1*B11
        U[0,0,0]=1; V[0,0,0]=1; W[0,0,0]=1
        # c=1: A2*B21
        U[0,1,1]=1; V[1,0,1]=1; W[0,0,1]=1
        # c=2: A1*B12
        U[0,0,2]=1; V[0,1,2]=1; W[1,0,2]=1
        # c=3: A2*B22
        U[0,1,3]=1; V[1,1,3]=1; W[1,0,3]=1
        
        return U, V, W, 4

    def get_block_2x1x2():
        # Rank 4
        U = np.zeros((2, 1, 4))
        V = np.zeros((1, 2, 4))
        W = np.zeros((2, 2, 4))
        
        # c=0: A11*B11
        U[0,0,0]=1; V[0,0,0]=1; W[0,0,0]=1
        # c=1: A21*B11
        U[1,0,1]=1; V[0,0,1]=1; W[0,1,1]=1
        # c=2: A11*B12
        U[0,0,2]=1; V[0,1,2]=1; W[1,0,2]=1
        # c=3: A21*B12
        U[1,0,3]=1; V[0,1,3]=1; W[1,1,3]=1
        
        return U, V, W, 4

    def get_block_1x1x2():
        # Rank 2
        U = np.zeros((1, 1, 2))
        V = np.zeros((1, 2, 2))
        W = np.zeros((2, 1, 2))
        U[0,0,0]=1; V[0,0,0]=1; W[0,0,0]=1
        U[0,0,1]=1; V[0,1,1]=1; W[1,0,1]=1
        return U, V, W, 2

    def get_block_2x1x1():
        # Rank 2
        U = np.zeros((2, 1, 2))
        V = np.zeros((1, 1, 2))
        W = np.zeros((1, 2, 2))
        U[0,0,0]=1; V[0,0,0]=1; W[0,0,0]=1
        U[1,0,1]=1; V[0,0,1]=1; W[0,1,1]=1
        return U, V, W, 2

    def get_block_1x2x1():
        # Rank 2
        U = np.zeros((1, 2, 2))
        V = np.zeros((2, 1, 2))
        W = np.zeros((1, 1, 2))
        U[0,0,0]=1; V[0,0,0]=1; W[0,0,0]=1
        U[0,1,1]=1; V[1,0,1]=1; W[0,0,1]=1
        return U, V, W, 2

    def get_block_1x1x1():
        # Rank 1
        U = np.zeros((1, 1, 1))
        V = np.zeros((1, 1, 1))
        W = np.zeros((1, 1, 1))
        U[0,0,0]=1; V[0,0,0]=1; W[0,0,0]=1
        return U, V, W, 1

    # Pre-compute block data
    blocks = [
        (2, 2, 2) + get_block_2x2x2_strassen(),
        (1, 2, 2) + get_block_1x2x2(),
        (2, 1, 2) + get_block_2x1x2(),
        (1, 1, 2) + get_block_1x1x2(),
        (2, 1, 1) + get_block_2x1x1(),
        (1, 2, 1) + get_block_1x2x1(),
        (1, 1, 1) + get_block_1x1x1()
    ]
    
    # Sort blocks by efficiency (rank / volume) to prioritize good fits
    # Volume = si * sj * sk
    # Efficiency = rank / volume
    blocks.sort(key=lambda b: b[3] / (b[0]*b[1]*b[2]))

    # State: 3D grid of booleans, initially False (uncovered)
    # We will use backtracking search to cover the grid.
    
    grid = np.zeros((n, m, p), dtype=bool)
    
    # We need to track the global U, V, W matrices being built.
    # Since we don't know the final rank, we'll build lists of (row, col_val) for each matrix
    # and then assemble them.
    # Actually, easier: build the matrices dynamically.
    # But rank is the sum of block ranks.
    
    current_U = []
    current_V = []
    current_W = []
    current_rank = 0
    
    # To make backtracking efficient, we'll store the operations applied.
    # Operation: (block_type, i_start, j_start, k_start)
    
    def get_first_uncovered():
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if not grid[i, j, k]:
                        return i, j, k
        return None

    def place_block(b_idx, i0, j0, k0):
        si, sj, sk, rank, U_b, V_b, W_b = blocks[b_idx]
        
        # Check bounds
        if i0 + si > n or j0 + sj > m or k0 + sk > p:
            return False
        
        # Check overlap
        for di in range(si):
            for dj in range(sj):
                for dk in range(sk):
                    if grid[i0+di, j0+dj, k0+dk]:
                        return False
        
        # Place
        for di in range(si):
            for dj in range(sj):
                for dk in range(sk):
                    grid[i0+di, j0+dj, k0+dk] = True
                    
        # Record to global matrices
        # We need to append columns to U, V, W
        # U is (n*m, rank). Rows are i*m+j.
        # V is (m*p, rank). Rows are j*p+k.
        # W is (p*n, rank). Rows are k*n+i.
        
        for c in range(rank):
            u_col = [0.0] * (n * m)
            v_col = [0.0] * (m * p)
            w_col = [0.0] * (p * n)
            
            for di in range(si):
                for dj in range(sj):
                    val_u = U_b[di, dj, c]
                    if val_u != 0:
                        u_col[(i0+di)*m + (j0+dj)] = val_u
                        
            for dj in range(sj):
                for dk in range(sk):
                    val_v = V_b[dj, dk, c]
                    if val_v != 0:
                        v_col[(j0+dj)*p + (k0+dk)] = val_v
                        
            for dk in range(sk):
                for di in range(si):
                    val_w = W_b[dk, di, c]
                    if val_w != 0:
                        w_col[(k0+dk)*n + (i0+di)] = val_w
                        
            current_U.append(u_col)
            current_V.append(v_col)
            current_W.append(w_col)
            
        return True

    def remove_block(b_idx, i0, j0, k0):
        si, sj, sk, rank, U_b, V_b, W_b = blocks[b_idx]
        
        # Unmark grid
        for di in range(si):
            for dj in range(sj):
                for dk in range(sk):
                    grid[i0+di, j0+dj, k0+dk] = False
                    
        # Remove last 'rank' columns from current_U, V, W
        for _ in range(rank):
            current_U.pop()
            current_V.pop()
            current_W.pop()

    def search(depth=0):
        nonlocal best_rank, best_U, best_V, best_W
        
        if time.time() > deadline:
            return False
            
        pos = get_first_uncovered()
        if pos is None:
            # All covered
            if len(current_U) < best_rank:
                best_rank = len(current_U)
                best_U = [col[:] for col in current_U]
                best_V = [col[:] for col in current_V]
                best_W = [col[:] for col in current_W]
            return True # Found a valid solution, but continue to find better? 
                        # Backtracking will find all, but we can prune.
            
        i0, j0, k0 = pos
        
        # Try all blocks that can fit at (i0, j0, k0)
        # We should try blocks that cover this position.
        # Since (i0, j0, k0) is the first uncovered, any block placed must cover it.
        # The block must have i_start <= i0, j_start <= j0, k_start <= k0.
        # And i_start + si > i0, etc.
        
        for b_idx in range(len(blocks)):
            si, sj, sk, rank, U_b, V_b, W_b = blocks[b_idx]
            
            # The block must cover (i0, j0, k0).
            # So i0 is in [i_start, i_start+si)
            # i_start <= i0 < i_start + si  =>  i0 - si + 1 <= i_start <= i0
            # Similarly for j, k.
            
            min_i = max(0, i0 - si + 1)
            max_i = min(i0, n - si)
            min_j = max(0, j0 - sj + 1)
            max_j = min(j0, m - sj)
            min_k = max(0, k0 - sk + 1)
            max_k = min(k0, p - sk)
            
            for ni in range(min_i, max_i + 1):
                for nj in range(min_j, max_j + 1):
                    for nk in range(min_k, max_k + 1):
                        if place_block(b_idx, ni, nj, nk):
                            # Pruning: if current rank + minimum possible remaining rank > best_rank, skip
                            # Minimum remaining rank is at least number of uncovered cells / max_volume_efficiency
                            # For now, just recurse
                            if search(depth + 1):
                                # If we found a solution, we could return True to stop,
                                # but we want the MINIMAL rank.
                                # So we continue searching to find smaller ranks.
                                pass
                            remove_block(b_idx, ni, nj, nk)
                            
        return False

    # Run the search
    # To avoid infinite loops or excessive time, we rely on the deadline.
    # We might want to limit the depth or number of solutions found.
    # Since the space is small (3x4x6=72 cells), and blocks are large, depth should be small.
    
    search()
    
    # If no solution found (shouldn't happen, schoolbook is always a solution), 
    # the best_rank should still be 72 from initialization.
    
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }
