"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix
    by a 3x8 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    We use Strassen-like 2x2x2 block decompositions to reduce the rank
    below the schoolbook rank of n*m*p = 72.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Helper to verify a decomposition
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if total != 1:
                        return False
        return True
    
    # Helper to try a candidate decomposition
    def try_candidate(U, V, W):
        """Try a candidate decomposition, return True if valid."""
        rank = len(U[0])
        if len(U) != n * m:
            return False
        if len(V) != m * p:
            return False
        if len(W) != p * n:
            return False
        if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
            return False
        return verify_decomposition(U, V, W, rank)
    
    # Strassen 2x2x2 block: 7 factors instead of 8 schoolbook
    # For C = A*B where A is 2x2, B is 2x2, C is 2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def build_strassen_block(i0, j0, k0):
        """
        Build the 7 Strassen factors for a 2x2x2 block at (i0, j0, k0).
        Returns list of (U_entry, V_entry, W_entries) tuples.
        U_entry: (i, j) index in A
        V_entry: (j, k) index in B  
        W_entries: list of (w_index, coefficient) where w_index = k*n+i
        """
        factors = []
        
        # M1 = (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        # Contributes to: C[i0,k0], C[i0+1,k0+1]
        factors.append({
            'U': [(i0, j0, 1), (i0+1, j0+1, 1)],
            'V': [(j0, k0, 1), (j0+1, k0+1, 1)],
            'W': [(k0*n+i0, 1), ((k0+1)*n+i0+1, 1)]
        })
        
        # M2 = (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        # Contributes to: C[i0+1,k0]
        factors.append({
            'U': [(i0+1, j0, 1), (i0+1, j0+1, 1)],
            'V': [(j0, k0, 1)],
            'W': [(k0*n+i0+1, 1)]
        })
        
        # M3 = A[i0,j0] * (B[j0+1,k0]-B[j0+1,k0+1])
        # Contributes to: C[i0,k0+1]
        factors.append({
            'U': [(i0, j0, 1)],
            'V': [(j0+1, k0, 1), (j0+1, k0+1, -1)],
            'W': [((k0+1)*n+i0, 1)]
        })
        
        # M4 = A[i0+1,j0+1] * (B[j0,k0+1]-B[j0,k0])
        # Contributes to: C[i0,k0], C[i0+1,k0]
        factors.append({
            'U': [(i0+1, j0+1, 1)],
            'V': [(j0, k0+1, 1), (j0, k0, -1)],
            'W': [(k0*n+i0, 1), (k0*n+i0+1, 1)]
        })
        
        # M5 = (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        # Contributes to: C[i0,k0+1], C[i0,k0]
        factors.append({
            'U': [(i0, j0, 1), (i0, j0+1, 1)],
            'V': [(j0+1, k0+1, 1)],
            'W': [((k0+1)*n+i0, 1), (k0*n+i0, -1)]
        })
        
        # M6 = (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0+1,k0])
        # Contributes to: C[i0+1,k0+1]
        factors.append({
            'U': [(i0+1, j0, 1), (i0, j0, -1)],
            'V': [(j0, k0, 1), (j0+1, k0, 1)],
            'W': [((k0+1)*n+i0+1, 1)]
        })
        
        # M7 = (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0,k0+1]+B[j0+1,k0+1])
        # Contributes to: C[i0,k0], C[i0+1,k0+1]
        factors.append({
            'U': [(i0, j0+1, 1), (i0+1, j0+1, -1)],
            'V': [(j0, k0+1, 1), (j0+1, k0+1, 1)],
            'W': [(k0*n+i0, 1), ((k0+1)*n+i0+1, 1)]
        })
        
        return factors
    
    def build_decomposition_with_strassen(strassen_blocks):
        """
        Build a decomposition using the given Strassen blocks and schoolbook for the rest.
        strassen_blocks: list of (i0, j0, k0) tuples
        Returns (U, V, W, rank) or None if invalid.
        """
        # Determine which (i,j,k) entries are covered by Strassen blocks
        covered = set()
        for i0, j0, k0 in strassen_blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i0+di, j0+dj, k0+dk))
        
        # Initialize U, V, W with zeros
        # We'll build them incrementally
        U = {}
        V = {}
        W = {}
        
        # First, add Strassen factors
        factor_idx = 0
        for i0, j0, k0 in strassen_blocks:
            factors = build_strassen_block(i0, j0, k0)
            for f in factors:
                # Add to U
                for (i, j, coeff) in f['U']:
                    key = (i*m+j, factor_idx)
                    U[key] = U.get(key, 0) + coeff
                # Add to V
                for (j, k, coeff) in f['V']:
                    key = (j*p+k, factor_idx)
                    V[key] = V.get(key, 0) + coeff
                # Add to W
                for (w_idx, coeff) in f['W']:
                    key = (w_idx, factor_idx)
                    W[key] = W.get(key, 0) + coeff
                factor_idx += 1
        
        # Then add schoolbook for uncovered entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        # Schoolbook factor
                        key_u = (i*m+j, factor_idx)
                        U[key_u] = U.get(key_u, 0) + 1
                        key_v = (j*p+k, factor_idx)
                        V[key_v] = V.get(key_v, 0) + 1
                        key_w = (k*n+i, factor_idx)
                        W[key_w] = W.get(key_w, 0) + 1
                        factor_idx += 1
        
        rank = factor_idx
        
        # Convert to lists
        U_list = [[0] * rank for _ in range(n * m)]
        V_list = [[0] * rank for _ in range(m * p)]
        W_list = [[0] * rank for _ in range(p * n)]
        
        for (row, col), val in U.items():
            U_list[row][col] = val
        for (row, col), val in V.items():
            V_list[row][col] = val
        for (row, col), val in W.items():
            W_list[row][col] = val
        
        return U_list, V_list, W_list, rank
    
    # Try various combinations of Strassen blocks
    # Valid placements: i0 in {0,1}, j0 in {0,1}, k0 in {0,...,6}
    # But we need to ensure blocks don't overlap in a way that breaks the decomposition
    
    # Let's try all single Strassen blocks first
    for i0 in range(n-1):
        for j0 in range(m-1):
            for k0 in range(p-1):
                U, V, W, rank = build_decomposition_with_strassen([(i0, j0, k0)])
                if try_candidate(U, V, W):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U, "V": V, "W": W}
    
    # Now try pairs of non-overlapping Strassen blocks
    # Two blocks (i0,j0,k0) and (i1,j1,k1) are compatible if they don't share
    # any (i,j,k) entries
    def blocks_compatible(b1, b2):
        i0, j0, k0 = b1
        i1, j1, k1 = b2
        # Check if the 2x2x2 blocks overlap
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    entry1 = (i0+di, j0+dj, k0+dk)
                    for di2 in range(2):
                        for dj2 in range(2):
                            for dk2 in range(2):
                                entry2 = (i1+di2, j1+dj2, k1+dk2)
                                if entry1 == entry2:
                                    return False
        return True
    
    all_blocks = []
    for i0 in range(n-1):
        for j0 in range(m-1):
            for k0 in range(p-1):
                all_blocks.append((i0, j0, k0))
    
    # Try pairs of compatible blocks
    for idx1 in range(len(all_blocks)):
        for idx2 in range(idx1+1, len(all_blocks)):
            if blocks_compatible(all_blocks[idx1], all_blocks[idx2]):
                U, V, W, rank = build_decomposition_with_strassen([all_blocks[idx1], all_blocks[idx2]])
                if try_candidate(U, V, W):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U, "V": V, "W": W}
    
    # Try triples of mutually compatible blocks
    for idx1 in range(len(all_blocks)):
        for idx2 in range(idx1+1, len(all_blocks)):
            if not blocks_compatible(all_blocks[idx1], all_blocks[idx2]):
                continue
            for idx3 in range(idx2+1, len(all_blocks)):
                if blocks_compatible(all_blocks[idx1], all_blocks[idx3]) and \
                   blocks_compatible(all_blocks[idx2], all_blocks[idx3]):
                    U, V, W, rank = build_decomposition_with_strassen([
                        all_blocks[idx1], all_blocks[idx2], all_blocks[idx3]
                    ])
                    if try_candidate(U, V, W):
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
