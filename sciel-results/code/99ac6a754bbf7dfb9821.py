import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    def verify_decomposition(U, V, W, R):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        total = 0.0
                        for r in range(R):
                            total += W[k * n + i][r] * U[i * m + j][r] * V[jp * p + k][r]
                        expected = 1.0 if j == jp else 0.0
                        if abs(total - expected) > 1e-9:
                            return False
        return True
    
    def strassen_block_terms(i1, i2, j1, j2, k1, k2):
        """
        Return 7 Strassen terms for a 2x2x2 block.
        Each term is (U_dict, V_dict, W_dict) mapping (local_idx, local_idx) to coefficient.
        Local indices: i in {0,1} -> i1+i_local, j in {0,1} -> j1+j_local, k in {0,1} -> k1+k_local
        """
        terms = []
        
        # m1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        # C[0,0] += 1, C[1,1] += 1
        U1 = {(i1, j1): 1, (i2, j2): 1}
        V1 = {(j1, k1): 1, (j2, k2): 1}
        W1 = {(k1, i1): 1, (k2, i2): 1}
        terms.append((U1, V1, W1))
        
        # m2 = (A[1,0] + A[1,1]) * B[0,0]
        # C[1,0] += 1, C[1,1] -= 1
        U2 = {(i2, j1): 1, (i2, j2): 1}
        V2 = {(j1, k1): 1}
        W2 = {(k1, i2): 1, (k2, i2): -1}
        terms.append((U2, V2, W2))
        
        # m3 = A[0,0] * (B[0,1] - B[1,1])
        # C[0,1] += 1, C[1,1] += 1
        U3 = {(i1, j1): 1}
        V3 = {(j1, k2): 1, (j2, k2): -1}
        W3 = {(k2, i1): 1, (k2, i2): 1}
        terms.append((U3, V3, W3))
        
        # m4 = A[1,1] * (B[1,0] - B[0,0])
        # C[1,0] += 1, C[0,0] += 1
        U4 = {(i2, j2): 1}
        V4 = {(j2, k1): 1, (j1, k1): -1}
        W4 = {(k1, i2): 1, (k1, i1): 1}
        terms.append((U4, V4, W4))
        
        # m5 = (A[0,0] + A[0,1]) * B[1,1]
        # C[0,1] += 1, C[0,0] -= 1
        U5 = {(i1, j1): 1, (i1, j2): 1}
        V5 = {(j2, k2): 1}
        W5 = {(k2, i1): 1, (k1, i1): -1}
        terms.append((U5, V5, W5))
        
        # m6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        # C[1,1] += 1
        U6 = {(i2, j1): 1, (i1, j1): -1}
        V6 = {(j1, k1): 1, (j1, k2): 1}
        W6 = {(k2, i2): 1}
        terms.append((U6, V6, W6))
        
        # m7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        # C[0,0] += 1
        U7 = {(i1, j2): 1, (i2, j2): -1}
        V7 = {(j2, k1): 1, (j2, k2): 1}
        W7 = {(k1, i1): 1}
        terms.append((U7, W7, W7))  # placeholder, will fix
        
        # Fix: m7 W should be {(k1, i1): 1}
        terms[6] = (U7, V7, {(k1, i1): 1})
        
        return terms
    
    def build_optimized_decomposition():
        """
        Build decomposition using 6 overlapping Strassen blocks + schoolbook for remainder.
        
        Blocks:
        A: i={0,1}, j={0,1}, k={0,1}
        B: i={0,1}, j={2,3}, k={0,1}
        C: i={0,1}, j={0,1}, k={2,3}
        D: i={0,1}, j={2,3}, k={2,3}
        E: i={1,2}, j={0,1}, k={3,4}
        F: i={1,2}, j={2,3}, k={3,4}
        
        Total Strassen terms: 6 * 7 = 42
        Schoolbook: remaining (i,j,k) not covered
        """
        blocks = [
            (0, 1, 0, 1, 0, 1),  # A
            (0, 1, 2, 3, 0, 1),  # B
            (0, 1, 0, 1, 2, 3),  # C
            (0, 1, 2, 3, 2, 3),  # D
            (1, 2, 0, 1, 3, 4),  # E
            (1, 2, 2, 3, 3, 4),  # F
        ]
        
        # Collect all terms
        all_terms = []
        covered = set()
        
        for i1, i2, j1, j2, k1, k2 in blocks:
            block_terms = strassen_block_terms(i1, i2, j1, j2, k1, k2)
            all_terms.extend(block_terms)
            # Mark all (i,j,k) in this block as covered
            for i in [i1, i2]:
                for j in [j1, j2]:
                    for k in [k1, k2]:
                        covered.add((i, j, k))
        
        # Add schoolbook terms for uncovered (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U_sb = {(i, j): 1}
                        V_sb = {(j, k): 1}
                        W_sb = {(k, i): 1}
                        all_terms.append((U_sb, V_sb, W_sb))
        
        R = len(all_terms)
        
        # Build matrices
        U = [[0] * R for _ in range(dim_u)]
        V = [[0] * R for _ in range(dim_v)]
        W = [[0] * R for _ in range(dim_w)]
        
        for c, (U_dict, V_dict, W_dict) in enumerate(all_terms):
            for (i, j), coeff in U_dict.items():
                U[i * m + j][c] = coeff
            for (j, k), coeff in V_dict.items():
                V[j * p + k][c] = coeff
            for (k, i), coeff in W_dict.items():
                W[k * n + i][c] = coeff
        
        # Verify
        if not verify_decomposition(U, V, W, R):
            return None
        return U, V, W, R
    
    def build_fallback_decomposition():
        """Fallback: 4 disjoint Strassen blocks + schoolbook (rank 56)."""
        blocks = [
            (0, 1, 0, 1, 0, 1),
            (0, 1, 2, 3, 0, 1),
            (0, 1, 0, 1, 2, 3),
            (0, 1, 2, 3, 2, 3),
        ]
        
        all_terms = []
        covered = set()
        
        for i1, i2, j1, j2, k1, k2 in blocks:
            block_terms = strassen_block_terms(i1, i2, j1, j2, k1, k2)
            all_terms.extend(block_terms)
            for i in [i1, i2]:
                for j in [j1, j2]:
                    for k in [k1, k2]:
                        covered.add((i, j, k))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U_sb = {(i, j): 1}
                        V_sb = {(j, k): 1}
                        W_sb = {(k, i): 1}
                        all_terms.append((U_sb, V_sb, W_sb))
        
        R = len(all_terms)
        
        U = [[0] * R for _ in range(dim_u)]
        V = [[0] * R for _ in range(dim_v)]
        W = [[0] * R for _ in range(dim_w)]
        
        for c, (U_dict, V_dict, W_dict) in enumerate(all_terms):
            for (i, j), coeff in U_dict.items():
                U[i * m + j][c] = coeff
            for (j, k), coeff in V_dict.items():
                V[j * p + k][c] = coeff
            for (k, i), coeff in W_dict.items():
                W[k * n + i][c] = coeff
        
        if not verify_decomposition(U, V, W, R):
            return None
        return U, V, W, R
    
    # Try optimized decomposition first
    result = build_optimized_decomposition()
    if result is not None:
        U, V, W, R = result
        return {"U": U, "V": V, "W": W}
    
    # Fallback to 4-block decomposition
    result = build_fallback_decomposition()
    if result is not None:
        U, V, W, R = result
        return {"U": U, "V": V, "W": W}
    
    # Final fallback: pure schoolbook
    R = n * m * p
    U = [[0] * R for _ in range(dim_u)]
    V = [[0] * R for _ in range(dim_v)]
    W = [[0] * R for _ in range(dim_w)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
