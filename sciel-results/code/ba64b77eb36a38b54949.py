import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    deadline = time.time() + 140.0
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W decompose the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                ui = i * m + j
                for k in range(p):
                    vk = j * p + k
                    wk = k * n + i
                    total = 0.0
                    for r in range(rank):
                        total += U[ui][r] * V[vk][r] * W[wk][r]
                    if abs(total - 1.0) > tol:
                        return False
        return True
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    def add_strassen_2x2x2(terms, r1, c1, k1):
        """Add Strassen terms for a 2x2x2 block starting at (r1,c1,k1)."""
        A00 = r1 * m + c1
        A01 = r1 * m + (c1 + 1)
        A10 = (r1 + 1) * m + c1
        A11 = (r1 + 1) * m + (c1 + 1)
        
        B00 = c1 * p + k1
        B01 = c1 * p + (k1 + 1)
        B10 = (c1 + 1) * p + k1
        B11 = (c1 + 1) * p + (k1 + 1)
        
        C00_w = k1 * n + r1
        C01_w = (k1 + 1) * n + r1
        C10_w = k1 * n + (r1 + 1)
        C11_w = (k1 + 1) * n + (r1 + 1)
        
        terms.append(({A00: 1, A11: 1}, {B00: 1, B11: 1}, {C00_w: 1, C11_w: 1}))
        terms.append(({A10: 1, A11: 1}, {B00: 1}, {C10_w: 1, C11_w: -1}))
        terms.append(({A00: 1}, {B01: 1, B11: -1}, {C01_w: 1, C11_w: 1}))
        terms.append(({A11: 1}, {B10: 1, B00: -1}, {C00_w: 1, C10_w: 1}))
        terms.append(({A00: 1, A01: 1}, {B11: 1}, {C00_w: -1, C01_w: 1}))
        terms.append(({A10: 1, A00: -1}, {B00: 1, B01: 1}, {C11_w: 1}))
        terms.append(({A01: 1, A11: -1}, {B10: 1, B11: 1}, {C00_w: 1}))
    
    # Strassen-3 algorithm for 3x3 matrix multiplication (rank 23)
    # Based on the decomposition by B. Strassen (1969)
    def add_strassen_3x3x3(terms, r1, c1, k1):
        """Add Strassen-3 terms for a 3x3x3 block starting at (r1,c1,k1)."""
        # Index helper functions
        def A(i, j): return (r1 + i) * m + (c1 + j)
        def B(i, j): return (c1 + i) * p + (k1 + j)
        def C(i, j): return (k1 + i) * n + (r1 + j)
        
        # The 23 terms of the Strassen-3 algorithm
        # Each term is (U_dict, V_dict, W_dict)
        
        # Term 1: M1 = A11*(B11 - B21)
        terms.append(({A(0,0): 1}, {B(0,0): 1, B(0,1): -1}, {C(0,0): 1, C(1,0): -1}))
        
        # Term 2: M2 = (A21 + A11)*B11
        terms.append(({A(1,0): 1, A(0,0): 1}, {B(0,0): 1}, {C(1,0): 1}))
        
        # Term 3: M3 = (A11 - A21)*(B21 + B31)
        terms.append(({A(0,0): 1, A(1,0): -1}, {B(1,0): 1, B(2,0): 1}, {C(1,1): 1, C(2,1): -1}))
        
        # Term 4: M4 = A33*(B33 - B23)
        terms.append(({A(2,2): 1}, {B(2,2): 1, B(1,2): -1}, {C(2,2): 1, C(1,2): -1}))
        
        # Term 5: M5 = (A23 + A33)*B33
        terms.append(({A(1,2): 1, A(2,2): 1}, {B(2,2): 1}, {C(1,2): 1}))
        
        # Term 6: M6 = (A33 - A23)*(B23 + B13)
        terms.append(({A(2,2): 1, A(1,2): -1}, {B(1,2): 1, B(0,2): 1}, {C(1,1): 1, C(0,1): -1}))
        
        # Term 7: M7 = (A11 + A33)*(B11 + B33)
        terms.append(({A(0,0): 1, A(2,2): 1}, {B(0,0): 1, B(2,2): 1}, {C(0,0): 1, C(2,2): 1}))
        
        # Term 8: M8 = (A12 - A32)*(B22 + B32)
        terms.append(({A(0,1): 1, A(2,1): -1}, {B(1,1): 1, B(2,1): 1}, {C(0,1): 1, C(2,1): -1}))
        
        # Term 9: M9 = (A31 + A11 - A21 - A33)*(B33 + B23)
        terms.append(({A(2,0): 1, A(0,0): 1, A(1,0): -1, A(2,2): -1}, {B(2,2): 1, B(1,2): 1}, {C(2,0): 1, C(1,0): -1}))
        
        # Term 10: M10 = (A21 + A22 - A12 - A23)*(B13 + B23)
        terms.append(({A(1,0): 1, A(1,1): 1, A(0,1): -1, A(1,2): -1}, {B(0,2): 1, B(1,2): 1}, {C(1,0): 1, C(1,1): -1}))
        
        # Term 11: M11 = (A22 + A33 - A13 - A31)*(B11 + B12)
        terms.append(({A(1,1): 1, A(2,2): 1, A(0,2): -1, A(2,0): -1}, {B(0,0): 1, B(0,1): 1}, {C(1,1): 1, C(0,1): -1}))
        
        # Term 12: M12 = (A11 + A12 - A22 - A13)*(B22 + B21)
        terms.append(({A(0,0): 1, A(0,1): 1, A(1,1): -1, A(0,2): -1}, {B(1,1): 1, B(1,0): 1}, {C(0,1): 1, C(0,0): -1}))
        
        # Term 13: M13 = (A12 + A22 - A11 - A21)*(B12 + B22)
        terms.append(({A(0,1): 1, A(1,1): 1, A(0,0): -1, A(1,0): -1}, {B(0,1): 1, B(1,1): 1}, {C(0,0): 1, C(1,0): -1}))
        
        # Term 14: M14 = (A22 + A12 - A32 - A23)*(B23 + B33)
        terms.append(({A(1,1): 1, A(0,1): 1, A(2,1): -1, A(1,2): -1}, {B(1,2): 1, B(2,2): 1}, {C(1,2): 1, C(2,2): -1}))
        
        # Term 15: M15 = (A33 + A23 - A13 - A31)*(B31 + B11)
        terms.append(({A(2,2): 1, A(1,2): 1, A(0,2): -1, A(2,0): -1}, {B(2,0): 1, B(0,0): 1}, {C(2,0): 1, C(0,0): -1}))
        
        # Term 16: M16 = (A11 + A13 - A31 - A21)*(B31 + B21)
        terms.append(({A(0,0): 1, A(0,2): 1, A(2,0): -1, A(1,0): -1}, {B(2,0): 1, B(1,0): 1}, {C(0,0): 1, C(1,0): -1}))
        
        # Term 17: M17 = (A31 + A33 - A11 - A22)*(B11 + B13)
        terms.append(({A(2,0): 1, A(2,2): 1, A(0,0): -1, A(1,1): -1}, {B(0,0): 1, B(0,2): 1}, {C(2,0): 1, C(0,2): -1}))
        
        # Term 18: M18 = (A21 + A23 - A33 - A12)*(B33 + B31)
        terms.append(({A(1,0): 1, A(1,2): 1, A(2,2): -1, A(0,1): -1}, {B(2,2): 1, B(2,0): 1}, {C(1,2): 1, C(2,0): -1}))
        
        # Term 19: M19 = (A13 + A33 - A23 - A12)*(B22 + B32)
        terms.append(({A(0,2): 1, A(2,2): 1, A(1,2): -1, A(0,1): -1}, {B(1,1): 1, B(2,1): 1}, {C(0,2): 1, C(2,1): -1}))
        
        # Term 20: M20 = (A32 + A12 - A22 - A33)*(B11 + B21)
        terms.append(({A(2,1): 1, A(0,1): 1, A(1,1): -1, A(2,2): -1}, {B(0,0): 1, B(1,0): 1}, {C(2,1): 1, C(0,0): -1}))
        
        # Term 21: M21 = (A11 + A21 - A31 - A32)*(B22 + B32)
        terms.append(({A(0,0): 1, A(1,0): 1, A(2,0): -1, A(2,1): -1}, {B(1,1): 1, B(2,1): 1}, {C(0,0): 1, C(2,1): -1}))
        
        # Term 22: M22 = (A22 + A32 - A12 - A13)*(B11 + B13)
        terms.append(({A(1,1): 1, A(2,1): 1, A(0,1): -1, A(0,2): -1}, {B(0,0): 1, B(0,2): 1}, {C(1,1): 1, C(0,2): -1}))
        
        # Term 23: M23 = (A33 + A13 - A23 - A22)*(B11 + B21)
        terms.append(({A(2,2): 1, A(0,2): 1, A(1,2): -1, A(1,1): -1}, {B(0,0): 1, B(1,0): 1}, {C(2,2): 1, C(1,0): -1}))
    
    def build_strassen3_plus_strassen2():
        """Build decomposition: Strassen-3 on indices {0,1,2} + Strassen-2 on index {3,4}."""
        terms = []
        
        # Strassen-3 for the 3x3 block at (0,0)
        add_strassen_3x3x3(terms, 0, 0, 0)
        
        # Strassen-2 for the 2x2 block at (3,3)
        add_strassen_2x2x2(terms, 3, 3, 3)
        
        # Cross terms: schoolbook for mixed blocks
        # A is 3x2 (rows 0-2, cols 3-4), B is 2x3 (rows 3-4, cols 0-2)
        # C contributions: for i in 0-2, j in 3-4, k in 0-2
        for i in range(3):
            for j in range(3, 5):
                for k in range(3):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        
        # A is 2x3 (rows 3-4, cols 0-2), B is 3x2 (rows 0-2, cols 3-4)
        # C contributions: for i in 3-4, j in 0-2, k in 3-4
        for i in range(3, 5):
            for j in range(3):
                for k in range(3, 5):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        
        # A is 3x3 (rows 0-2, cols 0-2), B is 3x3 (rows 0-2, cols 0-2)
        # But we already have Strassen-3 for this, so no additional terms needed
        
        # A is 2x2 (rows 3-4, cols 3-4), B is 2x2 (rows 3-4, cols 3-4)
        # But we already have Strassen-2 for this, so no additional terms needed
        
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for r in range(rank):
            U_dict, V_dict, W_dict = terms[r]
            for idx, val in U_dict.items():
                U[idx][r] = val
            for idx, val in V_dict.items():
                V[idx][r] = val
            for idx, val in W_dict.items():
                W[idx][r] = val
        
        return U, V, W, terms
    
    def build_block_decomposition(partition):
        """Build a block decomposition with Strassen for 2x2x2 blocks."""
        terms = []
        blocks = partition
        
        for r1, r2 in blocks:
            for c1, c2 in blocks:
                for k1, k2 in blocks:
                    ar = r2 - r1
                    ac = c2 - c1
                    bc = k2 - k1
                    
                    if ar == 2 and ac == 2 and bc == 2:
                        add_strassen_2x2x2(terms, r1, c1, k1)
                    else:
                        for ii in range(r1, r2):
                            for jj in range(c1, c2):
                                for kk in range(k1, k2):
                                    u_idx = ii * m + jj
                                    v_idx = jj * p + kk
                                    w_idx = kk * n + ii
                                    terms.append(({u_idx: 1}, {v_idx: 1}, {w_idx: 1}))
        
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for r in range(rank):
            U_dict, V_dict, W_dict = terms[r]
            for idx, val in U_dict.items():
                U[idx][r] = val
            for idx, val in V_dict.items():
                V[idx][r] = val
            for idx, val in W_dict.items():
                W[idx][r] = val
        
        return U, V, W, terms
    
    # Try different approaches
    best_U, best_V, best_W = get_schoolbook()
    best_rank = len(best_U[0])
    
    # First try the Strassen-3 + Strassen-2 combination
    if time.time() < deadline - 30.0:
        U, V, W, terms = build_strassen3_plus_strassen2()
        rank = len(U[0])
        if verify(U, V, W):
            if rank < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = rank
    
    # Try different block partitions
    partitions = [
        [(0, 2), (2, 4), (4, 5)],  # [2,2,1]
        [(0, 4), (4, 5)],          # [4,1]
        [(0, 3), (3, 5)],          # [3,2]
    ]
    
    for partition in partitions:
        if time.time() > deadline - 20.0:
            break
        U, V, W, terms = build_block_decomposition(partition)
        rank = len(U[0])
        if verify(U, V, W):
            if rank < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = rank
    
    # Try greedy local search: merge pairs of terms
    if best_rank < n * m * p and time.time() < deadline - 30.0:
        # Rebuild terms from best decomposition
        terms = []
        for r in range(best_rank):
            U_dict = {}
            V_dict = {}
            W_dict = {}
            for idx in range(n * m):
                if best_U[idx][r] != 0:
                    U_dict[idx] = best_U[idx][r]
            for idx in range(m * p):
                if best_V[idx][r] != 0:
                    V_dict[idx] = best_V[idx][r]
            for idx in range(p * n):
                if best_W[idx][r] != 0:
                    W_dict[idx] = best_W[idx][r]
            terms.append((U_dict, V_dict, W_dict))
        
        def can_merge(term1, term2):
            """Check if term1 + term2 is a rank-1 tensor."""
            t1 = {}
            for u_idx, u_val in term1[0].items():
                for v_idx, v_val in term1[1].items():
                    for w_idx, w_val in term1[2].items():
                        key = (u_idx, v_idx, w_idx)
                        t1[key] = t1.get(key, 0) + u_val * v_val * w_val
            
            t2 = {}
            for u_idx, u_val in term2[0].items():
                for v_idx, v_val in term2[1].items():
                    for w_idx, w_val in term2[2].items():
                        key = (u_idx, v_idx, w_idx)
                        t2[key] = t2.get(key, 0) + u_val * v_val * w_val
            
            combined = {}
            for k, v in t1.items():
                combined[k] = v
            for k, v in t2.items():
                combined[k] = combined.get(k, 0) + v
            
            # Check if combined is rank-1 by checking if all 2x2 minors are zero
            u_indices = sorted(set(k[0] for k in combined.keys()))
            v_indices = sorted(set(k[1] for k in combined.keys()))
            w_indices = sorted(set(k[2] for k in combined.keys()))
            
            # For rank-1, the tensor must be outer product of three vectors
            # Check by trying to find vectors u, v, w such that T[i,j,k] = u[i]*v[j]*w[k]
            if not u_indices or not v_indices or not w_indices:
                return False
            
            # Pick a reference index with non-zero value
            ref_key = None
            for key, val in combined.items():
                if abs(val) > 1e-12:
                    ref_key = key
                    break
            
            if ref_key is None:
                return True  # All zeros, trivially rank 0
            
            u0, v0, w0 = ref_key
            ref_val = combined[ref_key]
            
            # Try to determine u, v, w from ref
            # u[u0] * v[v0] * w[w0] = ref_val
            # We need to check consistency
            
            # For each u_i, find a (v_j, w_k) with non-zero to determine u_i
            # This is complex; use a simpler check: all 2x2 minors in each mode must be zero
            
            # Check u-mode 2x2 minors
            for i1 in u_indices:
                for i2 in u_indices:
                    if i1 >= i2:
                        continue
                    for j1 in v_indices:
                        for j2 in v_indices:
                            if j1 >= j2:
                                continue
                            for k1 in w_indices:
                                for k2 in w_indices:
                                    if k1 >= k2:
                                        continue
                                    # 2x2 minor in u-v plane at fixed w
                                    a = combined.get((i1, j1, k1), 0)
                                    b = combined.get((i1, j2, k1), 0)
                                    c = combined.get((i2, j1, k1), 0)
                                    d = combined.get((i2, j2, k1), 0)
                                    if abs(a * d - b * c) > 1e-9:
                                        return False
                                    # 2x2 minor in u-w plane at fixed v
                                    a = combined.get((i1, j1, k1), 0)
                                    b = combined.get((i1, j1, k2), 0)
                                    c = combined.get((i2, j1, k1), 0)
                                    d = combined.get((i2, j1, k2), 0)
                                    if abs(a * d - b * c) > 1e-9:
                                        return False
                                    # 2x2 minor in v-w plane at fixed u
                                    a = combined.get((i1, j1, k1), 0)
                                    b = combined.get((i1, j1, k2), 0)
                                    c = combined.get((i1, j2, k1), 0)
                                    d = combined.get((i1, j2, k2), 0)
                                    if abs(a * d - b * c) > 1e-9:
                                        return False
            return True
        
        # Try to merge pairs
        merged = True
        while merged and time.time() < deadline - 10.0:
            merged = False
            new_terms = list(terms)
            for i in range(len(new_terms)):
                if time.time() > deadline - 10.0:
                    break
                for j in range(i + 1, len(new_terms)):
                    if time.time() > deadline - 10.0:
                        break
                    if can_merge(new_terms[i], new_terms[j]):
                        # Merge them
                        U_dict = {}
                        V_dict = {}
                        W_dict = {}
                        for idx, val in new_terms[i][0].items():
                            U_dict[idx] = U_dict.get(idx, 0) + val
                        for idx, val in new_terms[j][0].items():
                            U_dict[idx] = U_dict.get(idx, 0) + val
                        for idx, val in new_terms[i][1].items():
                            V_dict[idx] = V_dict.get(idx, 0) + val
                        for idx, val in new_terms[j][1].items():
                            V_dict[idx] = V_dict.get(idx, 0) + val
                        for idx, val in new_terms[i][2].items():
                            W_dict[idx] = W_dict.get(idx, 0) + val
                        for idx, val in new_terms[j][2].items():
                            W_dict[idx] = W_dict.get(idx, 0) + val
                        
                        # Remove zeros
                        U_dict = {k: v for k, v in U_dict.items() if abs(v) > 1e-12}
                        V_dict = {k: v for k, v in V_dict.items() if abs(v) > 1e-12}
                        W_dict = {k: v for k, v in W_dict.items() if abs(v) > 1e-12}
                        
                        new_terms = [t for idx, t in enumerate(new_terms) if idx != i and idx != j]
                        new_terms.append((U_dict, V_dict, W_dict))
                        terms = new_terms
                        merged = True
                        break
                if merged:
                    break
        
        # Rebuild matrices
        rank = len(terms)
        if rank < best_rank:
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            for r in range(rank):
                U_dict, V_dict, W_dict = terms[r]
                for idx, val in U_dict.items():
                    U[idx][r] = val
                for idx, val in V_dict.items():
                    V[idx][r] = val
                for idx, val in W_dict.items():
                    W[idx][r] = val
            if verify(U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
