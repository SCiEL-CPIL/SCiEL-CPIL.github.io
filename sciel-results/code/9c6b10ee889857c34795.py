"""Deterministic Kronecker-product decomposition for the 2x5 x 5x6 tensor."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Target tensor: T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # We want to express T as sum of rank-1 tensors: U[:,r] * V[:,r] * W[:,r]
    
    # Strategy: Greedy set-cover over candidate rank-1 tensors
    # A rank-1 tensor is defined by (u, v, w) where u in R^nm, v in R^mp, w in R^pn
    # The tensor product u⊗v⊗w has entries u[a]*v[b]*w[c]
    
    # First, let's try to find a decomposition using structured candidates
    # based on Strassen-like patterns and linear combinations
    
    # Build candidate pool: use small support patterns
    # For 2x5x6, we can try various block structures
    
    # Let's use a more systematic approach: 
    # Try to find rank-r decomposition by solving linear systems
    
    # The tensor T has dimensions (10, 30, 12)
    # Flatten: T_flat[a, b, c] where a in 0..9, b in 0..29, c in 0..11
    
    # For rank r, we need U (10 x r), V (30 x r), W (12 x r)
    # such that sum_r U[:,r] * V[:,r] * W[:,r] = T
    
    # This is a bilinear system. Let's use a greedy approach:
    # At each step, pick the rank-1 tensor that best approximates the residual
    
    # Initialize target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Greedy: iteratively pick best rank-1 approximation
    # To make this tractable, we'll use a restricted candidate set
    
    # Candidate generation: for each (i,j,k) triple, the standard rank-1 term
    # Also try Strassen-like combinations for 2x2x2 sub-blocks
    
    # Let's first try the naive approach with rank = n*m*p = 60
    # Then try to reduce using Strassen blocks
    
    # Actually, let's try a direct optimization approach:
    # Use alternating minimization or a greedy algorithm
    
    # Given time constraints, let's implement a careful greedy set-cover:
    # 1. Generate candidate rank-1 tensors from structured patterns
    # 2. Greedily select those that cover the most residual mass
    
    # Candidate 1: Standard terms (one per (i,j,k))
    candidates = []
    
    # Standard basis: for each (i,j,k), u has 1 at i*m+j, v has 1 at j*p+k, w has 1 at k*n+i
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(nm)
                v = np.zeros(mp)
                w = np.zeros(pn)
                u[i*m+j] = 1.0
                v[j*p+k] = 1.0
                w[k*n+i] = 1.0
                candidates.append((u, v, w))
    
    # Now add Strassen-like candidates for 2x2x2 blocks
    # For blocks where i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    def strassen_2x2x2_candidates(i0, j0, k0):
        """Generate 7 Strassen rank-1 tensors for a 2x2x2 block."""
        cands = []
        
        # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
        # Map to global: gi = i0+li, gj = j0+lj, gk = k0+lk
        
        # T1: (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
        u1 = np.zeros(nm); v1 = np.zeros(mp); w1 = np.zeros(pn)
        u1[(i0)*m + j0] = 1.0; u1[(i0+1)*m + (j0+1)] = 1.0
        v1[j0*p + k0] = 1.0; v1[(j0+1)*p + (k0+1)] = 1.0
        w1[k0*n + i0] = 1.0; w1[(k0+1)*n + (i0+1)] = 1.0
        cands.append((u1, v1, w1))
        
        # T2: (a10+a11)*b00 -> c10 += 1, c11 -= 1
        u2 = np.zeros(nm); v2 = np.zeros(mp); w2 = np.zeros(pn)
        u2[(i0+1)*m + j0] = 1.0; u2[(i0+1)*m + (j0+1)] = 1.0
        v2[j0*p + k0] = 1.0
        w2[k0*n + (i0+1)] = 1.0; w2[(k0+1)*n + (i0+1)] = -1.0
        cands.append((u2, v2, w2))
        
        # T3: a00*(b01-b11) -> c01 += 1, c11 += 1
        u3 = np.zeros(nm); v3 = np.zeros(mp); w3 = np.zeros(pn)
        u3[i0*m + j0] = 1.0
        v3[j0*p + (k0+1)] = 1.0; v3[(j0+1)*p + (k0+1)] = -1.0
        w3[(k0+1)*n + i0] = 1.0; w3[(k0+1)*n + (i0+1)] = 1.0
        cands.append((u3, v3, w3))
        
        # T4: a11*(b10-b00) -> c00 += 1, c10 += 1
        u4 = np.zeros(nm); v4 = np.zeros(mp); w4 = np.zeros(pn)
        u4[(i0+1)*m + (j0+1)] = 1.0
        v4[(j0+1)*p + k0] = 1.0; v4[j0*p + k0] = -1.0
        w4[k0*n + i0] = 1.0; w4[k0*n + (i0+1)] = 1.0
        cands.append((u4, v4, w4))
        
        # T5: (a00+a01)*b11 -> c00 -= 1, c01 += 1
        u5 = np.zeros(nm); v5 = np.zeros(mp); w5 = np.zeros(pn)
        u5[i0*m + j0] = 1.0; u5[i0*m + (j0+1)] = 1.0
        v5[(j0+1)*p + (k0+1)] = 1.0
        w5[k0*n + i0] = -1.0; w5[(k0+1)*n + i0] = 1.0
        cands.append((u5, v5, w5))
        
        # T6: (a10-a00)*(b00+b01) -> c11 += 1
        u6 = np.zeros(nm); v6 = np.zeros(mp); w6 = np.zeros(pn)
        u6[(i0+1)*m + j0] = 1.0; u6[i0*m + j0] = -1.0
        v6[j0*p + k0] = 1.0; v6[j0*p + (k0+1)] = 1.0
        w6[(k0+1)*n + (i0+1)] = 1.0
        cands.append((u6, v6, w6))
        
        # T7: (a01-a11)*(b10+b11) -> c00 += 1
        u7 = np.zeros(nm); v7 = np.zeros(mp); w7 = np.zeros(pn)
        u7[i0*m + (j0+1)] = 1.0; u7[(i0+1)*m + (j0+1)] = -1.0
        v7[(j0+1)*p + k0] = 1.0; v7[(j0+1)*p + (k0+1)] = 1.0
        w7[k0*n + i0] = 1.0
        cands.append((u7, v7, w7))
        
        return cands
    
    # Add Strassen candidates for all valid 2x2x2 blocks
    for i0 in range(n-1):
        for j0 in range(m-1):
            for k0 in range(p-1):
                candidates.extend(strassen_2x2x2_candidates(i0, j0, k0))
    
    # Also add some mixed candidates that span multiple blocks
    # For the 2x5x6 case, we can try candidates that cover 2x2x3 or 2x3x2 blocks
    
    # Greedy set-cover: at each step, pick the candidate that minimizes ||T - T_approx||_F^2
    # i.e., maximizes the reduction in residual
    
    residual = T.copy()
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Precompute tensor products for efficiency
    # For each candidate (u,v,w), compute the tensor u⊗v⊗w
    # and its inner product with the residual
    
    max_rank = 54  # Start with upper bound
    min_rank_found = max_rank
    best_decomp = None
    
    # Try greedy from different starting points
    for max_iter in [max_rank, 50, 45, 40]:
        if time.time() > deadline - 30:
            break
            
        residual = T.copy()
        U_cols = []
        V_cols = []
        W_cols = []
        
        for step in range(max_iter):
            if time.time() > deadline - 30:
                    break
                
            best_score = -np.inf
            best_idx = -1
            
            for idx, (u, v, w) in enumerate(candidates):
                # Compute tensor product u⊗v⊗w
                # tensor[a,b,c] = u[a]*v[b]*w[c]
                # Inner product with residual: sum_{a,b,c} residual[a,b,c] * u[a]*v[b]*w[c]
                # = sum_a u[a] * (sum_b v[b] * (sum_c residual[a,b,c]*w[c]))
                
                # Efficient computation:
                # First: temp[b,c] = v[b]*w[c] (outer product)
                # Then: temp2[a,c] = sum_b residual[a,b,c] * v[b] 
                # Then: score = sum_{a,c} u[a] * temp2[a,c] * w[c]
                
                # Even better: 
                # score = sum_{a,b,c} residual[a,b,c] * u[a] * v[b] * w[c]
                # = u^T * (sum_a residual * u) contracted with v and w
                
                # Let's compute: R_u[b,c] = sum_a residual[a,b,c] * u[a]
                # Then: score = sum_{b,c} R_u[b,c] * v[b] * w[c]
                
                R_u = np.tensordot(residual, u, axes=([0], [0]))  # shape (mp, pn)
                score = np.sum(R_u * np.outer(v, w))
                
                # The reduction in ||residual||^2 is 2*score - ||u⊗v⊗w||^2
                # ||u⊗v⊗w||^2 = ||u||^2 * ||v||^2 * ||w||^2
                norm_sq = np.sum(u**2) * np.sum(v**2) * np.sum(w**2)
                reduction = 2*score - norm_sq
                
                if reduction > best_score:
                    best_score = reduction
                    best_idx = idx
            
            if best_idx < 0 or best_score < 1e-10:
                break
            
            u, v, w = candidates[best_idx]
            residual -= np.einsum('a,b,c->abc', u, v, w)
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)
        
        if len(U_cols) < min_rank_found:
            min_rank_found = len(U_cols)
            best_decomp = (U_cols, V_cols, W_cols)
        
        if time.time() > deadline - 20:
            break
    
    # If greedy didn't find anything better, fall back to the original 54-rank construction
    if best_decomp is None or min_rank_found >= 54:
        # Use the original deterministic construction
        rank = 54
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        col = 0
        # S1: 2 x 1 x 6
        for i in range(n):
            for k in range(p):
                U[i*m+0][col] = 1
                V[0*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
        
        # S2: 2 x 4 x 6 using Strassen
        def strassen_2x2x2_terms():
            return [
                ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
                ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
                ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
                ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
                ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
                ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
                ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
            ]
        
        terms = strassen_2x2x2_terms()
        for j_block in range(2):
            j_base = 1 + j_block * 2
            for k_block in range(3):
                k_base = k_block * 2
                for u_list, v_list, w_list in terms:
                    for (li, lj, val) in u_list:
                        U[(li)*m + (j_base+lj)][col] += val
                    for (lj, lk, val) in v_list:
                        V[(j_base+lj)*p + (k_base+lk)][col] += val
                    for (li, lk, val) in w_list:
                        W[(k_base+lk)*n + li][col] += val
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Convert best_decomp to list-of-lists format
    U_cols, V_cols, W_cols = best_decomp
    rank = len(U_cols)
    U = [[0]*rank for _ in range(nm)]
    V = [[0]*rank for _ in range(mp)]
    W = [[0]*rank for _ in range(pn)]
    
    for r in range(rank):
        for a in range(nm):
            U[a][r] = U_cols[r][a]
        for b in range(mp):
            V[b][r] = V_cols[r][b]
        for c in range(pn):
            W[c][r] = W_cols[r][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
