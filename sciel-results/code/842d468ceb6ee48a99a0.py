import time
import itertools
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
                nonzero_indices.append((idx_u, idx_v, idx_w))
    
    # Start with naive rank-12 decomposition
    rank_naive = n * m * p
    U_naive = [[0] * rank_naive for _ in range(n * m)]
    V_naive = [[0] * rank_naive for _ in range(m * p)]
    W_naive = [[0] * rank_naive for _ in range(p * n)]
    
    r = 0
    for (idx_u, idx_v, idx_w) in nonzero_indices:
        U_naive[idx_u][r] = 1
        V_naive[idx_v][r] = 1
        W_naive[idx_w][r] = 1
        r += 1
    
    # Try to find lower-rank decompositions using structured search
    # Strategy: For each candidate rank r < 12, try to find a decomposition
    
    def verify_decomposition(U, V, W, T):
        """Verify that U, V, W exactly decompose T"""
        rank = len(U[0])
        for idx_u in range(n * m):
            for idx_v in range(m * p):
                for idx_w in range(p * n):
                    val = 0.0
                    for r in range(rank):
                        val += U[idx_u][r] * V[idx_v][r] * W[idx_w][r]
                    expected = T[idx_u, idx_v, idx_w]
                    if abs(val - expected) > 1e-9:
                        return False
        return True
    
    # Generate candidate rank-1 tensors with structured patterns
    # Key insight: For matrix multiplication, we can use patterns that exploit
    # the structure of the tensor
    
    def generate_structured_candidates():
        """Generate structured candidate rank-1 tensors"""
        candidates = []
        
        # Pattern 1: Single entry (naive)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    u[i * m + j] = 1
                    v[j * p + k] = 1
                    w[k * n + i] = 1
                    candidates.append((u, v, w))
        
        # Pattern 2: Strassen-like for 2x2 blocks
        # Try combinations that cover multiple (i,j,k) triples
        
        # For 2x3x4, we can try to use the fact that:
        # (a+b)(c+d) = ac + ad + bc + bd
        # This suggests we can combine terms
        
        # Try rank-2 patterns for pairs of k values
        for i in range(n):
            for j in range(m):
                for k1 in range(p):
                    for k2 in range(p):
                        if k1 >= k2:
                            continue
                        # Try: (a_j) * (b_{k1} + b_{k2})
                        u = [0] * (n * m)
                        v = [0] * (m * p)
                        w = [0] * (p * n)
                        u[i * m + j] = 1
                        v[j * p + k1] = 1
                        v[j * p + k2] = 1
                        w[k1 * n + i] = 1
                        w[k2 * n + i] = 1
                        candidates.append((u, v, w))
                        
                        # Try: (a_j) * (b_{k1} - b_{k2})
                        u2 = [0] * (n * m)
                        v2 = [0] * (m * p)
                        w2 = [0] * (p * n)
                        u2[i * m + j] = 1
                        v2[j * p + k1] = 1
                        v2[j * p + k2] = -1
                        w2[k1 * n + i] = 1
                        w2[k2 * n + i] = -1
                        candidates.append((u2, v2, w2))
        
        # Try rank-2 patterns for pairs of i values
        for j in range(m):
            for k in range(p):
                for i1 in range(n):
                    for i2 in range(n):
                        if i1 >= i2:
                            continue
                        # Try: (a_{i1} + a_{i2}) * b_j * (c_k)
                        u = [0] * (n * m)
                        v = [0] * (m * p)
                        w = [0] * (p * n)
                        u[i1 * m + j] = 1
                        u[i2 * m + j] = 1
                        v[j * p + k] = 1
                        w[k * n + i1] = 1
                        w[k * n + i2] = 1
                        candidates.append((u, v, w))
                        
                        # Try: (a_{i1} - a_{i2}) * b_j * (c_k)
                        u2 = [0] * (n * m)
                        v2 = [0] * (m * p)
                        w2 = [0] * (p * n)
                        u2[i1 * m + j] = 1
                        u2[i2 * m + j] = -1
                        v2[j * p + k] = 1
                        w2[k * n + i1] = 1
                        w2[k * n + i2] = -1
                        candidates.append((u2, v2, w2))
        
        # Try rank-2 patterns for pairs of j values
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        if j1 >= j2:
                            continue
                        # Try: a_i * (b_{j1} + b_{j2}) * c_k
                        u = [0] * (n * m)
                        v = [0] * (m * p)
                        w = [0] * (p * n)
                        u[i * m + j1] = 1
                        u[i * m + j2] = 1
                        v[j1 * p + k] = 1
                        v[j2 * p + k] = 1
                        w[k * n + i] = 1
                        candidates.append((u, v, w))
                        
                        # Try: a_i * (b_{j1} - b_{j2}) * c_k
                        u2 = [0] * (n * m)
                        v2 = [0] * (m * p)
                        w2 = [0] * (p * n)
                        u2[i * m + j1] = 1
                        u2[i * m + j2] = -1
                        v2[j1 * p + k] = 1
                        v2[j2 * p + k] = -1
                        w2[k * n + i] = 1
                        candidates.append((u2, v2, w2))
        
        return candidates
    
    # Greedy selection: try to find a small set of rank-1 tensors that span T
    def greedy_select(candidates, T, max_rank=11):
        """Greedy selection of rank-1 tensors"""
        # Flatten T
        T_flat = T.flatten()
        
        # Compute residual
        residual = T_flat.copy()
        selected = []
        
        for _ in range(max_rank):
            if time.time() > deadline - 10:
                break
            
            best_idx = -1
            best_reduction = 0
            
            for idx, (u, v, w) in enumerate(candidates):
                # Compute rank-1 tensor
                rank1 = np.outer(np.outer(u, v), w)
                rank1_flat = rank1.flatten()
                
                # Project residual onto this tensor
                # coefficient = <residual, rank1> / ||rank1||^2
                dot = np.dot(residual, rank1_flat)
                norm_sq = np.dot(rank1_flat, rank1_flat)
                
                if norm_sq < 1e-12:
                    continue
                
                coeff = dot / norm_sq
                
                # Reduction in residual norm
                new_residual = residual - coeff * rank1_flat
                reduction = np.dot(residual, residual) - np.dot(new_residual, new_residual)
                
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_idx = idx
                    best_coeff = coeff
            
            if best_idx < 0 or best_reduction < 1e-12:
                break
            
            u, v, w = candidates[best_idx]
            selected.append((u, v, w, best_coeff))
            residual -= best_coeff * np.outer(np.outer(u, v), w).flatten()
            
            if np.linalg.norm(residual) < 1e-9:
                break
        
        return selected
    
    # Try to find lower-rank decompositions
    best_result = None
    best_rank = rank_naive
    
    # First, try greedy selection with structured candidates
    candidates = generate_structured_candidates()
    
    # Deduplicate candidates (normalize them)
    unique_candidates = []
    seen = set()
    for u, v, w in candidates:
        # Create a canonical form
        key = (tuple(u), tuple(v), tuple(w))
        if key not in seen:
            seen.add(key)
            unique_candidates.append((u, v, w))
    
    candidates = unique_candidates
    
    # Try greedy selection
    selected = greedy_select(candidates, T, max_rank=11)
    
    if selected:
        rank = len(selected)
        if rank < best_rank:
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            for r_idx, (u, v, w, coeff) in enumerate(selected):
                for idx in range(n * m):
                    U[idx][r_idx] = u[idx] * coeff
                for idx in range(m * p):
                    V[idx][r_idx] = v[idx] * coeff
                for idx in range(p * n):
                    W[idx][r_idx] = w[idx] * coeff
            
            # Verify
            if verify_decomposition(U, V, W, T):
                best_rank = rank
                best_result = {"U": U, "V": V, "W": W}
    
    # If greedy didn't work, try a more systematic approach
    # For each candidate rank r from 7 to 11, try to find a decomposition
    
    for target_rank in range(7, min(12, rank_naive)):
        if time.time() > deadline - 20:
            break
        
        # Try to find a rank-r decomposition using linear programming
        # This is heuristic - we'll try random combinations of candidates
        
        import random
        random.seed(42)
        
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            
            # Select random subset of candidates
            if len(candidates) < target_rank:
                break
            
            subset = random.sample(candidates, target_rank)
            
            # Build linear system: T = sum_r c_r * (u_r ⊗ v_r ⊗ w_r)
            # This is linear in c_r
            
            # Flatten all rank-1 tensors
            A = np.zeros((n * m * m * p * p * n, target_rank))
            for r_idx, (u, v, w) in enumerate(subset):
                rank1 = np.outer(np.outer(u, v), w)
                A[:, r_idx] = rank1.flatten()
            
            T_flat = T.flatten()
            
            # Solve least squares
            try:
                coeffs, residuals, rank_ls, sv = np.linalg.lstsq(A, T_flat, rcond=None)
                
                # Check if solution is good
                residual = np.linalg.norm(A @ coeffs - T_flat)
                
                if residual < 1e-6:
                    # Found a decomposition
                    U = [[0] * target_rank for _ in range(n * m)]
                    V = [[0] * target_rank for _ in range(m * p)]
                    W = [[0] * target_rank for _ in range(p * n)]
                    
                    for r_idx, (u, v, w) in enumerate(subset):
                        c = coeffs[r_idx]
                        for idx in range(n * m):
                            U[idx][r_idx] = u[idx] * c
                        for idx in range(m * p):
                            V[idx][r_idx] = v[idx] * c
                        for idx in range(p * n):
                            W[idx][r_idx] = w[idx] * c
                    
                    if verify_decomposition(U, V, W, T):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_result = {"U": U, "V": V, "W": W}
                        break
            except:
                pass
    
    # Return best result found
    if best_result is not None:
        return best_result
    else:
        return {"U": U_naive, "V": V_naive, "W": W_naive}
