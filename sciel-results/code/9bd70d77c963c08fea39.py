import numpy as np
import time
from scipy.optimize import linprog

DIMENSIONS = [3, 5, 5]

def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W form a valid tensor decomposition for matrix multiplication.
    """
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_reshaped = U_np.reshape(n, m, rank)
    V_reshaped = V_np.reshape(m, p, rank)
    W_reshaped = W_np.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_reshaped[i, j, r] * V_reshaped[j, k, r] * W_reshaped[k, i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_candidate_terms(n, m, p):
    """
    Build a set of candidate rank-1 terms for the tensor decomposition.
    Each term is defined by (i, j, k) indices.
    """
    candidates = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                candidates.append((i, j, k))
    return candidates

def _solve_ilp_rank_reduction(n, m, p, max_rank=None):
    """
    Use integer linear programming to find a minimal set of rank-1 terms
    that span the tensor's support.
    """
    candidates = _build_candidate_terms(n, m, p)
    num_candidates = len(candidates)
    
    if max_rank is None:
        max_rank = n * m * p
    
    # We need to select at least one term for each (i,j,k) in the support
    # This is a set cover problem
    
    # Binary variable x_c = 1 if candidate c is selected
    # Objective: minimize sum(x_c)
    c_obj = np.ones(num_candidates)
    
    # Constraints: for each (i,j,k), at least one selected term must cover it
    # A term (i', j', k') covers (i,j,k) if i'=i, j'=j, k'=k
    # So we need exactly the terms where i=i', j=j', k=k'
    
    # Actually, this is simpler: we need to select terms such that for each output element,
    # the sum of products equals 1.
    
    # Let's think differently. The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For the trivial decomposition, each term (i,j,k) contributes 1 to T[i,j,k].
    
    # To reduce rank, we need to find a subset of terms and assign weights such that
    # the weighted sum still gives T[i,j,k] = 1 for all (i,j,k).
    
    # This is equivalent to finding a sparse representation.
    
    # For now, let's try a greedy approach with ILP relaxation
    
    # Build constraint matrix: A[i,j,k,c] = 1 if term c can contribute to (i,j,k)
    # Term (i',j',k') contributes to (i,j,k) only if i=i', j=j', k=k'
    
    num_constraints = n * m * p
    A = np.zeros((num_constraints, num_candidates))
    
    for idx_c, (i, j, k) in enumerate(candidates):
        idx_constraint = i * m * p + j * p + k
        A[idx_constraint, idx_c] = 1.0
    
    # b = 1 for all constraints
    b = np.ones(num_constraints)
    
    # Bounds: 0 <= x_c <= 1
    bounds = [(0, 1)] * num_candidates
    
    # Solve LP relaxation
    try:
        result = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        
        if result.status == 0:
            # Get the solution
            x = result.x
            selected = [i for i in range(num_candidates) if x[i] > 0.5]
            
            if len(selected) < num_candidates:
                # Build the decomposition using only selected terms
                rank = len(selected)
                U = [[0] * rank for _ in range(n * m)]
                V = [[0] * rank for _ in range(m * p)]
                W = [[0] * rank for _ in range(p * n)]
                
                for col, idx in enumerate(selected):
                    i, j, k = candidates[idx]
                    weight = x[idx]
                    U[i * m + j][col] = weight
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                
                # Verify
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
        
    except Exception:
        pass
    
    return None

def _try_greedy_reduction(n, m, p):
    """
    Try greedy rank reduction by iteratively removing terms.
    """
    candidates = _build_candidate_terms(n, m, p)
    num_candidates = len(candidates)
    
    # Start with all terms
    selected = list(range(num_candidates))
    
    # Try to remove terms one by one
    for attempt in range(min(50, num_candidates)):  # Limit attempts
        removed = False
        for idx in list(selected):
            if len(selected) <= 1:
                break
            
            # Try removing this term
            new_selected = [i for i in selected if i != idx]
            
            # Build decomposition with new_selected
            rank = len(new_selected)
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            
            for col, i_idx in enumerate(new_selected):
                i, j, k = candidates[i_idx]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
            
            # Verify
            if _verify_decomposition(U, V, W, n, m, p):
                selected = new_selected
                removed = True
                break
        
        if not removed:
            break
    
    if len(selected) < num_candidates:
        rank = len(selected)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for col, i_idx in enumerate(selected):
            i, j, k = candidates[i_idx]
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
        
        return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Start with trivial decomposition
    best_result = _trivial_decomposition(n, m, p)
    
    # Try ILP-based reduction
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:  # Leave 30s margin
        ilp_result = _solve_ilp_rank_reduction(n, m, p)
        if ilp_result is not None:
            if len(ilp_result['U'][0]) < len(best_result['U'][0]):
                best_result = ilp_result
    
    # Try greedy reduction
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:  # Leave 15s margin
        greedy_result = _try_greedy_reduction(n, m, p)
        if greedy_result is not None:
            if len(greedy_result['U'][0]) < len(best_result['U'][0]):
                best_result = greedy_result
    
    # Final verification
    U, V, W = best_result['U'], best_result['V'], best_result['W']
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to trivial
        return _trivial_decomposition(n, m, p)
    
    return best_result
