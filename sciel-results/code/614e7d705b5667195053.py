"""Matrix multiplication 4x5x6 tensor decomposition - search for rank-minimal."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_arr[i * m + j]
                v_row = V_arr[j * p + k]
                w_row = W_arr[k * n + i]
                val = np.dot(u_row, v_row * w_row)
                if abs(val - 1.0) > 1e-9:
                    return False, (i, j, k, val)
    return True, None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def _try_strassen_embedding(n, m, p):
    """
    Try to embed the 4x5x6 multiplication into a larger structured decomposition.
    Use a block-diagonal approach with small Strassen-like blocks.
    """
    # This is a heuristic attempt; return None if it doesn't work
    pass

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = len(U[0])
    best_result = (U, V, W)
    
    # Strategy: Try to find a lower rank decomposition using a greedy/iterative approach
    # The tensor T[i,j,k] = 1 for all (i,j,k) in [n]x[m]x[p]
    # We want rank-r tensors U (nm x r), V (mp x r), W (pn x r) such that
    # for all (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Approach: Start with schoolbook and try to merge columns
    # Two columns c1, c2 can be merged if there exist scalars such that
    # the combined column still satisfies all constraints.
    
    # Better approach: Use a randomized search for structured decompositions
    # Try ranks from a lower bound upward
    
    # Theoretical lower bound: rank >= max(n*m, m*p, p*n) / something
    # Actually rank >= max(n*m, m*p, p*n) is not tight
    # Known: rank(4x5x6) is at most 120 (schoolbook)
    # Let's try to find something better
    
    # Method: Iterative column elimination
    # For each pair of columns, check if one can be removed by adjusting others
    # This is expensive, so use a greedy approach
    
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    current_rank = best_rank
    
    # Try to reduce rank by removing redundant columns
    # A column c is redundant if the tensor it contributes can be represented
    # by the other columns
    
    # For efficiency, work with numpy
    U_arr = np.array(current_U, dtype=np.float64)
    V_arr = np.array(current_V, dtype=np.float64)
    W_arr = np.array(current_W, dtype=np.float64)
    
    # Try iterative column removal
    # For each column, check if removing it still allows exact reconstruction
    # This is a hard problem, so use a heuristic:
    # Try to find linear dependencies among the "rank-1 tensors"
    
    # Alternative: Try specific structured decompositions
    # For 4x5x6, we can try a block approach
    
    # Let's try: split into blocks and use Strassen-like recursion
    # 4x5 can be seen as blocks, 5x6 as blocks
    
    # Actually, let's try a direct optimization approach
    # Start from schoolbook and try to find a lower-rank equivalent
    
    # Key insight: The matrix multiplication tensor T_{i,j,k} = 1 
    # can be decomposed as sum of rank-1 tensors
    # A rank-1 tensor is u_i * v_j * w_k
    
    # Let's try to find a decomposition with rank < 120
    
    # Use a greedy algorithm:
    # 1. Start with the full tensor
    # 2. Find the best rank-1 approximation
    # 3. Subtract and repeat
    
    # But this gives approximate, not exact decompositions
    
    # For exact decompositions with integer/half-integer entries,
    # we need a different approach
    
    # Let's try: enumerate small-rank decompositions with entries in {0, 1, -1, 1/2, -1/2}
    # This is a combinatorial search, so use randomization
    
    # Try ranks from 100 downward (optimistic)
    # Actually, let's be more systematic
    
    # Known results: 
    # - 3x3x3 has rank 7 (Strassen)
    # - 4x4x4 has rank 49 (or lower with newer results)
    # - For 4x5x6, no known sub-schoolbook rank is widely established
    
    # Let's try a structured approach using the fact that
    # 4x5x6 can be related to 4x4x4, 5x5x5, 6x6x6
    
    # Padding approach: pad 4x5x6 to 6x6x6, use a 6x6x6 decomposition,
    # then restrict to the 4x5x6 sub-tensor
    
    # For 6x6x6, schoolbook is 216. If we have a better 6x6x6, we could use it.
    
    # Let's try a different approach: use the fact that
    # T[i,j,k] = 1 = (sum over some structured basis)
    
    # Try: T[i,j,k] = sum_{a=0}^{n-1} sum_{b=0}^{m-1} ... 
    # This is just schoolbook.
    
    # Let's try a randomized search for small improvements
    # Start from schoolbook and try to merge pairs of columns
    
    # For merging columns c1 and c2:
    # We need to find new values for all other columns such that
    # the sum still equals 1 for all (i,j,k)
    # This is a system of linear equations
    
    # Let's implement a simple column-merging heuristic
    
    def try_merge_columns(U_arr, V_arr, W_arr, c1, c2, n, m, p):
        """Try to merge columns c1 and c2. Returns True if successful."""
        nm, mp, pn = n * m, m * p, p * n
        r = U_arr.shape[1]
        
        # The contribution of column c is:
        # T_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # We want to absorb c2 into c1 (and possibly adjust other columns)
        
        # Simple approach: set column c2 to zero and see if we can adjust c1
        # to compensate. This only works if the tensors are linearly dependent.
        
        # Check if tensor c2 is a scalar multiple of tensor c1
        t1 = np.outer(np.outer(U_arr[:, c1], V_arr[:, c2]), W_arr[:, c1])  # wrong
        # Let's compute properly
        t1 = np.zeros((n, m, p))
        t2 = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    t1[i, j, k] = U_arr[i*m+j, c1] * V_arr[j*p+k, c1] * W_arr[k*n+i, c1]
                    t2[i, j, k] = U_arr[i*m+j, c2] * V_arr[j*p+k, c2] * W_arr[k*n+i, c2]
        
        # Check if t2 is a scalar multiple of t1
        nonzero1 = np.abs(t1) > 1e-12
        nonzero2 = np.abs(t2) > 1e-12
        
        if not np.any(nonzero1):
            # t1 is zero, just remove c2
            U_arr[:, c2] = 0
            V_arr[:, c2] = 0
            W_arr[:, c2] = 0
            return True
        
        if not np.any(nonzero2):
            # t2 is zero, nothing to do
            return True
        
        # Check if t2 = alpha * t1
        alpha_candidates = []
        for idx in np.where(nonzero1 & nonzero2):
            i, j, k = idx
            alpha_candidates.append(t2[i,j,k] / t1[i,j,k])
        
        if not alpha_candidates:
            # No overlap, can't merge
            return False
        
        alpha = alpha_candidates[0]
        for ac in alpha_candidates[1:]:
            if abs(ac - alpha) > 1e-9:
                return False
        
        # t2 = alpha * t1, so we can add alpha*t1 to... wait, we need to remove t2
        # The total contribution is t1 + t2 + others = (1+alpha)*t1 + others
        # We can scale column c1 by (1+alpha) and zero out c2
        # But scaling column c1 by (1+alpha) changes its contribution to (1+alpha)*t1
        # which is exactly t1 + t2. 
        
        U_arr[:, c1] *= (1 + alpha)
        V_arr[:, c1] *= (1 + alpha)  # wait, this would make it (1+alpha)^2
        # No! Scaling U, V, W all by (1+alpha) scales the tensor by (1+alpha)^3
        # We need to scale the product by (1+alpha)
        # So we can scale U by (1+alpha) and leave V, W unchanged
        # Then tensor becomes (1+alpha)*t1 = t1 + t2. 
        
        U_arr[:, c1] *= (1 + alpha)
        U_arr[:, c2] = 0
        V_arr[:, c2] = 0
        W_arr[:, c2] = 0
        return True
    
    # Try merging columns greedily
    # This is O(r^2) pairs, each check is O(nmp)
    # r=120, so 120^2/2 = 7200 pairs, each O(120) = 864000 ops total
    # That's fast enough
    
    merged = True
    while merged and (time.time() - start_time) < DEADLINE:
        merged = False
        r = U_arr.shape[1]
        for c1 in range(r):
            for c2 in range(c1 + 1, r):
                if (time.time() - start_time) > DEADLINE:
                    break
                # Check if columns c1 and c2 can be merged
                # First check if they have the same support pattern
                t1 = np.zeros((n, m, p))
                t2 = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            t1[i, j, k] = U_arr[i*m+j, c1] * V_arr[j*p+k, c1] * W_arr[k*n+i, c1]
                            t2[i, j, k] = U_arr[i*m+j, c2] * V_arr[j*p+k, c2] * W_arr[k*n+i, c2]
                
                # Check if t2 = alpha * t1 for some alpha
                nonzero1 = np.abs(t1) > 1e-12
                nonzero2 = np.abs(t2) > 1e-12
                
                if np.any(nonzero2) and not np.any(nonzero1):
                    continue
                if not np.any(nonzero2):
                    # t2 is zero, remove c2
                    U_arr[:, c2] = 0
                    V_arr[:, c2] = 0
                    W_arr[:, c2] = 0
                    merged = True
                    break
                
                if np.any(nonzero1) and not np.any(nonzero2):
                    continue
                    
                # Both have support, check if proportional
                overlap = nonzero1 & nonzero2
                if not np.any(overlap):
                    continue
                    
                alpha_candidates = []
                for idx in np.where(overlap):
                    i, j, k = idx
                    alpha_candidates.append(t2[i,j,k] / t1[i,j,k])
                
                alpha = alpha_candidates[0]
                is_proportional = True
                for ac in alpha_candidates[1:]:
                    if abs(ac - alpha) > 1e-9:
                        is_proportional = False
                        break
                
                if not is_proportional:
                    continue
                
                # Also need to check: where t1 is nonzero but t2 is zero, and vice versa
                # If t2 = alpha * t1 everywhere, then where t1=0, t2 must also be 0
                # and where t2=0, t1 must also be 0 (if alpha != 0)
                
                if abs(alpha) > 1e-12:
                    # Check: where t1 is zero, t2 should be zero
                    if np.any(nonzero2 & ~nonzero1):
                        continue
                    # Where t2 is zero, t1 should be zero  
                    if np.any(nonzero1 & ~nonzero2):
                        continue
                
                # Merge: scale c1 by (1+alpha), zero out c2
                U_arr[:, c1] *= (1 + alpha)
                U_arr[:, c2] = 0
                V_arr[:, c2] = 0
                W_arr[:, c2] = 0
                merged = True
                break
            if merged:
                break
    
    # Remove zero columns
    nonzero_cols = [c for c in range(U_arr.shape[1]) 
                    if np.any(np.abs(U_arr[:, c]) > 1e-12) or 
                       np.any(np.abs(V_arr[:, c]) > 1e-12) or 
                       np.any(np.abs(W_arr[:, c]) > 1e-12)]
    
    U_final = U_arr[:, nonzero_cols]
    V_final = V_arr[:, nonzero_cols]
    W_final = W_arr[:, nonzero_cols]
    
    final_rank = len(nonzero_cols)
    
    # Verify
    valid, err = _verify(U_final.tolist(), V_final.tolist(), W_final.tolist(), n, m, p)
    
    if valid and final_rank < best_rank:
        best_rank = final_rank
        best_result = (U_final.tolist(), V_final.tolist(), W_final.tolist())
    elif not valid:
        # Fall back to schoolbook
        best_result = _schoolbook(n, m, p)
        best_rank = n * m * p
    
    # Check deadline
    if (time.time() - start_time) > DEADLINE:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # Try additional optimization: look for more complex merges
    # Try to find 3-column merges where the sum of 3 rank-1 tensors equals another
    
    # For now, return the best we found
    U, V, W = best_result
    
    # Convert to list of lists for the interface
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
