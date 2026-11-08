"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=int)
        V = np.zeros((mp, rank), dtype=int)
        W = np.zeros((pn, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Precompute indices for faster checking
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W form a valid decomposition."""
        for i, j, k in indices:
            val = 0
            for c in range(rank):
                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            if val != 1:
                return False
        return True
    
    # Greedy construction: build columns one at a time
    # Each column contributes to a set of (i,j,k) triples
    # We want to cover all (i,j,k) with minimal columns
    
    def greedy_construct(max_rank=95):
        """Try to construct a decomposition greedily."""
        # Track current sum for each (i,j,k)
        current_sum = np.zeros((n, m, p), dtype=int)
        U_cols = []
        V_cols = []
        W_cols = []
        
        # Uncovered entries
        uncovered = set(indices)
        
        for c in range(max_rank):
            if time.time() > deadline:
                break
            if not uncovered:
                break
            
            # Try to find a good column that covers many uncovered entries
            best_col = None
            best_count = 0
            
            # Try various patterns for this column
            # A column is defined by which (i,j) pairs are active in U,
            # which (j,k) pairs are active in V, and which (k,i) pairs are active in W
            
            # Simple approach: try random sparse patterns
            for _ in range(50):
                if time.time() > deadline:
                    break
                
                # Randomly select some (i,j) for U
                u_active = set()
                for i in range(n):
                    for j in range(m):
                        if random.random() < 0.3:
                            u_active.add((i, j))
                
                # Randomly select some (j,k) for V
                v_active = set()
                for j in range(m):
                    for k in range(p):
                        if random.random() < 0.3:
                            v_active.add((j, k))
                
                # Randomly select some (k,i) for W
                w_active = set()
                for k in range(p):
                    for i in range(n):
                        if random.random() < 0.3:
                            w_active.add((k, i))
                
                # Count how many uncovered entries this column would cover
                # An entry (i,j,k) is covered if (i,j) in u_active, (j,k) in v_active, (k,i) in w_active
                count = 0
                for (i, j, k) in uncovered:
                    if (i, j) in u_active and (j, k) in v_active and (k, i) in w_active:
                        count += 1
                
                if count > best_count:
                    best_count = count
                    best_col = (u_active, v_active, w_active)
            
            if best_col is None or best_count == 0:
                break
            
            u_active, v_active, w_active = best_col
            
            # Build the column vectors
            u_col = np.zeros(nm, dtype=int)
            v_col = np.zeros(mp, dtype=int)
            w_col = np.zeros(pn, dtype=int)
            
            for (i, j) in u_active:
                u_col[i*m + j] = 1
            for (j, k) in v_active:
                v_col[j*p + k] = 1
            for (k, i) in w_active:
                w_col[k*n + i] = 1
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
            
            # Update current sum and uncovered
            for (i, j, k) in list(uncovered):
                if (i, j) in u_active and (j, k) in v_active and (k, i) in w_active:
                    current_sum[i, j, k] += 1
                    if current_sum[i, j, k] == 1:
                        uncovered.discard((i, j, k))
        
        if not uncovered and len(U_cols) > 0:
            rank = len(U_cols)
            U = np.column_stack(U_cols)
            V = np.column_stack(V_cols)
            W = np.column_stack(W_cols)
            return U, V, W, rank
        
        return None
    
    # Try greedy construction
    result = greedy_construct()
    if result is not None:
        U, V, W, rank = result
        if check_decomposition(U, V, W, rank) and rank < best_rank:
            best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Also try some structured approaches
    # Approach: use the fact that we can decompose into blocks
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
