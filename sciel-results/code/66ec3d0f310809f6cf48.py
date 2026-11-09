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
    
    def tensor_sum(U, V, W, rank):
        """Compute the tensor sum for all (i,j,k)."""
        tensor = np.zeros((n, m, p), dtype=int)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    tensor[i, j, k] = s
        return tensor
    
    def try_local_search(U, V, W, rank, max_iters=5000):
        """Try to reduce rank by removing or merging columns."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        current_rank = rank
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            if current_rank <= best_rank:
                break
            
            # Try removing each column
            for c in range(current_rank):
                if time.time() > deadline:
                    break
                
                # Check if column c can be removed
                # Compute contribution of column c
                # For each (i,j,k), check if removing col c still gives sum=1
                
                removable = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            contrib = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            if contrib != 0:
                                # This column contributes to (i,j,k)
                                # Check if other columns sum to 1 - contrib
                                other_sum = 0
                                for c2 in range(current_rank):
                                    if c2 != c:
                                        other_sum += U[i*m+j, c2] * V[j*p+k, c2] * W[k*n+i, c2]
                                if other_sum != 1 - contrib:
                                    removable = False
                                    break
                        if not removable:
                            break
                    if not removable:
                        break
                
                if removable:
                    # Remove column c
                    U = np.delete(U, c, axis=1)
                    V = np.delete(V, c, axis=1)
                    W = np.delete(W, c, axis=1)
                    current_rank -= 1
                    break
            else:
                # No column could be removed, try perturbation
                if time.time() > deadline:
                    break
                
                # Try modifying a random column
                c = random.randint(0, current_rank - 1)
                # Randomly perturb the column
                for _ in range(3):
                    # Try changing a random entry
                    idx = random.randint(0, U.shape[0] - 1)
                    old_val = U[idx, c]
                    U[idx, c] = random.choice([-1, 0, 1])
                    
                    if check_decomposition(U, V, W, current_rank):
                        break
                    else:
                        U[idx, c] = old_val
                
                # Also try swapping two columns or combining
                if random.random() < 0.3 and current_rank >= 2:
                    c1 = random.randint(0, current_rank - 1)
                    c2 = random.randint(0, current_rank - 1)
                    if c1 != c2:
                        # Try merging: set col2 to be a combination
                        # For now, just try random modifications
                        pass
        
        if current_rank < rank and check_decomposition(U, V, W, current_rank):
            return U, V, W, current_rank
        
        return None
    
    # Try local search from schoolbook
    result = try_local_search(best_U, best_V, best_W, best_rank, max_iters=3000)
    if result is not None:
        U, V, W, rank = result
        if rank < best_rank:
            best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try a structured approach: exploit the fact that m=4
    # We can think of this as multiplying by 4 intermediate values
    # Try to find a decomposition with rank < 96
    
    # Strategy: Use the structure of matrix multiplication
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # We want to express this as sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    # Try a block-based approach
    def try_block_decomposition():
        """Try to find a decomposition using block structure."""
        # For m=4, we can potentially use rank-4 decompositions for sub-problems
        # But this is complex. Let's try a simpler approach.
        
        # Start with schoolbook and try to merge columns that have
        # complementary patterns
        
        U = best_U.copy()
        V = best_V.copy()
        W = best_W.copy()
        rank = best_rank
        
        # Group columns by their (i,j) pattern in U
        # If two columns have the same U pattern but different V/W patterns,
        # they might be mergeable
        
        # For now, just return schoolbook
        return None
    
    result2 = try_block_decomposition()
    if result2 is not None:
        U, V, W, rank = result2
        if check_decomposition(U, V, W, rank) and rank < best_rank:
            best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
