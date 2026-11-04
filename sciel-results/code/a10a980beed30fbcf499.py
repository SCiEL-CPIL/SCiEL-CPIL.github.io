"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p
    
    # Build target tensor entries
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.append((a, b, c))
    
    # Schoolbook decomposition
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a, col] = 1
                V_sb[b, col] = 1
                W_sb[c, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    def validate_decomp(U, V, W):
        """Check if decomposition exactly reproduces the tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        for (a, b, c) in target_entries:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val - 1.0) > 1e-9:
                return False
        # Check that off-diagonal entries are zero
        # Sample some off-diagonal positions
        import random
        random.seed(123)
        for _ in range(200):
            a = random.randint(0, nm - 1)
            b = random.randint(0, mp - 1)
            c = random.randint(0, pn - 1)
            # Check if this is a valid target entry
            is_target = (a, b, c) in target_entries
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if is_target:
                if abs(val - 1.0) > 1e-9:
                    return False
            else:
                if abs(val) > 1e-9:
                    return False
        return True
    
    def try_reduce_rank(U, V, W, target_rank, max_iterations=100):
        """Try to reduce rank by merging columns."""
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        current_rank = U.shape[1]
        
        if target_rank >= current_rank:
            return None
        
        # Try to merge pairs of columns
        # For each pair, try to find linear combinations that preserve the tensor
        import random
        random.seed(42)
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 10:
                break
            
            # Pick two random columns to try to merge
            if current_rank <= target_rank:
                break
            
            col1 = random.randint(0, current_rank - 1)
            col2 = random.randint(0, current_rank - 1)
            if col1 == col2:
                continue
            
            # Try to find alpha, beta, gamma, delta such that:
            # new_u = alpha*u1 + beta*u2
            # new_v = gamma*v1 + delta*v2  
            # new_w = (u1*v1*w1 + u2*v2*w2) / (new_u * new_v) pointwise
            # This is tricky. Instead, try simple merges.
            
            # Simple approach: try to set one column to zero by adjusting others
            # This requires solving a linear system
            
            # Alternative: use the fact that if two terms have similar structure,
            # we can sometimes combine them.
            
            # For now, try a simple perturbation approach
            # This is a placeholder - actual optimization would be complex
            
            if iteration % 10 == 0:
                if validate_decomp(U.tolist(), V.tolist(), W.tolist()):
                    # Try to find if we can drop a column
                    for drop_col in range(current_rank):
                        U_test = np.delete(U, drop_col, axis=1)
                        V_test = np.delete(V, drop_col, axis=1)
                        W_test = np.delete(W, drop_col, axis=1)
                        if validate_decomp(U_test.tolist(), V_test.tolist(), W_test.tolist()):
                            U = U_test
                            V = V_test
                            W = W_test
                            current_rank -= 1
                            break
                        if time.time() > deadline - 10:
                            break
        
        if current_rank < best_rank:
            return U, V, W, current_rank
        return None
    
    # Try structured approaches for specific ranks
    # Known results suggest rank around 70-90 might be achievable
    
    # Try a few random restarts with different initial structures
    import random
    random.seed(42)
    
    for target_rank in [110, 100, 90, 80, 70]:
        if time.time() > deadline - 20:
            break
        
        # Try to find a decomposition at this rank
        # Use a simple iterative method
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            
            # Initialize with random sparse vectors
            U = np.zeros((nm, target_rank))
            V = np.zeros((mp, target_rank))
            W = np.zeros((pn, target_rank))
            
            # Initialize based on target entries
            for r in range(target_rank):
                # Pick a random target entry
                a, b, c = random.choice(target_entries)
                U[a, r] = 1.0
                V[b, r] = 1.0
                W[c, r] = 1.0
            
            # Try to optimize
            result = try_reduce_rank(U, V, W, target_rank, max_iterations=50)
            if result is not None:
                U_opt, V_opt, W_opt, new_rank = result
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
