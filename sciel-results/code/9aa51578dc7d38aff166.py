"""Bootstrap SEARCH for rank-minimal exact tensor decomposition of 2x5 * 5x6."""
import time
import numpy as np
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _verify_rank(U, V, W, n=2, m=5, p=6):
    """Check if given factor matrices produce correct matmul tensor."""
    rank = len(U[0])
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # T[i*m+j, j*p+k, k*n+i] should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal: T[a, b, c] = 0 unless indices align
    # a = i*m+j, b = j'*p+k, c = k'*n+i'
    # Non-zero only if j==j', k==k', i==i'
    for a in range(n*m):
        for b in range(m*p):
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            if j_a != j_b:
                continue
            for c in range(p*n):
                k_c = c // n
                i_c = c % n
                if k_b != k_c or i_a != i_c:
                    val = np.sum(U_np[a] * V_np[b] * W_np[c])
                    if abs(val) > 1e-9:
                        return False
    return True

def _schoolbook_rank():
    """Return the standard schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _compute_tensor(U, V, W, n=2, m=5, p=6):
    """Compute the full tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]"""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = len(U[0])
    
    # Build tensor efficiently using einsum or broadcasting
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is a 3-way outer product summed over r
    
    # For small dimensions, we can compute directly
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U_np[:,r], V_np[:,r], W_np[:,r])
    return T

def _get_constraint_indices(n=2, m=5, p=6):
    """Get indices for the 60 'one' constraints and count off-diagonal constraints"""
    ones = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ones.append((i*m+j, j*p+k, k*n+i))
    return ones

def construct():
    start_time = time.time()
    deadline = start_time + 128  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    best = _schoolbook_rank()
    best_rank = n * m * p  # 60
    
    # Verify schoolbook works
    if not _verify_rank(best["U"], best["V"], best["W"]):
        return best
    
    random.seed(42)
    
    # Strategy: Start with schoolbook and try to merge columns
    # Each column r corresponds to a rank-1 term
    # We can try to eliminate a column by adjusting others
    
    # Key insight: For 2x5*5x6, the tensor has structure
    # T[i,j,k] = 1 if (i,j,k) forms a valid matmul triple
    # We want to find minimal rank decomposition
    
    # Approach 1: Try to find a lower rank by random search with better initialization
    # Approach 2: Try structured patterns
    
    # Let's use a hybrid approach:
    # 1. Try random search with various value sets
    # 2. Try to merge columns from schoolbook
    
    def _check_valid(U, V, W, rank, tol=1e-8):
        """Fast validity check"""
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check the 60 ones
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U_np[u_idx] * V_np[v_idx] * W_np[w_idx])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check off-diagonals (sample for speed, full check at end)
        # For now, just check ones; full verification later
        return True
    
    def _full_verify(U, V, W):
        """Full verification"""
        return _verify_rank(U, V, W)
    
    # Try to find better decompositions
    # Value sets to try
    value_sets = [
        [0, 0.5, -0.5, 1, -1],
        [0, 0.5, -0.5, 1, -1, 1.5, -1.5],
        [0, 1, -1],
        [0, 0.5, -0.5, 1, -1, 2, -2, 0.25, -0.25],
    ]
    
    # Try ranks from 59 down to 30
    # For each rank, try multiple random candidates
    for target_rank in range(59, 29, -1):
        if time.time() > deadline:
            break
        
        # Number of attempts per rank (adjust based on remaining time)
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
        
        # More attempts for higher ranks (easier to find)
        attempts = min(200, max(20, int(remaining_time * 5)))
        
        found = False
        for vs_idx, values in enumerate(value_sets):
            if time.time() > deadline:
                break
            
            for attempt in range(attempts // len(value_sets)):
                if time.time() > deadline:
                    break
                
                # Generate random candidate
                U = np.array([[random.choice(values) for _ in range(target_rank)] for _ in range(n*m)], dtype=float)
                V = np.array([[random.choice(values) for _ in range(target_rank)] for _ in range(m*p)], dtype=float)
                W = np.array([[random.choice(values) for _ in range(target_rank)] for _ in range(p*n)], dtype=float)
                
                # Quick check
                if _check_valid(U, V, W, target_rank):
                    # Full verification
                    if _full_verify(U.tolist(), V.tolist(), W.tolist()):
                        best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = target_rank
                        found = True
                        break
            
            if found:
                break
    
    # Also try structured approach: start from schoolbook and try to merge
    if time.time() < deadline:
        # Try to find a rank-59 decomposition by merging two schoolbook terms
        # This is a local search approach
        for merge_attempt in range(50):
            if time.time() > deadline:
                break
            
            # Start with schoolbook
            U = np.zeros((n*m, 60))
            V = np.zeros((m*p, 60))
            W = np.zeros((p*n, 60))
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
            
            # Try to eliminate one column by adjusting others
            # Pick a random column to eliminate
            target_col = random.randint(0, 59)
            
            # Get the (i,j,k) triple for this column
            # col = i*m*p + j*p + k? No, col = i*(m*p) + j*p + k
            i_target = target_col // (m*p)
            rem = target_col % (m*p)
            j_target = rem // p
            k_target = rem % p
            
            # Try to express this term as a combination of others
            # This is complex; skip for now
            
            # Instead, try a different approach: random perturbation
            # Take schoolbook and randomly modify some entries
            U_mod = U.copy()
            V_mod = V.copy()
            W_mod = W.copy()
            
            # Zero out the target column
            U_mod[:, target_col] = 0
            V_mod[:, target_col] = 0
            W_mod[:, target_col] = 0
            
            # Try to recover by adjusting other columns
            # Add random perturbations to other columns
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                U_test = U_mod.copy()
                V_test = V_mod.copy()
                W_test = W_mod.copy()
                
                # Randomly adjust some entries
                for _ in range(10):
                    mat = random.choice([U_test, V_test, W_test])
                    row = random.randint(0, len(mat)-1)
                    col_idx = random.randint(0, 58)  # Not target_col
                    if col_idx != target_col:
                        mat[row, col_idx] += random.choice([-1, -0.5, 0.5, 1])
                
                # Check if still valid
                if _check_valid(U_test, V_test, W_test, 59):
                    if _full_verify(U_test.tolist(), V_test.tolist(), W_test.tolist()):
                        # Found a rank-59 decomposition
                        # Trim to 59 columns
                        keep_cols = [c for c in range(60) if c != target_col]
                        best = {
                            "U": U_test[:, keep_cols].tolist(),
                            "V": V_test[:, keep_cols].tolist(),
                            "W": W_test[:, keep_cols].tolist()
                        }
                        best_rank = 59
                        break
                if best_rank == 59:
                    break
            if best_rank == 59:
                break
    
    return best

# EVOLVE-BLOCK-END
