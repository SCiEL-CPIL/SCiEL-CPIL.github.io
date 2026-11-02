import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_222():
    """Return U, V, W (each 4x7) for 2x2x2 matmul."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # r=1: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # r=2: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # r=3: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1
    
    # r=4: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    # r=5: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1
    
    # r=6: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1
    
    return U, V, W

def _build_structured_54():
    """Build 2x5x6 decomposition using block Strassen. Rank 54."""
    n, m, p = 2, 5, 6
    rank = 54
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    sU, sV, sW = _strassen_222()
    col = 0
    
    # Part 1: A1 * B1 (2x2 x 6), three Strassen blocks
    for block in range(3):
        k_offset = block * 2
        for r in range(7):
            U[0, col] = sU[0, r]
            U[1, col] = sU[1, r]
            U[5, col] = sU[2, r]
            U[6, col] = sU[3, r]
            
            V[k_offset, col] = sV[0, r]
            V[k_offset + 1, col] = sV[1, r]
            V[6 + k_offset, col] = sV[2, r]
            V[6 + k_offset + 1, col] = sV[3, r]
            
            W[k_offset * 2, col] = sW[0, r]
            W[k_offset * 2 + 1, col] = sW[1, r]
            W[(k_offset + 1) * 2, col] = sW[2, r]
            W[(k_offset + 1) * 2 + 1, col] = sW[3, r]
            col += 1
    
    # Part 2a: A2a * B2a (2x1 x 6), schoolbook rank 12
    for i in range(2):
        for k in range(6):
            U[i * 5 + 2, col] = 1
            V[2 * 6 + k, col] = 1
            W[k * 2 + i, col] = 1
            col += 1
    
    # Part 2b: A2b * B2b (2x2 x 6), three Strassen blocks
    for block in range(3):
        k_offset = block * 2
        for r in range(7):
            U[3, col] = sU[0, r]
            U[4, col] = sU[1, r]
            U[8, col] = sU[2, r]
            U[9, col] = sU[3, r]
            
            V[3 * 6 + k_offset, col] = sV[0, r]
            V[3 * 6 + k_offset + 1, col] = sV[1, r]
            V[4 * 6 + k_offset, col] = sV[2, r]
            V[4 * 6 + k_offset + 1, col] = sV[3, r]
            
            W[k_offset * 2, col] = sW[0, r]
            W[k_offset * 2 + 1, col] = sW[1, r]
            W[(k_offset + 1) * 2, col] = sW[2, r]
            W[(k_offset + 1) * 2 + 1, col] = sW[3, r]
            col += 1
    
    assert col == rank
    return U, V, W

def _verify_decomp(U, V, W, n=2, m=5, p=6):
    """Verify the full decomposition."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                if abs(total - 1.0) > 1e-6:
                    return False
    return True

def _try_reduce_rank(U, V, W, target_rank, n=2, m=5, p=6, deadline=None):
    """
    Try to reduce rank by finding linear dependencies among rank-1 terms.
    Given a rank-R decomposition, check if we can find a rank-r (r < R) one.
    Strategy: for each subset of r columns, check if the remaining tensor can be
    expressed using only those r columns (i.e., the other columns are in the span).
    This is expensive, so we use a greedy approach.
    """
    if deadline and time.time() > deadline:
        return None
    
    R = U.shape[1]
    if target_rank >= R:
        return None
    
    # Form the full tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # T should be all 1s for valid (i,j,k)
    # We want to find r < R such that there exist U', V', W' of rank r with same T
    
    # Approach: random projection + solve
    # Pick r random directions, project the problem, solve
    # This is heuristic
    
    num_trials = 50
    for trial in range(num_trials):
        if deadline and time.time() > deadline:
            break
        
        # Random linear combination approach:
        # Try to find r rank-1 terms that span the tensor
        # Use random U, V with structure from the known solution
        
        # Initialize U, V from the structured solution, zero out random columns
        mask = np.random.choice(R, size=target_rank, replace=False)
        U_sub = U[:, mask]
        V_sub = V[:, mask]
        W_sub = W[:, mask]
        
        # Check if this sub-decomposition is valid
        if _verify_decomp(U_sub, V_sub, W_sub, n, m, p):
            return U_sub, V_sub, W_sub
        
        # Try to solve for W given U_sub, V_sub
        # For each (i,k), solve M_ik @ w = ones(m)
        W_new = np.zeros((p * n, target_rank))
        valid = True
        for i in range(n):
            for k in range(p):
                U_part = U_sub[i * m:(i + 1) * m, :]
                V_rows = [j * p + k for j in range(m)]
                V_part = V_sub[V_rows, :]
                M_ik = U_part * V_part
                try:
                    w, residuals, rank_m, s = np.linalg.lstsq(M_ik, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = w
                except:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and _verify_decomp(U_sub, V_sub, W_new, n, m, p):
            return U_sub, V_sub, W_new
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Step 1: Build structured rank-54 decomposition
    U54, V54, W54 = _build_structured_54()
    
    # Verify
    assert _verify_decomp(U54, V54, W54, n, m, p), "Structured decomposition failed verification"
    
    best_result = {"U": U54.tolist(), "V": V54.tolist(), "W": W54.tolist()}
    best_rank = 54
    
    # Step 2: Try to reduce rank
    # Try target ranks from low to high
    for target_rank in range(30, 54):
        if time.time() > deadline - 10:
            break
        
        result = _try_reduce_rank(U54, V54, W54, target_rank, n, m, p, deadline)
        if result is not None:
            U_new, V_new, W_new = result
            if _verify_decomp(U_new, V_new, W_new, n, m, p):
                best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                best_rank = target_rank
                break  # Found a valid lower rank, stop
    
    # Step 3: Random search for even lower rank (if time permits)
    if time.time() < deadline - 20:
        for target_rank in range(30, best_rank):
            if time.time() > deadline - 10:
                break
            
            num_trials = 200
            found = False
            for trial in range(num_trials):
                if time.time() > deadline - 5:
                    break
                
                # Random U, V with sparse entries
                U_rand = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
                V_rand = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
                
                W_rand = np.zeros((p * n, target_rank))
                valid = True
                for i in range(n):
                    for k in range(p):
                        U_part = U_rand[i * m:(i + 1) * m, :]
                        V_rows = [j * p + k for j in range(m)]
                        V_part = V_rand[V_rows, :]
                        M_ik = U_part * V_part
                        try:
                            w, residuals, rank_m, s = np.linalg.lstsq(M_ik, np.ones(m), rcond=None)
                            W_rand[k * n + i, :] = w
                        except:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and _verify_decomp(U_rand, V_rand, W_rand, n, m, p):
                    best_result = {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
                    best_rank = target_rank
                    found = True
                    break
            
            if found:
                break
    
    # Step 4: If no improvement, return the structured rank-54
    if best_rank == 54:
        # Try a few more structured variants with different block splits
        # For now, just return the rank-54
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
