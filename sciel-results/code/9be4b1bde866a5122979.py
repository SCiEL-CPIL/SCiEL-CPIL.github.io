import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Fallback: Schoolbook
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try to find a better decomposition using block structure
    # Split p=6 into three blocks of size 2
    # Each block is a 2x3x2 multiplication
    # If we find a rank r decomposition for 2x3x2, total rank is 3r
    
    # Search for 2x3x2 decomposition with rank 11
    decomp_232 = _search_232_decomposition(11, deadline)
    if decomp_232 is not None:
        # Construct full 2x3x6 decomposition
        result = _expand_decomposition(decomp_232, n, m, p, block_size=2)
        if result is not None:
            best_result = result
            best_rank = 3 * 11
    
    # If 11 failed, try rank 12 (which is schoolbook for 232, so no gain, but safe)
    # Or try other ranks
    
    return best_result

def _schoolbook_decomposition(n, m, p):
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

def _search_232_decomposition(target_rank, deadline):
    """
    Search for a rank target_rank decomposition of 2x3x2 matrix multiplication.
    Returns U, V, W matrices or None.
    """
    n_sub, m_sub, p_sub = 2, 3, 2
    r = target_rank
    
    # Number of attempts
    max_attempts = 2000
    attempts = 0
    
    while attempts < max_attempts:
        if time.time() > deadline:
            break
        
        # Generate random U, V with small integers
        U = np.random.randint(-2, 3, size=(n_sub * m_sub, r))
        V = np.random.randint(-2, 3, size=(m_sub * p_sub, r))
        
        # Solve for W
        W = np.zeros((p_sub * n_sub, r))
        valid = True
        
        for i in range(n_sub):
            for k in range(p_sub):
                # Matrix M for this i,k pair
                # M[j, c] = U[i*m_sub + j, c] * V[j*p_sub + k, c]
                M = np.zeros((m_sub, r))
                for j in range(m_sub):
                    for c in range(r):
                        M[j, c] = U[i * m_sub + j, c] * V[j * p_sub + k, c]
                
                # Solve M * W_vec = 1
                # M is m_sub x r (3 x 11)
                # We need a solution W_vec of size r
                # Since r > m_sub, there are infinite solutions.
                # We want W_vec to be half-integers.
                # We can use least squares or find a particular solution.
                # Let's try to find a solution with small norm.
                try:
                    # Use pseudo-inverse to get a particular solution
                    W_particular = np.linalg.pinv(M) @ np.ones(m_sub)
                    
                    # Check if W_particular is close to half-integers
                    # Or just check if it is rational with small denominator?
                    # For now, let's just check if it's integer or half-integer
                    # This is a heuristic.
                    
                    # A better approach:
                    # Fix some columns of W to be 0 or 1 to make the system determined?
                    # Or just check if the entries of W_particular are close to half-integers.
                    
                    # If we find a valid W, we need to ensure it's half-integer.
                    # We can round to nearest half-integer and verify.
                    W_rounded = np.round(W_particular * 2) / 2
                    
                    # Verify if M @ W_rounded == 1
                    if np.allclose(M @ W_rounded, np.ones(m_sub)):
                        W[k * n_sub + i, :] = W_rounded
                    else:
                        valid = False
                        break
                except:
                    valid = False
                    break
            
            if not valid:
                break
        
        if valid:
            # Final verification
            if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n_sub, m_sub, p_sub):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        attempts += 1
    
    return None

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W decompose the n x m x p multiplication tensor.
    """
    rank = len(U[0])
    # Convert to numpy for speed
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over c
                val = 0
                for c in range(rank):
                    val += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                if val != 1:
                    return False
    return True

def _expand_decomposition(decomp_232, n, m, p, block_size):
    """
    Expand a 2x3x2 decomposition to 2x3x6 by repeating it.
    """
    U_232 = decomp_232["U"]
    V_232 = decomp_232["V"]
    W_232 = decomp_232["W"]
    
    r_232 = len(U_232[0])
    r_total = r_232 * (p // block_size)
    
    U_full = [[0] * r_total for _ in range(n * m)]
    V_full = [[0] * r_total for _ in range(m * p)]
    W_full = [[0] * r_total for _ in range(p * n)]
    
    for block in range(p // block_size):
        offset = block * r_232
        
        # Copy U
        for i in range(n * m):
            for c in range(r_232):
                U_full[i][offset + c] = U_232[i][c]
        
        # Copy V (shift columns)
        for j in range(m):
            for k in range(block_size):
                for c in range(r_232):
                    V_full[j * p + block * block_size + k][offset + c] = V_232[j * block_size + k][c]
        
        # Copy W (shift rows)
        for i in range(n):
            for k in range(block_size):
                for c in range(r_232):
                    W_full[(block * block_size + k) * n + i][offset + c] = W_232[k * n + i][c]
    
    return {"U": U_full, "V": V_full, "W": W_full}

# EVOLVE-BLOCK-END
