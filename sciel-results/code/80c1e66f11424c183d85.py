"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication
    of a 2x3 matrix by a 3x6 matrix.
    
    Uses block structure exploiting 6 = 2*3, with systematic search for lower rank.
    """
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p  # 36
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, column] = 1.0
                V_sb[j * p + k, column] = 1.0
                W_sb[k * n + i, column] = 1.0
                column += 1
    
    best_rank = rank_schoolbook
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Fast verification using vectorized operations
    def verify_full(U, V, W, tol=1e-9):
        """Check all valid and invalid positions."""
        # Valid positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = np.dot(U[i * m + j, :], np.multiply(V[j * p + k, :], W[k * n + i, :]))
                    if abs(actual - 1.0) > tol:
                        return False
        
        # Invalid positions: check all triples where indices don't align
        # a = i1*m + j1, b = j2*p + k1, c = k2*n + i2
        # Invalid if i1 != i2 or j1 != j2 or k1 != k2
        for i1 in range(n):
            for j1 in range(m):
                a = i1 * m + j1
                for j2 in range(m):
                    for k1 in range(p):
                        b = j2 * p + k1
                        for k2 in range(p):
                            for i2 in range(n):
                                c = k2 * n + i2
                                if i1 == i2 and j1 == j2 and k1 == k2:
                                    continue  # valid
                                actual = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
                                if abs(actual) > tol:
                                    return False
        return True
    
    def verify_sample(U, V, W, tol=1e-9, n_samples=500):
        """Fast verification with sampling of invalid positions."""
        # Check all valid positions (only 36)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = np.dot(U[i * m + j, :], np.multiply(V[j * p + k, :], W[k * n + i, :]))
                    if abs(actual - 1.0) > tol:
                        return False
        
        # Sample invalid positions
        for _ in range(n_samples):
            a = random.randint(0, n*m - 1)
            b = random.randint(0, m*p - 1)
            c = random.randint(0, p*n - 1)
            i1, j1 = divmod(a, m)
            j2, k1 = divmod(b, p)
            k2, i2 = divmod(c, n)
            if i1 == i2 and j1 == j2 and k1 == k2:
                continue
            actual = np.dot(U[a, :], np.multiply(V[b, :], W[c, :]))
            if abs(actual) > tol:
                return False
        return True
    
    # Block-structured construction: split p into two halves of size 3
    # Each half is a 2x3x3 matrix multiplication
    # Rank of each half: try to find minimal, default schoolbook = 18
    
    def construct_block_2x3x3(rank, seed=None):
        """
        Construct a rank-r decomposition for 2x3 * 3x3 -> 2x3.
        Returns (U_block, V_block, W_block) with shapes (6, r), (9, r), (6, r).
        U_block[i*3+j, r], V_block[j*3+k, r], W_block[k*2+i, r]
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState(random.randint(0, 2**31 - 1))
        
        U_b = np.zeros((6, rank))
        V_b = np.zeros((9, rank))
        W_b = np.zeros((6, rank))
        
        if rank == 18:
            # Schoolbook
            col = 0
            for i in range(2):
                for j in range(3):
                    for k in range(3):
                        U_b[i*3+j, col] = 1.0
                        V_b[j*3+k, col] = 1.0
                        W_b[k*2+i, col] = 1.0
                        col += 1
            return U_b, V_b, W_b
        
        # For rank < 18, use random initialization and check
        for _ in range(200):
            U_b[:] = rng.randn(6, rank)
            V_b[:] = rng.randn(9, rank)
            W_b[:] = rng.randn(6, rank)
            
            # Check valid
            ok = True
            for i in range(2):
                for j in range(3):
                    for k in range(3):
                        val = np.dot(U_b[i*3+j, :], np.multiply(V_b[j*3+k, :], W_b[k*2+i, :]))
                        if abs(val - 1.0) > 1e-6:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            
            # Sample invalid
            ok = True
            for _ in range(200):
                a = rng.randint(0, 5)
                b = rng.randint(0, 8)
                c = rng.randint(0, 5)
                i1, j1 = divmod(a, 3)
                j2, k1 = divmod(b, 3)
                k2, i2 = divmod(c, 2)
                if i1 == i2 and j1 == j2 and k1 == k2:
                    continue
                val = np.dot(U_b[a, :], np.multiply(V_b[b, :], W_b[c, :]))
                if abs(val) > 1e-6:
                    ok = False
                    break
            if not ok:
                continue
            
            return U_b, V_b, W_b
        
        return None
    
    # Try to find lower-rank decompositions for each 2x3x3 block
    # Then combine
    
    # First, try the simple block concatenation with schoolbook rank for each block
    # This gives rank 36, same as overall schoolbook
    
    # Try to find a rank-r decomposition for the full problem using block structure
    # where each rank element only affects one half
    
    for target_rank in range(18, rank_schoolbook + 1):
        if time.time() > deadline - 15:
            break
        
        max_attempts = max(3, 30 // max(1, target_rank - 17))
        
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 15:
                break
            
            # Split rank between two halves
            if attempt % 3 == 0:
                r1 = target_rank // 2
                r2 = target_rank - r1
            elif attempt % 3 == 1:
                r1 = max(1, target_rank // 3)
                r2 = target_rank - r1
            else:
                r1 = random.randint(1, max(1, target_rank - 1))
                r2 = target_rank - r1
            
            if r1 < 1 or r2 < 1:
                continue
            
            U = np.zeros((n * m, target_rank))
            V = np.zeros((m * p, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # First half: k in 0..2
            for r in range(r1):
                for i in range(n):
                    for j in range(m):
                        U[i*m+j, r] = random.gauss(0, 1)
                for j in range(m):
                    for k in range(3):
                        V[j*p+k, r] = random.gauss(0, 1)
                for k in range(3):
                    for i in range(n):
                        W[k*n+i, r] = random.gauss(0, 1)
            
            # Second half: k in 3..5
            for r in range(r1, target_rank):
                for i in range(n):
                    for j in range(m):
                        U[i*m+j, r] = random.gauss(0, 1)
                for j in range(m):
                    for k in range(3, p):
                        V[j*p+k, r] = random.gauss(0, 1)
                for k in range(3, p):
                    for i in range(n):
                        W[k*n+i, r] = random.gauss(0, 1)
            
            if verify_sample(U, V, W, tol=1e-8, n_samples=1000):
                # Full verification
                if verify_full(U, V, W, tol=1e-8):
                    best_rank = target_rank
                    best_decomp = (U.copy(), V.copy(), W.copy())
                    found = True
                    break
        
        if found:
            break
    
    if best_rank < rank_schoolbook:
        return {"U": best_decomp[0].tolist(), "V": best_decomp[1].tolist(), "W": best_decomp[2].tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
