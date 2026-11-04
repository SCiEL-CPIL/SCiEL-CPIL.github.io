import numpy as np
import time

n, m, p = 2, 5, 6
# U: (n*m, rank) = (10, rank)
# V: (m*p, rank) = (30, rank)
# W: (p*n, rank) = (12, rank)
# Tensor T[i, j, k] = 1 corresponds to C_{i,k} += A_{i,j} * B_{j,k}
# U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    def try_rank(R):
        # Random search with random restarts for a fixed rank R
        # We use integer/half-integer values, so we scale by 2 to use integers
        # U, V, W have values in {0, 0.5, 1, 1.5, ...}
        # Actually, let's try a specific known pattern or random search.
        # For 2x5x6, the schoolbook rank is 60.
        # Is it possible to do better?
        # Lower bound for 2x5x6:
        # rank >= ceil(np.log2(2) * np.log2(5) * np.log2(6)) / (np.log2(2)*np.log2(5)*np.log2(6)) * 2*5*6? No, Strassen bound.
        # Simple lower bound: rank >= max(n*m, m*p, p*n) = max(10, 30, 12) = 30.
        # Also rank >= n*p = 12.
        # Known best for 2x5x6?
        # Let's try random search for rank < 60.
        
        best_result = None
        attempts = 0
        while time.time() < deadline:
            attempts += 1
            # Random matrix generation with small integers or half-integers
            # Let's try {0, 1} first, then {0, 1, -1}, then wider range.
            # For simplicity, let's use {0, 1} and {0, 1, -1}.
            # If R is close to 60, {0, 1} might work.
            # Let's use random entries from a small set.
            
            # Scale entries by 2 to handle half-integers.
            # Let's stick to integers for now to avoid precision issues, 
            # then divide by 2 at the end if we found a solution with even sums?
            # No, the problem allows half-integers.
            
            # Let's use integer entries in U, V, W.
            # The tensor equation is sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
            # This is a system of 2*5*6 = 60 equations.
            # Variables: R * (10 + 30 + 12) = 52R variables.
            # If R=30, 52*30 = 1560 variables.
            # We can use random initialization and then refine?
            # Or just random search might be too hard.
            
            # Let's try a structured approach.
            # For R=60, schoolbook works.
            # Let's check if R=59 works?
            # The task is to find a rank-minimal decomposition.
            # Since I don't have a search algorithm that can find R<60 quickly, 
            # and the previous program returned 60, I should return 60 unless I can prove/find better.
            # However, the prompt says "investigate improvements within its family".
            # The "family" is the schoolbook decomposition.
            # Can we merge terms?
            # Merging terms requires U, V, W to be non-zero in multiple places.
            # If we merge two terms, we save 1 rank.
            # But we need to satisfy all 60 equations.
            
            # Let's just return the schoolbook decomposition with rank 60, 
            # as it's guaranteed correct.
            # If I have time, I can try to search for R=59.
            
            if R < 60:
                # Try random search for R < 60
                # This is likely to fail quickly, but let's try a few random attempts.
                for _ in range(10):
                    U = np.random.randint(-2, 3, size=(10, R))
                    V = np.random.randint(-2, 3, size=(30, R))
                    W = np.random.randint(-2, 3, size=(12, R))
                    
                    # Check tensor equation
                    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # We need T[i, j, k] = 1 for all i, j, k.
                    
                    valid = True
                    for i in range(2):
                        for j in range(5):
                            for k in range(6):
                                val = 0
                                for r in range(R):
                                    val += U[i*5+j, r] * V[j*6+k, r] * W[k*2+i, r]
                                if val != 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        return U, V, W
            else:
                # R=60, return schoolbook
                U = np.zeros((10, 60))
                V = np.zeros((30, 60))
                W = np.zeros((12, 60))
                col = 0
                for i in range(2):
                    for j in range(5):
                        for k in range(6):
                            U[i*5+j, col] = 1
                            V[j*6+k, col] = 1
                            W[k*2+i, col] = 1
                            col += 1
                return U, V, W
        return None

    # Try ranks from 30 to 60
    for R in range(30, 61):
        result = try_rank(R)
        if result is not None:
            U, V, W = result
            # Convert to lists of lists
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U = np.zeros((10, 60))
    V = np.zeros((30, 60))
    W = np.zeros((12, 60))
    col = 0
    for i in range(2):
        for j in range(5):
            for k in range(6):
                U[i*5+j, col] = 1
                V[j*6+k, col] = 1
                W[k*2+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
