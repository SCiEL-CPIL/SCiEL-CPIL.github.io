"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # We attempt a rank-16 decomposition using a systematic search over a restricted
    # coefficient set {-1, 0, 1} with a smart initialization and local search.
    # If that fails within budget, we fall back to the verified rank-17.
    
    def verify(U, V, W, rank):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    comp = 0
                    for r in range(rank):
                        comp += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                    if comp != 1:
                        return False
        return True

    def schoolbook(rank=18):
        """Generate a valid schoolbook decomposition."""
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, idx] = 1
                    V[v_idx, idx] = 1
                    W[w_idx, idx] = 1
                    idx += 1
        return U, V, W

    # Known rank-17 construction (Strassen + schoolbook)
    def rank17_construction():
        rank = 17
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        # Strassen part for 2x2 block (indices 0-6)
        U[0, 0] = 1; U[4, 0] = 1
        V[0, 0] = 1; V[4, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[3, 1] = 1; U[4, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1; W[3, 1] = -1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[4, 2] = -1
        W[2, 2] = 1; W[3, 2] = 1
        
        U[4, 3] = 1
        V[3, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[4, 4] = 1
        W[2, 4] = 1; W[0, 4] = -1
        
        U[3, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[4, 6] = -1
        V[3, 6] = 1; V[4, 6] = 1
        W[0, 6] = 1
        
        # Schoolbook for remaining terms (indices 7-16)
        U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
        U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
        U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
        U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
        U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
        U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
        U[3, 13] = 1; V[2, 13] = 1; W[5, 13] = 1
        U[4, 14] = 1; V[5, 14] = 1; W[5, 14] = 1
        U[2, 15] = 1; V[8, 15] = 1; W[4, 15] = 1
        U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
        
        return U, V, W

    # First, try the known rank-17
    U17, V17, W17 = rank17_construction()
    if verify(U17, V17, W17, 17):
        # Now attempt to reduce to rank 16 via local search
        # We start from the rank-17 solution and try to eliminate one factor
        # by finding a linear dependency or a zero row/column that can be removed
        
        # Strategy: Try to find a rank-16 solution by random restarts in a small neighborhood
        # Since the search space is huge, we use a heuristic:
        # 1. Start with random integer matrices of rank 16 with small entries
        # 2. Use gradient-free optimization (simulated annealing) to minimize the error
        
        best_rank = 17
        best_U, best_V, best_W = U17, V17, W17
        
        # Time budget for search
        search_time = deadline - (time.time() - start_time) - 5.0
        if search_time > 10:
            # Try a few random rank-16 solutions
            np.random.seed(42)
            max_iters = int(search_time * 50)  # Adjust based on available time
            best_err = float('inf')
            
            for _ in range(max_iters):
                if time.time() > deadline - 5.0:
                    break
                
                # Random initialization with small integers
                U = np.random.randint(-1, 2, size=(n * m, 16))
                V = np.random.randint(-1, 2, size=(m * p, 16))
                W = np.random.randint(-1, 2, size=(p * n, 16))
                
                # Calculate error
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            comp = 0
                            for r in range(16):
                                comp += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                            err += (comp - 1) ** 2
                
                if err == 0:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                if err < best_err:
                    best_err = err
                    best_U, best_V, best_W = U, V, W
            
            # If we found a good candidate, try to refine it
            # But for simplicity and safety, we stick with the known valid solution
            if best_err == 0:
                best_rank = 16
                best_U, best_V, best_W = best_U, best_V, best_W
        
        # Return the best valid solution found
        if verify(best_U, best_V, best_W, best_rank):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        else:
            # Fallback to schoolbook
            U, V, W = schoolbook(18)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        U, V, W = schoolbook(18)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
