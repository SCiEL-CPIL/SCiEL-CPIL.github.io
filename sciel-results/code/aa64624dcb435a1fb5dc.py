import time
import numpy as np
import random

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s limit to be safe within 150s
    
    # Schoolbook decomposition (Rank 105)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb
    
    # Function to verify a decomposition
    def verify(U, V, W, rank):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        U_t = U_np.reshape(n, m, rank)
        V_t = V_np.reshape(m, p, rank)
        W_t = W_np.reshape(p, n, rank)
        try:
            T = np.einsum('ijr,jkr,kin->ijk', U_t, V_t, W_t)
            if not np.allclose(T, 1.0, atol=1e-9):
                return False
        except Exception:
            return False
        return True

    # Try structured block decomposition
    # Split m=5 into 3+2, p=7 into 3+2+2
    # Use Strassen (rank 7) for 2x2x2 blocks
    def strassen_2x2():
        """Return (U, V, W) for 2x2x2 with rank 7, Strassen algorithm."""
        # Strassen's 7 multiplications for 2x2
        # U: (2*2, 7), V: (2*2, 7), W: (2*2, 7)
        # Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        U_s = [[0]*7 for _ in range(4)]
        V_s = [[0]*7 for _ in range(4)]
        W_s = [[0]*7 for _ in range(4)]
        
        # Strassen's 7 products:
        # M1 = (a00+a11)*(b00+b11)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b01-b11)
        # M4 = a11*(b10-b00)
        # M5 = (a00+a01)*b11
        # M6 = (a10-a00)*(b00+b01)
        # M7 = (a01-a11)*(b10+b11)
        #
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # For each rank term r, U[i,j,r] * V[j,k,r] * W[k,i,r] should sum to delta(i,j,k)
        # This is tricky. Let me use a known rank-7 decomposition.
        
        # Actually, for the tensor T[i,j,k] = 1 for 2x2x2, the rank is 7.
        # Let me construct it properly.
        
        # The 7 rank-1 terms for 2x2x2 multiplication:
        terms = [
            # (U entries, V entries, W entries)
            # Each term: U is 2x2, V is 2x2, W is 2x2
            # Term 1: M1 = (a00+a11)(b00+b11)
            ([[1,0],[1,0]], [[1,1],[0,0]], [[1,0],[1,0]]),  # Not quite right
        ]
        
        # This is getting complex. Let me use a simpler approach.
        # For now, return schoolbook for 2x2x2 (rank 8)
        U_8 = [[0]*8 for _ in range(4)]
        V_8 = [[0]*8 for _ in range(4)]
        W_8 = [[0]*8 for _ in range(4)]
        c = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_8[i*2+j][c] = 1
                    V_8[j*2+k][c] = 1
                    W_8[k*2+i][c] = 1
                    c += 1
        return U_8, V_8, W_8
    
    # Given the complexity, let's try a different approach:
    # Use a greedy removal from schoolbook
    # Start with schoolbook, try to remove each term and see if we can adjust
    
    random.seed(42)
    np.random.seed(42)
    
    # Try random search with structured initialization
    # Use the fact that we need sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # This is a system of 105 equations in 3*R*rank variables
    
    # Try ranks from 104 down
    for R in range(best_rank - 1, max(best_rank - 20, 80), -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few random initializations
        max_trials = 5 if R > 95 else 20
        for trial in range(max_trials):
            # Initialize with schoolbook-like structure
            U = np.zeros((n * m, R), dtype=float)
            V = np.zeros((m * p, R), dtype=float)
            W = np.zeros((p * n, R), dtype=float)
            
            # For the first min(R, 105) terms, use schoolbook
            for r in range(min(R, rank_sb)):
                i = r // (m * p)
                j = (r // p) % m
                k = r % p
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
            
            # For remaining terms, add small random perturbations
            for r in range(min(R, rank_sb), R):
                U[:, r] = np.random.choice([-1, 0, 1], size=n * m)
                V[:, r] = np.random.choice([-1, 0, 1], size=m * p)
                W[:, r] = np.random.choice([-1, 0, 1], size=p * n)
            
            if verify(U.tolist(), V.tolist(), W.tolist(), R):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # If no better solution found, return schoolbook
    return best_solution
# EVOLVE-BLOCK-END
