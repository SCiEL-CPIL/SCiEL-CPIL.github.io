import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    r = U.shape[1]
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            s = np.dot(U[a * m + b, :], np.multiply(V[c * p + d, :], W[k * n + i, :]))
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Strategy: 
    # 1. Try to find a rank-11 decomposition for M_{2,3,3}.
    # 2. If found, use it twice to construct M_{2,3,6} with rank 22.
    # 3. If not, try to find a rank-12 decomposition for M_{2,3,3}.
    # 4. If found, use it twice to construct M_{2,3,6} with rank 24.
    # 5. Fallback to rank 34 solution.
    
    def search_m233_rank(target_rank, time_budget):
        """
        Attempts to find a rank 'target_rank' decomposition for 2x3x3 matrix multiplication.
        Returns (U, V, W) if found, else None.
        """
        if time.time() > start_time + time_budget:
            return None
            
        n2, m2, p2 = 2, 3, 3
        U_dim = n2 * m2 # 6
        V_dim = m2 * p2 # 9
        W_dim = p2 * n2 # 6
        
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Initialize with schoolbook (rank 18) and zero out columns?
        # Or random init.
        
        for restart in range(20):
            if time.time() > start_time + time_budget:
                break
                
            # Random init
            U = np.random.randn(U_dim, target_rank) * 0.1
            V = np.random.randn(V_dim, target_rank) * 0.1
            W = np.random.randn(W_dim, target_rank) * 0.1
            
            current_error = float('inf')
            
            # Simulated Annealing
            T = 1.0
            cool = 0.95
            steps = 0
            max_steps = 5000
            
            while steps < max_steps and T > 1e-5:
                if time.time() > start_time + time_budget:
                    return None if best_error > 1e-9 else (best_U, best_V, best_W)
                
                # Pick a random matrix and entry to perturb
                mat_idx = random.randint(0, 2)
                if mat_idx == 0:
                    mat = U
                    dim = U_dim
                elif mat_idx == 1:
                    mat = V
                    dim = V_dim
                else:
                    mat = W
                    dim = W_dim
                    
                row = random.randint(0, dim - 1)
                col = random.randint(0, target_rank - 1)
                
                old_val = mat[row, col]
                new_val = old_val + np.random.randn() * 0.1
                mat[row, col] = new_val
                
                # Calculate error
                err = 0.0
                for a in range(n2):
                    for b in range(m2):
                        for c in range(m2):
                            for d in range(p2):
                                for i in range(n2):
                                    for k in range(p2):
                                        target = 1.0 if (a == i and c == k and b == d) else 0.0
                                        s = np.dot(U[a * m2 + b, :], np.multiply(V[c * p2 + d, :], W[k * n2 + i, :]))
                                        err += (s - target) ** 2
                                        
                if err < current_error:
                    current_error = err
                    if err < 1e-9:
                        if best_error > current_error:
                            best_error = current_error
                            best_U = U.copy()
                            best_V = V.copy()
                            best_W = W.copy()
                            if current_error < 1e-12:
                                return (best_U, best_V, best_W)
                    T *= cool
                else:
                    delta_E = err - current_error
                    if np.random.rand() < np.exp(-delta_E / T):
                        T *= cool
                    else:
                        mat[row, col] = old_val
                        T *= cool
                
                steps += 1
                
            if best_error > 1e-9:
                continue
                
            if verify_decomposition(best_U, best_V, best_W, n2, m2, p2):
                return (best_U, best_V, best_W)
                
        return None

    # Try to find rank 11 for 2x3x3
    # Budget: 60 seconds
    m233_sol = search_m233_rank(11, 60)
    
    if m233_sol is not None:
        U233, V233, W233 = m233_sol
        # Construct M_{2,3,6}
        # U shape: (6, 22)
        # V shape: (18, 22)
        # W shape: (12, 22)
        
        U_comb = np.zeros((6, 22))
        V_comb = np.zeros((18, 22))
        W_comb = np.zeros((12, 22))
        
        # Block 1 (d=0,1,2 -> k=0,1,2)
        U_comb[:, 0:11] = U233
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, 0:11] = V233[j * 3 + k]
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, 0:11] = W233[k * 2 + i]
                
        # Block 2 (d=3,4,5 -> k=3,4,5)
        U_comb[:, 11:22] = U233
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + (3 + k), 11:22] = V233[j * 3 + k]
        for k in range(3):
            for i in range(2):
                W_comb[(3 + k) * 2 + i, 11:22] = W233[k * 2 + i]
                
        # Verify full solution
        if verify_decomposition(U_comb, V_comb, W_comb, n, m, p):
            return {
                "U": U_comb.tolist(),
                "V": V_comb.tolist(),
                "W": W_comb.tolist()
            }
            
    # Try to find rank 12 for 2x3x3
    m233_sol_12 = search_m233_rank(12, 60)
    
    if m233_sol_12 is not None:
        U233, V233, W233 = m233_sol_12
        # Construct M_{2,3,6}
        # U shape: (6, 24)
        # V shape: (18, 24)
        # W shape: (12, 24)
        
        U_comb = np.zeros((6, 24))
        V_comb = np.zeros((18, 24))
        W_comb = np.zeros((12, 24))
        
        # Block 1
        U_comb[:, 0:12] = U233
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, 0:12] = V233[j * 3 + k]
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, 0:12] = W233[k * 2 + i]
                
        # Block 2
        U_comb[:, 12:24] = U233
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + (3 + k), 12:24] = V233[j * 3 + k]
        for k in range(3):
            for i in range(2):
                W_comb[(3 + k) * 2 + i, 12:24] = W233[k * 2 + i]
                
        # Verify full solution
        if verify_decomposition(U_comb, V_comb, W_comb, n, m, p):
            return {
                "U": U_comb.tolist(),
                "V": V_comb.tolist(),
                "W": W_comb.tolist()
            }
            
    # Fallback: Rank 34 solution from current program
    U233_seed = np.zeros((6, 17))
    V233_seed = np.zeros((9, 17))
    W233_seed = np.zeros((6, 17))
    
    U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
    V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
    W233_seed[0, 0] = 1
    W233_seed[3, 0] = 1
    
    U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
    V233_seed[0, 1] = 1
    W233_seed[1, 1] = 1
    W233_seed[3, 1] = -1
    
    U233_seed[0, 2] = 1
    V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
    W233_seed[2, 2] = 1
    W233_seed[3, 2] = 1
    
    U233_seed[4, 3] = 1
    V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
    W233_seed[0, 3] = 1
    W233_seed[1, 3] = 1
    
    U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
    V233_seed[4, 4] = 1
    W233_seed[0, 4] = -1
    W233_seed[2, 4] = 1
    
    U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
    V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
    W233_seed[3, 5] = 1
    
    U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
    V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
    W233_seed[0, 6] = 1
    
    U233_seed[0, 7] = 1
    V233_seed[2, 7] = 1
    W233_seed[4, 7] = 1
    
    U233_seed[1, 8] = 1
    V233_seed[5, 8] = 1
    W233_seed[4, 8] = 1
    
    U233_seed[3, 9] = 1
    V233_seed[2, 9] = 1
    W233_seed[5, 9] = 1
    
    U233_seed[4, 10] = 1
    V233_seed[5, 10] = 1
    W233_seed[5, 10] = 1
    
    U233_seed[2, 11] = 1
    V233_seed[6, 11] = 1
    W233_seed[0, 11] = 1
    
    U233_seed[2, 12] = 1
    V233_seed[7, 12] = 1
    W233_seed[2, 12] = 1
    
    U233_seed[2, 13] = 1
    V233_seed[8, 13] = 1
    W233_seed[4, 13] = 1
    
    U233_seed[5, 14] = 1
    V233_seed[6, 14] = 1
    W233_seed[1, 14] = 1
    
    U233_seed[5, 15] = 1
    V233_seed[7, 15] = 1
    W233_seed[3, 15] = 1
    
    U233_seed[5, 16] = 1
    V233_seed[8, 16] = 1
    W233_seed[5, 16] = 1
    
    U_comb = np.zeros((6, 34))
    V_comb = np.zeros((18, 34))
    W_comb = np.zeros((12, 34))
    
    U_comb[:, :17] = U233_seed
    U_comb[:, 17:] = U233_seed
    
    for j in range(3):
        for k in range(3):
            V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
            V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
            
    for k in range(3):
        for i in range(2):
            W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
            W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
            
    return {
        "U": U_comb.tolist(),
        "V": V_comb.tolist(),
        "W": W_comb.tolist()
    }

if __name__ == "__main__":
    construct()
