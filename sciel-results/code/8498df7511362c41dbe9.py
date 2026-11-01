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
    # Tensor T_{a,b,c,d,i,k} is 1 if a==i, b==d, c==k else 0
    # Decomposition: sum_l U_{a*m+b, l} V_{c*p+d, l} W_{k*n+i, l}
    
    # For efficiency, we can check specific non-trivial combinations
    # Or just iterate over all if dimensions are small.
    # n=2, m=3, p=6 -> 2*3*3*6*2*6 = 1296 checks.
    # Each check sums over rank r. If r=33, 33 ops. ~43k ops. Very fast.
    
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            s = 0.0
                            # Vectorized dot product for speed
                            s = np.dot(U[a * m + b, :], np.multiply(V[c * p + d, :], W[k * n + i, :]))
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # We want to find a low rank decomposition for M_{2,3,6}.
    # We hypothesize that M_{2,3,6} can be decomposed into 3 blocks of M_{2,3,2}.
    # If we find a rank R decomposition for M_{2,3,2}, the total rank is 3*R.
    # Schoolbook for M_{2,3,2} is 12. Total 36.
    # Current best is 34 (from M_{2,3,3} rank 17 * 2).
    # If we find M_{2,3,2} rank 11, total is 33.
    # If we find M_{2,3,2} rank 10, total is 30.
    
    # Strategy: Search for rank 11 decomposition of M_{2,3,2} using Simulated Annealing.
    # If successful, construct the full M_{2,3,6} solution.
    # If not, fall back to the known rank 34 solution.
    
    def search_m232_rank(target_rank, time_budget):
        """
        Attempts to find a rank 'target_rank' decomposition for 2x3x2 matrix multiplication.
        Returns (U, V, W) if found, else None.
        """
        if time.time() > start_time + time_budget:
            return None
            
        n2, m2, p2 = 2, 3, 2
        U_dim = n2 * m2 # 6
        V_dim = m2 * p2 # 6
        W_dim = p2 * n2 # 4
        
        best_error = float('inf')
        best_U = None
        best_V = None
        best_W = None
        
        # Initialize with schoolbook (rank 12) and zero out 12-target_rank random entries?
        # Or just random init with small values?
        # Random init in high dimensions rarely finds exact integer solutions.
        # Better: Start from schoolbook rank 12, try to nullify terms.
        
        # Schoolbook for 2x3x2:
        # U rows: (a,b), V rows: (c,d), W rows: (k,i)
        # Terms: a=0,1; b=0,1,2; c=0,1,2; d=0,1; k=0,1; i=0,1
        # Standard basis: U_{a,b} = e_{a,b}, V_{c,d} = e_{c,d}, W_{k,i} = e_{k,i}
        # This gives rank 12.
        
        # We want rank 11. We need to merge 2 terms into 1, or find a structure.
        
        # Let's try a direct optimization on a rank-11 matrix.
        # Objective: sum of squared errors + sparsity penalty?
        # No, we need EXACT.
        
        # Simulated Annealing on the entries.
        # We will use a "structured" random init.
        
        for restart in range(50):
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
            max_steps = 20000
            
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
                # Perturb
                new_val = old_val + np.random.randn() * 0.1
                mat[row, col] = new_val
                
                # Calculate error
                # To speed up, we only calculate error for a subset of tensors?
                # No, for exactness we need full check eventually.
                # For SA, we can use a fast approximate error or full error if small.
                # Full error is ~1296 * 11 ops = 14k flops. 20k steps * 14k = 280M flops.
                # In Python/NumPy, this might be slow.
                # Let's optimize the error calculation.
                
                # Precompute? No, it's dynamic.
                # Let's just do it.
                err = 0.0
                # We can vectorize the tensor check?
                # T is sparse.
                # Let's just iterate.
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
                        # Found exact solution!
                        # Try to snap to integers/half-integers?
                        # If it's exact, it's good enough for the evaluator if it's close.
                        # But we want to return the matrices.
                        if best_error > current_error:
                            best_error = current_error
                            best_U = U.copy()
                            best_V = V.copy()
                            best_W = W.copy()
                            # Check if we can return immediately?
                            # Yes, if error is 0.
                            if current_error < 1e-12:
                                return (best_U, best_V, best_W)
                    T *= cool
                else:
                    # Accept with probability exp(-deltaE / T)
                    delta_E = err - current_error
                    if np.random.rand() < np.exp(-delta_E / T):
                        T *= cool
                    else:
                        mat[row, col] = old_val
                        T *= cool
                
                steps += 1
                
            if best_error > 1e-9:
                continue
                
            # Check if best is valid
            if verify_decomposition(best_U, best_V, best_W, n2, m2, p2):
                return (best_U, best_V, best_W)
                
        return None

    # Try to find rank 11 for 2x3x2
    # Budget: 60 seconds
    m232_sol = search_m232_rank(11, 60)
    
    if m232_sol is not None:
        U232, V232, W232 = m232_sol
        # Construct M_{2,3,6}
        # U shape: (6, 33)
        # V shape: (18, 33)
        # W shape: (12, 33)
        
        U_comb = np.zeros((6, 33))
        V_comb = np.zeros((18, 33))
        W_comb = np.zeros((12, 33))
        
        # Block 1 (d=0,1 -> k=0,1)
        U_comb[:, 0:11] = U232
        for j in range(3):
            for k in range(2):
                V_comb[j * 6 + k, 0:11] = V232[j * 2 + k]
        for k in range(2):
            for i in range(2):
                W_comb[k * 2 + i, 0:11] = W232[k * 2 + i]
                
        # Block 2 (d=2,3 -> k=2,3)
        U_comb[:, 11:22] = U232
        for j in range(3):
            for k in range(2):
                V_comb[j * 6 + (2 + k), 11:22] = V232[j * 2 + k]
        for k in range(2):
            for i in range(2):
                W_comb[(2 + k) * 2 + i, 11:22] = W232[k * 2 + i]
                
        # Block 3 (d=4,5 -> k=4,5)
        U_comb[:, 22:33] = U232
        for j in range(3):
            for k in range(2):
                V_comb[j * 6 + (4 + k), 22:33] = V232[j * 2 + k]
        for k in range(2):
            for i in range(2):
                W_comb[(4 + k) * 2 + i, 22:33] = W232[k * 2 + i]
                
        # Verify full solution
        if verify_decomposition(U_comb, V_comb, W_comb, n, m, p):
            return {
                "U": U_comb.tolist(),
                "V": V_comb.tolist(),
                "W": W_comb.tolist()
            }
            
    # Fallback: Rank 34 solution from current program
    # This is the verified solution from the prompt's current program.
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
