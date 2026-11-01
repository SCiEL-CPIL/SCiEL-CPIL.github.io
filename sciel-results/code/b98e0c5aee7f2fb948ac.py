"""Matrix multiplication 2x3x4 rank-minimal tensor decomposition."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    # Build the target tensor entries
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                tensor_entries.append((u_idx, v_idx, w_idx))
    
    num_entries = len(tensor_entries)  # 24
    
    # Schoolbook fallback (rank 24)
    best_rank = n * m * p
    U_sb = [[0] * best_rank for _ in range(nm)]
    V_sb = [[0] * best_rank for _ in range(mp)]
    W_sb = [[0] * best_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    def verify_solution(U, V, W, rank):
        """Check if the decomposition is exact."""
        # Check tensor entries
        for (u, v, w) in tensor_entries:
            val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
            if abs(val - 1.0) > 1e-10:
                return False
        # Check non-entries
        entry_set = set(tensor_entries)
        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    if (u, v, w) not in entry_set:
                        val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
                        if abs(val) > 1e-10:
                            return False
        return True

    # Create target tensor T
    T = np.zeros((nm, mp, pn))
    for (u, v, w) in tensor_entries:
        T[u, v, w] = 1.0

    # Search for minimal rank
    # We try ranks from 18 to 23
    for target_rank in range(18, best_rank):
        if time.time() > deadline - 5:
            break
        
        found = False
        np.random.seed(42 + target_rank)
        
        # Number of random initializations
        num_attempts = 200
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 2:
                break
            
            # Initialize U, V, W with small random values
            # Using a range that allows for half-integers after rounding
            # e.g. [-2, 2]
            U = np.random.uniform(-2, 2, size=(nm, target_rank))
            V = np.random.uniform(-2, 2, size=(mp, target_rank))
            W = np.random.uniform(-2, 2, size=(pn, target_rank))
            
            # Gradient Descent
            # We use a simple adaptive learning rate
            alpha = 0.1
            for it in range(100):
                # Compute S = sum_r U_r V_r W_r
                S = np.einsum('ur,vr,wr->uvw', U, V, W)
                error = np.sum((S - T) ** 2)
                
                if error < 1e-10:
                    break
                
                # Compute gradients
                G_U = 2 * np.einsum('uvw,vr,wr->ur', S - T, V, W)
                G_V = 2 * np.einsum('uvw,ur,wr->vr', S - T, U, W)
                G_W = 2 * np.einsum('uvw,ur,vr->wr', S - T, U, V)
                
                # Update
                U -= alpha * G_U
                V -= alpha * G_V
                W -= alpha * G_W
                
                # Optional: clamp values to prevent divergence
                U = np.clip(U, -10, 10)
                V = np.clip(V, -10, 10)
                W = np.clip(W, -10, 10)
            
            # Check if error is small
            S = np.einsum('ur,vr,wr->uvw', U, V, W)
            error = np.sum((S - T) ** 2)
            
            if error > 1e-6:
                continue
            
            # Round to half-integers
            U_check = np.round(U * 2) / 2.0
            V_check = np.round(V * 2) / 2.0
            W_check = np.round(W * 2) / 2.0
            
            # Verify
            if verify_solution(U_check.tolist(), V_check.tolist(), W_check.tolist(), target_rank):
                best_rank = target_rank
                best_result = {
                    "U": U_check.tolist(),
                    "V": V_check.tolist(),
                    "W": W_check.tolist()
                }
                found = True
                break
        
        if found:
            break
    
    if best_rank < n * m * p:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
