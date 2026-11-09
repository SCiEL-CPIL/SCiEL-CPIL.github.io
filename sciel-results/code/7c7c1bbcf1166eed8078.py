"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    def to_half_integers(U, V, W):
        """Convert to half-integer representation (multiply by 2, round, then divide)."""
        U2 = np.round(U * 2) / 2
        V2 = np.round(V * 2) / 2
        W2 = np.round(W * 2) / 2
        return U2, V2, W2
    
    # Try to reduce rank by attempting to eliminate terms
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Try removing one term at a time
    for attempt in range(min(100, int(deadline - time.time() - 5))):
        if time.time() > deadline - 5:
            break
        
        # Pick a random term to try to eliminate
        c_to_remove = random.randint(0, best_rank - 1)
        
        # Save current state
        U_test = best_U.copy()
        V_test = best_V.copy()
        W_test = best_W.copy()
        
        # Zero out the term
        U_test[:, c_to_remove] = 0
        V_test[:, c_to_remove] = 0
        W_test[:, c_to_remove] = 0
        
        # Check if we can compensate by adjusting nearby terms
        # This is a simplified approach - just check if verification still passes
        # (which it won't, so this is just a placeholder for a more sophisticated search)
        
        # For now, just try small random perturbations
        improved = False
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            
            # Randomly perturb some entries
            for _ in range(5):
                r = random.randint(0, best_rank - 1)
                if r == c_to_remove:
                    continue
                if random.random() < 0.5:
                    U_test[random.randint(0, n*m-1), r] += random.choice([-0.5, 0.5])
                else:
                    V_test[random.randint(0, m*p-1), r] += random.choice([-0.5, 0.5])
            
            # Convert to half-integers
            U_test, V_test, W_test = to_half_integers(U_test, V_test, W_test)
            
            # Verify
            if verify(U_test, V_test, W_test, best_rank):
                # Check if we can actually remove the term
                # Try to set the removed term to zero and verify
                U_test2 = U_test.copy()
                V_test2 = V_test.copy()
                W_test2 = W_test.copy()
                U_test2[:, c_to_remove] = 0
                V_test2[:, c_to_remove] = 0
                W_test2[:, c_to_remove] = 0
                
                # Compact: remove the zero column
                mask = np.ones(best_rank, dtype=bool)
                mask[c_to_remove] = False
                U_compact = U_test2[:, mask]
                V_compact = V_test2[:, mask]
                W_compact = W_test2[:, mask]
                
                if verify(U_compact, V_compact, W_compact, best_rank - 1):
                    best_U, best_V, best_W = U_compact, V_compact, W_compact
                    best_rank -= 1
                    improved = True
                    break
        
        if improved:
            continue
    
    # Convert to lists for the output
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
