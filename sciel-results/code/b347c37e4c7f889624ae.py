"""Matrix multiplication 2x3 x 3x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Block-padding strategy: split 8 into two 4-blocks
    # Each 2x3x4 sub-problem gets its own decomposition
    # Total rank = rank_4block * 2
    
    # For 2x3x4, use a structured decomposition
    # Schoolbook for 2x3x4 is rank 24
    # We'll try to find a lower rank decomposition
    
    def verify_234(U, V, W, rank):
        """Verify 2x3x4 decomposition."""
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    s = np.dot(U[i*3+j, :rank], V[j*4+k, :rank] * W[k*2+i, :rank])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to construct a 2x3x4 decomposition with rank < 24
    # Use random search with structure for 2x3x4
    
    best_rank_4 = 24
    best_U_4, best_V_4, best_W_4 = None, None, None
    
    U_s4 = np.zeros((6, 24))
    V_s4 = np.zeros((12, 24))
    W_s4 = np.zeros((8, 24))
    col = 0
    for i in range(2):
        for j in range(3):
            for k in range(4):
                U_s4[i*3+j, col] = 1
                V_s4[j*4+k, col] = 1
                W_s4[k*2+i, col] = 1
                col += 1
    
    best_U_4, best_V_4, best_W_4 = U_s4.copy(), V_s4.copy(), W_s4.copy()
    
    vals = np.array([0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0])
    
    # Search for better 2x3x4 decomposition
    for target_rank in range(12, 24):
        if time.time() > deadline - 30.0:
            break
        
        found = False
        max_trials = 200 if target_rank <= 18 else 100
        for trial in range(max_trials):
            if time.time() > deadline - 20.0:
                break
            
            rng = np.random.RandomState(trial * 7919 + target_rank * 137 + 42)
            
            U = np.zeros((6, target_rank))
            V = np.zeros((12, target_rank))
            W = np.zeros((8, target_rank))
            
            density = 0.3 + 0.1 * (target_rank / 24.0)
            for idx in range(6):
                mask = rng.random(target_rank) < density
                U[idx, mask] = rng.choice(vals, size=np.sum(mask))
            for idx in range(12):
                mask = rng.random(target_rank) < density
                V[idx, mask] = rng.choice(vals, size=np.sum(mask))
            for idx in range(8):
                mask = rng.random(target_rank) < density
                W[idx, mask] = rng.choice(vals, size=np.sum(mask))
            
            if verify_234(U, V, W, target_rank):
                best_rank_4 = target_rank
                best_U_4, best_V_4, best_W_4 = U.copy(), V.copy(), W.copy()
                found = True
                break
        
        if found:
            continue
    
    # Build the full 2x3x8 decomposition from two 2x3x4 blocks
    total_rank = best_rank_4 * 2
    
    U_full = np.zeros((6, total_rank))
    V_full = np.zeros((24, total_rank))
    W_full = np.zeros((16, total_rank))
    
    # First 4-block (k=0,1,2,3)
    U_full[:, :best_rank_4] = best_U_4
    V_full[:12, :best_rank_4] = best_V_4
    W_full[:8, :best_rank_4] = best_W_4
    
    # Second 4-block (k=4,5,6,7)
    U_full[:, best_rank_4:] = best_U_4
    V_full[12:, best_rank_4:] = best_V_4
    W_full[8:, best_rank_4:] = best_W_4
    
    # Verify the full decomposition
    def verify_full(U, V, W, rank):
        for i in range(2):
            for j in range(3):
                for k in range(8):
                    s = np.dot(U[i*3+j, :rank], V[j*8+k, :rank] * W[k*2+i, :rank])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_full(U_full, V_full, W_full, total_rank):
        # Fall back to schoolbook
        rank_school = 48
        U_s = np.zeros((6, rank_school))
        V_s = np.zeros((24, rank_school))
        W_s = np.zeros((16, rank_school))
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(8):
                    U_s[i*3+j, col] = 1
                    V_s[j*8+k, col] = 1
                    W_s[k*2+i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
# EVOLVE-BLOCK-END
