import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 3, 3, 6
    
    def generate_schoolbook_3x3x3():
        rank = 27
        U = [[0] * rank for _ in range(9)]
        V = [[0] * rank for _ in range(9)]
        W = [[0] * rank for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i * 3 + j][col] = 1
                    V[j * 3 + k][col] = 1
                    W[i * 3 + k][col] = 1
                    col += 1
        return U, V, W

    def verify_3x3x3(U, V, W):
        R = len(U[0])
        for i in range(3):
            for ip in range(3):
                for k in range(3):
                    for kp in range(3):
                        for j in range(3):
                            for jp in range(3):
                                expected = 1.0 if (j == jp and i == ip and k == kp) else 0.0
                                actual = 0.0
                                for r in range(R):
                                    actual += U[i * 3 + j][r] * V[jp * 3 + k][r] * W[ip * 3 + kp][r]
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True

    def extend_to_3x3x6(U_core, V_core, W_core):
        R = len(U_core[0])
        total_rank = 2 * R
        U_tot = [[0] * total_rank for _ in range(9)]
        V_tot = [[0] * total_rank for _ in range(18)]
        W_tot = [[0] * total_rank for _ in range(18)]
        
        for i in range(3):
            for j in range(3):
                for r in range(R):
                    U_tot[i * 3 + j][r] = U_core[i * 3 + j][r]
                    U_tot[i * 3 + j][R + r] = U_core[i * 3 + j][r]
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for r in range(R):
                        V_tot[j * 6 + k][r] = V_core[j * 3 + k][r]
                        W_tot[k * 3 + i][r] = W_core[k * 3 + i][r]
                        
                        V_tot[j * 6 + k + 3][R + r] = V_core[j * 3 + k][r]
                        W_tot[(k + 3) * 3 + i][R + r] = W_core[k * 3 + i][r]
        
        return {"U": U_tot, "V": V_tot, "W": W_tot}

    best_rank = 54
    best_result = None
    
    # Try to find a lower rank 3x3x3 decomposition
    # Start with schoolbook rank 27
    U_cur, V_cur, W_cur = generate_schoolbook_3x3x3()
    
    # Try to reduce rank by eliminating terms
    # This is a heuristic search
    for target_rank in range(27, 22, -1):
        if time.time() > deadline - 10:
            break
            
        U_try, V_try, W_try = generate_schoolbook_3x3x3()
        # Randomly shuffle and try to zero out (27 - target_rank) terms
        # This is very naive.
        
        # Instead, let's just return the schoolbook extension (rank 54) if no better found.
        # But we know rank 54 is the current score.
        # We need to find < 54.
        # If we find rank 26, we get 52.
        
        pass
        
    # Fallback: Return schoolbook 3x3x3 extended (Rank 54)
    # But we can try to find a rank 27 decomposition that is "sparser" or allows packing?
    # No, packing requires disjoint supports.
    
    # Let's try to find a rank 26 solution by random search with local improvement
    # We will skip this for now and just return 54? No, we need to improve.
    
    # Actually, if we use the schoolbook 3x3x3 (rank 27) and extend it, we get rank 54.
    # Is there a way to get rank 52?
    # If we can find a rank 26 3x3x3 decomposition.
    
    # I will implement a simple local search to try to find rank 26.
    # If it fails, return 54.
    
    found_better = False
    for attempt in range(100):
        if time.time() > deadline - 20:
            break
        U_try, V_try, W_try = generate_schoolbook_3x3x3()
        # Randomly perturb
        for _ in range(10):
             # Try to zero out a random term
             r = random.randint(0, 26)
             # Check if we can solve for W with term r zeroed?
             # This is complex.
             pass
             
    # Just return the known best: Schoolbook 3x3x3 extended to 3x3x6 is Rank 54.
    # Wait, the current program returns 54.
    # I need to return < 54.
    # If I cannot find < 54, I should return 54.
    # But the task asks to improve.
    # Maybe I can return a rank 53 decomposition?
    
    # Let's try to construct a rank 53 decomposition directly?
    # No, I will stick to the 3x3x3 core search.
    
    # I will use the schoolbook 3x3x3 and return rank 54.
    # This is the same as the current score.
    # I need to find a better one.
    
    # I will try to find a rank 26 3x3x3 decomposition using a more effective search.
    # I will use a random integer search with a larger budget.
    
    best_core = None
    for R in range(26, 22, -1):
        if time.time() > deadline - 30:
            break
        for attempt in range(500):
            if time.time() > deadline - 30:
                break
            U_r = np.random.randint(-1, 2, size=(9, R))
            V_r = np.random.randint(-1, 2, size=(9, R))
            W_c = np.zeros((9, R))
            valid = True
            for i in range(3):
                for k in range(3):
                    M = np.zeros((9, R))
                    for j in range(3):
                        for jp in range(3):
                            M[j * 3 + jp, :] = U_r[i * 3 + j, :] * V_r[jp * 3 + k, :]
                    rhs = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float)
                    try:
                        w_vec, residuals, rank_m, s = np.linalg.lstsq(M, rhs, rcond=None)
                    except:
                        valid = False
                        break
                    w_doubled = w_vec * 2
                    closest_ints = np.round(w_doubled)
                    error = np.max(np.abs(w_doubled - closest_ints))
                    if error < 1e-9:
                        W_c[i * 3 + k, :] = w_vec
                    else:
                        valid = False
                        break
            if valid:
                # Verify
                if verify_3x3x3(U_r.tolist(), V_r.tolist(), W_c.tolist()):
                    best_core = (U_r.tolist(), V_r.tolist(), W_c.tolist())
                    break
        if best_core:
            break

    if best_core:
        U_core, V_core, W_core = best_core
        R = len(U_core[0])
        total_rank = 2 * R
        if total_rank < best_rank:
            best_result = extend_to_3x3x6(U_core, V_core, W_core)
            best_rank = total_rank
    else:
        # Fallback to schoolbook
        U_core, V_core, W_core = generate_schoolbook_3x3x3()
        best_result = extend_to_3x3x6(U_core, V_core, W_core)
        best_rank = 54

    return best_result
