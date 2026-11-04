import numpy as np
import random
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Build the target tensor T
    # T[u, v, w] where u in [0, nm), v in [0, mp), w in [0, pn)
    # u = i*m + j, v = j*p + k, w = k*n + i
    # T[u, v, w] = 1 if j_u == j_v and k_v == k_w and i_w == i_u else 0
    T = np.zeros((nm, mp, pn), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p

    def verify_full(U, V, W, rank):
        """Full tensor verification."""
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        
        for w in range(pn):
            M = np.zeros((nm, mp), dtype=np.int64)
            for c in range(rank):
                if W_np[w, c] != 0:
                    M += U_np[:, c] * V_np[:, c] * W_np[w, c]
            if not np.array_equal(M, T[:, :, w]):
                return False
        return True

    # Precompute: for each w, the list of (u, v) pairs where T[u,v,w] = 1
    T_slices = []
    for w in range(pn):
        k = w // n
        i = w % n
        pairs = []
        for j in range(m):
            u = i*m + j
            v = j*p + k
            pairs.append((u, v))
        T_slices.append(pairs)

    # Strategy: For each target rank, try structured U, V patterns
    # Key idea: U and V should have structure that respects the index decomposition
    
    def try_rank(target_rank, num_attempts=50):
        """Try to find a decomposition of given rank."""
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Generate U, V with small integer entries
            # Use a mix of structured and random patterns
            if attempt < num_attempts // 3:
                # Structured: based on index patterns
                U = [[0] * target_rank for _ in range(nm)]
                V = [[0] * target_rank for _ in range(mp)]
                
                # Assign random small values to create structure
                for u in range(nm):
                    i = u // m
                    j = u % m
                    for c in range(target_rank):
                        # Pattern based on i, j, c
                        val = random.choice([-2, -1, 0, 1, 2])
                        if (i + j + c) % 3 == 0:
                            val = 1
                        elif (i + j + c) % 3 == 1:
                            val = -1
                        U[u][c] = val
                
                for v in range(mp):
                    j = v // p
                    k = v % p
                    for c in range(target_rank):
                        val = random.choice([-2, -1, 0, 1, 2])
                        if (j + k + c) % 3 == 0:
                            val = 1
                        elif (j + k + c) % 3 == 1:
                            val = -1
                        V[v][c] = val
            elif attempt < 2 * num_attempts // 3:
                # Semi-structured
                U = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(nm)]
                V = [[random.choice([-1, 0, 1]) for _ in range(target_rank)] for _ in range(mp)]
            else:
                # Random with small integers
                U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(nm)]
                V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(mp)]
            
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            
            W = [[0] * target_rank for _ in range(pn)]
            valid = True
            
            # Solve for W for each w
            for w in range(pn):
                # Construct A_w (nm*mp x target_rank)
                # A_w[u*mp + v, c] = U[u, c] * V[v, c]
                A_w = np.zeros((nm * mp, target_rank), dtype=np.float64)
                for u in range(nm):
                    for v in range(mp):
                        A_w[u * mp + v, :] = U_np[u, :] * V_np[v, :]
                
                # b_w: the slice of T
                b_w = np.zeros(nm * mp, dtype=np.float64)
                for (u, v) in T_slices[w]:
                    b_w[u * mp + v] = 1.0
                
                # Solve A_w W_w = b_w using least squares
                try:
                    W_w, residuals, rank_w, s = np.linalg.lstsq(A_w, b_w, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Check residual
                if residuals.size > 0 and residuals[0] > 1e-8:
                    valid = False
                    break
                    
                # Check if solution is close to half-integers
                W_w_snapped = np.round(W_w * 2) / 2.0
                
                # Verify snapped solution
                residual_check = np.linalg.norm(A_w.dot(W_w_snapped) - b_w)
                if residual_check > 1e-8:
                    valid = False
                    break
                
                for c in range(target_rank):
                    W[w][c] = W_w_snapped[c]
            
            if valid:
                # Final verification
                if verify_full(U, V, W, target_rank):
                    return {"U": U, "V": V, "W": W}
        
        return None

    # Try ranks from lower to upper
    # Lower bound: at least max(nm, mp, pn) = 21, but tensor rank can be lower
    # For 2x3x7, known results suggest rank around 21-30
    min_rank = 18
    
    for target_rank in range(min_rank, best_rank):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(target_rank, num_attempts=30)
        if result is not None:
            best = result
            best_rank = target_rank
            # Continue trying lower ranks
    
    return best
# EVOLVE-BLOCK-END
