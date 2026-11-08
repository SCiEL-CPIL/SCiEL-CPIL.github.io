import numpy as np
import time
import random

DIMENSIONS = [2, 4, 7]
DEADLINE = 125  # safety margin under 150s

def _check_decomposition(U, V, W, n, m, p):
    """
    Verify sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    """
    rank = len(U[0])
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Reshape
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # Compute S[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
    # Using einsum
    # U3: (n, m, r)
    # V3: (m, p, r)
    # W3: (p, n, r)
    # We want result (n, m, p)
    # S[i, j, k] = sum_c U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
    
    try:
        S = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
        if not np.allclose(S, 1.0):
            return False
        return True
    except Exception:
        return False

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # 1. Generate Schoolbook
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best_result = {"U": U, "V": V, "W": W}
    best_rank = rank_schoolbook

    # 2. Search for lower rank
    # We iterate r downwards.
    # For each r, we try to find U, V such that we can solve for W.
    
    def try_rank(r):
        nonlocal best_rank, best_result
        if r >= best_rank:
            return False
        
        # Time limit for this rank: 10 seconds
        end_time = time.time() + 10
        
        # Possible values for U, V
        # Using {-1, 0, 1} is a good start.
        # We can also include 2, -2 to increase chances of S being 1 or 2.
        # But larger values make S larger, making it harder to get 1 with small W.
        # Let's stick to {-1, 0, 1} first.
        
        while time.time() < end_time:
            # Random U, V
            U_rand = np.random.choice([-1, 0, 1], size=(n * m, r))
            V_rand = np.random.choice([-1, 0, 1], size=(m * p, r))
            
            # Compute S[i, k, c] = sum_j U[i, j, c] V[j, k, c]
            # U3: (n, m, r)
            # V3: (m, p, r)
            U3 = U_rand.reshape(n, m, r)
            V3 = V_rand.reshape(m, p, r)
            
            # S_ikc = sum_j U3[i, j, c] V3[j, k, c]
            # einsum: 'ijc,jkc->ikc'
            S = np.einsum('ijc,jkc->ikc', U3, V3)
            
            # Now we need to find W[k, i, c] such that
            # sum_c S[i, k, c] W[k, i, c] = 1 for all i, k.
            # Note: S is indexed [i, k, c]. W is indexed [k, i, c] in the tensor sense?
            # The problem defines W shape (p*n, r). Index k*n+i.
            # So W_row = k*n + i.
            # The equation is sum_c S[i, k, c] * W[k*n+i, c] = 1.
            
            W_rand = np.zeros((p * n, r))
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # We have vector S_vec = S[i, k, :] of length r.
                    # We need w_vec = W[k*n+i, :] such that dot(S_vec, w_vec) = 1.
                    # w_vec entries must be in {0, 0.5, 1, ...} or {-1, -0.5, ...}
                    # Let's check if any single element works.
                    # If S_vec[c] == 1, w_vec[c] = 1 works.
                    # If S_vec[c] == 2, w_vec[c] = 0.5 works.
                    # If S_vec[c] == -1, w_vec[c] = -1 works.
                    # If S_vec[c] == -2, w_vec[c] = -0.5 works.
                    
                    found = False
                    for c in range(r):
                        s_val = S[i, k, c]
                        if s_val == 1:
                            W_rand[k * n + i, c] = 1.0
                            found = True
                            break
                        elif s_val == 2:
                            W_rand[k * n + i, c] = 0.5
                            found = True
                            break
                        elif s_val == -1:
                            W_rand[k * n + i, c] = -1.0
                            found = True
                            break
                        elif s_val == -2:
                            W_rand[k * n + i, c] = -0.5
                            found = True
                            break
                    
                    if not found:
                        # Try pairs
                        # We need a*S1 + b*S2 = 1 with a,b in half-integers.
                        # This is a small search.
                        # Since r is small enough, we can check all pairs if needed,
                        # but it's O(r^2).
                        # Let's just check if any pair of small integers sums to 1.
                        # Actually, if no single element works, it's less likely to find a pair quickly.
                        # We can mark this attempt as failed for this (i,k).
                        valid = False
                        break
                    
                if not valid:
                    break
            
            if valid:
                # Verify the full decomposition to be safe
                if _check_decomposition(U_rand, V_rand, W_rand, n, m, p):
                    best_rank = r
                    # Convert to lists of floats
                    best_U = [list(row) for row in U_rand]
                    best_V = [list(row) for row in V_rand]
                    best_W = [list(row) for row in W_rand]
                    best_result = {"U": best_U, "V": best_V, "W": best_W}
                    return True
        return False

    # Try ranks lower than schoolbook
    # Start from 55 down to 40
    for r in range(rank_schoolbook - 1, 39, -1):
        if time.time() > DEADLINE:
            break
        if try_rank(r):
            print(f"Found rank {r}")
            return best_result

    return best_result
