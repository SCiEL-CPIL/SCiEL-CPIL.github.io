"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the multiplication tensor T[i,j,k] = delta_{i,k} * delta_{j,j}
    # For matrix multiplication C = A @ B where A is n x m, B is m x p, C is n x p
    # T[a, b, c] = 1 if a == c (row of A matches row of C) and b == b (col of A matches row of B)
    # Actually, the tensor for matrix mult is: T[i, j, k] = 1 if i == k (output row i depends on input row k)
    # and the contraction is over j.
    
    # The decomposition says: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where i indexes A (n*m), j indexes B (m*p), k indexes C (p*n)
    
    # W uses trace convention: W[k*n+i, r] where k is column of C, i is row of C
    # So W index = k*n + i
    
    # Let's verify by checking all entries
    rank = U.shape[1]
    errors = 0
    max_err = 0
    
    for i in range(n * m):
        ai, bi = i // m, i % m  # A row ai, col bi
        for j in range(m * p):
            bj, cj = j // p, j % p  # B row bj, col cj
            # Output C[ai, cj] gets contribution if bi == bj
            expected = 1.0 if bi == bj else 0.0
            # Sum over r: U[i,r] * V[j,r] * W[cj*n + ai, r]
            actual = 0.0
            for r in range(rank):
                actual += U[i, r] * V[j, r] * W[cj * n + ai, r]
            err = abs(actual - expected)
            if err > 1e-9:
                errors += 1
                max_err = max(max_err, err)
    
    return errors == 0, max_err

def _build_strassen_decomposition(n, m, p):
    """Build the Strassen-based rank-42 decomposition."""
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    for u in range(6):
        base_v = 2 * u
        base_w = 2 * u * n
        for t in range(7):
            c = u * 7 + t
            for i in range(4):
                U[i][c] = U_forms[t][i]
            
            # V updates
            if t == 0:
                V[base_v][c] += 1
                V[p + base_v + 1][c] += 1
            elif t == 1:
                V[base_v][c] += 1
            elif t == 2:
                V[base_v + 1][c] += 1
                V[p + base_v + 1][c] -= 1
            elif t == 3:
                V[p + base_v][c] += 1
                V[base_v][c] -= 1
            elif t == 4:
                V[p + base_v + 1][c] += 1
            elif t == 5:
                V[base_v][c] += 1
                V[base_v + 1][c] += 1
            elif t == 6:
                V[p + base_v][c] += 1
                V[p + base_v + 1][c] += 1
                
            # W updates
            if t == 0:
                W[base_w + 0][c] += 1
                W[base_w + n + 1][c] += 1
            elif t == 1:
                W[base_w + 1][c] += 1
                W[base_w + n + 1][c] -= 1
            elif t == 2:
                W[base_w + n + 0][c] += 1
                W[base_w + n + 1][c] += 1
            elif t == 3:
                W[base_w + 0][c] += 1
                W[base_w + 1][c] += 1
            elif t == 4:
                W[base_w + 0][c] -= 1
                W[base_w + n + 0][c] += 1
            elif t == 5:
                W[base_w + n + 1][c] += 1
            elif t == 6:
                W[base_w + 0][c] += 1

    return U, V, W, rank

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Start with the known rank-42 solution
    U, V, W, rank = _build_strassen_decomposition(n, m, p)
    valid, max_err = _verify_decomposition(U, V, W, n, m, p)
    
    if not valid:
        # Fallback: return the construction as-is (should be valid)
        return {"U": U, "V": V, "W": W}
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find a lower rank by attempting rank reduction
    # Strategy: Try rank 40, 41 by attempting to express some columns as linear combinations
    # of others and eliminating them
    
    for target_rank in range(rank - 1, 38, -1):
        if time.time() > deadline:
            break
        
        # Try to reduce from current best to target_rank
        # We attempt to find linear dependencies among the rank components
        # by checking if removing any column still gives a valid decomposition
        
        U_arr = np.array(best_U, dtype=float)
        V_arr = np.array(best_V, dtype=float)
        W_arr = np.array(best_W, dtype=float)
        
        current_rank = U_arr.shape[1]
        
        # Try removing each column and checking validity
        reduced = False
        for col_to_remove in range(current_rank - 1, target_rank - 1, -1):
            if time.time() > deadline:
                break
            
            # Remove this column
            mask = [i for i in range(current_rank) if i != col_to_remove]
            U_test = U_arr[:, mask]
            V_test = V_arr[:, mask]
            W_test = W_arr[:, mask]
            
            valid_test, _ = _verify_decomposition(U_test, V_test, W_test, n, m, p)
            
            if valid_test:
                best_U = U_test.tolist()
                best_V = V_test.tolist()
                best_W = W_test.tolist()
                best_rank = len(mask)
                reduced = True
                break
        
        if not reduced:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
