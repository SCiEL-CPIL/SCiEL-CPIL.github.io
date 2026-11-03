"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    schoolbook_rank = n * m * p  # 144
    
    # Try to find a lower rank solution using local search
    best_solution = None
    best_rank = schoolbook_rank
    
    for target_rank in range(130, 115, -1):
        solution = _find_decomposition(n, m, p, target_rank)
        if solution is not None:
            U, V, W = solution
            if _verify(U, V, W, n, m, p):
                best_rank = target_rank
                best_solution = solution
                break
    
    if best_solution is None:
        return _schoolbook(n, m, p)
    
    U, V, W = best_solution
    U_list = [[_round_half(U[i][c]) for c in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[_round_half(V[i][c]) for c in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[_round_half(W[i][c]) for c in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _round_half(x):
    r = round(x * 2) / 2
    if abs(x - r) < 1e-6:
        return r
    return x

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    P = np.zeros((n, m, p))
    for c in range(rank):
        P += np.einsum('ij,jk,ki->ijk', U3[:,:,c], V3[:,:,c], W3[:,:,c])
    return np.allclose(P, 1.0, atol=1e-5)

def _find_decomposition(n, m, p, target_rank, max_iters=5000):
    np.random.seed(12345)
    
    # Start from schoolbook and try to compress
    # First, create a schoolbook-like initialization with target_rank columns
    # by randomly selecting which (i,j,k) triples to cover
    
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    W = np.zeros((p*n, target_rank))
    
    # Initialize: assign each column a random (i,j,k)
    for c in range(target_rank):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        U[i*m+j, c] = 1.0
        V[j*p+k, c] = 1.0
        W[k*n+i, c] = 1.0
    
    U3 = U.reshape(n, m, target_rank)
    V3 = V.reshape(m, p, target_rank)
    W3 = W.reshape(p, n, target_rank)
    
    best_error = _compute_error(U3, V3, W3, n, m, p, target_rank)
    
    for it in range(max_iters):
        c = np.random.randint(target_rank)
        mat_choice = np.random.randint(3)
        
        if mat_choice == 0:
            row = np.random.randint(n*m)
            i, j = divmod(row, m)
            old_val = U[row, c]
            new_val = float(np.random.choice([0, 1, -1]))
            if new_val != old_val:
                U[row, c] = new_val
                U3[i, j, c] = new_val
                new_error = _compute_error(U3, V3, W3, n, m, p, target_rank)
                if new_error < best_error:
                    best_error = new_error
                else:
                    U[row, c] = old_val
                    U3[i, j, c] = old_val
        elif mat_choice == 1:
            row = np.random.randint(m*p)
            j, k = divmod(row, p)
            old_val = V[row, c]
            new_val = float(np.random.choice([0, 1, -1]))
            if new_val != old_val:
                V[row, c] = new_val
                V3[j, k, c] = new_val
                new_error = _compute_error(U3, V3, W3, n, m, p, target_rank)
                if new_error < best_error:
                    best_error = new_error
                else:
                    V[row, c] = old_val
                    V3[j, k, c] = old_val
        else:
            row = np.random.randint(p*n)
            k, i = divmod(row, n)
            old_val = W[row, c]
            new_val = float(np.random.choice([0, 1, -1]))
            if new_val != old_val:
                W[row, c] = new_val
                W3[k, i, c] = new_val
                new_error = _compute_error(U3, V3, W3, n, m, p, target_rank)
                if new_error < best_error:
                    best_error = new_error
                else:
                    W[row, c] = old_val
                    W3[k, i, c] = old_val
        
        if best_error < 1e-10:
            return U, V, W
    
    return None

def _compute_error(U3, V3, W3, n, m, p, rank):
    P = np.zeros((n, m, p))
    for c in range(rank):
        P += np.einsum('ij,jk,ki->ijk', U3[:,:,c], V3[:,:,c], W3[:,:,c])
    return np.sum((P - 1) ** 2)
# EVOLVE-BLOCK-END
