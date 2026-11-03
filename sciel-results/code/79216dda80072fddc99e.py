"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank as fallback
    rank_schoolbook = n * m * p
    
    # Deadline guard: stop after 120s, leave margin
    t_start = time.time()
    deadline = 120.0
    
    # Try to find lower-rank decompositions using random search
    # The tensor T[i,j,k] = delta_{i,j,k} where (i,j) index U, (j,k) index V, (k,i) index W
    # We need U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) valid else 0
    # Actually for matrix mult: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal
    # the tensor entry. For standard matrix mult, T[i,j,k] = 1 for all valid (i,j,k)
    # Wait no - the matrix multiplication tensor has T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
    # because C[i,k] = sum_j A[i,j] * B[j,k]
    
    # So we need: for all (i,j,k) in [n]x[m]x[p], sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # And for all other combinations of indices that don't correspond to valid (i,j,k), the sum should be 0.
    # But actually, U only has indices (i,j) where i in [n], j in [m]
    # V only has indices (j,k) where j in [m], k in [p]
    # W only has indices (k,i) where k in [p], i in [n]
    # So the constraint is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # This is equivalent to: the tensor T with entries T[i,j,k]=1 for all valid (i,j,k)
    # must have rank <= r.
    
    # Let's try random initialization and gradient-free optimization
    # or use the fact that this is a structured tensor
    
    # Try ranks from some lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But tensor rank can be lower than these bounds in some cases
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Try a few random approaches
    for trial_rank in range(28, min(60, rank_schoolbook)):
        if time.time() - t_start > deadline - 10:
            break
        for attempt in range(50):
            if time.time() - t_start > deadline - 10:
                break
            
            # Random initialization with small values
            np.random.seed(42 + trial_rank * 1000 + attempt)
            U = np.random.randn(n * m, trial_rank) * 0.5
            V = np.random.randn(m * p, trial_rank) * 0.5
            W = np.random.randn(p * n, trial_rank) * 0.5
            
            # Check if this gives approximately correct tensor
            # Compute the reconstructed tensor
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-6:
                best_rank = trial_rank
                best_result = (U, V, W, trial_rank)
                break
        if best_result is not None:
            break
    
    if best_result is not None:
        U, V, W, r = best_result
        # Convert to list of lists with rounding for half-integers
        U_list = [[float(round(U[i][c], 10)) for c in range(r)] for i in range(n*m)]
        V_list = [[float(round(V[i][c], 10)) for c in range(r)] for i in range(m*p)]
        W_list = [[float(round(W[i][c], 10)) for c in range(r)] for i in range(p*n)]
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    rank = rank_schoolbook
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
# EVOLVE-BLOCK-END
