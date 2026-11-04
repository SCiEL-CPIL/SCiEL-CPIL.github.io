"""Matrix multiplication 3,4,7 rank-minimal decomposition."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    U_rows = n * m  # 12
    V_rows = m * p  # 28
    W_rows = p * n  # 21
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 84
    
    # Theoretical lower bound
    lower_bound = max(n*m, m*p, p*n)  # max(12, 28, 21) = 28
    
    # Try to find a decomposition with rank < schoolbook_rank
    # Using a random search approach with rational coefficients
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # First, let's try a structured approach
    # We know that for matrix multiplication, the tensor has a certain structure
    # Let's try to find a decomposition by solving linear constraints
    
    # The condition for a valid decomposition of rank r is:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # For all other (i', j', k') combinations where the tensor entry is 0:
    # sum_c U[i'*m+j', c] * V[j'*p+k', c] * W[k'*n+i', c] = 0
    
    # This is a system of polynomial equations. Let's try a different approach:
    # Use the fact that we can construct the decomposition by choosing U and V
    # and then solving for W, or vice versa.
    
    # Actually, let's try a more direct approach:
    # We'll parameterize the problem and use a greedy/iterative method.
    
    # Strategy: Try to find a low-rank decomposition using a structured basis.
    # For matrix multiplication, there's a known decomposition using the
    # "Strassen-like" ideas for rectangular matrices.
    
    # Let's try a random search with a time limit.
    # We'll try ranks from lower_bound up to schoolbook_rank - 1
    
    # For efficiency, let's use a simplified check:
    # The tensor entries for M_{n,m,p} are:
    # T[i,j,k] = 1 if we're multiplying (i,j) of A with (j,k) of B to get (i,k) of C
    # In the W convention: W[k*n+i, r], so W index is k*n + i
    
    # The bilinear form condition:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: sum_j T[i,j,k] * A[i,j] * B[j,k]
    # Where T[i,j,k] = 1 for valid (i,j,k) triples
    
    # In the factorization:
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_k V[j*p+k, r] * B[j,k])
    
    # Wait, the standard form is:
    # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_l V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
    # No, that's not quite right. Let me reconsider.
    
    # The standard tensor decomposition for matrix multiplication:
    # We have tensors U, V, W such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
    # where T[i,j,k] = 1 if (i,j,k) is a valid index triple for multiplication
    
    # For M_{3,4,7}:
    # i in 0..2, j in 0..3, k in 0..6
    # T[i,j,k] = 1 for all valid triples (all 3*4*7 = 84 triples)
    
    # So we need: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of 84 polynomial equations in 12*r + 28*r + 21*r = 61*r variables.
    
    # Let's try a numerical approach: fix some variables and solve for others.
    
    # Approach: Use a structured ansatz.
    # Try rank = 28 (the lower bound) first, then increase if needed.
    
    # Actually, let's try a different strategy: use known decompositions.
    # For M_{2,2,2}, rank 7 is known (Strassen).
    # For M_{3,3,3}, rank 23 is the best known.
    # For M_{3,4,7}, I'm not sure of the best known rank.
    
    # Let's try a random search approach with a time limit.
    
    def try_random_decomposition(rank, max_iterations=10000):
        """Try to find a rank-decomposition using random initialization."""
        if rank < lower_bound:
            return None
        
        # Randomly initialize U, V, W with small integers
        np.random.seed(42)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
            
            # Try different initialization strategies
            strategy = random.randint(0, 2)
            
            if strategy == 0:
                # Sparse random
                U = np.zeros((U_rows, rank), dtype=object)
                V = np.zeros((V_rows, rank), dtype=object)
                W = np.zeros((W_rows, rank), dtype=object)
                for c in range(rank):
                    # Pick random non-zero positions
                    u_idx = random.randint(0, U_rows - 1)
                    v_idx = random.randint(0, V_rows - 1)
                    w_idx = random.randint(0, W_rows - 1)
                    U[u_idx, c] = random.choice([-1, 1])
                    V[v_idx, c] = random.choice([-1, 1])
                    W[w_idx, c] = random.choice([-1, 1])
            elif strategy == 1:
                # Dense random with small values
                U = np.array([[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(U_rows)], dtype=object)
                V = np.array([[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(V_rows)], dtype=object)
                W = np.array([[random.choice([-1, 0, 1]) for _ in range(rank)] for _ in range(W_rows)], dtype=object)
            else:
                # Structured: try to use a Kronecker-like structure
                U = np.zeros((U_rows, rank), dtype=object)
                V = np.zeros((V_rows, rank), dtype=object)
                W = np.zeros((W_rows, rank), dtype=object)
                for c in range(rank):
                    # Distribute entries
                    n_u = random.randint(1, 3)
                    n_v = random.randint(1, 3)
                    n_w = random.randint(1, 3)
                    u_indices = random.sample(range(U_rows), min(n_u, U_rows))
                    v_indices = random.sample(range(V_rows), min(n_v, V_rows))
                    w_indices = random.sample(range(W_rows), min(n_w, W_rows))
                    for idx in u_indices:
                        U[idx, c] = random.choice([-1, 1])
                    for idx in v_indices:
                        V[idx, c] = random.choice([-1, 1])
                    for idx in w_indices:
                        W[idx, c] = random.choice([-1, 1])
            
            # Check if this decomposition works
            # We need: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0
                        for c in range(rank):
                            total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if total != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
                if not valid:
                    break
            
            if valid:
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return (U_list, V_list, W_list)
        
        return None
    
    # Try ranks from lower_bound upward
    for rank in range(lower_bound, schoolbook_rank):
        if time.time() > deadline - 10:
            break
        
        result = try_random_decomposition(rank, max_iterations=5000)
        if result is not None:
            U_list, V_list, W_list = result
            best_rank = rank
            best_U = U_list
            best_V = V_list
            best_W = W_list
            break
    
    # If we didn't find anything better, return schoolbook
    if best_U is None:
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(U_rows)]
        V = [[0] * rank for _ in range(V_rows)]
        W = [[0] * rank for _ in range(W_rows)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

