"""Matrix multiplication 3x5x7: rank-minimal exact tensor decomposition search."""
import time
import random
import itertools
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]
n, m, p = DIMENSIONS
# Index ranges
# A is n x m, B is m x p, C is n x p
# Tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
# Decomposition: T[i,j,k] = sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
# Note: The task specifies W[k*n+i, r]. Let's verify indices.
# U shape: (n*m, R), index i*m+j
# V shape: (m*p, R), index j*p+k
# W shape: (p*n, R), index k*n+i

def _verify(U, V, W):
    """
    Verify if the decomposition is exact.
    U, V, W are lists of lists (or arrays).
    Returns True if valid.
    """
    R = len(U[0])
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    
    # Convert to numpy for speed if needed, but we need exactness.
    # The task allows integer/half-integer entries.
    # We can verify by reconstructing the tensor.
    # T[i,j,k] should be 1 for all i,j,k.
    
    # Optimization: Use numpy for the sum, then check if it's close to 1?
    # No, "exactly decompose". 
    # If entries are half-integers, we can multiply by 2 to make them integers.
    # Let's check if all entries are multiples of 0.5.
    
    # First, let's just do a numerical check with high precision or exact fractions if possible.
    # Given the small dimensions, we can try to verify exactly using integers if we detect half-integers.
    
    # Let's assume the evaluator checks exactness. We should ensure our output is exact.
    # For the search, we will use integers or half-integers.
    
    # Let's convert to integers by multiplying by 2.
    U_int = np.array([[int(2 * x) for x in row] for row in U])
    V_int = np.array([[int(2 * x) for x in row] for row in V])
    W_int = np.array([[int(2 * x) for x in row] for row in W])
    
    # Check if conversion was lossless
    if not np.allclose(np.array(U, dtype=float) * 2, U_int, atol=1e-9): return False
    if not np.allclose(np.array(V, dtype=float) * 2, V_int, atol=1e-9): return False
    if not np.allclose(np.array(W, dtype=float) * 2, W_int, atol=1e-9): return False
    
    # Reconstruct tensor T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We want T[i,j,k] == 1 for all i,j,k.
    # Since we multiplied by 2, the sum will be 2^3 = 8 if the original was 1.
    # Sum_r U_int[i,j,r] * V_int[j,k,r] * W_int[k,i,r] should be 8.
    
    # Efficient check:
    # For each i, j, k, compute the dot product.
    # This is O(n*m*p*R). With small dims, this is fine.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_int[i*m+j, :]
                v_val = V_int[j*p+k, :]
                w_val = W_int[k*n+i, :]
                s = np.dot(np.dot(u_val, v_val), w_val)
                if s != 8:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Known lower bounds for 3x5x7
    # Rank of 3x5 is 7 (or 8?). 3x4 is 6? No, 3x4 is 6. 3x5 is likely 7 or 8.
    # Strassen for 4x4 is 7. 3x3 is 5.
    # Let's try to find a rank.
    # Schoolbook is 105.
    # We can try to improve using known algorithms or random search.
    
    # Strategy:
    # 1. Try to find a low rank decomposition using random initialization and gradient descent?
    #    But the space is discrete (integers/half-integers).
    # 2. Use the "Laser method" or similar algebraic constructions? Hard to code.
    # 3. Random search for small ranks.
    
    # Let's start with a reasonable upper bound and try to lower it.
    # For 3x5x7, the rank is at most 3*5*7 = 105.
    # We can try to find a rank < 105.
    
    # Let's try a randomized search for rank R.
    # We will iterate R from a lower bound upwards.
    # Lower bound: max(n*m, m*p, n*p) = max(15, 35, 21) = 35.
    # Actually, the rank must be at least max(nm, mp, np) is not a strict lower bound for tensor rank, 
    # but it is for the "border rank" or similar? 
    # The rank of the multiplication tensor is at least the max of the ranks of the slices.
    # Slices are rank 1 matrices? No.
    # The tensor T is the identity map on the space of matrices? No.
    
    # Let's just try to find *any* valid decomposition with a rank lower than 105.
    # We can use a heuristic: start with a random rank R, initialize U, V, W randomly with small integers,
    # and try to satisfy the equations.
    
    # However, solving the system of polynomial equations is hard.
    
    # Alternative: Use the "Strassen-like" approach if dimensions allow.
    # 3x5x7 is not a power of 2.
    
    # Let's try a simple random search with local optimization.
    # We define the error as the max absolute difference from 1.
    # We try to minimize this error.
    
    # To keep it fast, we will use numpy for the evaluation of the error.
    
    def eval_error(U, V, W):
        """
        U, V, W are numpy arrays of shape (n*m, R), (m*p, R), (p*n, R).
        Returns the max absolute error of T[i,j,k] - 1.
        """
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # We can compute this efficiently?
        # For each r, we add U_r * V_r * W_r to the tensor.
        # U_r is a vector of length n*m. We need to reshape it to (n,m).
        # V_r is a vector of length m*p. Reshape to (m,p).
        # W_r is a vector of length p*n. Reshape to (p,n). Note index k*n+i.
        
        # T = zeros(n,m,p)
        # For r in range(R):
        #   Ur = U[:,r].reshape(n,m)
        #   Vr = V[:,r].reshape(m,p)
        #   Wr = W[:,r].reshape(p,n)
        #   T += np.einsum('ij,jk,ki->ik', Ur, Vr, Wr) # Wait, indices.
        #   Ur[i,j], Vr[j,k], Wr[k,i] -> C[i,k]?
        #   Let's check: sum_j Ur[i,j] Vr[j,k] = (Ur Vr)[i,k].
        #   Then multiply by Wr?
        #   The term is Ur[i,j] * Vr[j,k] * Wr[k,i].
        #   This is not a standard matrix multiplication product.
        #   It is a tensor contraction.
        
        # Let's use einsum:
        # 'ij,jk,ki->ik' ?
        # i from Ur, j from Ur/Vr, k from Vr/Wr, i from Wr.
        # Result is a matrix of shape (n, p) for each r?
        # Sum over r gives the final C matrix?
        # No, the tensor T is 3D. The product of three rank-1 tensors is a rank-1 tensor.
        # We are summing rank-1 tensors to form T.
        # T[i,j,k] = 1.
        
        # So we need to check if sum_r Ur[i,j] Vr[j,k] Wr[k,i] == 1 for all i,j,k.
        
        # We can compute the whole tensor T (n,m,p) and check if it's all 1s.
        # T = np.zeros((n,m,p))
        # for r in range(R):
        #    Ur = U[:,r].reshape(n,m)
        #    Vr = V[:,r].reshape(m,p)
        #    Wr = W[:,r].reshape(p,n)
        #    T += Ur[:, :, None] * Vr[None, :, :] * Wr.T[None, :, :] # Wr.T is (n,p), we need (i,k)?
        #    Wait, Wr index is k*n+i. So Wr[k,i].
        #    We want to multiply Ur[i,j] * Vr[j,k] * Wr[k,i].
        #    Let's broadcast:
        #    Ur: (n, m, 1)
        #    Vr: (1, m, p)
        #    Wr: (n, 1, p) ? No, Wr[k,i] means row k, col i.
        #    So Wr.T is (n, p) where Wr.T[i,k] = Wr[k,i].
        #    So we can do:
        #    T += Ur[:, :, None] * Vr[None, :, :] * Wr.T[None, None, :] # (n,m,p)
        
        # This is O(R * n * m * p).
        
        T = np.zeros((n, m, p))
        for r in range(U.shape[1]):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)
            # Wr.T is (n, p)
            T += Ur[:, :, None] * Vr[None, :, :] * Wr.T[None, None, :]
        
        return np.max(np.abs(T - 1.0))

    # Random Search
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # We will try to find a solution for rank R.
    # Start with R = 35 (lower bound) and go up.
    # If we find one, we return it.
    # To make it more likely to find a solution, we can use random restarts.
    
    # However, random search in high dimensions is unlikely to find an exact integer solution.
    # We need a smarter approach.
    
    # Let's try to use the known result for 3x3x3 (rank 5) and 4x4x4 (rank 7).
    # Can we embed 3x3x3 into 3x5x7?
    # If we have a rank 5 decomposition for 3x3x3, can we extend it to 3x5x7?
    # Not directly.
    
    # Let's try a simple local search.
    # Initialize U, V, W with small random integers.
    # Compute error.
    # Try to change one entry at a time to reduce error.
    
    # Since the space is discrete, we can use a hill-climbing algorithm.
    
    def random_init(R):
        U = np.random.randint(-2, 3, (n*m, R))
        V = np.random.randint(-2, 3, (m*p, R))
        W = np.random.randint(-2, 3, (p*n, R))
        return U, V, W

    def hill_climb(U, V, W, max_steps=1000):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        R = U.shape[1]
        
        current_error = eval_error(U, V, W)
        
        for step in range(max_steps):
            if current_error < 1e-9:
                return U, V, W, current_error
            
            # Pick a random matrix and a random entry
            mat_idx = np.random.randint(3)
            if mat_idx == 0:
                i = np.random.randint(n*m)
                j = np.random.randint(R)
                old_val = U[i, j]
                # Try changing U[i,j]
                for delta in [-2, -1, 1, 2]:
                    new_val = old_val + delta
                    U[i, j] = new_val
                    new_error = eval_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        break
                    else:
                        U[i, j] = old_val
            elif mat_idx == 1:
                i = np.random.randint(m*p)
                j = np.random.randint(R)
                old_val = V[i, j]
                for delta in [-2, -1, 1, 2]:
                    new_val = old_val + delta
                    V[i, j] = new_val
                    new_error = eval_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        break
                    else:
                        V[i, j] = old_val
            else:
                i = np.random.randint(p*n)
                j = np.random.randint(R)
                old_val = W[i, j]
                for delta in [-2, -1, 1, 2]:
                    new_val = old_val + delta
                    W[i, j] = new_val
                    new_error = eval_error(U, V, W)
                    if new_error < current_error:
                        current_error = new_error
                        break
                    else:
                        W[i, j] = old_val
                        
            if time.time() > deadline:
                break
        
        return U, V, W, current_error

    # Try different ranks
    # Start from a lower bound.
    # The rank of 3x5x7 is unknown, but likely around 30-40.
    # Let's try R from 30 to 50.
    
    for R in range(30, 60):
        if time.time() > deadline - 10:
            break
            
        for restart in range(5):
            if time.time() > deadline - 5:
                break
                
            U, V, W = random_init(R)
            U, V, W, err = hill_climb(U, V, W, max_steps=200)
            
            if err < 1e-9:
                # Found a solution!
                # Convert to list of lists
                best_rank = R
                best_U = U.tolist()
                best_V = V.tolist()
                best_W = W.tolist()
                return {"U": best_U, "V": best_V, "W": best_W}
        
        # If we didn't find a solution for this R, try next.
        # But hill climbing might get stuck.
        
    # If no better solution found, return the schoolbook decomposition.
    # This ensures we always return a valid result.
    
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
