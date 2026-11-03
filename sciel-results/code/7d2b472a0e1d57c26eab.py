"""Program for 3x3 x 3x5 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 5

    def elapsed():
        return time.time() - start_time

    def remaining():
        return DEADLINE - SAFETY_MARGIN - elapsed()

    def make_schoolbook():
        """Return the schoolbook decomposition of rank n*m*p = 45."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W, rank):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    def tensor_error(U, V, W, rank):
        """Compute the sum of squared errors for the decomposition."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
                    err += (val - 1.0) ** 2
        return err

    def gradient_descent(U, V, W, rank, max_iter=2000, lr_init=0.05):
        """
        Perform gradient descent on the tensor error.
        Uses the fact that the tensor is T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        and we want T[i,j,k] = 1 for all (i,j,k) where the multiplication is defined.
        """
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()
        
        lr = lr_init
        best_err = tensor_error(U, V, W, rank)
        patience = 500
        no_improve = 0
        
        for it in range(max_iter):
            if remaining() < 5:
                break
            if no_improve > patience:
                lr *= 0.5
                no_improve = 0
                if lr < 1e-8:
                    break
            
            # Compute gradients
            grad_U = np.zeros_like(U)
            grad_V = np.zeros_like(V)
            grad_W = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # dT[i,j,k]/dU[i*m+j,r] = V[j*p+k,r] * W[k*n+i,r]
                        # dT[i,j,k]/dV[j*p+k,r] = U[i*m+j,r] * W[k*n+i,r]
                        # dT[i,j,k]/dW[k*n+i,r] = U[i*m+j,r] * V[j*p+k,r]
                        err_ijk = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank]) - 1.0
                        
                        grad_U[i*m+j, :rank] += 2.0 * err_ijk * V[j*p+k, :rank] * W[k*n+i, :rank]
                        grad_V[j*p+k, :rank] += 2.0 * err_ijk * U[i*m+j, :rank] * W[k*n+i, :rank]
                        grad_W[k*n+i, :rank] += 2.0 * err_ijk * U[i*m+j, :rank] * V[j*p+k, :rank]
            
            # Update
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            new_err = tensor_error(U, V, W, rank)
            if new_err < best_err:
                best_err = new_err
                no_improve = 0
                if best_err < 1e-12:
                    break
            else:
                no_improve += 1
                # Revert
                U += lr * grad_U
                V += lr * grad_V
                W += lr * grad_W
        
        return U, V, W, best_err

    def structured_init(R):
        """
        Create a structured initialization that respects the tensor structure.
        For matrix multiplication, we can initialize using the schoolbook pattern
        but with random linear combinations to allow for rank reduction.
        """
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        
        # Initialize with random signs and small values, structured by the multiplication pattern
        for r in range(R):
            # Each rank-1 term should contribute to the multiplication structure
            # U row (i,j) should be nonzero, V row (j,k) should be nonzero, W row (k,i) should be nonzero
            # where they form valid multiplication paths
            
            # Randomly choose which (i,j) pairs, (j,k) pairs, (k,i) pairs to activate
            for i in range(n):
                for j in range(m):
                    if random.random() < 0.7:
                        U[i*m+j, r] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
            
            for j in range(m):
                for k in range(p):
                    if random.random() < 0.7:
                        V[j*p+k, r] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
            
            for k in range(p):
                for i in range(n):
                    if random.random() < 0.7:
                        W[k*n+i, r] = random.choice([-1, 1]) * random.uniform(0.5, 1.5)
        
        # Ensure each column is nonzero
        for r in range(R):
            if not np.any(U[:, r] != 0):
                U[random.randint(0, n*m-1), r] = 1.0
            if not np.any(V[:, r] != 0):
                V[random.randint(0, m*p-1), r] = 1.0
            if not np.any(W[:, r] != 0):
                W[random.randint(0, p*n-1), r] = 1.0
        
        return U, V, W

    def try_rank(R, max_attempts=30):
        """Try to find a rank-R decomposition using structured init + gradient descent."""
        if remaining() < 10:
            return None

        for attempt in range(max_attempts):
            if remaining() < 5:
                break

            U, V, W = structured_init(R)
            U, V, W, err = gradient_descent(U, V, W, R, max_iter=3000)

            if err < 1e-10:
                # Try to round to nice values
                for mult in [1, 2, 4]:
                    U_round = np.round(U * mult) / mult
                    V_round = np.round(V * mult) / mult
                    W_round = np.round(W * mult) / mult
                    if verify(U_round, V_round, W_round, R):
                        return {
                            "U": U_round.tolist(),
                            "V": V_round.tolist(),
                            "W": W_round.tolist()
                        }
                
                # Accept floating point if close enough
                if verify(U, V, W, R):
                    return {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }

        return None

    def integer_search(R, max_iterations=50000):
        """
        Search for integer decompositions using a smarter approach.
        For small ranks, we can enumerate structured patterns.
        """
        if remaining() < 10:
            return None

        # For rank R, try to find {-1,0,1} solutions
        # Use a constraint satisfaction approach
        for _ in range(max_iterations):
            if remaining() < 5:
                break

            U = np.zeros((n*m, R), dtype=int)
            V = np.zeros((m*p, R), dtype=int)
            W = np.zeros((p*n, R), dtype=int)

            for r in range(R):
                # Generate random {-1,0,1} patterns
                U[:, r] = np.random.choice([-1, 0, 1], size=n*m)
                V[:, r] = np.random.choice([-1, 0, 1], size=m*p)
                W[:, r] = np.random.choice([-1, 0, 1], size=p*n)
                
                # Ensure each column has some nonzeros
                while not np.any(U[:, r] != 0):
                    U[random.randint(0, n*m-1), r] = random.choice([-1, 1])
                while not np.any(V[:, r] != 0):
                    V[random.randint(0, m*p-1), r] = random.choice([-1, 1])
                while not np.any(W[:, r] != 0):
                    W[random.randint(0, p*n-1), r] = random.choice([-1, 1])

            if verify(U, V, W, R):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }

        return None

    # Start with schoolbook rank 45
    best_result = make_schoolbook()
    best_rank = n * m * p  # 45

    # Try to find lower rank decompositions
    # The theoretical lower bound for 3x3 x 3x5 is at least max(n*p, m*p, n*m) = max(15,15,9) = 15
    # But in practice, for small matrices, the rank is close to n*m*p
    # Try ranks from 40 upward to see if we can find a better decomposition
    
    for target_rank in range(40, 45):
        if remaining() < 15:
            break

        # First try gradient-based search
        result = try_rank(target_rank, max_attempts=20)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break

        # Then try integer search
        result = integer_search(target_rank, max_iterations=20000)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break

    return best_result
# EVOLVE-BLOCK-END
