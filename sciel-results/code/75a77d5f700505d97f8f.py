import time
import numpy as np
import sys

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit (150s hard kill)

    # 1. Build the target tensor T of shape (n*m, m*p, p*n)
    # Indices: i in [0, n), j in [0, m), k in [0, p)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1

    def verify_decomposition(U, V, W):
        """Check if U, V, W exactly decompose T."""
        # Convert to float for computation
        U_f = np.array(U, dtype=float)
        V_f = np.array(V, dtype=float)
        W_f = np.array(W, dtype=float)
        T_approx = np.einsum('ic,jc,kc->ijk', U_f, V_f, W_f)
        return np.allclose(T_approx, T, atol=1e-9)

    def schoolbook():
        """Fallback schoolbook decomposition (rank 18)."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    # 2. Try to find a rank-10 decomposition using Integer Linear Programming
    # We restrict to integer coefficients for simplicity and robustness in MILP.
    # If an integer solution exists for rank 10, it is a valid half-integer solution.
    
    def try_milp_rank(rank, max_time=60):
        """
        Attempt to find an integer decomposition of the given rank using scipy.optimize.milp.
        Variables: u[i,r], v[j,r], w[k,r] for all indices and rank positions.
        Constraints: sum_r u[i,r]*v[j,r]*w[k,r] = T[i,j,k]
        This is a cubic problem. To make it linear for MILP, we introduce auxiliary variables.
        
        Let x_{i,j,k,r} = u_{i,r} * v_{j,r} * w_{k,r}.
        Then sum_r x_{i,j,k,r} = T_{i,j,k}.
        
        However, linking x to u,v,w is non-linear.
        Standard MILP cannot handle product of 3 integer variables directly without linearization tricks that blow up complexity or require binary expansion.
        
        Alternative: Since rank is small (10) and dimensions are small, we can iterate over possible "bases" or use a different formulation.
        
        Actually, a simpler structural approach for small ranks is to use the fact that the tensor is sparse.
        Let's try a different MILP formulation:
        We know the support of the tensor.
        
        Let's try a randomized restart with a tighter bound on coefficients, but using a local search that is smarter:
        "Compressed Sensing" style or "Alternating Least Squares" with hard rounding?
        
        Wait, the current program fails because random local search is inefficient.
        Let's try a **Structured Decomposition** based on the block structure.
        
        The tensor M_{2,3,3} can be viewed as:
        [ A  B ]   [ C  D  E ]   [ G  H ]
        [ F  G ]   [ I  J  K ]   [ L  M ]
        
        Actually, let's look at the known rank 10 construction.
        It is not trivial to hardcode.
        
        Let's try a **Genetic Algorithm** with a fitness function based on the residual error, but with a strong penalty for non-half-integer values, and elitism.
        
        Or better: **Simulated Annealing** on the half-integer lattice.
        The space of half-integers is discrete. We can define moves as flipping a value by +/-0.5.
        Energy = || T - sum u v w ||^2.
        
        Let's implement a fast Simulated Annealing search on the discrete half-integer lattice.
        """
        if time.time() > deadline - 10:
            return None
        
        # Initialize with a random half-integer matrix
        # To improve chances, we can initialize with the schoolbook solution and then merge columns?
        # No, schoolbook is rank 18. We need rank 10.
        
        # Let's use a random initialization but bias towards small values.
        # Values in {-1, -0.5, 0, 0.5, 1} are most likely.
        
        U = np.random.randint(-2, 3, size=(n*m, rank)) / 2.0
        V = np.random.randint(-2, 3, size=(m*p, rank)) / 2.0
        W = np.random.randint(-2, 3, size=(p*n, rank)) / 2.0
        
        # Calculate initial error
        def calc_error(U, V, W):
            T_approx = np.einsum('ic,jc,kc->ijk', U, V, W)
            return np.sum((T - T_approx)**2)
        
        current_error = calc_error(U, V, W)
        best_error = current_error
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Simulated Annealing parameters
        T_start = 1.0
        T_end = 1e-6
        alpha = 0.99
        max_iter = 20000
        
        # Precompute indices for fast access
        # We will pick a random matrix (U, V, or W), a random row, and a random column to perturb.
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # Temperature decay
            T_curr = T_start * (alpha ** it)
            
            # Pick a random variable to perturb
            # 1/3 chance for U, 1/3 for V, 1/3 for W
            choice = np.random.randint(0, 3)
            
            # Store old value
            old_val = 0.0
            if choice == 0:
                i, r = np.random.randint(n*m), np.random.randint(rank)
                old_val = U[i, r]
                delta = np.random.choice([-0.5, 0.5])
                U[i, r] = old_val + delta
            elif choice == 1:
                j, r = np.random.randint(m*p), np.random.randint(rank)
                old_val = V[j, r]
                delta = np.random.choice([-0.5, 0.5])
                V[j, r] = old_val + delta
            else:
                k, r = np.random.randint(p*n), np.random.randint(rank)
                old_val = W[k, r]
                delta = np.random.choice([-0.5, 0.5])
                W[k, r] = old_val + delta
            
            new_error = calc_error(U, V, W)
            delta_e = new_error - current_error
            
            # Acceptance criterion
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T_curr):
                current_error = new_error
                if new_error < best_error:
                    best_error = new_error
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                    # Check for exact solution
                    if best_error < 1e-9:
                        # Verify it's exactly on half-integers (it is by construction)
                        return best_U, best_V, best_W
            else:
                # Revert
                if choice == 0:
                    U[i, r] = old_val
                elif choice == 1:
                    V[j, r] = old_val
                else:
                    W[k, r] = old_val
                    
        if best_error < 1e-9:
            return best_U, best_V, best_W
        return None

    # Try rank 10 with multiple restarts of Simulated Annealing
    # The search space is large, but the target is specific.
    # We will try 5 restarts for rank 10.
    
    found = False
    for restart in range(5):
        if time.time() > deadline - 15:
            break
            
        # Set seed for reproducibility in debugging, but vary it
        np.random.seed(42 + restart)
        result = try_milp_rank(10) # Name is misleading, it's SA
        
        if result is not None:
            U, V, W = result
            if verify_decomposition(U, V, W):
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                best_rank = 10
                found = True
                break

    if not found:
        # If rank 10 fails, try rank 11?
        # Given the time budget, rank 10 is the primary target.
        # If we have time, we could try 11.
        if time.time() < deadline - 20:
            for restart in range(2):
                if time.time() > deadline - 10:
                    break
                np.random.seed(100 + restart)
                result = try_milp_rank(11)
                if result is not None:
                    U, V, W = result
                    if verify_decomposition(U, V, W):
                        best_result = {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        best_rank = 11
                        found = True
                        break

    return best_result
# EVOLVE-BLOCK-END
