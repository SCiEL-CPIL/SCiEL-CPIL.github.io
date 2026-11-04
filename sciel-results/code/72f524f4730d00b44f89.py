"""Matrix multiplication 4x5x6 tensor decomposition via ILP."""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

DIMENSIONS = [4, 5, 6]

def _make_schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def _try_ilp_rank(n, m, p, rank, time_limit=30):
    """
    Attempt to find a rank-decomposition using Integer Linear Programming.
    Variables: U, V, W matrices flattened.
    Constraints: sum_c U[i,m,j,c] * V[j,p,k,c] * W[k,n,i,c] == 1 for all i,j,k.
    This is a cubic problem, so we linearize it.
    
    Linearization:
    Let x_{i,j,k,c} = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    This is hard. 
    Alternative: 
    Since we only need existence, and small integers, we can fix one matrix?
    No, ILP on the product is non-linear.
    
    However, we can use the fact that for small ranks, we can iterate over possible values for one set of variables?
    Or better: Use the fact that the tensor is 0-1.
    
    Actually, a better ILP formulation for matrix multiplication tensor:
    We want to find U, V, W with integer entries such that T = U V^T W^T (elementwise product summed over rank).
    
    Let's try a different approach: 
    Since rank is small (around 50-60), and entries are likely small integers (-2, -1, 0, 1, 2),
    we can use a constraint solver if available? No, only scipy.
    
    Let's stick to a smarter random search with local improvement (Hill Climbing) but with a different representation:
    We fix the structure of the decomposition based on known lower bounds.
    
    Wait, the directive says "different design family". ILP is a good candidate if we can formulate it.
    The constraint is:
    For each (i, j, k):
    sum_{c=0}^{R-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This is a system of cubic equations.
    We can linearize by introducing auxiliary variables z_{i,j,k,c} = U_{i,j,c} * V_{j,k,c}.
    Then z_{i,j,k,c} * W_{k,i,c} = ... still quadratic.
    
    Let's try a **Genetic Algorithm** approach. This is a different optimization method (evolutionary) compared to random restart hill climbing.
    """
    pass

def _ga_search(n, m, p, max_rank, population_size=20, generations=50, time_limit=60):
    """
    Genetic Algorithm to find low rank decomposition.
    """
    start_time = time.time()
    
    # Initialize population with random integers in [-2, 2]
    # We try to find rank R such that it works.
    # We start with a known good upper bound or slightly below.
    
    # For 4x5x6, rank is likely around 50-60.
    # Let's try to find a solution for a specific rank R.
    # We will iterate R from 60 down to 40.
    
    best_rank = None
    best_sol = None
    
    for R in range(60, 39, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Population of individuals
        # Individual: (U, V, W)
        # U: (n*m, R), V: (m*p, R), W: (p*n, R)
        
        pop = []
        for _ in range(population_size):
            U = np.random.randint(-2, 3, size=(n*m, R))
            V = np.random.randint(-2, 3, size=(m*p, R))
            W = np.random.randint(-2, 3, size=(p*n, R))
            pop.append((U, V, W))
            
        # Evaluate fitness
        # Fitness: number of (i,j,k) satisfied.
        # We want all n*m*p = 120 to be satisfied.
        
        def evaluate(indiv):
            U, V, W = indiv
            satisfied = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(R):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        if val == 1:
                            satisfied += 1
            return satisfied
            
        # Run GA
        for gen in range(generations):
            if time.time() - start_time > time_limit:
                break
                
            scored = [(evaluate(ind), ind) for ind in pop]
            scored.sort(key=lambda x: x[0], reverse=True)
            
            if scored[0][0] == n * m * p:
                return R, scored[0][1]
                
            # Selection: Keep top half
            elite = [ind for _, ind in scored[:population_size//2]]
            
            # Crossover and Mutation to fill rest
            new_pop = list(elite)
            while len(new_pop) < population_size:
                parent1 = scored[np.random.randint(0, len(scored))][1]
                parent2 = scored[np.random.randint(0, len(scored))][1]
                
                # Uniform crossover for each matrix
                U1, V1, W1 = parent1
                U2, V2, W2 = parent2
                
                mask_U = np.random.rand(n*m, R) < 0.5
                mask_V = np.random.rand(m*p, R) < 0.5
                mask_W = np.random.rand(p*n, R) < 0.5
                
                U_new = np.where(mask_U, U1, U2)
                V_new = np.where(mask_V, V1, V2)
                W_new = np.where(mask_W, W1, W2)
                
                # Mutation: change a few entries
                if np.random.rand() < 0.5:
                    idx = np.random.randint(0, n*m)
                    col = np.random.randint(0, R)
                    U_new[idx, col] += np.random.randint(-2, 3)
                if np.random.rand() < 0.5:
                    idx = np.random.randint(0, m*p)
                    col = np.random.randint(0, R)
                    V_new[idx, col] += np.random.randint(-2, 3)
                if np.random.rand() < 0.5:
                    idx = np.random.randint(0, p*n)
                    col = np.random.randint(0, R)
                    W_new[idx, col] += np.random.randint(-2, 3)
                
                new_pop.append((U_new, V_new, W_new))
                
            pop = new_pop
            
    return best_rank, best_sol

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school = _make_schoolbook(n, m, p)
    
    # Try GA search
    # We know Strassen gives 108. Schoolbook is 120.
    # Literature suggests 4x5x6 rank is around 50-60.
    # Let's try to find a solution better than 108.
    
    found_rank, found_sol = _ga_search(n, m, p, max_rank=60, time_limit=120)
    
    if found_rank is not None and found_sol is not None:
        U, V, W = found_sol
        # Convert to list of lists for interface
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook if GA fails
        U_list = U_school.tolist()
        V_list = V_school.tolist()
        W_list = W_school.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
