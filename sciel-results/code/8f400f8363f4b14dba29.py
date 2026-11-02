import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[a, b, c, d] = 1 if b == c else 0
    T = np.zeros((n, m, m, p), dtype=np.int8)
    for b in range(m):
        for c in range(m):
            if b == c:
                T[:, b, c, :] = 1

    # Precompute indices for efficient error calculation
    # We want to check if sum_r U[a*m+b, r] * V[c*p+d, r] * W[d*n+a, r] == T[a,b,c,d]
    
    # We will search for solutions over integers {-1, 0, 1} (subset of half-integers)
    # This corresponds to F3 arithmetic effectively.
    
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.int8)
        V = np.zeros((m * p, rank), dtype=np.int8)
        W = np.zeros((p * n, rank), dtype=np.int8)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    start_time = time.time()
    time_limit = 110  # Leave 10s buffer for overhead
    
    # Genetic Algorithm Parameters
    pop_size = 50
    max_generations = 200
    mutation_rate = 0.1
    crossover_rate = 0.7
    
    # We try ranks from 25 to 35
    # If we find a valid solution, we return it.
    # Since exact integer solutions are hard, we start with higher ranks and try to lower,
    # or just find any valid one below schoolbook.
    
    # Let's try ranks 25 to 35.
    # For each rank, run GA.
    
    for target_rank in range(25, 36):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize population
        population = []
        for _ in range(pop_size):
            U = np.zeros((n * m, target_rank), dtype=np.int8)
            V = np.zeros((m * p, target_rank), dtype=np.int8)
            W = np.zeros((p * n, target_rank), dtype=np.int8)
            
            # Random sparse initialization with values in {-1, 1}
            # Density ~ 0.2
            U[np.random.rand(n * m, target_rank) < 0.2] = np.random.choice([-1, 1], size=(n * m, target_rank))
            V[np.random.rand(m * p, target_rank) < 0.2] = np.random.choice([-1, 1], size=(m * p, target_rank))
            W[np.random.rand(p * n, target_rank) < 0.2] = np.random.choice([-1, 1], size=(p * n, target_rank))
            
            population.append((U, V, W))

        best_fitness = float('inf')
        best_solution = None
        
        for gen in range(max_generations):
            if time.time() - start_time > time_limit:
                break
                
            # Evaluate population
            evaluated_pop = []
            for ind in population:
                U, V, W = ind
                # Compute error
                # Reshape for einsum
                U_r = U.reshape(n, m, target_rank)
                V_r = V.reshape(m, p, target_rank)
                W_r = W.reshape(p, n, target_rank)
                
                # S[a,b,c,d] = sum_r U[a,b,r] V[c,d,r] W[d,a,r]
                # Using einsum
                S = np.einsum('abr,cdr,dar->abcd', U_r, V_r, W_r)
                
                # Check exact match
                # T is 0 or 1. S must be 0 or 1.
                # Error: number of mismatches
                mismatch = np.sum(S != T)
                
                # Penalize rank usage? No, rank is fixed in this loop.
                # Fitness: mismatch count. If 0, we found a solution.
                fitness = mismatch
                
                evaluated_pop.append((fitness, U, V, W))
                
                if fitness == 0:
                    # Found exact solution
                    # Verify once more to be safe
                    U_chk = U.reshape(n, m, target_rank)
                    V_chk = V.reshape(m, p, target_rank)
                    W_chk = W.reshape(p, n, target_rank)
                    S_chk = np.einsum('abr,cdr,dar->abcd', U_chk, V_chk, W_chk)
                    if np.all(S_chk == T):
                        # Return immediately
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                        
            # Sort by fitness
            evaluated_pop.sort(key=lambda x: x[0])
            best_fitness = evaluated_pop[0][0]
            
            if best_fitness == 0:
                # Should have returned already, but just in case
                _, U, V, W = evaluated_pop[0]
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }

            # Selection: Keep top 50%
            next_pop = []
            half = pop_size // 2
            for i in range(half):
                _, U, V, W = evaluated_pop[i]
                next_pop.append((U.copy(), V.copy(), W.copy()))
            
            # Crossover and Mutation to fill rest
            while len(next_pop) < pop_size:
                if len(next_pop) < pop_size:
                    # Pick two parents from top half
                    p1 = evaluated_pop[random.randint(0, half - 1)]
                    p2 = evaluated_pop[random.randint(0, half - 1)]
                    
                    U1, V1, W1 = p1[1], p1[2], p1[3]
                    U2, V2, W2 = p2[1], p2[2], p2[3]
                    
                    # Uniform crossover
                    U_child = np.zeros_like(U1)
                    V_child = np.zeros_like(V1)
                    W_child = np.zeros_like(W1)
                    
                    mask_U = np.random.rand(*U1.shape) < 0.5
                    mask_V = np.random.rand(*V1.shape) < 0.5
                    mask_W = np.random.rand(*W1.shape) < 0.5
                    
                    U_child[mask_U] = U1[mask_U]
                    U_child[~mask_U] = U2[~mask_U]
                    
                    V_child[mask_V] = V1[mask_V]
                    V_child[~mask_V] = V2[~mask_V]
                    
                    W_child[mask_W] = W1[mask_W]
                    W_child[~mask_W] = W2[~mask_W]
                    
                    # Mutation
                    # Flip some values or set to 0
                    if np.random.rand() < mutation_rate:
                        # Randomly perturb a few entries
                        for _ in range(5):
                            mat = random.choice([U_child, V_child, W_child])
                            idx = np.unravel_index(np.random.randint(mat.size), mat.shape)
                            # Change to 0 or flip sign or random {-1,0,1}
                            mat[idx] = random.choice([-1, 0, 1])
                    
                    next_pop.append((U_child, V_child, W_child))
            
            population = next_pop
            
            # Elitism: ensure best is kept (already done in selection)
            
        # End generations for this rank
        
    # If no solution found in GA, return schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
