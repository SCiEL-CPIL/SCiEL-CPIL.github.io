import numpy as np
from itertools import product
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor: T[a, b, c, d] = 1 if b == c else 0
    T = np.zeros((n, m, m, p))
    for b in range(m):
        for c in range(m):
            if b == c:
                T[:, b, c, :] = 1.0
    
    # Try ranks from 20 to 36
    for rank in range(20, 37):
        # Genetic algorithm to find sparse binary support patterns
        # Each individual is a tuple of three bitmasks (for U, V, W columns)
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        
        # We'll work with the actual factor matrices but enforce sparsity
        # and try to find solutions with integer/half-integer values
        
        for restart in range(50):
            # Initialize population of random sparse matrices
            pop_size = 20
            population = []
            
            for _ in range(pop_size):
                U = np.random.randint(0, 2, (n * m, rank))
                V = np.random.randint(0, 2, (m * p, rank))
                W = np.random.randint(0, 2, (p * n, rank))
                
                # Make sparse: keep ~30% of entries
                U[np.random.rand(n * m, rank) < 0.7] = 0
                V[np.random.rand(m * p, rank) < 0.7] = 0
                W[np.random.rand(p * n, rank) < 0.7] = 0
                
                # Ensure at least one non-zero per column
                for j in range(rank):
                    if U[:, j].sum() == 0:
                        U[random.randint(0, n*m-1), j] = 1
                    if V[:, j].sum() == 0:
                        V[random.randint(0, m*p-1), j] = 1
                    if W[:, j].sum() == 0:
                        W[random.randint(0, p*n-1), j] = 1
                
                population.append((U, V, W))
            
            # Evolution loop
            for gen in range(200):
                # Evaluate fitness
                fitnesses = []
                for U, V, W in population:
                    U_r = U.reshape(n, m, rank)
                    V_r = V.reshape(m, p, rank)
                    W_r = W.reshape(p, n, rank)
                    S = np.einsum('abr,cdr,dar->abcd', U_r, V_r, W_r)
                    err = np.sum((S - T) ** 2)
                    fitnesses.append(err)
                
                # Check for solution
                for i, err in enumerate(fitnesses):
                    if err < 1e-10:
                        U, V, W = population[i]
                        # Convert to half-integers (they're already integers)
                        return {
                            "U": U.tolist(),
                            "V": V.tolist(),
                            "W": W.tolist()
                        }
                
                # Select best individuals
                sorted_idx = np.argsort(fitnesses)
                elite = [population[i] for i in sorted_idx[:pop_size // 2]]
                
                # Create new population
                new_population = list(elite)
                
                while len(new_population) < pop_size:
                    # Crossover
                    if random.random() < 0.7:
                        parent1 = random.choice(elite)
                        parent2 = random.choice(elite)
                        
                        # Uniform crossover
                        U1, V1, W1 = parent1
                        U2, V2, W2 = parent2
                        
                        U = np.where(np.random.rand(n * m, rank) < 0.5, U1, U2)
                        V = np.where(np.random.rand(m * p, rank) < 0.5, V1, V2)
                        W = np.where(np.random.rand(p * n, rank) < 0.5, W1, W2)
                    else:
                        # Clone
                        parent = random.choice(elite)
                        U, V, W = parent[0].copy(), parent[1].copy(), parent[2].copy()
                    
                    # Mutation
                    if random.random() < 0.8:
                        # Flip random bits
                        for _ in range(random.randint(1, 3)):
                            if random.random() < 0.33:
                                i, j = random.randint(0, n*m-1), random.randint(0, rank-1)
                                U[i, j] = 1 - U[i, j]
                            elif random.random() < 0.5:
                                i, j = random.randint(0, m*p-1), random.randint(0, rank-1)
                                V[i, j] = 1 - V[i, j]
                            else:
                                i, j = random.randint(0, p*n-1), random.randint(0, rank-1)
                                W[i, j] = 1 - W[i, j]
                    
                    # Ensure at least one non-zero per column
                    for j in range(rank):
                        if U[:, j].sum() == 0:
                            U[random.randint(0, n*m-1), j] = 1
                        if V[:, j].sum() == 0:
                            V[random.randint(0, m*p-1), j] = 1
                        if W[:, j].sum() == 0:
                            W[random.randint(0, p*n-1), j] = 1
                    
                    new_population.append((U, V, W))
                
                population = new_population
            
            # End evolution loop
        
        # End restart loop
    
    # If no solution found, return schoolbook
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
# EVOLVE-BLOCK-END
