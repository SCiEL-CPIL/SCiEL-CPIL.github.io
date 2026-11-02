import numpy as np

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

    # Precompute einsum indices
    # U: (n, m, rank), V: (m, p, rank), W: (p, n, rank)
    # S[a, b, c, d] = sum_r U[a, b, r] V[c, d, r] W[d, a, r]
    # Note: V index in problem is c*p+d -> V[c,d]. W index is d*n+a -> W[d,a].
    
    def compute_error(U_flat, V_flat, W_flat, rank):
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        S = np.einsum('abr,cdr,dar->abcd', U, V, W)
        return np.sum((S - T) ** 2)

    def compute_rounding_penalty(U_flat, V_flat, W_flat):
        # Penalty for not being close to half-integers
        U = U_flat
        V = V_flat
        W = W_flat
        pU = np.sum((U - np.round(U * 2) / 2) ** 2)
        pV = np.sum((V - np.round(V * 2) / 2) ** 2)
        pW = np.sum((W - np.round(W * 2) / 2) ** 2)
        return pU + pV + pW

    def compute_l1(U_flat, V_flat, W_flat):
        return np.sum(np.abs(U_flat)) + np.sum(np.abs(V_flat)) + np.sum(np.abs(W_flat))

    # Genetic Algorithm Parameters
    pop_size = 50
    generations = 200
    mutation_rate = 0.1
    crossover_rate = 0.8
    
    # Try ranks from 20 to 35
    for rank in range(20, 36):
        # Initialize population
        # Use sparse random initialization to help sparsity
        population = []
        for _ in range(pop_size):
            U = np.random.randn(n * m * rank)
            V = np.random.randn(m * p * rank)
            W = np.random.randn(p * n * rank)
            
            # Sparsify: keep only ~20-30% non-zeros
            mask_U = np.random.rand(n * m * rank) < 0.25
            mask_V = np.random.rand(m * p * rank) < 0.25
            mask_W = np.random.rand(p * n * rank) < 0.25
            
            U[~mask_U] = 0
            V[~mask_V] = 0
            W[~mask_W] = 0
            
            # Scale to reasonable magnitude
            U *= 1.0
            V *= 1.0
            W *= 1.0
            
            population.append([U, V, W])

        best_individual = None
        best_fitness = float('inf')

        for gen in range(generations):
            new_population = []
            
            # Evaluate fitness
            fitnesses = []
            for ind in population:
                U, V, W = ind
                err = compute_error(U, V, W, rank)
                if err < 1e-6:
                    # Check rounding
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    err_r = compute_error(U_r, V_r, W_r, rank)
                    if err_r < 1e-6:
                        # Found a valid solution!
                        return {
                            "U": U_r.reshape(n*m, rank).tolist(),
                            "V": V_r.reshape(m*p, rank).tolist(),
                            "W": W_r.reshape(p*n, rank).tolist()
                        }
                
                # Fitness: Error + Rounding Penalty + L1
                r_pen = compute_rounding_penalty(U, V, W)
                l1_pen = compute_l1(U, V, W) * 0.01
                fitness = err + 10.0 * r_pen + l1_pen
                fitnesses.append(fitness)

            # Sort population by fitness
            indices = np.argsort(fitnesses)
            population = [population[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]
            
            if fitnesses[0] < best_fitness:
                best_fitness = fitnesses[0]
                best_individual = population[0]
            
            # Early stop if very good
            if best_fitness < 1e-4:
                # Try to polish the best individual
                U, V, W = best_individual
                # Local refinement with rounding
                for _ in range(50):
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    err_r = compute_error(U_r, V_r, W_r, rank)
                    if err_r < 1e-6:
                        return {
                            "U": U_r.reshape(n*m, rank).tolist(),
                            "V": V_r.reshape(m*p, rank).tolist(),
                            "W": W_r.reshape(p*n, rank).tolist()
                        }
                    # Add small noise to escape local minima
                    U += np.random.randn(n * m * rank) * 0.01
                    V += np.random.randn(m * p * rank) * 0.01
                    W += np.random.randn(p * n * rank) * 0.01
                    U = np.where(np.abs(U) < 0.1, 0, U)
                    V = np.where(np.abs(V) < 0.1, 0, V)
                    W = np.where(np.abs(W) < 0.1, 0, W)

            # Elitism: Keep top 10%
            elite_count = max(1, pop_size // 10)
            new_population.extend(population[:elite_count])
            
            # Generate offspring
            while len(new_population) < pop_size:
                # Tournament selection
                def select():
                    i = np.random.randint(0, pop_size, 3)
                    return population[i[np.argmin(fitnesses[i])]]
                
                parent1 = select()
                parent2 = select()
                
                if np.random.rand() < crossover_rate:
                    # Uniform crossover
                    child_U = np.random.choice([parent1[0], parent2[0]], size=len(parent1[0]))
                    child_V = np.random.choice([parent1[1], parent2[1]], size=len(parent1[1]))
                    child_W = np.random.choice([parent1[2], parent2[2]], size=len(parent1[2]))
                else:
                    child_U = parent1[0].copy()
                    child_V = parent1[1].copy()
                    child_W = parent1[2].copy()
                
                # Mutation
                if np.random.rand() < mutation_rate:
                    # Gaussian mutation
                    child_U += np.random.randn(len(child_U)) * 0.5
                    child_V += np.random.randn(len(child_V)) * 0.5
                    child_W += np.random.randn(len(child_W)) * 0.5
                    # Sparsify
                    child_U = np.where(np.abs(child_U) < 0.2, 0, child_U)
                    child_V = np.where(np.abs(child_V) < 0.2, 0, child_V)
                    child_W = np.where(np.abs(child_W) < 0.2, 0, child_W)
                
                new_population.append([child_U, child_V, child_W])
            
            population = new_population

    # Fallback to schoolbook if GA fails
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
