import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

def _build_tensor():
    """Build the 4x4x8 multiplication tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i."""
    n, m, p = 4, 4, 8
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify(U, V, W, T):
    """Check if U, V, W exactly decompose T."""
    # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T_recon, T, atol=1e-9)

def _to_half_integer(mat):
    """Convert matrix to half-integer representation if possible."""
    mat2 = mat * 2
    rounded = np.round(mat2)
    if np.allclose(mat2, rounded, atol=1e-6):
        return (rounded / 2).tolist()
    rounded_int = np.round(mat)
    if np.allclose(mat, rounded_int, atol=1e-6):
        return rounded_int.astype(int).tolist()
    return mat.tolist()

def _schoolbook():
    """Return the schoolbook rank-128 decomposition."""
    n, m, p = 4, 4, 8
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen (rank 7)."""
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (A00+A11)(B00+B11)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[3,0] = 1
    W[0,0] = 1; W[3,0] = 1; W[1,0] = -1; W[2,0] = 1
    
    # M2 = (A10+A11)B00
    U[2,1] = 1; U[3,1] = 1
    V[0,1] = 1
    W[2,1] = 1; W[3,1] = -1
    
    # M3 = A00(B10-B11)
    U[0,2] = 1
    V[2,2] = 1; V[3,2] = -1
    W[0,2] = -1; W[1,2] = 1
    
    # M4 = A11(B01-B00)
    U[3,3] = 1
    V[1,3] = 1; V[0,3] = -1
    W[1,3] = -1; W[3,3] = 1
    
    # M5 = (A00+A01)B11
    U[0,4] = 1; U[1,4] = 1
    V[3,4] = 1
    W[0,4] = 1; W[1,4] = -1; W[3,4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2,5] = 1; U[0,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[2,5] = -1; W[3,5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1,6] = 1; U[3,6] = -1
    V[2,6] = 1; V[3,6] = 1
    W[0,6] = 1; W[2,6] = -1
    
    return U, V, W

def _build_4x4x4_49():
    """Build 4x4x4 rank-49 decomposition via Strassen recursion."""
    U2, V2, W2 = _strassen_2x2x2()
    U4 = np.zeros((16, 49), dtype=np.float64)
    V4 = np.zeros((16, 49), dtype=np.float64)
    W4 = np.zeros((16, 49), dtype=np.float64)
    
    def offset(b):
        if b == 0: return 0
        if b == 1: return 2
        if b == 2: return 8
        if b == 3: return 10
        return 0

    A_blocks = [
        [(0, 1), (3, 1)],
        [(2, 1), (3, 1)],
        [(0, 1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (0, -1)],
        [(1, 1), (3, -1)]
    ]
    B_blocks = [
        [(0, 1), (3, 1)],
        [(0, 1)],
        [(2, 1), (3, -1)],
        [(1, 1), (0, -1)],
        [(3, 1)],
        [(0, 1), (1, 1)],
        [(2, 1), (3, 1)]
    ]
    C_blocks = [
        [(0, 1), (3, 1), (1, -1), (2, 1)],
        [(2, 1), (3, -1)],
        [(0, -1), (1, 1)],
        [(1, -1), (3, 1)],
        [(0, 1), (1, -1), (3, 1)],
        [(2, -1), (3, 1)],
        [(0, 1), (2, -1)]
    ]
    
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            u_vals = U2[:, s]
            v_vals = V2[:, s]
            w_vals = W2[:, s]
            
            for i_local in range(4):
                val = u_vals[i_local]
                if val != 0:
                    for bA, sA in A_blocks[t]:
                        g_idx = offset(bA) + i_local
                        U4[g_idx, col] += val * sA
            
            for j_local in range(4):
                val = v_vals[j_local]
                if val != 0:
                    for bB, sB in B_blocks[t]:
                        g_idx = offset(bB) + j_local
                        V4[g_idx, col] += val * sB
            
            for k_local in range(4):
                val = w_vals[k_local]
                if val != 0:
                    for bC, sC in C_blocks[t]:
                        g_idx = offset(bC) + k_local
                        W4[g_idx, col] += val * sC
                        
    return U4, V4, W4

def _build_4x4x8_98():
    """Build 4x4x8 rank-98 decomposition via two 4x4x4 blocks."""
    U4, V4, W4 = _build_4x4x4_49()
    U = np.zeros((16, 98), dtype=np.float64)
    V = np.zeros((32, 98), dtype=np.float64)
    W = np.zeros((32, 98), dtype=np.float64)
    
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    U[:, 49:98] = U4
    for j in range(4):
        for k_local in range(4):
            v_idx = j * 4 + k_local
            v_target = j * 8 + k_local + 4
            V[v_target, 49:98] = V4[v_idx, 0:49]
            
    for k_local in range(4):
        for i in range(4):
            w_idx = k_local * 4 + i
            w_target = (k_local + 4) * 4 + i
            W[w_target, 49:98] = W4[w_idx, 0:49]
            
    return U, V, W

def _ga_search(T, target_rank, pop_size=50, generations=100, deadline=None):
    """
    Genetic algorithm search for a rank-r decomposition.
    Each individual is a set of r terms, each term being (u, v, w) sparse vectors.
    We use a structured initialization from the 98-rank solution and mutate by:
    1. Removing a term
    2. Adding a new random sparse term
    3. Mutating coefficients
    """
    if deadline is None:
        deadline = time.time() + 120
    
    n, m, p = 4, 4, 8
    dim_a = n * m  # 16
    dim_b = m * p  # 32
    dim_c = p * n  # 32
    
    # Start from the 98-rank solution and try to compress
    U98, V98, W98 = _build_4x4x8_98()
    
    # Initialize population: start with 98-rank, then randomly remove terms
    population = []
    
    # Individual 0: the 98-rank solution
    population.append((U98.copy(), V98.copy(), W98.copy()))
    
    # Other individuals: remove random terms from 98-rank
    for _ in range(pop_size - 1):
        U_ind = U98.copy()
        V_ind = V98.copy()
        W_ind = W98.copy()
        
        # Remove a random subset of columns
        num_remove = np.random.randint(0, 42)  # Try to get down to ~56
        cols_to_remove = np.random.choice(98, size=num_remove, replace=False)
        cols_to_keep = np.setdiff1d(np.arange(98), cols_to_remove)
        
        U_ind = U98[:, cols_to_keep]
        V_ind = V98[:, cols_to_keep]
        W_ind = W98[:, cols_to_keep]
        
        # Mutate some coefficients
        for _ in range(10):
            r = np.random.randint(U_ind.shape[1])
            a = np.random.randint(dim_a)
            U_ind[a, r] += np.random.choice([-1, -0.5, 0.5, 1])
            b = np.random.randint(dim_b)
            V_ind[b, r] += np.random.choice([-1, -0.5, 0.5, 1])
            c = np.random.randint(dim_c)
            W_ind[c, r] += np.random.choice([-1, -0.5, 0.5, 1])
        
        population.append((U_ind, V_ind, W_ind))
    
    best_rank = 98
    best_solution = (U98, V98, W98)
    
    def fitness(U, V, W):
        """Lower is better: number of terms + reconstruction error."""
        rank = U.shape[1]
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.sum((T_recon - T)**2)
        # Penalize non-exact solutions heavily
        if error > 1e-6:
            return rank + error * 1000
        return rank
    
    for gen in range(generations):
        if time.time() > deadline - 5:
            break
        
        # Evaluate fitness
        fitnesses = [fitness(U, V, W) for U, V, W in population]
        best_idx = np.argmin(fitnesses)
        best_fit = fitnesses[best_idx]
        
        if best_fit < best_rank:
            best_rank = int(best_fit)
            best_solution = (population[best_idx][0].copy(), 
                            population[best_idx][1].copy(), 
                            population[best_idx][2].copy())
        
        # Check if we found an exact solution
        U_best, V_best, W_best = population[best_idx]
        if _verify(U_best, V_best, W_best, T):
            return U_best, V_best, W_best
        
        # Selection: keep top half
        sorted_indices = np.argsort(fitnesses)
        survivors = [population[i] for i in sorted_indices[:pop_size//2]]
        
        # Crossover and mutation to fill population
        new_population = list(survivors)
        while len(new_population) < pop_size:
            if time.time() > deadline - 5:
                break
            
            # Pick two parents
            i1 = np.random.randint(len(survivors))
            i2 = np.random.randint(len(survivors))
            U1, V1, W1 = survivors[i1]
            U2, V2, W2 = survivors[i2]
            
            # Crossover: take columns from each parent
            min_rank = min(U1.shape[1], U2.shape[1])
            if min_rank == 0:
                continue
            
            cross_point = np.random.randint(1, min_rank + 1)
            U_child = np.hstack([U1[:, :cross_point], U2[:, cross_point:]])
            V_child = np.hstack([V1[:, :cross_point], V2[:, cross_point:]])
            W_child = np.hstack([W1[:, :cross_point], W2[:, cross_point:]])
            
            # Mutation
            for _ in range(5):
                if U_child.shape[1] == 0:
                    break
                r = np.random.randint(U_child.shape[1])
                a = np.random.randint(dim_a)
                U_child[a, r] += np.random.choice([-1, -0.5, 0.5, 1])
                b = np.random.randint(dim_b)
                V_child[b, r] += np.random.choice([-1, -0.5, 0.5, 1])
                c = np.random.randint(dim_c)
                W_child[c, r] += np.random.choice([-1, -0.5, 0.5, 1])
            
            new_population.append((U_child, V_child, W_child))
        
        population = new_population
    
    return best_solution

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Build the tensor
    T = _build_tensor()
    
    # Fallback: Schoolbook (rank 128)
    U_school, V_school, W_school = _schoolbook()
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = 128
    
    # Try 98-rank constructive solution
    if time.time() < deadline - 10:
        try:
            U98, V98, W98 = _build_4x4x8_98()
            if _verify(U98, V98, W98, T):
                best_U, best_V, best_W = U98, V98, W98
                best_rank = 98
        except Exception:
            pass
    
    # Try GA search for lower rank
    if time.time() < deadline - 20:
        try:
            U_ga, V_ga, W_ga = _ga_search(T, target_rank=56, pop_size=30, generations=50, deadline=deadline - 10)
            if _verify(U_ga, V_ga, W_ga, T):
                rank_ga = U_ga.shape[1]
                if rank_ga < best_rank:
                    best_U, best_V, best_W = U_ga, V_ga, W_ga
                    best_rank = rank_ga
        except Exception:
            pass
    
    # Convert to half-integer representation
    U_out = _to_half_integer(best_U)
    V_out = _to_half_integer(best_V)
    W_out = _to_half_integer(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
