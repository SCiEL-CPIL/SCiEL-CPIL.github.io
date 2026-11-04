import random
import numpy as np
import time
import sys
from scipy.optimize import milp, LinearConstraint, Bounds

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # Build the target tensor T as a sparse list of (u, v, w) triples
    # T[u, v, w] = 1 if indices match
    tensor_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                tensor_triples.append((u, v, w))
    
    # Also need the "zero" constraints: for all (u,v,w) NOT in tensor_triples, 
    # the sum must be 0.
    # Total number of (u,v,w) combos is nm * mp * pn = 6 * 21 * 14 = 1764
    # Number of ones is n*m*p = 42
    # So we have 42 positive constraints and 1722 negative constraints.
    
    # Strategy:
    # We search for rank r starting from a lower bound.
    # Lower bound for matmul rank: max(nm, mp, pn) = max(6, 21, 14) = 21.
    # Actually, for 2x3x7, the Strassen-like lower bound is higher, but 21 is a safe start.
    # Known lower bounds for 2x3x7:
    # The rank of 2x3x7 is known to be at least 24? Or 25?
    # Let's start from 21 and go up. If we find a solution at r, it's optimal for that r.
    # We assume entries are in {-1, 0, 1} first. If no solution, we might need to expand range,
    # but for small matrices, {-1, 0, 1} often suffices for minimal rank if it exists.
    # However, the problem allows half-integers. 
    # The interface requires integer/half-integer entries.
    # We can scale the problem by 2. Let U', V', W' be 2*U, 2*V, 2*W.
    # Then U', V', W' have integer entries.
    # The equation is sum U'[u,c] V'[v,c] W'[w,c] = 4 * T[u,v,w].
    # This might make the ILP harder due to larger coefficients.
    # Let's stick to integer entries {-1, 0, 1} first. If that fails, we can try {-2, -1, 0, 1, 2}.
    
    # Variables:
    # U has nm * r entries.
    # V has mp * r entries.
    # W has pn * r entries.
    # Total vars = (nm + mp + pn) * r = (6 + 21 + 14) * r = 41 * r.
    # For r=21, vars = 861.
    # For r=25, vars = 1025.
    # This is manageable for MILP if the structure is good.
    
    # Constraint formulation:
    # For each (u, v, w):
    # sum_c U[u,c] * V[v,c] * W[w,c] = T[u,v,w]
    # This is a CUBIC constraint. MILP handles linear constraints.
    # We need to linearize the product U*V*W.
    # This is generally hard.
    
    # Alternative Strategy: Fix U and V, solve for W?
    # If we fix U and V, the equation for each w is:
    # sum_c (U[:,c] * V[:,c])_uv * W[w,c] = T[u,v,w]
    # This is linear in W.
    # So we can use a hybrid approach:
    # 1. Iterate over possible ranks r.
    # 2. Use a local search or simulated annealing on U and V (binary/ternary vectors).
    # 3. For each candidate (U, V), solve for W using linear algebra.
    # 4. Check if W is integer/half-integer.
    
    # This is similar to the current program but with a more structured search on U, V.
    # The current program uses random U, V. We can use a more systematic search, 
    # e.g., hill-climbing on the "integer-ness" of W.
    
    # Let's try a different angle: 
    # The rank of 2x3x7 is known to be 25? 
    # Actually, let's look up the known rank.
    # Rank(2x3x7) is not trivially known.
    # But we can use the "border rank" or just search.
    
    # Let's implement a faster version of the random search but with better heuristics.
    # Heuristic: 
    # 1. Start with r = 24 (a reasonable guess).
    # 2. Generate random U, V with entries in {-1, 0, 1}.
    # 3. Solve for W.
    # 4. If W is close to integer, snap and verify.
    # 5. If not, use the residual to guide a local mutation of U or V.
    
    # However, the directive asks for a MATERIALLY DIFFERENT strategy.
    # "Formulate ... as a linear code ... Berlekamp-Massey" was the suggestion.
    # That is very specific and might not be applicable or efficient here.
    
    # Let's try the **Gröbner Basis / Polynomial Ideal** approach?
    # Or **Tensor Rank via Linear Programming Relaxation**?
    
    # Let's go with a **Structured Random Search with Local Optimization**.
    # The current program is pure random. 
    # I will implement a **Simulated Annealing** on the entries of U and V, 
    # minimizing the distance of W to the nearest half-integer lattice.
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p
    
    def verify_full(U, V, W, rank):
        U_np = np.array(U, dtype=np.int64)
        V_np = np.array(V, dtype=np.int64)
        W_np = np.array(W, dtype=np.int64)
        for w in range(pn):
            M = np.zeros((nm, mp), dtype=np.int64)
            for c in range(rank):
                if W_np[w, c] != 0:
                    M += U_np[:, c] * V_np[:, c] * W_np[w, c]
            if not np.array_equal(M, T[:, :, w]):
                return False
        return True
    
    # Precompute T as a 3D array for fast access in verification
    T = np.zeros((nm, mp, pn), dtype=np.int32)
    for u, v, w in tensor_triples:
        T[u, v, w] = 1
        
    # Precompute U*V outer products for each column c?
    # For a given U, V, the matrix A_w for solving W is:
    # A_w[u*mp+v, c] = U[u,c] * V[v,c]
    
    # Let's try ranks from 24 down to 21? No, rank must be minimal.
    # We don't know the minimal rank.
    # We know rank <= 42 (schoolbook).
    # We know rank >= max(nm, mp, pn) = 21.
    # Let's search r from 21 to 30.
    
    # To make the search faster, we can use the fact that 
    # U, V, W are defined up to a linear transformation of the factors.
    # We can fix some entries to break symmetry.
    # E.g., fix the first row of U to be [1, 0, ..., 0] ? No, that's too restrictive.
    
    # Let's use a **Genetic Algorithm** on U and V.
    
    def evaluate_uv(U, V, r):
        """
        Given U, V, solve for W and return the error (distance to half-integers)
        and the candidate W.
        """
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        
        W_candidates = []
        max_err = 0.0
        
        for w in range(pn):
            # Build A_w
            A_w = np.zeros((nm * mp, r), dtype=np.float64)
            for u in range(nm):
                for v in range(mp):
                    A_w[u * mp + v, :] = U_np[u, :] * V_np[v, :]
            
            b_w = T[:, :, w].flatten().astype(np.float64)
            
            # We need to solve A_w * W_w = b_w
            # Use least squares since it might be overdetermined or inconsistent
            try:
                W_w, residuals, rank_A, sv = np.linalg.lstsq(A_w, b_w, rcond=None)
            except:
                return np.inf, None
            
            # Check if W_w is close to half-integers
            W_w_snapped = np.round(W_w * 2) / 2.0
            err = np.linalg.norm(A_w.dot(W_w_snapped) - b_w)
            if err > 1e-6:
                # Not a valid solution even after snapping
                # But for fitness, we want to know how close it is
                # Actually, if it's not close, the fitness is bad.
                # We can use the distance to the nearest half-integer as fitness
                dist_to_half = np.linalg.norm(W_w - W_w_snapped)
                max_err = max(max_err, dist_to_half)
            else:
                max_err = max(max_err, 0.0)
            
            W_candidates.append(W_w_snapped)
        
        return max_err, W_candidates
    
    # Genetic Algorithm
    def random_uv(r, entries=[-1, 0, 1]):
        U = np.array([[random.choice(entries) for _ in range(r)] for _ in range(nm)])
        V = np.array([[random.choice(entries) for _ in range(r)] for _ in range(mp)])
        return U, V
    
    def mutate(U, V, r, entries=[-1, 0, 1]):
        U_new = U.copy()
        V_new = V.copy()
        # Flip a few random entries
        for _ in range(5):
            if random.random() < 0.5:
                u_idx, c_idx = random.randint(0, nm-1), random.randint(0, r-1)
                U_new[u_idx, c_idx] = random.choice(entries)
            else:
                v_idx, c_idx = random.randint(0, mp-1), random.randint(0, r-1)
                V_new[v_idx, c_idx] = random.choice(entries)
        return U_new, V_new
    
    def crossover(U1, V1, U2, V2, r):
        # Uniform crossover
        U_new = np.where(np.random.rand(nm, r) < 0.5, U1, U2)
        V_new = np.where(np.random.rand(mp, r) < 0.5, V1, V2)
        return U_new, V_new

    # Search loop
    # We will try ranks from 21 to 25.
    # If we find a solution, we return it.
    
    # To speed up, we can precompute the list of (u,v) for each w that are non-zero in T.
    # But lstsq on the full system is robust.
    
    # Optimization: Use integer arithmetic for verification.
    
    for r in range(21, 26):
        if time.time() > deadline - 20:
            break
            
        # Initialize population
        pop_size = 20
        population = []
        for _ in range(pop_size):
            U, V = random_uv(r)
            err, W = evaluate_uv(U, V, r)
            if err < 1e-6:
                # Found a solution!
                # Convert to list of lists of ints
                U_list = U.astype(int).tolist()
                V_list = V.astype(int).tolist()
                W_list = W[0].astype(int).tolist() # Wait, W is a list of arrays
                W_list = [w_arr.astype(int).tolist() for w_arr in W]
                # Verify
                if verify_full(U_list, V_list, W_list, r):
                    return {"U": U_list, "V": V_list, "W": W_list}
            population.append((err, U, V))
        
        population.sort(key=lambda x: x[0])
        
        # Evolution
        max_generations = 50
        for gen in range(max_generations):
            if time.time() > deadline - 10:
                break
                
            # Selection
            parents = population[:pop_size//2]
            
            # Crossover and Mutation
            new_pop = []
            while len(new_pop) < pop_size:
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                U1, V1 = p1[1], p1[2]
                U2, V2 = p2[1], p2[2]
                
                U_new, V_new = crossover(U1, V1, U2, V2, r)
                U_new, V_new = mutate(U_new, V_new, r)
                
                err, W = evaluate_uv(U_new, V_new, r)
                if err < 1e-6:
                    U_list = U_new.astype(int).tolist()
                    V_list = V_new.astype(int).tolist()
                    W_list = [w_arr.astype(int).tolist() for w_arr in W]
                    if verify_full(U_list, V_list, W_list, r):
                        return {"U": U_list, "V": V_list, "W": W_list}
                
                new_pop.append((err, U_new, V_new))
            
            population = new_pop
            population.sort(key=lambda x: x[0])
            
            # If best error is very small, try snapping and verifying more aggressively?
            # The evaluate function already snaps.
            
    # If no better rank found, return schoolbook
    return best
# EVOLVE-BLOCK-END
