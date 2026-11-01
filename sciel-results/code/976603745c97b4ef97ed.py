import numpy as np
import time
import itertools
import sympy as sp

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Schoolbook decomposition (rank 12)
    rank_school = n * m * p
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    best_rank = rank_school

    def get_target_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T
    
    T_target = get_target_tensor()
    
    def verify_exact(U, V, W):
        """
        Verify if U, V, W (lists of lists of Fractions or numbers) 
        exactly decompose the target tensor.
        """
        nm, mp, pn = n*m, m*p, p*n
        rank = len(U[0])
        
        # Check shapes
        if len(U) != nm or len(V) != mp or len(W) != pn:
            return False
        if any(len(row) != rank for row in U): return False
        if any(len(row) != rank for row in V): return False
        if any(len(row) != rank for row in W): return False

        # Compute tensor
        T_prime = np.zeros((nm, mp, pn), dtype=object)
        for r in range(rank):
            u_col = [row[r] for row in U]
            v_col = [row[r] for row in V]
            w_col = [row[r] for row in W]
            
            # Outer product
            for i in range(nm):
                ui = u_col[i]
                if ui == 0: continue
                for j in range(mp):
                    vj = v_col[j]
                    if vj == 0: continue
                    prod = ui * vj
                    if prod == 0: continue
                    for k in range(pn):
                        wk = w_col[k]
                        T_prime[i, j, k] += prod * wk

        # Compare with target
        for i in range(nm):
            for j in range(mp):
                for k in range(pn):
                    target_val = 1 if (i // m == (k // n)) and (i % m == (j // p)) and (j % p == (k % n)) else 0
                    # Actually, simpler check:
                    # T[i,j,k] = 1 if i,j,k correspond to (row, col, out) where out = row * B_col? 
                    # Wait, the target tensor definition in the seed:
                    # T[a, b, c] = 1.0 where a=i*m+j, b=j*p+k, c=k*n+i
                    # This implies A[i,j] * B[j,k] contributes to C[k,i].
                    # So T[i*m+j, j*p+k, k*n+i] = 1.
                    
                    # Let's just check against the precomputed T_target
                    if T_prime[i, j, k] != T_target[i, j, k]:
                        return False
        return True

    def try_rank_exact(rank):
        """
        Try to find an exact decomposition of given rank using Groebner basis.
        We iterate over all possible support patterns (which entries are non-zero).
        """
        nm, mp, pn = n*m, m*p, p*n
        
        # We will use sympy variables
        # For efficiency, we only consider supports where each tensor column has exactly one non-zero entry?
        # No, that's too restrictive. But for rank 9-11, the supports are likely sparse.
        # Actually, a known strategy for small matrix multiplication is that the tensors are often "simple" or have specific structures.
        # However, brute forcing all supports is 2^(nm*rank). 
        # nm=4, rank=9 -> 2^36 supports. Too many.
        
        # Alternative: Use a known lower bound or specific structure.
        # For 2x2x2, rank 7 (Strassen).
        # For 2x2x3, let's look for rank 10 or 11.
        
        # Instead of iterating all supports, we can use a randomized approach with exact arithmetic?
        # Or we can use the fact that if a solution exists, it might be found by fixing some variables to 0/1 and solving the rest.
        
        # Let's try a different exact approach: 
        # Assume the decomposition uses integer/half-integer entries.
        # We can set up the equations and use sympy's solve on a subset of variables.
        
        # Actually, the most robust "different" strategy that fits the 120s budget for this small problem:
        # Use `sympy.solvers.solveset` or `groebner` on a system derived from a random linear projection?
        # No, Groebner basis on the full system for rank 10 is likely too slow (explosion of polynomials).
        
        # Let's try a hybrid: 
        # 1. Use the current DE/L-BFGS to find a *candidate* solution.
        # 2. If DE finds a solution with low error, extract the support (non-zero pattern).
        # 3. Fix that support and solve the exact system using Groebner basis.
        # 4. If Groebner finds a rational solution, verify it.
        
        # This combines the heuristic search for the "shape" with exact solving for the "values".
        
        pass # Placeholder, logic implemented in main loop

    # We need to find a candidate first to get the support.
    # Let's reuse the DE part from the seed to get a candidate, but then switch to exact solving.
    
    from scipy.optimize import differential_evolution, minimize
    
    def compute_tensor(U, V, W):
        return np.einsum('ir,jr,kr->ijk', U, V, W)

    def objective(params, target_rank):
            U_flat = params[:nm*target_rank]
            V_flat = params[nm*target_rank:nm*target_rank + mp*target_rank]
            W_flat = params[nm*target_rank + mp*target_rank:]
            
            U = U_flat.reshape(nm, target_rank)
            V = V_flat.reshape(mp, target_rank)
            W = W_flat.reshape(pn, target_rank)
            
            T_prime = compute_tensor(U, V, W)
            diff = T_prime - T_target
            return np.sum(diff**2)

    found = False
    
    # Try ranks 10 and 11. Rank 9 is theoretically possible but harder.
    # Given 2x2x2 is 7, 2x2x3 is likely 10 or 11.
    target_ranks = [10, 11]
    
    for target_rank in target_ranks:
        if found or time.time() > deadline - 20.0:
            break
        
        nm, mp, pn = n*m, m*p, p*n
        total_params = (nm + mp + pn) * target_rank
        
        # Step 1: Heuristic search for a good candidate and its support
        bounds = [(-5, 5)] * total_params
        
        # Use DE to find a candidate
        # Limit time for DE to 30s per rank
        de_start = time.time()
        try:
            res = differential_evolution(
                lambda p: objective(p, target_rank), 
                bounds, 
                maxiter=500, 
                tol=1e-8, 
                popsize=30,
                mutation=(0.5, 1.0),
                crossover=0.9,
                seed=42,
                polish=False
            )
            
            if res.fun < 1e-4:
                # Candidate found!
                U_flat = res.x[:nm*target_rank]
                V_flat = res.x[nm*target_rank:nm*target_rank + mp*target_rank]
                W_flat = res.x[nm*target_rank + mp*target_rank:]
                
                U_num = U_flat.reshape(nm, target_rank)
                V_num = V_flat.reshape(mp, target_rank)
                W_num = W_flat.reshape(pn, target_rank)
                
                # Determine support: entries with magnitude > 0.1
                thresh = 0.1
                U_support = np.abs(U_num) > thresh
                V_support = np.abs(V_num) > thresh
                W_support = np.abs(W_num) > thresh
                
                # Step 2: Exact solving with fixed support
                # We define variables only for the non-zero entries
                var_names = []
                var_map = {} # (tensor, i, j) -> sympy symbol
                symbols = []
                
                for t_name, supp in [('U', U_support), ('V', V_support), ('W', W_support)]:
                    rows = supp.shape[0]
                    cols = supp.shape[1]
                    for i in range(rows):
                        for j in range(cols):
                            if supp[i, j]:
                                sym = sp.Symbol(f"{t_name}_{i}_{j}")
                                var_map[(t_name, i, j)] = sym
                                symbols.append(sym)
                
                if len(symbols) == 0:
                    continue
                
                # Build equations
                # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r] - T_target[i,j,k] = 0
                equations = []
                
                # To keep the system small, we can select a subset of equations if the system is overdetermined?
                # No, for Groebner basis, we need enough equations to determine the variables.
                # The number of non-zero entries in T_target is n*m*p = 12.
                # The tensor has nm*mp*pn = 4*6*6 = 144 entries.
                # Most are 0.
                # We can use the 12 non-zero equations and a few zero equations to constrain the solution.
                
                # Actually, using all 144 equations might be too slow for Groebner.
                # Let's use the 12 non-zero ones and a random subset of 12 zero ones to fix the gauge?
                # Or just the 12 non-zero ones. If the solution is unique up to scaling, we might need more.
                # But tensor rank decomposition is not unique.
                # However, if we fix the support, the solution might be unique or finite.
                
                # Let's try with just the non-zero entries of the target tensor first.
                nz_indices = np.argwhere(T_target != 0)
                
                for idx in nz_indices:
                    i, j, k = idx
                    eq = 0
                    for r in range(target_rank):
                        u_sym = var_map.get(('U', i, r), 0)
                        v_sym = var_map.get(('V', j, r), 0)
                        w_sym = var_map.get(('W', k, r), 0)
                        eq += u_sym * v_sym * w_sym
                    equations.append(eq - 1)
                
                # Add some zero equations to break symmetry and reduce solution space
                # Pick a few random zero entries
                zero_indices = np.argwhere(T_target == 0)
                np.random.shuffle(zero_indices)
                for idx in zero_indices[:10]:
                    i, j, k = idx
                    eq = 0
                    for r in range(target_rank):
                        u_sym = var_map.get(('U', i, r), 0)
                        v_sym = var_map.get(('V', j, r), 0)
                        w_sym = var_map.get(('W', k, r), 0)
                        eq += u_sym * v_sym * w_sym
                    equations.append(eq)
                
                # Solve
                try:
                    # Use Groebner basis
                    G = sp.groebner(equations, *symbols, order='lex')
                    sols = sp.solve(G, *symbols, dict=True)
                    
                    if sols:
                        # Check if any solution has rational numbers
                        for sol in sols:
                            # Convert to fractions
                            U_sol = [[0]*target_rank for _ in range(nm)]
                            V_sol = [[0]*target_rank for _ in range(mp)]
                            W_sol = [[0]*target_rank for _ in range(pn)]
                            
                            valid = True
                            for (t_name, i, j), sym in var_map.items():
                                val = sol.get(sym, 0)
                                if val == 0:
                                    continue
                                # Check if val is rational
                                if not val.is_rational:
                                    valid = False
                                    break
                                if t_name == 'U':
                                    U_sol[i][j] = val
                                elif t_name == 'V':
                                    V_sol[i][j] = val
                                else:
                                    W_sol[i][j] = val
                            
                            if valid:
                                # Verify exactness
                                if verify_exact(U_sol, V_sol, W_sol):
                                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                                    best_rank = target_rank
                                    found = True
                                    break
                        if found:
                            break
                except Exception as e:
                    pass # Groebner might be too slow or fail
        
        except Exception as e:
            pass

    return {"U": best_U, "V": best_V, "W": best_W}
