import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def check_half_integer_solution(U, V, R):
        """
        Given U and V (lists of lists of numbers), checks if there exists W
        with half-integer entries such that the decomposition is exact.
        Returns W if found, else None.
        """
        W = [[0] * R for _ in range(pn)]
        
        # We need to solve M_{ik} w_{ik} = 1 for each i, k
        # M_{ik} is m x R. M_{ik}[j, r] = U[i*m+j][r] * V[j*p+k][r]
        # We solve using numpy lstsq, then check if solution is close to half-integers.
        # To be robust, we can also try to solve exactly if R is small enough or use fractions.
        # Given R <= 56, m=4, lstsq is fast.
        
        for i in range(n):
            for k in range(p):
                M = np.zeros((m, R))
                for j in range(m):
                    for r in range(R):
                        M[j, r] = U[i*m + j][r] * V[j*p + k][r]
                
                try:
                    # Solve M w = 1
                    w, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    
                    # Check residual
                    if residuals is not None and residuals[0] > 1e-6:
                        return None
                    
                    # Check if w is close to half-integers
                    w_check = [x * 2 for x in w]
                    for val in w_check:
                        if abs(val - round(val)) > 1e-6:
                            return None
                    
                    # Snap to half-integers
                    for r in range(R):
                        W[k*n + i][r] = round(w[r] * 2) / 2.0
                        
                except Exception:
                    return None
        
        # Final verification to ensure no floating point drift caused a false positive
        if verify_decomposition(U, V, W, R):
            return W
        return None

    def random_sparse_init(R, density=0.3, max_val=2):
        """
        Generates random U and V with sparse integer/half-integer entries.
        """
        U = [[0] * R for _ in range(nm)]
        V = [[0] * R for _ in range(mp)]
        
        # Use a set of allowed values: -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2
        # To keep it simple and exact, let's use integers for U and V, 
        # which implies W will be rational. If W ends up half-integer, we are good.
        # Actually, if U, V are integers, M is integer. 
        # If M is invertible (square case) or has a unique solution, W might be rational.
        # We specifically need W to be half-integer.
        
        values = [-2, -1, 0, 1, 2]
        
        for i in range(nm):
            for r in range(R):
                if random.random() < density:
                    U[i][r] = random.choice(values)
        
        for j in range(mp):
            for r in range(R):
                if random.random() < density:
                    V[j][r] = random.choice(values)
                    
        return U, V

    def search_rank(R, time_limit):
        """
        Attempts to find a valid decomposition of rank R by random sampling.
        """
        while time.time() < time_limit:
            U, V = random_sparse_init(R)
            
            # Quick check: ensure the system is not trivially singular
            # We just try to solve
            W = check_half_integer_solution(U, V, R)
            
            if W is not None:
                return {"U": U, "V": V, "W": W}
                
        return None

    def search_rank_local_search(R, time_limit):
        """
        Starts with random sparse init, then does local search on U and V.
        Goal: Minimize the "distance to half-integer solution" for W.
        """
        U, V = random_sparse_init(R, density=0.5)
        
        def compute_cost(U, V):
            """
            Computes cost based on residuals and non-half-integer-ness of W.
            """
            cost = 0.0
            for i in range(n):
                for k in range(p):
                    M = np.zeros((m, R))
                    for j in range(m):
                        for r in range(R):
                            M[j, r] = U[i*m + j][r] * V[j*p + k][r]
                    try:
                        w, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        if residuals is not None:
                            cost += residuals[0]
                        else:
                            cost += 1e6
                        
                        # Penalty for non-half-integer W
                        for val in w:
                            diff = abs(val * 2 - round(val * 2))
                            cost += diff * 10.0
                    except Exception:
                        cost += 1e6
            return cost

        current_cost = compute_cost(U, V)
        best_cost = current_cost
        best_U = [row[:] for row in U]
        best_V = [row[:] for row in V]
        
        T = 1.0
        cooling = 0.999
        values = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        
        while time.time() < time_limit:
            # Propose move
            if random.random() < 0.5:
                i = random.randint(0, nm - 1)
                r = random.randint(0, R - 1)
                old_val = U[i][r]
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                U[i][r] = new_val
                changed = ('U', i, r, old_val)
            else:
                j = random.randint(0, mp - 1)
                r = random.randint(0, R - 1)
                old_val = V[j][r]
                new_val = random.choice(values)
                if new_val == old_val:
                    continue
                V[j][r] = new_val
                changed = ('V', j, r, old_val)
            
            new_cost = compute_cost(U, V)
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_U = [row[:] for row in U]
                    best_V = [row[:] for row in V]
                    
                    if best_cost < 1e-6:
                        W = check_half_integer_solution(best_U, best_V, R)
                        if W is not None:
                            return {"U": best_U, "V": best_V, "W": W}
            else:
                if changed[0] == 'U':
                    U[changed[1]][changed[2]] = changed[3]
                else:
                    V[changed[1]][changed[2]] = changed[3]
            
            T *= cooling
            if T < 1e-4:
                T = 1e-4
        
        # Final check
        if best_cost < 1e-6:
            W = check_half_integer_solution(best_U, best_V, R)
            if W is not None:
                return {"U": best_U, "V": best_V, "W": W}
        return None

    best_result = schoolbook()
    best_rank = n * m * p
    
    min_possible_rank = max(nm, mp, pn)
    
    # Try ranks from lower bound up to schoolbook
    # We will allocate time to lower ranks first as they are more valuable
    for R in range(min_possible_rank, best_rank):
        if time.time() > deadline - 5.0:
            break
        
        # Allocate time: give more time to lower ranks
        # If R is close to lower bound, it's harder, give more time?
        # Or just fixed time per rank.
        time_limit = time.time() + 8.0
        
        # First try pure random sampling (fast)
        result = search_rank(R, time_limit - 4.0)
        if result is not None:
            best_result = result
            best_rank = R
            break
            
        # Then try local search (slower)
        result = search_rank_local_search(R, time_limit)
        if result is not None:
            best_result = result
            best_rank = R
            break
            
    return best_result
