"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    # 1. Precomputation
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Check symmetry (tai256c is symmetric)
    a_sym = np.array_equal(a, a.T)
    b_sym = np.array_equal(b, b.T)
    
    def calculate_cost(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    def hungarian_init():
        """
        Generate an initial assignment using the Hungarian algorithm on a surrogate cost matrix.
        Surrogate cost: C[i,j] = sum_k a[i,k]b[j,k] + sum_k a[k,i]b[k,j]
        This is a linearization of the QAP cost.
        """
        # Efficiently compute the linearized cost matrix
        # C[i,j] = sum_k a[i,k] b[j,k] + sum_k a[k,i] b[k,j]
        # Note: a and b are symmetric in this instance, so a[i,k]=a[k,i] and b[j,k]=b[k,j]
        # C[i,j] = 2 * sum_k a[i,k] b[j,k] = 2 * (a @ b)[i,j]
        # If not symmetric, we need the general form.
        if a_sym and b_sym:
             cost_matrix = a @ b
        else:
             # General case:
             # Term 1: sum_k a[i,k] b[j,k] -> (a @ b)[i,j]
             # Term 2: sum_k a[k,i] b[k,j] -> (a.T @ b.T)[i,j] = (b @ a)[i,j]
             cost_matrix = a @ b + b @ a
            
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p

    def hungarian_reopt(p_current):
        """
        Use Hungarian algorithm on a linearized cost matrix centered around current permutation to escape local optima.
        """
        B_p = b[np.ix_(np.arange(n), p_current)]
        # M1[i,j] = sum_k a[i,k]*b[j,p[k]]
        M1 = a @ B_p.T
        Bp2 = b[np.ix_(p_current, np.arange(n))]
        # M2[i,j] = sum_k a[k,i]*b[p[k],j]
        M2 = a.T @ Bp2
        C = M1 + M2
        row_ind, col_ind = linear_sum_assignment(C)
        new_p = np.empty(n, dtype=np.int64)
        new_p[row_ind] = col_ind
        return new_p

    # 2. Initial Solution
    # Start with Hungarian initialization as it's often better than rank matching
    p = hungarian_init()
    best_p = p.copy()
    best_cost = calculate_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    if a_sym and b_sym:
        a_diag = np.diag(a)
        b_diag = np.diag(b)
        
        # Optimized Tabu Search
        tabu = np.zeros((n, n), dtype=np.int32)
        tabu_tenure = 100
        iteration = 0
        
        def calc_delta(i, j, perm):
            u = perm[i]
            v = perm[j]
            
            # Diagonal part
            delta = (a_diag[i] - a_diag[j]) * (b_diag[v] - b_diag[u])
            
            # Off-diagonal part
            Ai = a[i]
            Aj = a[j]
            Bv = b[v, perm]
            Bu = b[u, perm]
            
            diff_A = Ai - Aj
            diff_B = Bv - Bu
            
            # Dot product
            total = np.dot(diff_A, diff_B)
            
            # Subtract terms for i and j
            term_i = (Ai[i] - Aj[i]) * (Bv[i] - Bu[i])
            term_j = (Ai[j] - Aj[j]) * (Bv[j] - Bu[j])
            
            off_diag_sum = total - term_i - term_j
            delta += 2 * off_diag_sum
            
            return int(delta)

        # Phase 1: Iterated Tabu Search with Hungarian Escapes
        # We run Tabu search for a while, then if we stagnate or periodically,
        # we use Hungarian Reopt to jump.
        
        stagnation_counter = 0
        best_cost_for_stagnation = best_cost
        last_hungarian_time = start_time
        
        while time.time() < time_limit - 10.0:
            # Run a chunk of Tabu Search
            chunk_end = time.time() + 2.0
            improved_in_chunk = False
            
            while time.time() < chunk_end:
                improved = False
                start_i = random.randint(0, n-1)
                
                for k in range(n):
                    i = (start_i + k) % n
                    if time.time() > chunk_end:
                        break
                    u = current_p[i]
                    start_j = random.randint(0, n-1)
                    for m in range(n-1):
                        j = (start_j + m) % n
                        if i == j:
                            continue
                        if time.time() > chunk_end:
                            break
                        
                        # Check tabu
                        if tabu[i, j] > iteration:
                            # Aspiration
                            delta = calc_delta(i, j, current_p)
                            if current_cost + delta < best_cost:
                                v = current_p[j]
                                current_p[i], current_p[j] = v, u
                                current_cost += delta
                                tabu[i, j] = iteration + tabu_tenure
                                tabu[j, i] = iteration + tabu_tenure
                                if current_cost < best_cost:
                                    best_cost = current_cost
                                    best_p = current_p.copy()
                                    improved_in_chunk = True
                                improved = True
                                break
                        else:
                            delta = calc_delta(i, j, current_p)
                            if delta < 0:
                                v = current_p[j]
                                current_p[i], current_p[j] = v, u
                                current_cost += delta
                                tabu[i, j] = iteration + tabu_tenure
                                tabu[j, i] = iteration + tabu_tenure
                                if current_cost < best_cost:
                                    best_cost = current_cost
                                    best_p = current_p.copy()
                                    improved_in_chunk = True
                                improved = True
                                break
                    if improved:
                        break
                iteration += 1
                
                if not improved:
                    # Local optimum, small kick
                    i_kick = random.randint(0, n-1)
                    j_kick = random.randint(0, n-1)
                    if i_kick != j_kick:
                        delta = calc_delta(i_kick, j_kick, current_p)
                        v = current_p[j_kick]
                        current_p[i_kick], current_p[j_kick] = v, current_p[i_kick]
                        current_cost += delta
                        tabu[i_kick, j_kick] = iteration + tabu_tenure
                        tabu[j_kick, i_kick] = iteration + tabu_tenure
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_p = current_p.copy()
                            improved_in_chunk = True
            # End Tabu Chunk
            
            # Check for stagnation or time for Hungarian escape
            if not improved_in_chunk:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            
            # If we are stuck for a few chunks, or if it's been > 10s since last reopt, try Hungarian
            if (stagnation_counter >= 3) or (time.time() - last_hungarian_time > 10.0):
                if time.time() < time_limit - 15.0: # Don't do heavy reopt too close to end
                    new_p = hungarian_reopt(current_p)
                    new_cost = calculate_cost(new_p)
                    if new_cost < best_cost + (best_cost * 0.05): # Accept if not too worse, or just accept to explore
                        # Actually, for QAP, even worse solutions can lead to better ones later.
                        # But we must be careful not to waste time.
                        # Let's accept if it's better, or if we are very stuck.
                        if new_cost < best_cost or stagnation_counter > 5:
                            current_p = new_p
                            current_cost = new_cost
                            tabu[:] = 0 # Reset tabu
                            iteration = 0
                            last_hungarian_time = time.time()
                            stagnation_counter = 0
                            # If it was better, update best
                            if new_cost < best_cost:
                                best_cost = new_cost
                                best_p = new_p.copy()

        # Phase 2: Steepest Descent Polish
        # Use remaining time for steepest descent
        while time.time() < time_limit - 2.0:
            best_delta = 0
            best_move = None
            # Evaluate all moves
            for i in range(n):
                u = current_p[i]
                for j in range(i+1, n):
                    if time.time() > time_limit - 2.0:
                        break
                    delta = calc_delta(i, j, current_p)
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, j)
                if time.time() > time_limit - 2.0:
                    break
            
            if best_move is None:
                break
                
            i, j = best_move
            v = current_p[j]
            u = current_p[i]
            current_p[i], current_p[j] = v, u
            current_cost += best_delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
                
    else:
        # Fallback for non-symmetric (unlikely for tai256c but safe)
        # Simple local search
        improved = True
        while improved and time.time() < time_limit - 5.0:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if time.time() > time_limit - 5.0:
                        break
                    old_cost = current_cost
                    current_p[i], current_p[j] = current_p[j], current_p[i]
                    new_cost = calculate_cost(current_p)
                    if new_cost < current_cost:
                        current_cost = new_cost
                        improved = True
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_p = current_p.copy()
                    else:
                        current_p[i], current_p[j] = current_p[j], current_p[i]
                if time.time() > time_limit - 5.0:
                    break

    if sorted(best_p.tolist()) != list(range(n)):
        valid_p = np.arange(n, dtype=np.int64)
        return valid_p.tolist()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
