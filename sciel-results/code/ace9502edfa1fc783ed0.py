"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    # 1. Initial Solution: Rank Matching
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.copy()

    def calculate_cost(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))

    best_p = p.copy()
    best_cost = calculate_cost(best_p)
    current_p = p.copy()
    current_cost = best_cost

    a_sym = np.array_equal(a, a.T)
    b_sym = np.array_equal(b, b.T)
    
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
            # diff_A = a[i, :] - a[j, :]
            # diff_B = b[v, perm] - b[u, perm]
            # total_dot = np.dot(diff_A, diff_B)
            # exclude_i = diff_A[i] * diff_B[i]
            # expire_j = diff_A[j] * diff_B[j]
            # off_diag_sum = total_dot - exclude_i - expire_j
            # delta += 2 * off_diag_sum
            
            # Optimized version to reduce overhead
            Ai = a[i]
            Aj = a[j]
            # b[v, perm] and b[u, perm]
            Bv = b[v, perm]
            Bu = b[u, perm]
            
            diff_A = Ai - Aj
            diff_B = Bv - Bu
            
            # Dot product
            total = np.dot(diff_A, diff_B)
            
            # Subtract terms for i and j
            # Note: perm[i]=u, perm[j]=v
            # diff_A[i] = a[i,i] - a[j,i]
            # diff_B[i] = b[v, u] - b[u, u]
            term_i = (Ai[i] - Aj[i]) * (Bv[i] - Bu[i])
            # diff_A[j] = a[i,j] - a[j,j]
            # diff_B[j] = b[v, v] - b[u, v]
            term_j = (Ai[j] - Aj[j]) * (Bv[j] - Bu[j])
            
            off_diag_sum = total - term_i - term_j
            delta += 2 * off_diag_sum
            
            return int(delta)

        # Phase 1: Tabu Search (First Improvement)
        tabu_end_time = start_time + time_limit * 0.8
        while time.time() < tabu_end_time:
            improved = False
            # Randomize start index to avoid bias
            start_i = random.randint(0, n-1)
            
            for k in range(n):
                i = (start_i + k) % n
                if time.time() > tabu_end_time:
                    break
                u = current_p[i]
                # Randomize start j
                start_j = random.randint(0, n-1)
                for m in range(n-1):
                    j = (start_j + m) % n
                    if i == j:
                        continue
                    if time.time() > tabu_end_time:
                        break
                    
                    # Check tabu
                    if tabu[i, j] > iteration:
                        # Aspiration
                        delta = calc_delta(i, j, current_p)
                        if current_cost + delta < best_cost:
                            # Accept
                            v = current_p[j]
                            current_p[i], current_p[j] = v, u
                            current_cost += delta
                            tabu[i, j] = iteration + tabu_tenure
                            tabu[j, i] = iteration + tabu_tenure
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_p = current_p.copy()
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
                            improved = True
                            break
                
                if improved:
                    break
                    
            iteration += 1
            
            if not improved:
                # Local optimum, kick
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
        pass

    if sorted(best_p.tolist()) != list(range(n)):
        valid_p = np.arange(n, dtype=np.int64)
        return valid_p.tolist()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
