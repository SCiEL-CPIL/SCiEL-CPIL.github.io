# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_cost(p):
        return int(np.sum(a * b[np.ix_(p, p)]))

    # 1. Linearized Greedy Construction
    # Build permutation sequentially, maintaining dynamic cost matrix C[i,j]
    # C[i,j] = marginal cost of assigning facility i to location j given current partial assignment
    # When facility k is placed at location l, for any future (i,j):
    # C[i,j] += A[i,k]*B[j,l] + A[k,i]*B[l,j]
    
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    
    p_greedy = np.full(n, -1, dtype=np.int64)
    assigned_fac = np.zeros(n, dtype=bool)
    assigned_loc = np.zeros(n, dtype=bool)
    
    # C[i,j] accumulates the interaction cost with already-assigned facilities
    C = np.zeros((n, n), dtype=np.float64)
    
    for step in range(n):
        if time.time() > deadline - 60.0:
            break
        
        # Find best (i,j): minimize C[i,j] among unassigned
        C_masked = C.copy()
        C_masked[assigned_fac, :] = np.inf
        C_masked[:, assigned_loc] = np.inf
        
        min_val = np.min(C_masked)
        idx = np.argmin(C_masked)
        i_best = idx // n
        j_best = idx % n
        
        # Assign facility i_best to location j_best
        p_greedy[i_best] = j_best
        assigned_fac[i_best] = True
        assigned_loc[j_best] = True
        
        # Update C for future assignments:
        # C[i,j] += A[i,i_best]*B[j,j_best] + A[i_best,i]*B[j_best,j]
        # Term 1: outer(A[:, i_best], B[:, j_best]) gives [i,j] = A[i,i_best]*B[j,j_best]
        # Term 2: outer(A[i_best, :], B[j_best, :]) gives [i,j] = A[i_best,i]*B[j_best,j]
        
        C += np.outer(A[:, i_best], B[:, j_best]) + np.outer(A[i_best, :], B[j_best, :])
    
    # If greedy didn't complete, fill remaining
    if np.any(p_greedy < 0):
        remaining_fac = np.where(p_greedy < 0)[0]
        remaining_loc = np.where(~assigned_loc)[0]
        if len(remaining_fac) > 0 and len(remaining_loc) > 0:
            p_greedy[remaining_fac] = remaining_loc
    
    p_init = p_greedy.copy()
    current_cost = compute_cost(p_init)
    best_cost = current_cost
    best_p = p_init.copy()
    
    # Also try rank matching as alternative initial
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    fac_order = np.argsort(-a_row_sums, kind="stable")
    loc_order = np.argsort(b_row_sums, kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[fac_order] = loc_order
    cost_rank = compute_cost(p_rank)
    if cost_rank < best_cost:
        best_cost = cost_rank
        best_p = p_rank.copy()
    
    # 2. Local Search: Vectorized 2-Swap
    p = best_p.copy()
    current_cost = best_cost

    # Precompute b_p = b[p, :] and b_pT = b[:, p] for faster access
    b_p = b[p, :]  # Shape (n, n): b_p[i, j] = b[p[i], j]
    b_pT = b[:, p] # Shape (n, n): b_pT[i, j] = b[i, p[j]]

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1

        for i in range(n):
            if time.time() > deadline - 5.0:
                break

            u = p[i]
            A_i = a[i, :]
            A_iT = a[:, i]

            # Precompute differences for this i
            # diff1[k] = b[v, p[k]] - b[u, p[k]] where v = p[j]
            # diff2[k] = b[p[k], v] - b[p[k], u]

            for j in range(i + 1, n):
                v = p[j]
                A_j = a[j, :]
                A_jT = a[:, j]

                D1 = A_i - A_j
                D2 = A_iT - A_jT

                # Vectorized sums
                diff1 = b[v, p] - b[u, p]
                diff2 = b[p, v] - b[p, u]

                Sum1_all = D1 @ diff1
                Sum2_all = D2 @ diff2

                # Subtract k=i and k=j terms
                # k=i:
                Sub1_i = D1[i] * (b[v, u] - b[u, u])
                Sub2_i = D2[i] * (b[u, v] - b[u, u])

                # k=j:
                Sub1_j = D1[j] * (b[v, v] - b[u, v])
                Sub2_j = D2[j] * (b[v, v] - b[v, u])

                Sum1 = Sum1_all - Sub1_i - Sub1_j
                Sum2 = Sum2_all - Sub2_i - Sub2_j

                Cross = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])

                delta = Cross + Sum1 + Sum2

                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j

        if best_delta < -1e-10:
            # Perform swap
            i, j = best_i, best_j
            p[i], p[j] = p[j], p[i]
            current_cost += best_delta

            # Update b_p and b_pT
            u, v = p[i], p[j]
            b_p[i, :] = b[v, :]
            b_p[j, :] = b[u, :]
            b_pT[:, i] = b[:, v]
            b_pT[:, j] = b[:, u]

            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
            improved = True
        else:
            break

    # 4. Simulated Annealing Phase
    # Use remaining time for SA to escape local minima
    sa_deadline = deadline - 2.0
    if time.time() < sa_deadline:
        p = best_p.copy()
        current_cost = best_cost
        b_p = b[p, :]
        b_pT = b[:, p]

        # Temperature schedule
        T_start = max(1.0, abs(best_cost) * 0.01)
        T_end = 1e-6
        cooling_rate = 0.995

        T = T_start
        rng = random.Random(42)

        while T > T_end and time.time() < sa_deadline:
            # Pick random pair
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue

            u, v = p[i], p[j]

            # Compute delta
            A_i = a[i, :]
            A_iT = a[:, i]
            A_j = a[j, :]
            A_jT = a[:, j]

            D1 = A_i - A_j
            D2 = A_iT - A_jT

            diff1 = b[v, p] - b[u, p]
            diff2 = b[p, v] - b[p, u]

            Sum1_all = D1 @ diff1
            Sum2_all = D2 @ diff2

            Sub1_i = D1[i] * (b[v, u] - b[u, u])
            Sub2_i = D2[i] * (b[u, v] - b[u, u])
            Sub1_j = D1[j] * (b[v, v] - b[u, v])
            Sub2_j = D2[j] * (b[v, v] - b[v, u])

            Sum1 = Sum1_all - Sub1_i - Sub1_j
            Sum2 = Sum2_all - Sub2_i - Sub2_j

            Cross = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])
            delta = Cross + Sum1 + Sum2

            # Accept or reject
            if delta < 0 or rng.random() < np.exp(-delta / T):
                p[i], p[j] = p[j], p[i]
                current_cost += delta

                # Update b_p and b_pT
                u, v = p[i], p[j]
                b_p[i, :] = b[v, :]
                b_p[j, :] = b[u, :]
                b_pT[:, i] = b[:, v]
                b_pT[:, j] = b[:, u]

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()

            T *= cooling_rate

    return best_p.tolist()
# EVOLVE-BLOCK-END
