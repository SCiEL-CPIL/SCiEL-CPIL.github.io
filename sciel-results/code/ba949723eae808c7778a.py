"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _qap_cost(a, b, p):
    """Compute exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    C = b[np.ix_(p, p)]
    return int(np.sum(a * C))

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit (killed at 150s, but keep buffer)
    
    # 1. Initial Solution: Rank Matching
    # Facilities with high flow sums should go to locations with low distance sums (central)
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

    # Precompute symmetry flags
    a_sym = np.array_equal(a, a.T)
    b_sym = np.array_equal(b, b.T)
    
    if a_sym and b_sym:
        a_diag = np.diag(a)
        b_diag = np.diag(b)
        
        # Optimized Tabu Search
        # Tabu list: tabu[i, j] stores the iteration until which swapping i and j is forbidden
        # We use a 2D array for O(1) lookup
        tabu = np.zeros((n, n), dtype=np.int32)
        tabu_tenure = 100
        iteration = 0
        
        def calc_delta(i, j, perm):
            u = perm[i]
            v = perm[j]
            
            # Diagonal part
            # Old: a[i,i]*b[u,u] + a[j,j]*b[v,v]
            # New: a[i,i]*b[v,v] + a[j,j]*b[u,u]
            # Delta_diag = a[i,i]*(b[v,v]-b[u,u]) + a[j,j]*(b[u,u]-b[v,v])
            #            = (a[i,i]-a[j,j])*(b[v,v]-b[u,u])
            delta = (a_diag[i] - a_diag[j]) * (b_diag[v] - b_diag[u])
            
            # Off-diagonal part
            # For symmetric a and b, the change in cost for swapping i and j is:
            # sum_{k != i, j} (a[i,k] - a[j,k]) * (b[v, perm[k]] - b[u, perm[k]]) * 2
            # Wait, let's derive carefully.
            # Cost = sum_{k,l} a[k,l] b[perm[k], perm[l]]
            # Swap i, j.
            # Terms involving i or j change.
            # Let's use the vectorized dot product approach which is robust.
            
            Ai = a[i]
            Aj = a[j]
            Bv = b[v, perm]
            Bu = b[u, perm]
            
            diff_A = Ai - Aj
            diff_B = Bv - Bu
            
            # total = sum_k (a[i,k] - a[j,k]) * (b[v, perm[k]] - b[u, perm[k]])
            total = np.dot(diff_A, diff_B)
            
            # We need to subtract the terms where k=i and k=j because they are handled separately or double counted in the derivation?
            # Let's check the expansion.
            # Delta = Sum_{k!=i,j} [ a[i,k]b[v,perm[k]] + a[k,i]b[perm[k],v] + a[j,k]b[u,perm[k]] + a[k,j]b[perm[k],u]
            #             - a[i,k]b[u,perm[k]] - a[k,i]b[perm[k],u] - a[j,k]b[v,perm[k]] - a[k,j]b[perm[k],v] ]
            #       + [a[i,j]b[v,u] + a[j,i]b[u,v] + a[i,i]b[v,v] + a[j,j]b[u,u]
            #          - a[i,j]b[u,v] - a[j,i]b[v,u] - a[i,i]b[u,u] - a[j,j]b[v,v]]
            
            # If a and b are symmetric:
            # a[i,k]=a[k,i], a[j,k]=a[k,j], b[perm[k],v]=b[v,perm[k]], etc.
            # Term for k (k!=i,j):
            # (a[i,k]-a[j,k])b[v,perm[k]] + (a[i,k]-a[j,k])b[v,perm[k]]
            # - (a[i,k]-a[j,k])b[u,perm[k]] - (a[i,k]-a[j,k])b[u,perm[k]]
            # = 2 (a[i,k]-a[j,k]) (b[v,perm[k]] - b[u,perm[k]])
            
            # So the sum over k != i, j is 2 * sum_{k!=i,j} (a[i,k]-a[j,k]) (b[v,perm[k]] - b[u,perm[k]])
            
            # The term calculated by np.dot(diff_A, diff_B) is sum_{k=0}^{n-1} (a[i,k]-a[j,k]) (b[v,perm[k]] - b[u,perm[k]])
            
            # This sum includes k=i and k=j.
            # Let's isolate k=i and k=j.
            # k=i: (a[i,i]-a[j,i]) * (b[v,perm[i]] - b[u,perm[i]])
            #      perm[i] = u
            #      = (a[i,i]-a[j,i]) * (b[v,u] - b[u,u])
            
            # k=j: (a[i,j]-a[j,j]) * (b[v,perm[j]] - b[u,perm[j]])
            #      perm[j] = v
            #      = (a[i,j]-a[j,j]) * (b[v,v] - b[u,v])
            
            term_i = (Ai[i] - Aj[i]) * (Bv[i] - Bu[i])
            term_j = (Ai[j] - Aj[j]) * (Bv[j] - Bu[j])
            
            off_diag_sum = total - term_i - term_j
            delta += 2 * off_diag_sum
            
            # Now add the specific i-j interaction terms and self terms?
            # The formula above 2 * off_diag_sum accounts for k != i, j.
            # We still need the terms where k=i and l=j, k=j and l=i, k=i and l=i, k=j and l=j.
            # Let's re-evaluate the "specific" part.
            # Old: a[i,j]b[u,v] + a[j,i]b[v,u] + a[i,i]b[u,u] + a[j,j]b[v,v]
            # New: a[i,j]b[v,u] + a[j,i]b[u,v] + a[i,i]b[v,v] + a[j,j]b[u,u]
            # Delta_specific = a[i,j](b[v,u]-b[u,v]) + a[j,i](b[u,v]-b[v,u]) + a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v])
            # Since symmetric:
            # a[i,j](b[v,u]-b[u,v]) - a[i,j](b[v,u]-b[u,v]) = 0 ? No.
            # a[j,i] = a[i,j], b[u,v]=b[v,u].
            # a[i,j](b[v,u]-b[u,v]) + a[i,j](b[u,v]-b[v,u]) = 0.
            # So the cross terms cancel out for symmetric matrices?
            # Let's check:
            # Old cross: a[i,j]b[u,v] + a[j,i]b[v,u] = 2 a[i,j]b[u,v]
            # New cross: a[i,j]b[v,u] + a[j,i]b[u,v] = 2 a[i,j]b[v,u] = 2 a[i,j]b[u,v]
            # Yes, they are identical. So delta_cross = 0.
            
            # So we only need the self terms which we already added in `delta` variable at the start?
            # delta = (a[i,i]-a[j,j])*(b[v,v]-b[u,u])
            # This is exactly a[i,i](b[v,v]-b[u,u]) + a[j,j](b[u,u]-b[v,v]).
            # Yes.
            
            return int(delta)

        # Phase 1: Tabu Search (First Improvement)
        tabu_end_time = start_time + time_limit * 0.7
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
                        # Aspiration: accept if it improves global best
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
                # Perform a random move to escape
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
        # Use remaining time for steepest descent to fine-tune
        polish_deadline = start_time + time_limit - 2.0
        while time.time() < polish_deadline:
            best_delta = 0
            best_move = None
            # Evaluate all moves
            for i in range(n):
                u = current_p[i]
                for j in range(i+1, n):
                    if time.time() > polish_deadline:
                        break
                    delta = calc_delta(i, j, current_p)
                    if delta < best_delta:
                        best_delta = delta
                        best_move = (i, j)
                if time.time() > polish_deadline:
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
        # Fallback for non-symmetric case: simple random swap local search
        # This is less efficient but handles the general case
        while time.time() < time_limit - 2.0:
            improved = False
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i != j:
                u = current_p[i]
                v = current_p[j]
                
                # General delta calculation
                # Re-compute cost is O(n^2) which is slow, but n=256 is okay if we don't do it too often?
                # Better to calculate delta directly.
                # delta = sum_{k,l} a[k,l] (b[p'_k, p'_l] - b[p_k, p_l])
                # Only k or l in {i, j} changes.
                
                # Let's just recompute cost for simplicity in fallback, or use the vectorized trick if we can adapt it.
                # Actually, for non-symmetric, the vectorized trick is harder.
                # Given time constraints and that QAPLIB instances are often symmetric or we can assume structure,
                # let's try a fast approximate delta or just accept the slowness of full recompute if it's rare.
                # But full recompute is 65536 ops. 1000 iterations = 65M ops. ~0.1s.
                # So we can do ~1000-2000 iterations.
                
                new_p = current_p.copy()
                new_p[i], new_p[j] = v, u
                new_cost = calculate_cost(new_p)
                
                if new_cost < current_cost:
                    current_p = new_p
                    current_cost = new_cost
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = current_p.copy()
                
                if not improved and random.random() < 0.01:
                    # Restart from best
                    current_p = best_p.copy()
                    current_cost = best_cost

    if sorted(best_p.tolist()) != list(range(n)):
        valid_p = np.arange(n, dtype=np.int64)
        return valid_p.tolist()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
