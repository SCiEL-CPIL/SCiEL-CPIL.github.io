"""Rewritten QAP solver using RC/CC matrix incremental 2-opt with tabu search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 146.0  # Safety margin for 150s limit

    # Ensure arrays are int64 for exact integer arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    # Spectral initialization
    try:
        A_sym = ((a + a.T) / 2.0).astype(np.float64)
        B_sym = ((b + b.T) / 2.0).astype(np.float64)
        
        # Use top eigenvectors
        eigvals_A, eigvecs_A = np.linalg.eigh(A_sym)
        eigvals_B, eigvecs_B = np.linalg.eigh(B_sym)
        
        v_A = eigvecs_A[:, np.argmax(eigvals_A)]
        v_B = eigvecs_B[:, np.argmax(eigvals_B)]
        
        # Also try rank matching as alternative and pick better
        fac_sorted = np.argsort(v_A)
        loc_sorted = np.argsort(v_B)
        p = np.empty(n, dtype=np.int64)
        p[fac_sorted] = loc_sorted
        
        # Alternative: reverse match
        p2 = np.empty(n, dtype=np.int64)
        p2[fac_sorted] = loc_sorted[::-1]
        
        cost1 = int(np.sum(a * b[p]))
        cost2 = int(np.sum(a * b[p2]))
        
        if cost2 < cost1:
            p = p2
            cost = cost2
        else:
            cost = cost1
            
    except Exception:
        # Fallback to rank matching
        a_sums = a.sum(axis=1)
        b_sums = b.sum(axis=1)
        heavy_idx = np.argsort(-a_sums, kind="stable")
        central_idx = np.argsort(b_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_idx] = central_idx
        cost = int(np.sum(a * b[p]))

    best_cost = cost
    best_p = p.copy()

    # Build RC and CC matrices
    # RC[i, l] = sum_j a[i, j] b[l, p[j]]
    # CC[k, l] = sum_i a[i, k] b[p[i], l]
    
    def build_RC_CC(p, a, b):
        """Build RC and CC matrices from scratch."""
        # B_p[j, l] = b[p[j], l]
        B_p = b[p, :]  # Shape (n, n): row j is b[p[j], :]
        # RC[i, l] = sum_j a[i, j] b[l, p[j]]
        # Note: b[l, p[j]] = b.T[p[j], l] if not symmetric, but b is symmetric so b[p[j], l]
        # Actually we need b[l, p[j]], which is element (l, p[j]) of b.
        # b[p[j], l] is element (p[j], l). Since b is symmetric, these are equal.
        # But let's be careful: we want RC[i, l] = sum_j a[i, j] * b[l, p[j]]
        # b[l, p[j]] = b.T[p[j], l]
        # So RC = a @ B_p.T where B_p[j, l] = b[p[j], l]
        # B_p.T[l, j] = b[p[j], l]
        # So RC = a @ B_p.T
        
        RC = a.astype(np.float64) @ B_p.T.astype(np.float64)
        
        # CC[k, l] = sum_i a[i, k] b[p[i], l]
        # This is B_p @ a.T
        # B_p[i, l] = b[p[i], l], a.T[k, i] = a[i, k]
        # So CC = B_p @ a.T
        CC = B_p.astype(np.float64) @ a.T.astype(np.float64)
        
        return RC.astype(np.int64), CC.astype(np.int64)

    RC, CC = build_RC_CC(p, a, b)

    # Tabu list
    tabu_len = 30
    tabu = {}  # (i, j) -> iteration when it becomes non-tabu
    current_iter = 0

    # Local search with tabu
    while time.time() - start_time < time_limit:
        # Find best non-tabu move
        best_delta = 0
        best_i, best_j = -1, -1
        
        # Sample random pairs to speed up
        n_samples = min(2000, n * (n - 1) // 2)
        sampled = False
        
        for _ in range(n_samples):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            
            key = (min(i, j), max(i, j))
            if key in tabu and tabu[key] > current_iter:
                continue
            
            u = p[i]
            v = p[j]
            
            if u == v:
                continue
            
            # Delta cost for swapping i and j
            # delta = (a[i,i] + a[j,j] - 2*a[i,j]) * (b[u,v]) 
            #       - RC[i, u] - RC[j, v] + RC[i, v] + RC[j, u]
            #       - CC[u, j] - CC[v, i] + CC[u, i] + CC[v, j]
            # Actually let me re-derive:
            # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
            # When we swap p[i]=u and p[j]=v:
            # Terms involving i or j change.
            # delta = sum_k a[i,k](b[v,p[k]] - b[u,p[k]]) + a[j,k](b[u,p[k]] - b[v,p[k]))
            #       + sum_k a[k,i](b[p[k],v] - b[p[k],u]) + a[k,j](b[p[k],u] - b[p[k],v])
            #       + a[i,j](b[v,u] - b[u,v]) + a[j,i](b[u,v] - b[v,u])  [these cancel]
            #       + a[i,i](b[v,v] - b[u,u]) + a[j,j](b[u,u] - b[v,v])
            
            # Using RC and CC:
            # RC[i, l] = sum_j a[i,j] b[l, p[j]]
            # So sum_k a[i,k] b[v, p[k]] - sum_k a[i,k] b[u, p[k]] = RC[i, v] - RC[i, u]
            # Similarly for j.
            # CC[k, l] = sum_i a[i,k] b[p[i], l]
            # So sum_k a[k,i] b[p[k], v] - sum_k a[k,i] b[p[k], u] = CC[i, v] - CC[i, u]
            # Wait, CC is indexed by the facility index in a and location in b.
            # CC[k, l] = sum_i a[i, k] b[p[i], l]
            # So sum_k a[k, i] b[p[k], v] = CC[i, v]
            # And sum_k a[k, i] b[p[k], u] = CC[i, u]
            
            delta = (RC[i, v] - RC[i, u]) + (RC[j, u] - RC[j, v]) + (CC[i, v] - CC[i, u]) + (CC[j, u] - CC[j, v])
            
            if delta < best_delta:
                best_delta = delta
                best_i, best_j = i, j
                sampled = True
        
        if not sampled or best_i == -1:
            # No improving move found, try full scan occasionally
            if current_iter % 50 == 0:
                for i in range(n):
                    for j in range(i + 1, n):
                        key = (i, j)
                        if key in tabu and tabu[key] > current_iter:
                            continue
                        u = p[i]
                        v = p[j]
                        if u == v:
                            continue
                        delta = (RC[i, v] - RC[i, u]) + (RC[j, u] - RC[j, v]) + (CC[i, v] - CC[i, u]) + (CC[j, u] - CC[j, v])
                        if delta < best_delta:
                            best_delta = delta
                            best_i, best_j = i, j
            
            if best_i == -1:
                # Stuck, restart with perturbation
                p = best_p.copy()
                cost = best_cost
                RC, CC = build_RC_CC(p, a, b)
                # Perturb
                for _ in range(n // 10):
                    i1 = random.randint(0, n - 1)
                    i2 = random.randint(0, n - 1)
                    if i1 != i2:
                        u, v = p[i1], p[i2]
                        p[i1], p[i2] = v, u
                        # Update cost
                        cost = int(np.sum(a * b[p]))
                        RC, CC = build_RC_CC(p, a, b)
                continue
        
        # Apply best move
        i, j = best_i, best_j
        u, v = p[i], p[j]
        p[i], p[j] = v, u
        cost += int(best_delta)
        
        # Update tabu
        key = (min(i, j), max(i, j))
        tabu[key] = current_iter + tabu_len
        
        # Update RC and CC incrementally
        # RC[i, :] changes, RC[j, :] changes
        # For RC[k, l] where k != i, j:
        # RC[k, l] = sum_j a[k, j] b[l, p[j]]
        # Only terms j=i and j=j change.
        # Old: ... + a[k,i] b[l,u] + a[k,j] b[l,v]
        # New: ... + a[k,i] b[l,v] + a[k,j] b[l,u]
        # Delta RC[k, l] = a[k,i](b[l,v] - b[l,u]) + a[k,j](b[l,u] - b[l,v])
        #                = (a[k,i] - a[k,j])(b[l,v] - b[l,u])
        
        delta_a_col = a[:, i] - a[:, j]  # Shape (n,)
        delta_b_row = b[:, v] - b[:, u]  # Shape (n,)
        RC += np.outer(delta_a_col, delta_b_row)
        
        # Similarly for CC
        # CC[k, l] = sum_i a[i, k] b[p[i], l]
        # Only i=i and i=j change.
        # Delta CC[k, l] = (a[i,k] - a[j,k])(b[v,l] - b[u,l])
        delta_a_row = a[i, :] - a[j, :]  # Shape (n,)
        delta_b_col = b[v, :] - b[u, :]  # Shape (n,)
        CC += np.outer(delta_a_row, delta_b_col)
        
        current_iter += 1
        
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
            # Reset tabu on improvement
            tabu.clear()

    return best_p.tolist()

# EVOLVE-BLOCK-END
