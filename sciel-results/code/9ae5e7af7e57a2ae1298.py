# EVOLVE-BLOCK-START
import numpy as np
import time


def _total_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def solve(a, b):
    n = len(a)
    deadline = time.time() + 110.0  # safety margin under 120s

    # --- Build a good initial solution via greedy construction ---
    # We build the assignment one facility at a time, placing each facility
    # at the location that minimizes the marginal cost given already-placed facilities.
    # Start from the rank-matching heuristic, then refine with a greedy pass.

    # Precompute
    a_rows = a  # a[i,:]
    b_mat = b

    # Initial: rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central

    # Greedy improvement pass: for each facility in order, try all locations and pick best
    # This is O(n^3) but n=100 so 1e6 * small constant, fine.
    # Actually let's do a few passes of "for each i, try all j, keep if better" (steepest descent)
    best_cost = _total_cost(a, b, p)

    # Precompute b rows/cols for fast delta
    # For swapping u and v, we need b[lv, p[j]] etc.
    
    # Steepest descent improvement (a few passes)
    for _pass in range(3):
        improved = False
        for u in range(n):
            if time.time() > deadline - 30:
                break
            lu = p[u]
            best_v = -1
            best_delta = 0
            for v in range(u + 1, n):
                lv = p[v]
                # Compute delta for swapping u and v
                # delta = sum over affected terms
                # Vectorized:
                # Terms: i=u row, i=v row, j=u col, j=v col, with care for double count
                # Use the formula:
                # delta = (a[u,:] @ (b[lv,:] - b[lu,:])) 
                #       + (a[v,:] @ (b[lu,:] - b[lv,:]))
                #       + (a[:,u] @ (b[:,lv] - b[:,lu]))
                #       + (a[:,v] @ (b[:,lu] - b[:,lv]))
                #       - corrections for double-counted (u,v),(v,u),(u,u),(v,v)
                # The four sums above each include the cross terms. Let's see:
                # a[u,:] @ (b[lv,:]-b[lu,:]) includes j=u: a[u,u]*(b[lv,u]-b[lu,u]) and j=v: a[u,v]*(b[lv,v]-b[lu,v])
                # a[v,:] @ (b[lu,:]-b[lv,:]) includes j=u: a[v,u]*(b[lu,u]-b[lv,u]) and j=v: a[v,v]*(b[lu,v]-b[lv,v])
                # a[:,u] @ (b[:,lv]-b[:,lu]) includes i=u: a[u,u]*(b[u,lv]-b[u,lu]) and i=v: a[v,u]*(b[v,lv]-b[v,lu])
                # a[:,v] @ (b[:,lu]-b[:,lv]) includes i=u: a[u,v]*(b[u,lu]-b[u,lv]) and i=v: a[v,v]*(b[v,lu]-b[v,lv])
                # So (u,u) appears in sum1 (j=u) and sum3 (i=u): 
                #   sum1 j=u: a[u,u]*(b[lv,u]-b[lu,u])
                #   sum3 i=u: a[u,u]*(b[u,lv]-b[u,lu])
                #   Together: a[u,u]*(b[lv,u]+b[u,lv]-b[lu,u]-b[u,lu])
                #   But true change for (u,u): a[u,u]*(b[lv,lv]-b[lu,lu])
                #   So we have extra: a[u,u]*(b[lv,u]+b[u,lv]-b[lu,u]-b[u,lu] - b[lv,lv]+b[lu,lu])
                # This is getting messy. Let me just compute delta directly by summing affected terms.
                
                # Direct computation:
                # Affected pairs: (u,u),(u,v),(v,u),(v,v) and (u,j),(v,j),(i,u),(i,v) for j,i not in {u,v}
                # delta = a[u,u]*(b[lv,lv]-b[lu,lu]) 
                #       + a[u,v]*(b[lv,lu]-b[lu,lv])
                #       + a[v,u]*(b[lu,lv]-b[lv,lu])
                #       + a[v,v]*(b[lu,lu]-b[lv,lv])
                #       + a[u,:].dot(b[lv,:]) - a[u,:].dot(b[lu,:])  # but this includes j=u,v
                #       ...
                # Let me just do: delta = (new cost of affected) - (old cost of affected)
                # Affected set S = {(i,j): i in {u,v} or j in {u,v}}
                # new_cost_S = sum_{(i,j) in S} a[i,j]*b[p'[i],p'[j]]
                # old_cost_S = sum_{(i,j) in S} a[i,j]*b[p[i],p[j]]
                # where p' is p with u,v swapped.
                
                # Vectorized approach: 
                # Let's compute the change as:
                # For i=u: p[u] changes lu->lv, so contribution change = sum_j a[u,j]*(b[lv,p[j]]-b[lu,p[j]])
                #   = a[u,:].dot(b[lv,p]) - a[u,:].dot(b[lu,p])
                # For i=v: p[v] changes lv->lu, so change = a[v,:].dot(b[lu,p]) - a[v,:].dot(b[lv,p])
                # For j=u: p[u] changes lu->lv, so change = sum_i a[i,u]*(b[p[i],lv]-b[p[i],lu])
                #   = a[:,u].dot(b[p,lv]) - a[:,u].dot(b[p,lu])
                # For j=v: p[v] changes lv->lu, so change = a[:,v].dot(b[p,lu]) - a[:,v].dot(b[p,lv])
                # Total = [a[u,:].dot(b[lv,p]) - a[u,:].dot(b[lu,p])]
                #       + [a[v,:].dot(b[lu,p]) - a[v,:].dot(b[lv,p])]
                #       + [a[:,u].dot(b[p,lv]) - a[:,u].dot(b[p,lu])]
                #       + [a[:,v].dot(b[p,lu]) - a[:,v].dot(b[p,lv])]
                # But this double counts the terms where both i and j are in {u,v}:
                # (u,u): counted in "i=u" (j=u term) and "j=u" (i=u term)
                #   i=u, j=u: a[u,u]*(b[lv,p[u]]-b[lu,p[u]]) = a[u,u]*(b[lv,lu]-b[lu,lu])
                #   j=u, i=u: a[u,u]*(b[p[u],lv]-b[p[u],lu]) = a[u,u]*(b[lu,lv]-b[lu,lu])
                #   Sum: a[u,u]*(b[lv,lu]+b[lu,lv]-2*b[lu,lu])
                #   True change for (u,u): a[u,u]*(b[lv,lv]-b[lu,lu])
                #   Extra: a[u,u]*(b[lv,lu]+b[lu,lv]-b[lu,lu]-b[lv,lv]) = a[u,u]*(b[lv,lu]-b[lv,lv]+b[lu,lv]-b[lu,lu])
                # Similarly for (u,v), (v,u), (v,v).
                # 
                # To avoid this, let me just compute the four corner terms separately and the rest vectorized.
                
                # Corners:
                d_uu = a[u,u] * (b[lv,lv] - b[lu,lu])
                d_uv = a[u,v] * (b[lv,lu] - b[lu,lv])
                d_vu = a[v,u] * (b[lu,lv] - b[lv,lu])
                d_vv = a[v,v] * (b[lu,lu] - b[lv,lv])
                
                # Row u (excluding j=u,v):
                # sum_{j not in {u,v}} a[u,j]*(b[lv,p[j]]-b[lu,p[j]])
                # = a[u,:].dot(b[lv,p]) - a[u,:].dot(b[lu,p]) - a[u,u]*(b[lv,lu]-b[lu,lu]) - a[u,v]*(b[lv,lv]-b[lu,lv])
                row_u = a[u,:].dot(b[lv,p]) - a[u,:].dot(b[lu,p]) - a[u,u]*(b[lv,lu]-b[lu,lu]) - a[u,v]*(b[lv,lv]-b[lu,lv])
                # Row v (excluding j=u,v):
                row_v = a[v,:].dot(b[lu,p]) - a[v,:].dot(b[lv,p]) - a[v,u]*(b[lu,lu]-b[lv,lu]) - a[v,v]*(b[lu,lv]-b[lv,lv])
                # Col u (excluding i=u,v):
                col_u = a[:,u].dot(b[p,lv]) - a[:,u].dot(b[p,lu]) - a[u,u]*(b[lu,lv]-b[lu,lu]) - a[v,u]*(b[lv,lv]-b[lv,lu])
                # Col v (excluding i=u,v):
                col_v = a[:,v].dot(b[p,lu]) - a[:,v].dot(b[p,lv]) - a[u,v]*(b[lu,lu]-b[lu,lv]) - a[v,v]*(b[lv,lu]-b[lv,lv])
                
                delta = d_uu + d_uv + d_vu + d_vv + row_u + row_v + col_u + col_v
                
                if delta < best_delta:
                    best_delta = delta
                    best_v = v
            
            if best_v >= 0 and best_delta < -1e-9:
                # Perform swap
                p[u], p[best_v] = p[best_v], p[u]
                best_cost += int(best_delta)
                improved = True
        if not improved:
            break

    # --- Simulated annealing ---
    # Precompute for fast delta
    a_rows = a  # (n,n)
    b_rows = b  # (n,n)
    
    # Current state
    cur_p = p.copy()
    cur_cost = best_cost
    
    # Temperature schedule
    T0 = max(1.0, abs(cur_cost) * 0.001)
    Tend = 1e-6
    # Number of iterations: we'll run until deadline
    alpha = 0.9999  # cooling rate
    
    T = T0
    rng = np.random.default_rng(42)
    
    # Precompute b[p,:] as a matrix for fast access: Bp[i] = b[p[i],:]
    # And b[:,p[i]]: BpT[i] = b[:,p[i]]
    # Actually, for delta we need b[lu, p[j]] for all j, which is b[lu, cur_p] (a vector).
    # And b[cur_p, lu] (a vector).
    # Let's precompute row_sums and col_sums? No, we need the full vectors.
    
    # For speed, precompute:
    # a_rows[i] = a[i,:] (already a)
    # a_cols[i] = a[:,i]
    a_cols = a.T  # a_cols[i] = a[:,i]
    
    # Precompute b_row[l] = b[l,:], b_col[l] = b[:,l]
    b_cols = b.T
    
    # For each step, pick random u,v, compute delta, accept if delta<0 or with prob exp(-delta/T)
    
    iter_count = 0
    while time.time() < deadline - 5.0:
        # Batch: do several swaps per time check for efficiency
        batch = 100
        for _ in range(batch):
            if time.time() > deadline - 5.0:
                break
            u = rng.integers(0, n)
            v = rng.integers(0, n)
            if u == v:
                continue
            lu = cur_p[u]
            lv = cur_p[v]
            
            # Compute delta
            d_uu = a[u,u] * (b[lv,lv] - b[lu,lu])
            d_uv = a[u,v] * (b[lv,lu] - b[lu,lv])
            d_vu = a[v,u] * (b[lu,lv] - b[lv,lu])
            d_vv = a[v,v] * (b[lu,lu] - b[lv,lv])
            
            row_u = a[u,:].dot(b[lv,cur_p]) - a[u,:].dot(b[lu,cur_p]) - a[u,u]*(b[lv,lu]-b[lu,lu]) - a[u,v]*(b[lv,lv]-b[lu,lv])
            row_v = a[v,:].dot(b[lu,cur_p]) - a[v,:].dot(b[lv,cur_p]) - a[v,u]*(b[lu,lu]-b[lv,lu]) - a[v,v]*(b[lu,lv]-b[lv,lv])
            col_u = a_cols[u].dot(b[lu,cur_p]) - a_cols[u].dot(b[lv,cur_p]) - a[u,u]*(b[lu,lv]-b[lu,lu]) - a[v,u]*(b[lv,lv]-b[lv,lu])
            col_v = a_cols[v].dot(b[lv,cur_p]) - a_cols[v].dot(b[lu,cur_p]) - a[u,v]*(b[lu,lu]-b[lu,lv]) - a[v,v]*(b[lv,lu]-b[lv,lv])
            
            delta = d_uu + d_uv + d_vu + d_vv + row_u + row_v + col_u + col_v
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                # Accept
                cur_p[u], cur_p[v] = lv, lu
                cur_cost += int(delta)
            
            iter_count += 1
        
        # Cool
        T *= alpha
        if T < Tend:
            T = Tend
    
    # Final steepest descent pass
    for _pass in range(5):
        improved = False
        for u in range(n):
            if time.time() > deadline - 2.0:
                break
            lu = cur_p[u]
            best_v = -1
            best_delta = 0
            for v in range(u + 1, n):
                lv = cur_p[v]
                d_uu = a[u,u] * (b[lv,lv] - b[lu,lu])
                d_uv = a[u,v] * (b[lv,lu] - b[lu,lv])
                d_vu = a[v,u] * (b[lu,lv] - b[lv,lu])
                d_vv = a[v,v] * (b[lu,lu] - b[lv,lv])
                row_u = a[u,:].dot(b[lv,cur_p]) - a[u,:].dot(b[lu,cur_p]) - a[u,u]*(b[lv,lu]-b[lu,lu]) - a[u,v]*(b[lv,lv]-b[lu,lv])
                row_v = a[v,:].dot(b[lu,cur_p]) - a[v,:].dot(b[lv,cur_p]) - a[v,u]*(b[lu,lu]-b[lv,lu]) - a[v,v]*(b[lu,lv]-b[lv,lv])
                col_u = a_cols[u].dot(b[lu,cur_p]) - a_cols[u].dot(b[lv,cur_p]) - a[u,u]*(b[lu,lv]-b[lu,lu]) - a[v,u]*(b[lv,lv]-b[lv,lu])
                col_v = a_cols[v].dot(b[lv,cur_p]) - a_cols[v].dot(b[lu,cur_p]) - a[u,v]*(b[lu,lu]-b[lu,lv]) - a[v,v]*(b[lv,lu]-b[lv,lv])
                delta = d_uu + d_uv + d_vu + d_vv + row_u + row_v + col_u + col_v
                if delta < best_delta:
                    best_delta = delta
                    best_v = v
            if best_v >= 0 and best_delta < -1e-9:
                cur_p[u], cur_p[best_v] = cur_p[best_v], cur_p[u]
                cur_cost += int(best_delta)
                improved = True
        if not improved:
            break
    
    return cur_p.tolist()
# EVOLVE-BLOCK-END
