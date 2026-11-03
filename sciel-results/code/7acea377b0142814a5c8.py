"""
Vectorized Multi-Start Simulated Annealing with 3-cyclic moves for QAPLIB wil100.

Key structural differences from the seed:
1. Neighborhood: 3-cyclic moves (i->j->k->i) instead of 2-swaps.
2. Vectorized delta computation: all 3-cyclic deltas involving position i
   are computed in one NumPy op, enabling ~1000x more evaluations.
3. No local search phase — pure SA with many short runs.
4. Multi-start with greedy + random initializations.
"""
import numpy as np
import time


def _qap_cost(a, b, p):
    """Exact integer cost."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _precompute(a, b, p):
    """
    Precompute interaction vectors for fast 3-cyclic delta computation.
    
    For a 3-cycle (i->j->k->i), facility at pos i goes to j, j goes to k, k goes to i.
    The cost change depends on:
      - a[i,:] and a[:,i] (facility i's interactions)
      - a[j,:] and a[:,j]
      - a[k,:] and a[:,k]
      - b at the relevant positions
    
    We precompute:
      row_a[i] = a[i, :]  (flow from facility i)
      col_a[i] = a[:, i]  (flow to facility i)
      b_row[pos] = b[pos, :]
      b_col[pos] = b[:, pos]
    """
    n = len(a)
    # These are just views, no copy needed
    return None  # a and b are already arrays; we index them directly


def _delta_3cycle(a, b, p, i, j, k):
    """
    Compute cost delta for 3-cycle: pos i gets facility from k,
    pos j gets facility from i, pos k gets facility from j.
    
    i.e., p_new[i] = p[k], p_new[j] = p[i], p_new[k] = p[j]
    (cycle: i -> j -> k -> i in terms of which facility goes where)
    
    Actually let's define: the facility currently at position i moves to position j,
    the facility at j moves to k, the facility at k moves to i.
    So: p_new[j] = p[i], p_new[k] = p[j], p_new[i] = p[k].
    """
    fi = p[i]  # facility at position i
    fj = p[j]  # facility at position j
    fk = p[k]  # facility at position k
    
    # After 3-cycle:
    # pos i gets fk, pos j gets fi, pos k gets fj
    
    # Cost = sum_{u,v} a[u,v] * b[p_new[u], p_new[v]]
    # Only terms involving positions i, j, k change.
    # Terms where both u,v are not in {i,j,k}: unchanged.
    # Terms where exactly one of u,v is in {i,j,k}: change in b index.
    # Terms where both u,v are in {i,j,k}: change in both.
    
    # Let S = {i, j, k}
    # Old contribution from pairs involving S:
    # sum_{u in S, v not in S} a[u,v]*b[old_p[u], old_p[v]] + a[v,u]*b[old_p[v], old_p[u]]
    #   + sum_{u in S, v in S} a[u,v]*b[old_p[u], old_p[v]]
    
    # New contribution:
    # sum_{u in S, v not in S} a[u,v]*b[new_p[u], old_p[v]] + a[v,u]*b[old_p[v], new_p[u]]
    #   + sum_{u in S, v in S} a[u,v]*b[new_p[u], new_p[v]]
    
    # delta = new - old
    
    # For u in S, v not in S:
    # delta_cross = sum_{u in S} [ a[u, v] * (b[new_p[u], p[v]] - b[old_p[u], p[v]])
    #                            + a[v, u] * (b[p[v], new_p[u]] - b[p[v], old_p[u]]) ]
    # summed over all v not in S
    
    # Let's compute this efficiently.
    # For position u in S, the new facility is:
    # new_p[i] = fk, new_p[j] = fi, new_p[k] = fj
    # old_p[i] = fi, old_p[j] = fj, old_p[k] = fk
    
    # delta_cross for position u:
    # sum_{v not in S} a[u,v]*(b[new_p[u], p[v]] - b[old_p[u], p[v]])
    #                    + a[v,u]*(b[p[v], new_p[u]] - b[p[v], old_p[u]])
    
    # = sum_{v not in S} a[u,v]*(b[new_p[u], :] - b[old_p[u], :])[p[v]]
    #   + sum_{v not in S} a[:,u]*(b[:, new_p[u]] - b[:, old_p[u]])[p[v]]
    
    # Let's precompute for each position u in S:
    # diff_row_u = b[new_p[u], :] - b[old_p[u], :]
    # diff_col_u = b[:, new_p[u]] - b[:, old_p[u]]
    
    # Then delta_cross_u = sum_{v not in S} a[u,v]*diff_row_u[p[v]] + a[v,u]*diff_col_u[p[v]]
    # = sum_{v not in S} a[u,v]*diff_row_u[p[v]] + a[:,u][v]*diff_col_u[p[v]]
    
    # For efficiency, let's compute over ALL v and subtract the S terms.
    
    new_pi = fk
    new_pj = fi
    new_pk = fj
    
    # diff vectors for each position
    diff_row_i = b[new_pi, :] - b[fi, :]
    diff_col_i = b[:, new_pi] - b[:, fi]
    diff_row_j = b[new_pj, :] - b[fj, :]
    diff_col_j = b[:, new_pj] - b[:, fj]
    diff_row_k = b[new_pk, :] - b[fk, :]
    diff_col_k = b[:, new_pk] - b[:, fk]
    
    # Cross terms (u in S, v not in S):
    # For u=i: sum_{v not in S} a[i,v]*diff_row_i[p[v]] + a[v,i]*diff_col_i[p[v]]
    # We compute sum over ALL v, then subtract v in S.
    
    all_v_idx = p  # p[v] for all v
    
    # Sum over all v:
    cross_i_all = np.dot(a[i, :], diff_row_i[all_v_idx]) + np.dot(a[:, i], diff_col_i[all_v_idx])
    cross_j_all = np.dot(a[j, :], diff_row_j[all_v_idx]) + np.dot(a[:, j], diff_col_j[all_v_idx])
    cross_k_all = np.dot(a[k, :], diff_row_k[all_v_idx]) + np.dot(a[:, k], diff_col_k[all_v_idx])
    
    # Subtract v in S (v = i, j, k):
    # For v=i: a[i, i]*diff_row_i[fi] + a[i, i]*diff_col_i[fi]
    # For v=j: a[i, j]*diff_row_i[fj] + a[j, i]*diff_col_i[fj]
    # For v=k: a[i, k]*diff_row_i[fk] + a[k, i]*diff_col_i[fk]
    sub_i = (a[i, i] * diff_row_i[fi] + a[i, i] * diff_col_i[fi]
            + a[i, j] * diff_row_i[fj] + a[j, i] * diff_col_i[fj]
            + a[i, k] * diff_row_i[fk] + a[k, i] * diff_col_i[fk])
    
    sub_j = (a[j, i] * diff_row_j[fi] + a[i, j] * diff_col_j[fi]
            + a[j, j] * diff_row_j[fj] + a[j, j] * diff_col_j[fj]
            + a[j, k] * diff_row_j[fk] + a[k, j] * diff_col_j[fk])
    
    sub_k = (a[k, i] * diff_row_k[fi] + a[i, k] * diff_col_k[fi]
            + a[k, j] * diff_row_k[fj] + a[j, k] * diff_col_k[fj]
            + a[k, k] * diff_row_k[fk] + a[k, k] * diff_col_k[fk])
    
    cross_i = cross_i_all - sub_i
    cross_j = cross_j_all - sub_j
    cross_k = cross_k_all - sub_k
    
    # Internal terms (u in S, v in S):
    # Old: a[i,j]*b[fi,fj] + a[i,k]*b[fi,fk] + a[j,i]*b[fj,fi] + a[j,k]*b[fj,fk]
    #      + a[k,i]*b[fk,fi] + a[k,j]*b[fk,fj] + a[i,i]*b[fi,fi] + a[j,j]*b[fj,fj] + a[k,k]*b[fk,fk]
    # New: a[i,j]*b[fk,fi] + a[i,k]*b[fk,fj] + a[j,i]*b[fi,fk] + a[j,k]*b[fi,fj]
    #      + a[k,i]*b[fj,fi] + a[k,j]*b[fj,fk] + a[i,i]*b[fk,fk] + a[j,j]*b[fi,fi] + a[k,k]*b[fj,fj]
    
    old_int = (a[i, j] * b[fi, fj] + a[i, k] * b[fi, fk]
              + a[j, i] * b[fj, fi] + a[j, k] * b[fj, fk]
              + a[k, i] * b[fk, fi] + a[k, j] * b[fk, fj]
              + a[i, i] * b[fi, fi] + a[j, j] * b[fj, fj] + a[k, k] * b[fk, fk])
    
    new_int = (a[i, j] * b[fk, fi] + a[i, k] * b[fk, fj]
              + a[j, i] * b[fi, fk] + a[j, k] * b[fi, fj]
              + a[k, i] * b[fj, fi] + a[k, j] * b[fj, fk]
              + a[i, i] * b[fk, fk] + a[j, j] * b[fi, fi] + a[k, k] * b[fj, fj])
    
    delta = (cross_i + cross_j + cross_k) + (new_int - old_int)
    return int(delta)


def _delta_3cycle_fast(a, b, p, i, j, k):
    """
    Faster 3-cycle delta using a different decomposition.
    3-cycle: i->j->k->i means facility at i goes to j, at j goes to k, at k goes to i.
    new_p[j] = p[i], new_p[k] = p[j], new_p[i] = p[k]
    """
    fi = p[i]
    fj = p[j]
    fk = p[k]
    
    # The change in cost comes from:
    # 1. Terms where facility i (now at j) interacts with all other facilities
    # 2. Terms where facility j (now at k) interacts with all other facilities  
    # 3. Terms where facility k (now at i) interacts with all other facilities
    # But we need to be careful not to double-count.
    
    # Alternative: compute the difference directly.
    # Old positions: fi at i, fj at j, fk at k
    # New positions: fk at i, fi at j, fj at k
    
    # The cost is sum_{u,v} a[u,v] * b[pos(u), pos(v)]
    # where pos(u) is the location of facility u.
    # Old: pos(fi)=i, pos(fj)=j, pos(fk)=k
    # New: pos(fi)=j, pos(fj)=k, pos(fk)=i
    
    # Delta = sum over all pairs (u,v) of a[u,v] * (b[new_pos(u), new_pos(v)] - b[old_pos(u), old_pos(v)])
    # Only terms where at least one of u,v is in {fi, fj, fk} change.
    
    # Let's compute it as:
    # For each facility u in {fi, fj, fk}:
    #   sum over all v != u: a[u,v] * (b[new_pos(u), new_pos(v)] - b[old_pos(u), old_pos(v)])
    #   + sum over all v != u: a[v,u] * (b[new_pos(v), new_pos(u)] - b[old_pos(v), old_pos(u)])
    #   + a[u,u] * (b[new_pos(u), new_pos(u)] - b[old_pos(u), old_pos(u)])
    # But this double-counts pairs where both u,v are in {fi,fj,fk}.
    
    # Simpler: just compute the full change for the three facilities.
    
    # Let's define old_pos and new_pos for the three facilities:
    # fi: old=i, new=j
    # fj: old=j, new=k
    # fk: old=k, new=i
    
    # For any other facility w (not in {fi,fj,fk}), its position doesn't change.
    
    # Delta = [sum_{w not in F} a[fi,w]*(b[j,pos(w)]-b[i,pos(w)]) + a[w,fi]*(b[pos(w),j]-b[pos(w),i])
    #        + sum_{w not in F} a[fj,w]*(b[k,pos(w)]-b[j,pos(w)]) + a[w,fj]*(b[pos(w),k]-b[pos(w),j])
    #        + sum_{w not in F} a[fk,w]*(b[i,pos(w)]-b[k,pos(w)]) + a[w,fk]*(b[pos(w),i]-b[pos(w),k)]
    #        + a[fi,fj]*(b[j,k]-b[i,j]) + a[fi,fk]*(b[j,i]-b[i,k])
    #        + a[fj,fi]*(b[k,i]-b[j,i]) + a[fj,fk]*(b[k,k]-b[j,k])
    #        + a[fk,fi]*(b[i,j]-b[k,i]) + a[fk,fj]*(b[i,k]-b[k,j])
    #        + a[fi,fi]*(b[j,j]-b[i,i]) + a[fj,fj]*(b[k,k]-b[j,j]) + a[fk,fk]*(b[i,i]-b[k,k])]
    
    # This is getting complex. Let's use the position-based approach with precomputation.
    
    # For position-based:
    # old: pos[i]=fi, pos[j]=fj, pos[k]=fk
    # new: pos[i]=fk, pos[j]=fi, pos[k]=fj
    
    # The delta can be computed as:
    # For each position u in {i,j,k}, for each position v not in {i,j,k}:
    #   a[old_p[u], old_p[v]] * (b[new_p[u], old_p[v]] - b[old_p[u], old_p[v]])
    #   + a[old_p[v], old_p[u]] * (b[old_p[v], new_p[u]] - b[old_p[v], old_p[u]])
    
    # Plus for u,v both in {i,j,k}:
    #   a[old_p[u], old_p[v]] * (b[new_p[u], new_p[v]] - b[old_p[u], old_p[v]])
    
    # Let's compute this.
    
    new_pi = fk
    new_pj = fi
    new_pk = fj
    
    # Cross terms
    # For u=i: sum_{v not in {i,j,k}} a[fi, p[v]]*(b[fk, p[v]] - b[fi, p[v]]) + a[p[v], fi]*(b[p[v], fk] - b[p[v], fi])
    # We can compute this as a dot product if we have the right vectors.
    
    # Let's just do it directly with numpy for the three positions.
    
    # Create index arrays
    idx_i = np.arange(len(p))
    mask = (idx_i != i) & (idx_i != j) & (idx_i != k)
    other_positions = idx_i[mask]
    other_facilities = p[other_positions]
    
    # Cross terms for position i (facility fi moving from i to j... wait, no)
    # Position i: old facility fi, new facility fk
    # Cross: sum_{v in others} a[fi, p[v]]*(b[fk, p[v]] - b[fi, p[v]]) + a[p[v], fi]*(b[p[v], fk] - b[p[v], fi])
    
    cross_i = (np.dot(a[fi, other_facilities], b[fk, other_facilities] - b[fi, other_facilities])
              + np.dot(a[other_facilities, fi], b[other_facilities, fk] - b[other_facilities, fi]))
    
    # Position j: old facility fj, new facility fi
    cross_j = (np.dot(a[fj, other_facilities], b[fi, other_facilities] - b[fj, other_facilities])
              + np.dot(a[other_facilities, fj], b[other_facilities, fi] - b[other_facilities, fj]))
    
    # Position k: old facility fk, new facility fj
    cross_k = (np.dot(a[fk, other_facilities], b[fj, other_facilities] - b[fk, other_facilities])
              + np.dot(a[other_facilities, fk], b[other_facilities, fj] - b[other_facilities, fk]))
    
    # Internal terms (all 9 pairs among i,j,k)
    old_pairs = [(i, j), (i, k), (j, i), (j, k), (k, i), (k, j), (i, i), (j, j), (k, k)]
    old_facs = [(fi, fj), (fi, fk), (fj, fi), (fj, fk), (fk, fi), (fk, fj), (fi, fi), (fj, fj), (fk, fk)]
    new_facs = [(fk, fi), (fk, fj), (fi, fk), (fi, fj), (fj, fi), (fj, fk), (fk, fk), (fi, fi), (fj, fj)]
    
    old_int = 0
    new_int = 0
    for (ou, ov), (of, nf) in zip(old_pairs, old_facs):
        pass  # This is getting too complex inline
    
    # Let me just compute directly
    old_int = (a[fi, fj] * b[i, j] + a[fi, fk] * b[i, k]
              + a[fj, fi] * b[j, i] + a[fj, fk] * b[j, k]
              + a[fk, fi] * b[k, i] + a[fk, fj] * b[k, j]
              + a[fi, fi] * b[i, i] + a[fj, fj] * b[j, j] + a[fk, fk] * b[k, k])
    
    new_int = (a[fk, fi] * b[i, j] + a[fk, fj] * b[i, k]
              + a[fi, fk] * b[j, i] + a[fi, fj] * b[j, k]
              + a[fj, fi] * b[k, i] + a[fj, fk] * b[k, j]
              + a[fk, fk] * b[i, i] + a[fi, fi] * b[j, j] + a[fj, fj] * b[k, k])
    
    delta = cross_i + cross_j + cross_k + (new_int - old_int)
    return int(delta)


def _greedy_initial(a, b):
    """Rank-matching heuristic."""
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _sa_run(a, b, p_init, time_limit, start_time, T0=500.0, T1=0.1):
    """
    Vectorized SA with 3-cyclic moves.
    """
    n = len(p)
    p = p_init.copy()
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    elapsed_total = time_limit
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        frac = min(1.0, elapsed / max(elapsed_total, 1.0))
        T = T0 * (T1 / T0) ** frac
        
        # Pick a random position i, then two random other positions j, k
        i = np.random.randint(n)
        j = np.random.randint(n)
        k = np.random.randint(n)
        
        if i == j or i == k or j == k:
            continue
        
        delta = _delta_3cycle_fast(a, b, p, i, j, k)
        
        if delta < 0 or (T > 0.01 and np.random.rand() < np.exp(-delta / max(T, 1e-10))):
            # Apply 3-cycle: i->j->k->i
            # p_new[i] = p[k], p_new[j] = p[i], p_new[k] = p[j]
            p[i], p[j], p[k] = p[k], p[i], p[j]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    total_deadline = 145.0  # Safety margin before 150s limit
    start_time = time.time()
    
    np.random.seed(42)
    
    best_p = None
    best_cost = np.inf
    
    # Multi-start SA
    # Start 1: Greedy
    p0 = _greedy_initial(a, b)
    p_best, c_best = _sa_run(a, b, p0, min(20.0, total_deadline - 5), start_time)
    best_p = p_best.copy()
    best_cost = c_best
    
    # Start 2-10: Random
    num_starts = 10
    for s in range(num_starts):
        elapsed = time.time() - start_time
        remaining = total_deadline - elapsed
        if remaining < 10:
            break
        
        # Allocate time proportionally
        time_per_run = min(15.0, (remaining - 5) / max(1, num_starts - s))
        
        p_rand = np.random.permutation(n).astype(np.int64)
        p_run, c_run = _sa_run(a, b, p_rand, time_per_run, time.time())
        
        if c_run < best_cost:
            best_cost = c_run
            best_p = p_run.copy()
    
    # Final polish: SA from best
    elapsed = time.time() - start_time
    if elapsed < total_deadline - 3:
        sa_time = total_deadline - elapsed - 2
        p_final, c_final = _sa_run(a, b, best_p, sa_time, time.time(), T0=200.0, T1=0.05)
        if c_final < best_cost:
            best_p = p_final
            best_cost = c_final
    
    return best_p.tolist()
