"""Spin-glass Ising formulation of QAP solved via Parallel Tempering (replica-exchange SA)."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import linear_sum_assignment


def _initial_solution(a, b):
    """Hungarian rank-matching: minimize sum of (row-total_a * row-total_b) assignment.
    Heavy-flow facilities go to low-distance (central) locations."""
    a_row = a.sum(axis=1).astype(np.float64)
    b_row = b.sum(axis=1).astype(np.float64)
    # Cost of assigning facility i to location j: a_row[i]*b_row[j]
    # Minimize total -> heavy a with small b
    cost_mat = np.outer(a_row, b_row)
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    p = np.empty(len(a), dtype=np.int64)
    p[row_ind] = col_ind
    return p


def _total_cost(p, a, b):
    """Exact cost sum_{i,j} a[i,j]*b[p[i],p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def solve(a, b):
    n = len(a)
    deadline = time.time() + 110.0   # 10s safety margin before 120s kill

    # Precompute row sums for O(N) delta-swap
    a_row = a.sum(axis=1).astype(np.float64)
    b_row = b.sum(axis=1).astype(np.float64)

    # Initial solution via Hungarian rank-matching
    p0 = _initial_solution(a, b)
    c0 = _total_cost(p0, a, b)

    best_p = p0.copy()
    best_c = c0

    # Number of replicas (parallel tempering chains)
    n_rep = 6
    # Geometric temperature ladder
    # Estimate scale of typical delta
    typical = abs(float(np.median(a_row)) * abs(float(np.median(b_row)))) * 2.0
    if typical <= 0:
        typical = 1.0
    T_max = typical * 8.0
    T_min = typical * 0.001
    ratio = (T_max / T_min) ** (1.0 / (n_rep - 1))
    temps = [T_min * (ratio ** i) for i in range(n_rep)]

    # Initialize replicas: first is best, others are perturbed copies
    reps_p = [None] * n_rep
    reps_c = [0] * n_rep
    reps_p[0] = p0.copy()
    reps_c[0] = c0
    rng = random.Random(12345)
    for r in range(1, n_rep):
        pr = p0.copy()
        # random perturbation
        for _ in range(n // 2):
            i, j = rng.randrange(n), rng.randrange(n)
            if i != j:
                pr[i], pr[j] = pr[j], pr[i]
        reps_p[r] = pr
        reps_c[r] = _total_cost(pr, a, b)

    # O(N) delta-swap cost update
    def delta_swap(p, c, i, j, a, b, a_row, b_row):
        """Return new cost after swapping locations of facilities i and j."""
        li = p[i]
        lj = p[j]
        if li == lj:
            return c
        # cost contribution change:
        # terms involving facility i with location li, and facility j with location lj
        # delta = a[i,i]*(b[li,li]-b[lj,lj]) + a[j,j]*(b[lj,lj]-b[li,li])
        #       + (a[i,j]+a[j,i])*(b[lj,li]-b[li,lj])
        #       + (a_row[i]-a[i,i]-a[i,j])*(b[li,li]-b[lj,lj])   ... wait need care
        # Let's derive: cost = sum_{k,l} a[k,l]*b[p[k],p[l]]
        # Only terms where p[k] or p[l] is li or lj change, i.e., k in {i,j} or l in {i,j}
        # Actually b[p[k],p[l]] changes only when p[k] in {li,lj} or p[l] in {li,lj}
        # p[k] in {li,lj} only for k in {i,j} (since p is a permutation)
        # So changed terms: (k,l) where k in {i,j} OR l in {i,j}
        # = terms with k=i or k=j or l=i or l=j
        # Let's compute directly the difference.
        # Old contribution of these terms minus new.
        # Simpler: recompute the affected sub-block.
        # Affected facilities: i and j. Affected locations: li and lj.
        # Old: a[i,li-side]... let's just do:
        # old = sum_{k in {i,j}} sum_{l} a[k,l]*b[p[k],p[l]] + sum_{k not in {i,j}} sum_{l in {i,j}} a[k,l]*b[p[k],p[l]]
        # This is getting complex. Use the standard formula.

        # Standard QAP 2-swap delta:
        # Let A = a[i,i], B = a[j,j], C = a[i,j], D = a[j,i]
        # Let s = b[li,li], t = b[lj,lj], u = b[li,lj], v = b[lj,li]
        # Let r_i = a_row[i] - A - C   (sum of a[i,l] for l not i,j)
        # Let r_j = a_row[j] - B - D
        # Old cost of affected part:
        #   A*s + B*t + C*u + D*v
        #   + r_i * (b[li,li] + b[li,lj])  ... no
        # Let me just recompute via the row structure.
        # cost = sum_k a_row[k] * b[p[k], p[k]]? No, that's only diagonal.

        # Given complexity and risk, fall back to a safe but slightly slower method:
        # recompute cost from scratch only when needed is too slow.
        # Instead use: new_cost = c + delta where delta computed by summing affected terms.
        # Affected (k,l) pairs: k in {i,j} or l in {i,j}.
        # Count distinct: { (i,i),(i,j),(j,i),(j,j) } plus { (k,i) for k not i,j } and { (k,j) for k not i,j } and { (i,l) for l not i,j } and { (j,l) for l not i,j }
        # That's O(N). Let's compute delta by summing old vs new for these.

        # To keep it correct, compute delta as:
        # sum over affected (k,l) of a[k,l]*(b[p_new[k],p_new[l]] - b[p[k],p[l]])
        # p_new differs from p only at indices i and j.
        # So b[p_new[k],p_new[l]] - b[p[k],p[l]] is nonzero only when k in {i,j} or l in {i,j}.
        # For such (k,l):
        #   if k in {i,j}: p_new[k] is swapped
        #   if l in {i,j}: p_new[l] is swapped

        # Let's define mapping: new_loc(i)=lj, new_loc(j)=li, others same.
        # delta = sum_{k,l: k in {i,j} or l in {i,j}} a[k,l] * (b[nl(k),nl(l)] - b[p[k],p[l]])

        # We can compute this in O(N) by iterating.
        # But to be safe and simple, I'll compute via full affected sum each time.
        # For n=150, O(N)=150 per swap is fine.

        nl_i = lj
        nl_j = li
        # terms where k==i
        # sum_l a[i,l]*(b[nl_i, nl(l)] - b[li, p[l]])
        # nl(l) = lj if l==j, li if l==i, p[l] otherwise
        # terms where k==j
        # sum_l a[j,l]*(b[nl_j, nl(l)] - b[lj, p[l]])
        # terms where l==i (and k not i,j to avoid double count)
        # sum_{k not i,j} a[k,i]*(b[p[k], nl_i] - b[p[k], li])
        # terms where l==j (and k not i,j)
        # sum_{k not i,j} a[k,j]*(b[p[k], nl_j] - b[p[k], lj])

        delta = 0.0
        # k==i row
        # l==i: a[i,i]*(b[nl_i,nl_i]-b[li,li]) = A*(b[lj,lj]-b[li,li]) = A*(t-s)
        delta += a[i, i] * (b[lj, lj] - b[li, li])
        # l==j: a[i,j]*(b[nl_i,nl_j]-b[li,lj]) = C*(b[lj,li]-b[li,lj]) = C*(v-u)
        delta += a[i, j] * (b[lj, li] - b[li, lj])
        # l not i,j: a[i,l]*(b[nl_i, p[l]] - b[li, p[l]]) = a[i,l]*(b[lj,p[l]]-b[li,p[l]])
        # sum over l not i,j
        for l in range(n):
            if l == i or l == j:
                continue
            delta += a[i, l] * (b[lj, p[l]] - b[li, p[l]])
        # k==j row
        # l==i: a[j,i]*(b[nl_j,nl_i]-b[lj,li]) = D*(b[li,lj]-b[lj,li]) = D*(u-v)
        delta += a[j, i] * (b[li, lj] - b[lj, li])
        # l==j: a[j,j]*(b[nl_j,nl_j]-b[lj,lj]) = B*(b[li,li]-b[lj,lj]) = B*(s-t)
        delta += a[j, j] * (b[li, li] - b[lj, lj])
        # l not i,j: a[j,l]*(b[li,p[l]]-b[lj,p[l]])
        for l in range(n):
            if l == i or l == j:
                continue
            delta += a[j, l] * (b[li, p[l]] - b[lj, p[l]])
        # l==i, k not i,j: a[k,i]*(b[p[k],nl_i]-b[p[k],li]) = a[k,i]*(b[p[k],lj]-b[p[k],li])
        for k in range(n):
            if k == i or k == j:
                continue
            delta += a[k, i] * (b[p[k], lj] - b[p[k], li])
        # l==j, k not i,j: a[k,j]*(b[p[k],nl_j]-b[p[k],lj]) = a[k,j]*(b[p[k],li]-b[p[k],lj])
        for k in range(n):
            if k == i or k == j:
                continue
            delta += a[k, j] * (b[p[k], li] - b[p[k], lj])

        return c + delta

    # Parallel tempering main loop
    exchange_interval = 200  # attempts between exchanges
    attempt = 0

    while time.time() < deadline - 2.0:
        for r in range(n_rep):
            if time.time() >= deadline - 2.0:
                break
            p = reps_p[r]
            c = reps_c[r]
            T = temps[r]
            # propose a swap
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            new_c = delta_swap(p, c, i, j, a, b, a_row, b_row)
            dc = new_c - c
            if dc <= 0 or (T > 0 and rng.random() < np.exp(-dc / T)):
                # accept
                p[i], p[j] = p[j], p[i]
                c = new_c
                reps_p[r] = p
                reps_c[r] = c
                if c < best_c:
                    best_c = c
                    best_p = p.copy()
            attempt += 1

        # Replica exchange
        if attempt % exchange_interval == 0:
            # try to swap adjacent replicas where hot is better than cold
            for r in range(n_rep - 1):
                # r is colder (lower T), r+1 is hotter
                if reps_c[r + 1] < reps_c[r]:
                    # hot replica has better (lower) cost; swap them
                    reps_p[r], reps_p[r + 1] = reps_p[r + 1], reps_p[r]
                    reps_c[r], reps_c[r + 1] = reps_c[r + 1], reps_c[r]

    return best_p.tolist()
# EVOLVE-BLOCK-END
