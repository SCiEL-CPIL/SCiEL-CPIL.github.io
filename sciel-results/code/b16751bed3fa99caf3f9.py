"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 145.0  # Leave 5s safety margin for 150s limit

    # Ensure arrays are int64 for exact integer arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)

    # 1. Initial Solution: Rank Matching (Robust baseline)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    heavy_idx = np.argsort(-a_sums, kind="stable")
    central_idx = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx
    
    # Calculate initial cost
    cost = int(np.sum(a * b[p]))
    best_cost = cost
    best_p = p.copy()

    # Precompute for fast delta calculation
    # We will use a vectorized delta calculation for 3-swaps.
    # For a 3-swap (i, j, k), the cost change depends on the interactions 
    # between {i,j,k} and the rest, and within {i,j,k}.
    
    # To make 3-swaps efficient, we can precompute the current contribution 
    # of each pair to the total cost, or simply compute the delta directly 
    # using matrix slicing.
    
    # Given n=150, a full cost evaluation is O(n^2) = 22500 ops.
    # A delta evaluation for 3-swap can be done in O(n) if we are careful, 
    # or O(1) if we precompute row/col sums, but the interaction terms are tricky.
    # Let's use a semi-vectorized approach:
    # The change in cost for swapping p[i], p[j], p[k] to new positions is:
    # We can compute the cost of the sub-problem involving {i,j,k} and the rest.
    
    # Actually, for simplicity and robustness within the time limit, 
    # let's use a "greedy 3-swap" or "random 3-swap with SA acceptance".
    # Computing the exact delta for 3-swap:
    # Let S = {i, j, k}.
    # Old cost contribution of S with all nodes (including S):
    # C_old = sum_{m in S, n in all} a[m,n] * b[p[m], p[n]]
    # New cost contribution:
    # C_new = sum_{m in S, n in all} a[m,n] * b[p'_m, p'_n]
    # where p' is the permuted assignment for S.
    # Delta = C_new - C_old.
    
    # This can be computed in O(n) for a given triple and permutation of S.
    # There are 6 permutations of 3 items.
    
    def get_delta_3swap(i, j, k, p, a, b, perm):
        """
        Compute delta for swapping p[i], p[j], p[k] according to perm.
        perm is a tuple of 3 indices (0,1,2) indicating which old position goes to which new slot.
        Actually, let's define it as: new_p[i] = old_p[perm[0]], new_p[j] = old_p[perm[1]], new_p[k] = old_p[perm[2]]
        Wait, that's not a swap of positions, that's a reassignment.
        Let's stick to standard 3-cycle or 3-swap (permutation of 3 facilities among 3 locations).
        
        Let u, v, w be the locations currently occupied by i, j, k.
        u = p[i], v = p[j], w = p[k].
        
        We consider all 6 permutations of assigning {i,j,k} to {u,v,w}.
        One is the identity (delta 0).
        The others are 2-swaps (3 of them) and 3-cycles (2 of them).
        
        For a specific permutation pi of {0,1,2}:
        new_p[i] = [u,v,w][pi[0]]
        new_p[j] = [u,v,w][pi[1]]
        new_p[k] = [u,v,w][pi[2]]
        
        Delta = sum_{m in {i,j,k}, n in all} a[m,n] * (b[new_p[m], new_p[n]] - b[p[m], p[n]])
        
        Note: For n in {i,j,k}, p[n] changes. For n not in {i,j,k}, p[n] stays same.
        So we can split the sum:
        Delta = sum_{m in S, n in S} a[m,n] * (b[new_p[m], new_p[n]] - b[p[m], p[n]])
              + sum_{m in S, n not in S} a[m,n] * (b[new_p[m], p[n]] - b[p[m], p[n]])
              + sum_{m not in S, n in S} a[m,n] * (b[p[m], new_p[n]] - b[p[m], p[n]])
        
        The first term is O(1) (3x3 matrix).
        The second term is O(n).
        The third term is O(n).
        
        Total O(n).
        """
        S = [i, j, k]
        old_locs = [p[i], p[j], p[k]]
        
        # new_locs[m] is the new location for facility S[m]
        # perm[0] is the index in old_locs for the new location of S[0] (i)
        # perm[1] is the index in old_locs for the new location of S[1] (j)
        # perm[2] is the index in old_locs for the new location of S[2] (k)
        new_locs = [old_locs[perm[0]], old_locs[perm[1]], old_locs[perm[2]]]
        
        # Term 1: Internal to S
        delta = 0
        for idx_m in range(3):
            for idx_n in range(3):
                m = S[idx_m]
                n = S[idx_n]
                old_b = b[p[m], p[n]]
                new_b = b[new_locs[idx_m], new_locs[idx_n]]
                delta += a[m, n] * (new_b - old_b)
        
        # Term 2 & 3: Interaction with rest
        # We need to sum over all n not in S.
        # sum_{n not in S} a[i,n]*(b[new_p[i], p[n]] - b[p[i], p[n]]) + ... for j, k
        # + sum_{m not in S} a[m,i]*(b[p[m], new_p[i]] - b[p[m], p[i]]) + ...
        
        # Vectorized approach:
        # Let mask be True for all indices, False for i,j,k
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        mask[k] = False
        
        # For m in S, n not in S:
        # a[i, mask] dot (b[new_p[i], p[mask]] - b[p[i], p[mask]])
        # Note: b[new_p[i], p[mask]] is a vector of length n-3
        
        # Let's construct the vectors
        # p_mask = p[mask]
        p_mask = p[mask]
        
        # Row part: m in S, n not in S
        # m=i: a[i, mask] * (b[new_p[i], p_mask] - b[p[i], p_mask])
        delta += np.dot(a[i, mask], b[new_locs[0], p_mask] - b[p[i], p_mask])
        delta += np.dot(a[j, mask], b[new_locs[1], p_mask] - b[p[j], p_mask])
        delta += np.dot(a[k, mask], b[new_locs[2], p_mask] - b[p[k], p_mask])
        
        # Col part: m not in S, n in S
        # n=i: a[mask, i] * (b[p_mask, new_p[i]] - b[p_mask, p[i]])
        delta += np.dot(a[mask, i], b[p_mask, new_locs[0]] - b[p_mask, p[i]])
        delta += np.dot(a[mask, j], b[p_mask, new_locs[1]] - b[p_mask, p[j]])
        delta += np.dot(a[mask, k], b[p_mask, new_locs[2]] - b[p_mask, p[k]])
        
        return int(delta)

    # Simulated Annealing
    # Initial temperature
    # Estimate cost scale
    max_cost_diff = int(np.abs(a).max() * np.abs(b).max() * n)
    T = max_cost_diff * 0.1
    T_min = 1.0
    cooling_rate = 0.999
    
    current_cost = cost
    current_p = p.copy()
    
    # Permutations of 3 elements, excluding identity (0,1,2)
    # (0,1,2) -> identity
    # (0,2,1) -> swap j,k
    # (1,0,2) -> swap i,j
    # (1,2,0) -> cycle i->j, j->k, k->i? 
    # Let's just list all 6 permutations and skip the first one (identity)
    perms = [
        (0, 1, 2), # Identity
        (0, 2, 1), # Swap j,k
        (1, 0, 2), # Swap i,j
        (1, 2, 0), # Cycle
        (2, 0, 1), # Cycle
        (2, 1, 0)  # Swap i,k
    ]
    # We will sample one of the non-identity permutations
    non_identity_perms = perms[1:]

    while time.time() - start_time < time_limit:
        # Pick random 3 distinct facilities
        i, j, k = random.sample(range(n), 3)
        
        # Pick a random non-identity permutation
        perm = random.choice(non_identity_perms)
        
        delta = get_delta_3swap(i, j, k, current_p, a, b, perm)
        
        # Acceptance criterion
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Apply the move
            # new_p[i] = old_p[perm[0]] etc.
            old_p_i = current_p[i]
            old_p_j = current_p[j]
            old_p_k = current_p[k]
            
            current_p[i] = old_p_i if perm[0]==0 else (old_p_j if perm[0]==1 else old_p_k)
            current_p[j] = old_p_i if perm[1]==0 else (old_p_j if perm[1]==1 else old_p_k)
            current_p[k] = old_p_i if perm[2]==0 else (old_p_j if perm[2]==1 else old_p_k)
            
            # Wait, the permutation logic in get_delta_3swap was:
            # new_locs[0] = old_locs[perm[0]]
            # So new_p[i] = old_locs[perm[0]]
            # My assignment above is correct.
            
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = current_p.copy()
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_p.tolist()

# EVOLVE-BLOCK-END
