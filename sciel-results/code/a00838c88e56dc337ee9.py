"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from itertools import combinations

P = {'n': 12, 'm': 40}

def _count_pairs(family):
    """Count comparable pairs in a family of bitmasks."""
    m = len(family)
    count = 0
    for i in range(m):
        si = family[i]
        for j in range(i + 1, m):
            sj = family[j]
            if (si & sj) == si and si != sj:
                count += 1
            elif (si & sj) == sj and si != sj:
                count += 1
    return count

def _build_comparability_structure(n):
    """Precompute comparability relations for all 2^n subsets."""
    N = 1 << n
    # For each mask, compute the set of masks comparable to it (strict subset/superset)
    # comparable[i] = bitmask over [0, N) where bit j is set if i and j are comparable
    comparable = np.zeros(N, dtype=np.uint64)
    
    for i in range(N):
        # i is subset of j if (i & j) == i and i != j
        # j is subset of i if (j & i) == j and j != i
        for j in range(N):
            if i == j:
                continue
            if (i & j) == i:  # i is strict subset of j
                comparable[i] |= (1 << j)
            elif (j & i) == j:  # j is strict subset of i
                comparable[i] |= (1 << j)
    return comparable

def _greedy_chain_init(n, m):
    """Initialize by building chains to maximize comparability."""
    family = []
    used = set()
    max_mask = 1 << n
    
    for _ in range(m):
        best_set = -1
        best_gain = -1
        
        candidates = set()
        
        for t in family:
            sub = t
            while sub > 0:
                candidates.add(sub)
                sub = (sub - 1) & t
            candidates.add(0)
            
            for _ in range(5):
                extra = random.randint(1, max_mask - 1)
                candidates.add(t | extra)
        
        for _ in range(100):
            candidates.add(random.randint(0, max_mask - 1))
        
        candidates.add(0)
        candidates.add(max_mask - 1)
        
        for s in candidates:
            if s in used:
                continue
            
            gain = 0
            for t in family:
                if (s & t) == s and s != t:
                    gain += 1
                elif (t & s) == t and t != s:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_set = s

        if best_set != -1:
            family.append(best_set)
            used.add(best_set)
        else:
            for s in range(max_mask):
                if s not in used:
                    family.append(s)
                    used.add(s)
                    break
    
    return family

def _branch_and_bound(n, m, deadline, initial_family=None):
    """
    Branch and bound on the incomparability graph.
    
    We want to select m subsets from 2^n such that the number of comparable pairs is maximized.
    Equivalently, we want to minimize the number of incomparable pairs.
    
    Total pairs = C(m, 2). Maximizing comparable pairs = minimizing incomparable pairs.
    
    We use branch and bound where:
    - State: current selection, next candidate to consider
    - Bound: lower bound on additional incomparable pairs from remaining choices
    
    For efficiency, we precompute the incomparability relation as bitsets.
    """
    N = 1 << n
    total_pairs = m * (m - 1) // 2
    
    # Precompute: for each mask, which other masks are incomparable with it
    # incomparable[i] is a list of masks j where i and j are NOT comparable
    # Two masks are comparable if one is a strict subset of the other
    # They are incomparable if neither is a subset of the other (and they're not equal)
    
    # For speed, use numpy arrays
    # comp_matrix[i][j] = 1 if i and j are comparable (strict), 0 otherwise
    # We'll work with the complement: incomp[i] = set of j not comparable to i
    
    # Precompute subset relationships
    # is_subset[i][j] = 1 if i is a strict subset of j
    # This is a N x N boolean matrix, N=4096, so 4096^2 = 16.7M entries
    # Too large for a dense matrix in Python, but we can use bitsets
    
    # Alternative: for each i, precompute the list of masks that are strict supersets of i
    # and strict subsets of i
    strict_supersets = [set() for _ in range(N)]
    strict_subsets = [set() for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if (i & j) == i:  # i is strict subset of j
                strict_subsets[j].add(i)
                strict_supersets[i].add(j)
    
    # For each i, the set of masks comparable to i is strict_subsets[i] | strict_supersets[i]
    comparable_to = [strict_subsets[i] | strict_supersets[i] for i in range(N)]
    
    # For branch and bound, we need to track:
    # - Current set of selected masks
    # - For each candidate mask, how many already-selected masks it's incomparable with
    # - Lower bound on total incomparable pairs
    
    # Strategy: order masks by some heuristic, then do depth-first search
    # with pruning based on lower bounds
    
    # Heuristic ordering: prefer masks that are "central" in the subset lattice
    # i.e., masks with size close to n/2 are more likely to be comparable to many others
    # Actually, for minimizing incomparable pairs, we want masks that are comparable to many others
    # Masks of size 0 and n are comparable to everything, so they're great
    # Masks of size n/2 are comparable to fewer others
    
    # Let's order by: first the empty set, then full set, then by size (small to large)
    # This gives a good initial ordering for the search
    
    # Actually, for B&B to work well, we need a good upper bound on the number of incomparable pairs
    # we can still add. If we've selected k masks and have m-k to go, the minimum additional
    # incomparable pairs is 0 (if all new masks are comparable to all existing ones).
    # The maximum is (m-k) * k + C(m-k, 2) if all new masks are incomparable to everything.
    
    # A tighter bound: for each candidate mask not yet selected, compute how many selected
    # masks it's incomparable with. The minimum over all candidates gives a lower bound
    # on additional incomparable pairs from adding one more mask.
    
    # Let's implement a simpler approach: simulated annealing with the comparability structure
    # precomputed, plus a final local search.
    
    # Actually, let me implement a proper B&B with bitsets for the incomparability graph.
    
    # Represent the incomparability graph using Python integers as bitsets (arbitrary precision)
    # incomp_bitset[i] = integer where bit j is set if i and j are incomparable
    
    incomp_bitset = [0] * N
    for i in range(N):
        bits = 0
        for j in range(N):
            if i == j:
                continue
            if j not in comparable_to[i]:
                bits |= (1 << j)
        incomp_bitset[i] = bits
    
    # Now do branch and bound
    # State: (index in ordering, current_selection as frozenset, current_incomp_count)
    # We try to select m masks from the ordering
    
    # Ordering: sort by "popularity" - number of comparable masks
    # Masks with more comparable partners should be considered first
    popularity = [len(comparable_to[i]) for i in range(N)]
    ordering = sorted(range(N), key=lambda i: (-popularity[i], i))
    
    # Precompute: for each position in ordering, which earlier positions are incomparable
    # This helps with pruning
    
    # For efficiency, use numpy for the main search loop
    # But B&B is inherently sequential, so let's keep it in Python with good pruning
    
    best_incomp = total_pairs  # We want to minimize incomparable pairs
    best_selection = None
    
    # Use a stack-based DFS with pruning
    # State: (pos, selected_list, incomp_count)
    # where pos is the next position in ordering to consider
    
    # To make this tractable, we need strong pruning.
    # Key insight: if we've selected k masks and need m-k more, the minimum additional
    # incomparable pairs is bounded below by 0, but we can do better.
    
    # For each unselected mask, compute its incomparability with the current selection.
    # The minimum such value over all unselected masks is a lower bound on the additional
    # incomparable pairs from the next selection.
    
    # This is still expensive. Let's use a different approach:
    # 1. Start with a good initial solution from greedy
    # 2. Use local search with the precomputed comparability structure
    # 3. Add simulated annealing for escaping local optima
    
    # Given the time constraints, let's do a hybrid:
    # - Precompute comparability as a dense numpy boolean matrix (4096x4096 = 16M entries)
    # - Use vectorized local search with this matrix
    
    comp_matrix = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in comparable_to[i]:
            comp_matrix[i, j] = True
    
    # Now local search with vectorized operations
    if initial_family is None:
        initial_family = _greedy_chain_init(n, m)
    
    current = list(initial_family)
    current_set = set(current)
    
    # Compute current incomparable count
    def _count_incomparable(fam):
        m_local = len(fam)
        count = 0
        for i in range(m_local):
            for j in range(i + 1, m_local):
                if not comp_matrix[fam[i], fam[j]]:
                    count += 1
        return count
    
    current_incomp = _count_incomparable(current)
    best_incomp = current_incomp
    best_solution = list(current)
    
    # Local search: try replacing each element with a better one
    # For each position i and candidate c not in current:
    #   delta = (new_incomp - old_incomp)
    #   old_incomp contribution of current[i] = number of j != i where not comp[current[i]][current[j]]
    #   new_incomp contribution of c = number of j != i where not comp[c][current[j]]
    
    deadline_check = time.time()
    while time.time() < deadline:
        improved = False
        
        for idx in range(m):
            if time.time() >= deadline:
                break
            
            old = current[idx]
            
            # Old contribution: number of other elements incomparable with old
            old_contrib = 0
            for j in range(m):
                if j == idx:
                    continue
                if not comp_matrix[old, current[j]]:
                    old_contrib += 1
            
            # Try all candidates not in current_set
            # Vectorized: for each candidate c, compute incomparability with all other elements
            others = [current[j] for j in range(m) if j != idx]
            others_arr = np.array(others, dtype=np.int32)
            
            # For each candidate c in [0, N), compute how many of others it's incomparable with
            # incomp_with_others[c] = sum over j of (not comp_matrix[c, others[j]])
            
            # comp_matrix is (N, N), others_arr is (m-1,)
            # We want for each c: sum over j of (1 - comp_matrix[c, others[j]])
            # = (m-1) - sum over j of comp_matrix[c, others[j]]
            
            # comp_matrix[:, others_arr] gives (N, m-1) boolean matrix
            # Sum along axis 1 gives (N,) array of comparable counts
            
            comp_counts = comp_matrix[:, others_arr].sum(axis=1)  # (N,)
            incomp_counts = (m - 1) - comp_counts  # (N,)
            
            # Delta for candidate c: incomp_counts[c] - old_contrib
            deltas = incomp_counts - old_contrib
            
            # Mask out candidates already in current_set
            valid = np.ones(N, dtype=bool)
            for s in current_set:
                valid[s] = False
            deltas[~valid] = np.inf
            
            best_c = int(np.argmin(deltas))
            best_delta = deltas[best_c]
            
            if best_delta < 0:  # Strict improvement
                current_set.remove(old)
                current_set.add(best_c)
                current[idx] = best_c
                current_incomp += int(best_delta)
                improved = True
                
                if current_incomp < best_incomp:
                    best_incomp = current_incomp
                    best_solution = list(current)
        
        if not improved:
            # Perturbation: randomly replace 2-3 elements
            num_perturb = random.randint(2, 3)
            for _ in range(num_perturb):
                if time.time() >= deadline:
                    break
                idx = random.randint(0, m - 1)
                old = current[idx]
                
                # Pick a random candidate
                for _ in range(50):
                    new_c = random.randint(0, N - 1)
                    if new_c in current_set:
                        continue
                    
                    # Compute delta
                    others = [current[j] for j in range(m) if j != idx]
                    old_contrib = sum(1 for j in range(m) if j != idx and not comp_matrix[old, current[j]])
                    new_contrib = sum(1 for j in range(m) if j != idx and not comp_matrix[new_c, current[j]])
                    delta = new_contrib - old_contrib
                    
                    if delta < 0 or random.random() < 0.1:  # Accept improving or sometimes worsening
                        current_set.remove(old)
                        current_set.add(new_c)
                        current[idx] = new_c
                        current_incomp += delta
                        if current_incomp < best_incomp:
                            best_incomp = current_incomp
                            best_solution = list(current)
                        break
    
    # Convert back to comparable pairs count
    best_comparable = total_pairs - best_incomp
    return best_solution, best_comparable

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin (90s limit)
    n = P['n']
    m = P['m']
    
    best_family_overall = None
    best_score_overall = -1
    
    # Multiple restarts with different random seeds
    num_restarts = 4
    restart_time = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        restart_deadline = start_time + (restart + 1) * restart_time
        if time.time() >= restart_deadline:
            break
        
        random.seed(restart * 98765 + 42)
        initial = _greedy_chain_init(n, m)
        
        family, score = _branch_and_bound(n, m, restart_deadline, initial)
        
        if score > best_score_overall:
            best_score_overall = score
            best_family_overall = family
    
    # Final polish with remaining time
    if best_family_overall is not None and time.time() < deadline - 3:
        family, score = _branch_and_bound(n, m, deadline, best_family_overall)
        if score > best_score_overall:
            best_family_overall = family
            best_score_overall = score
    
    if best_family_overall is None:
        best_family_overall = _greedy_chain_init(n, m)
    
    # Convert back to lists of ints
    result = []
    for s in best_family_overall:
        lst = []
        for b in range(n):
            if s & (1 << b):
                lst.append(b)
        result.append(lst)
    
    return result
# EVOLVE-BLOCK-END
