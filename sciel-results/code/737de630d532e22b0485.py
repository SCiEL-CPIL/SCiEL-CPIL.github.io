import time
import random
import heapq
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))
N = len(ALL_TRIPLES)  # 560

# Precompute: for each triple, the set of triples that would form K(2,2,2) with it
# Two triples (a,b,c) and (a',b',c') form K(2,2,2) iff a,a',b,b',c,c' are 6 distinct vertices
# and all 8 transversals exist. But for the conflict graph, we need:
# triple i and triple j are adjacent if there exist 4 more triples making all 8.
# Actually, the standard approach: precompute for each pair of "opposite" triples.
# 
# Better approach: for each triple (a,b,c), the "opposite" position is (a',b',c') where
# a',b',c' are distinct from a,b,c and each other. The 8 transversals are:
# (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
# 
# For the conflict graph: triple i=(a,b,c) conflicts with triple j=(x,y,z) if
# {x,y,z} is a valid "opposite" of {a,b,c} (i.e., x,y,z distinct from a,b,c and each other)
# AND all 6 remaining transversals are present. But that's state-dependent.
# 
# Actually for a STATIC conflict graph: i and j are adjacent if they CAN be part of a K(2,2,2).
# That is, if {a,b,c} and {x,y,z} are disjoint 3-sets (6 distinct vertices), then they are
# "potentially conflicting" - but they only actually conflict if the other 6 transversals exist.
#
# The correct approach for maximum independent set: the conflict is NOT static. Two triples
# only conflict if the other 6 transversals are also present. So this is NOT a simple graph
# independent set problem.
#
# Let me reconsider. The constraint is: no 6 vertices {a,a',b,b',c,c'} with all 8 transversals.
# So when we add triple (a,b,c), it creates a conflict only if for some a',b',c', the other
# 7 transversals are already present (or will be present).
#
# The current approach of checking at insertion time is correct. Let me optimize it.

# Precompute partner masks: for each ordered pair (u,v), which w form a present triple
# This is dynamic, so we maintain it during construction.

# Precompute the list of all possible "opposite" triples for each triple
# For triple (a,b,c), the opposite triples are (a',b',c') where a',b',c' are from {0..15}\{a,b,c}
# and distinct from each other.

def _precompute_opposites():
    """For each triple index, list of indices of triples that are 'opposite' (disjoint vertex sets)."""
    opposites = [[] for _ in range(N)]
    for i, (a, b, c) in enumerate(ALL_TRIPLES):
        used = {a, b, c}
        avail = [v for v in range(V) if v not in used]
        for j in range(N):
            if j == i:
                continue
            a2, b2, c2 = ALL_TRIPLES[j]
            if a2 not in used and b2 not in used and c2 not in used:
                opposites[i].append(j)
    return opposites

_OPPOSITES = _precompute_opposites()

def _check_conflict(a, b, c, partner_mask):
    """
    Check if adding triple (a,b,c) creates a K(2,2,2).
    partner_mask[u*V+v] is a bitmask of w such that (u,v,w) is present.
    """
    # a' candidates: w such that (w,b,c) present, w != a,b,c
    mask_a = partner_mask[b*V + c] & ~(1 << a) & ~(1 << b) & ~(1 << c)
    if not mask_a:
        return False
    
    # b' base: w such that (a,w,c) present, w != a,b,c
    mask_b_base = partner_mask[a*V + c] & ~(1 << a) & ~(1 << b) & ~(1 << c)
    # c' base: w such that (a,b,w) present, w != a,b,c
    mask_c_base = partner_mask[a*V + b] & ~(1 << a) & ~(1 << b) & ~(1 << c)
    
    while mask_a:
        lsb = mask_a & -mask_a
        ap = lsb.bit_length() - 1
        mask_a ^= lsb
        
        # b' must also satisfy (ap, b', c) present
        mask_b = mask_b_base & partner_mask[ap*V + c] & ~(1 << ap)
        if not mask_b:
            continue
        
        # c' must also satisfy (ap, b, c') present
        mask_c = mask_c_base & partner_mask[ap*V + b] & ~(1 << ap)
        if not mask_c:
            continue
        
        while mask_b:
            lsb_b = mask_b & -mask_b
            bp = lsb_b.bit_length() - 1
            mask_b ^= lsb_b
            
            # c' must satisfy (a, bp, c') and (ap, bp, c')
            mask_c_final = mask_c & partner_mask[a*V + bp] & partner_mask[ap*V + bp] & ~(1 << bp)
            if mask_c_final:
                return True
    return False

def _add(a, b, c, pm):
    pm[a*V+b] |= 1<<c; pm[b*V+a] |= 1<<c
    pm[a*V+c] |= 1<<b; pm[c*V+a] |= 1<<b
    pm[b*V+c] |= 1<<a; pm[c*V+b] |= 1<<a

def _remove(a, b, c, pm):
    pm[a*V+b] &= ~(1<<c); pm[b*V+a] &= ~(1<<c)
    pm[a*V+c] &= ~(1<<b); pm[c*V+a] &= ~(1<<b)
    pm[b*V+c] &= ~(1<<a); pm[c*V+b] &= ~(1<<a)

def _greedy_construct(order, pm, presence, deadline):
    """Add triples in given order, skipping conflicts."""
    count = 0
    for idx in order:
        if time.time() >= deadline:
            break
        a, b, c = ALL_TRIPLES[idx]
        if _check_conflict(a, b, c, pm):
            continue
        _add(a, b, c, pm)
        presence[idx] = True
        count += 1
    return count

def _local_search(pm, presence, count, deadline):
    """
    Local search: 
    1. Try to add any missing triple (forward scan)
    2. Try 1-swap: remove one present, try to add as many missing as possible, accept if net gain
    """
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: add any possible missing triple
        for idx in range(N):
            if time.time() >= deadline:
                return count, presence, pm
            if presence[idx]:
                continue
            a, b, c = ALL_TRIPLES[idx]
            if not _check_conflict(a, b, c, pm):
                _add(a, b, c, pm)
                presence[idx] = True
                count += 1
                improved = True
        
        if improved:
            continue
        
        # Phase 2: try swaps
        present_list = [i for i in range(N) if presence[i]]
        if not present_list:
            break
        
        random.shuffle(present_list)
        max_swaps = min(30, len(present_list))
        
        for idx_out in present_list[:max_swaps]:
            if time.time() >= deadline:
                return count, presence, pm
            
            a, b, c = ALL_TRIPLES[idx_out]
            # Save state
            pm_backup = pm[:]
            pres_backup = presence[:]
            count_backup = count
            
            _remove(a, b, c, pm)
            presence[idx_out] = False
            count -= 1
            
            # Try to add all possible missing triples
            for idx_in in range(N):
                if time.time() >= deadline:
                    pm[:] = pm_backup
                    presence[:] = pres_backup
                    count = count_backup
                    return count, presence, pm
                if presence[idx_in]:
                    continue
                ai, bi, ci = ALL_TRIPLES[idx_in]
                if not _check_conflict(ai, bi, ci, pm):
                    _add(ai, bi, ci, pm)
                    presence[idx_in] = True
                    count += 1
            
            if count > count_backup:
                improved = True
                break
            else:
                # Revert
                pm[:] = pm_backup
                presence[:] = pres_backup
                count = count_backup
    
    return count, presence, pm

def _weighted_greedy(weights, pm, presence, deadline):
    """
    Weighted greedy: repeatedly pick the highest-weight available triple (no conflict with selected).
    After each selection, update weights of remaining candidates.
    """
    selected = set()
    count = 0
    remaining = list(range(N))
    
    while remaining and time.time() < deadline:
        # Find best candidate
        best_idx = -1
        best_weight = -1
        for idx in remaining:
            if presence[idx]:
                continue
            a, b, c = ALL_TRIPLES[idx]
            if _check_conflict(a, b, c, pm):
                continue
            w = weights[idx]
            if w > best_weight:
                best_weight = w
                best_idx = idx
        
        if best_idx == -1:
            break
        
        a, b, c = ALL_TRIPLES[best_idx]
        _add(a, b, c, pm)
        presence[best_idx] = True
        selected.add(best_idx)
        count += 1
        remaining.remove(best_idx)
        
        # Update weights: decrease weight of triples that conflict with the new one
        # (This is heuristic - we don't precompute static conflicts)
    
    return count

def construct():
    start = time.time()
    deadline = start + 110  # safety margin for 120s limit
    
    best_count = 0
    best_presence = [False] * N
    
    random.seed(42)
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        pm = [0] * (V * V)
        presence = [False] * N
        
        # Strategy: random order greedy + local search
        order = list(range(N))
        random.shuffle(order)
        
        count = _greedy_construct(order, pm, presence, deadline)
        count, presence, pm = _local_search(pm, presence, count, deadline)
        
        if count > best_count:
            best_count = count
            best_presence = presence[:]
    
    result = []
    for i in range(N):
        if best_presence[i]:
            result.append(list(ALL_TRIPLES[i]))
    return result

random.seed(42)
