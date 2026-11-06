"""
Strategy: CSP-style backtracking with dynamic variable ordering + greedy set-cover construction
We combine a fast greedy construction (multiple random restarts) with a CSP-inspired
backtracking search that uses forward checking to prune the space, plus an intensive
local-search polish. The key insight is that for small grids (N=16, V=256), we can
afford a more principled search if we use bitsets aggressively.
"""

import itertools
import time
import random
import numpy as np

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
V = len(POINTS)

# Precompute squared distances
dist2 = [[0] * V for _ in range(V)]
for i in range(V):
    xi, yi = POINTS[i]
    for j in range(i + 1, V):
        xj, yj = POINTS[j]
        d = (xi - xj) ** 2 + (yi - yj) ** 2
        dist2[i][j] = d
        dist2[j][i] = d

# Precompute conflict bitsets using Python ints (arbitrary precision)
# conflict_pair[i][j] = bitset of all k such that triangle(i,j,k) is isosceles
conflict_pair = [[0] * V for _ in range(V)]
for i in range(V):
    for j in range(i + 1, V):
        d_ij = dist2[i][j]
        mask = 0
        for k in range(V):
            if k == i or k == j:
                continue
            d_ik = dist2[i][k]
            d_jk = dist2[j][k]
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                mask |= (1 << k)
        conflict_pair[i][j] = mask
        conflict_pair[j][i] = mask

# For each point p, precompute which pairs (i,j) would be "blocked" if p were added
# i.e., blocked_by[p] = set of (i,j) such that p in conflict_pair[i][j]
# Actually, we need: if S contains i and j, then p is blocked.
# So for a given S, p is blocked if there exists i,j in S with p in conflict_pair[i][j].

def _bits_to_list(bits):
    lst = []
    tmp = bits
    while tmp:
        lsb = tmp & (-tmp)
        idx = lsb.bit_length() - 1
        lst.append(idx)
        tmp ^= lsb
    return lst

def _is_independent_with_S(p, S_bits):
    """Check if adding p to S keeps S independent."""
    s_list = _bits_to_list(S_bits)
    for i in range(len(s_list)):
        for j in range(i + 1, len(s_list)):
            if (conflict_pair[s_list[i]][s_list[j]] >> p) & 1:
                return False
    return True

def _is_maximal(S_bits):
    """Check that S is maximal: no point can be added."""
    for p in range(V):
        if (S_bits >> p) & 1:
            continue
        if _is_independent_with_S(p, S_bits):
            return False
    return True

def _verify_independent(S_bits):
    """Check that S is an independent no-isosceles set."""
    s_list = _bits_to_list(S_bits)
    for i in range(len(s_list)):
        for j in range(i + 1, len(s_list)):
            for k in range(j + 1, len(s_list)):
                if (conflict_pair[s_list[i]][s_list[j]] >> s_list[k]) & 1:
                    return False
    return True

def _construct_greedy_coverage(seed=0):
    """
    Greedy set-cover style construction with bitset optimization.
    """
    random.seed(seed)
    S_bits = 0
    all_bits = (1 << V) - 1
    unblocked = all_bits
    
    while unblocked:
        best_p = -1
        best_score = -1
        
        ub_list = _bits_to_list(unblocked)
        
        for p in ub_list:
            if not _is_independent_with_S(p, S_bits):
                continue
            
            blocked_by_p = 0
            tmp_s = S_bits
            while tmp_s:
                lsb = tmp_s & (-tmp_s)
                s = lsb.bit_length() - 1
                blocked_by_p |= conflict_pair[p][s]
                tmp_s ^= lsb
            
            score = (unblocked & blocked_by_p & ~(1 << p)).bit_count()
            noise = random.random() * 0.5
            total_score = score + noise
            
            if total_score > best_score:
                best_score = total_score
                best_p = p
        
        if best_p == -1:
            break
            
        S_bits |= (1 << best_p)
        
        blocked_by_new = 0
        tmp_s = S_bits & ~(1 << best_p)
        while tmp_s:
            lsb = tmp_s & (-tmp_s)
            s = lsb.bit_length() - 1
            blocked_by_new |= conflict_pair[best_p][s]
            tmp_s ^= lsb
        
        unblocked &= ~(1 << best_p)
        unblocked &= ~blocked_by_new

    return S_bits

def _backtracking_search(deadline, initial_size_limit=None):
    """
    CSP-style backtracking search with dynamic variable ordering.
    We try to build a small maximal independent set by carefully choosing which points to include.
    
    The key idea: we process points in a dynamic order, and at each step we decide
    whether to include or exclude a point. Forward checking ensures that if we exclude
    a point, we check that it's still "coverable" (i.e., it will be blocked by some
    pair in the final set).
    
    For efficiency, we use a heuristic: prefer to include points that block many
    remaining candidates, and use branch-and-bound with the current best size.
    """
    best_bits = None
    best_size = V
    
    if initial_size_limit is not None:
        best_size = initial_size_limit
    
    # We'll do a limited backtracking search
    # State: S (bitset of included points), candidates (bitset of points not yet decided)
    # A point in candidates can be included if S U {p} is independent
    # At the end, S must be maximal
    
    def _search(S_bits, candidates, depth, max_depth):
        nonlocal best_bits, best_size
        
        if time.time() > deadline:
            return
        
        # If S is already larger than best, prune
        if S_bits.bit_count() >= best_size:
            return
        
        # If no candidates left, check if S is maximal
        if candidates == 0:
            if _is_maximal(S_bits):
                sz = S_bits.bit_count()
                if sz < best_size:
                    best_size = sz
                    best_bits = S_bits
            return
        
        if depth >= max_depth:
            # Check if current S can be extended to a maximal set
            # Quick check: if S is already maximal, we're done
            if _is_maximal(S_bits):
                sz = S_bits.bit_count()
                if sz < best_size:
                    best_size = sz
                    best_bits = S_bits
            return
        
        # Dynamic variable ordering: pick the candidate that blocks the most other candidates
        # This is a "most constrained" heuristic
        cand_list = _bits_to_list(candidates)
        
        # Score each candidate by how many other candidates it would block if added
        scored = []
        for p in cand_list:
            if not _is_independent_with_S(p, S_bits):
                continue
            # How many candidates would be blocked?
            blocked = 0
            tmp_s = S_bits
            while tmp_s:
                lsb = tmp_s & (-tmp_s)
                s = lsb.bit_length() - 1
                blocked |= conflict_pair[p][s]
                tmp_s ^= lsb
            blocked_count = (candidates & blocked & ~(1 << p)).bit_count()
            scored.append((blocked_count, p))
        
        if not scored:
            # No valid candidates to add; check if S is maximal
            if _is_maximal(S_bits):
                sz = S_bits.bit_count()
                if sz < best_size:
                    best_size = sz
                    best_bits = S_bits
            return
        
        # Sort by blocked_count descending (include most impactful first)
        scored.sort(reverse=True)
        
        # Try including the top few candidates (limit branching)
        max_branch = min(3, len(scored))
        for _, p in scored[:max_branch]:
            new_S = S_bits | (1 << p)
            
            # Update candidates: remove p and any that are now blocked
            blocked_by_p = 0
            tmp_s = new_S & ~(1 << p)
            while tmp_s:
                lsb = tmp_s & (-tmp_s)
                s = lsb.bit_length() - 1
                blocked_by_p |= conflict_pair[p][s]
                tmp_s ^= lsb
            
            new_candidates = candidates & ~(1 << p) & ~blocked_by_p
            
            _search(new_S, new_candidates, depth + 1, max_depth)
            
            if time.time() > deadline:
                return
            
            # Also try excluding p (but only if p is not essential for maximality)
            # This is expensive; only do it for the first candidate
            if p == scored[0][1]:
                # Try excluding p
                new_candidates_excl = candidates & ~(1 << p)
                # Check: is p "blocked" by current S? If not, excluding it might still allow maximality
                # For now, skip exclusion to keep search tractable
                pass
    
    # Start search with empty S and all candidates
    # Use a moderate max_depth to keep it tractable
    max_depth = 25
    _search(0, (1 << V) - 1, 0, max_depth)
    
    return best_bits, best_size

def _local_search_polish(S_bits, max_iters=30, time_limit=None):
    """
    Try to reduce S by swapping points.
    """
    current = S_bits
    current_size = current.bit_count()
    
    for iteration in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
        
        s_list = _bits_to_list(current)
        improved = False
        
        for remove_idx in range(len(s_list)):
            r = s_list[remove_idx]
            S_without_r = current & ~(1 << r)
            
            all_bits = (1 << V) - 1
            candidates = all_bits & ~S_without_r
            
            valid_candidates = []
            tmp_c = candidates
            while tmp_c:
                lsb = tmp_c & (-tmp_c)
                c = lsb.bit_length() - 1
                tmp_c ^= lsb
                
                if _is_independent_with_S(c, S_without_r):
                    valid_candidates.append(c)
            
            if not valid_candidates:
                continue
            
            best_c = -1
            best_block_count = -1
            
            for c in valid_candidates:
                new_S = S_without_r | (1 << c)
                
                # Check maximality
                is_maximal = True
                for p in range(V):
                    if (new_S >> p) & 1:
                        continue
                    if _is_independent_with_S(p, new_S):
                        is_maximal = False
                        break
                
                if not is_maximal:
                    continue
                
                new_s_list = _bits_to_list(new_S)
                block_count = 0
                for i in range(len(new_s_list)):
                    for j in range(i + 1, len(new_s_list)):
                        block_count += conflict_pair[new_s_list[i]][new_s_list[j]].bit_count()
                
                if block_count > best_block_count:
                    best_block_count = block_count
                    best_c = c
            
            if best_c != -1:
                current = S_without_r | (1 << best_c)
                if current.bit_count() < current_size:
                    current_size = current.bit_count()
                    improved = True
                    break
        
        if not improved:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_bits = None
    best_size = V
    
    # Phase 1: Greedy constructions with multiple seeds
    num_greedy = 50
    
    for seed in range(num_greedy):
        if time.time() > deadline - 60.0:
            break
            
        S_bits = _construct_greedy_coverage(seed)
        
        if not _verify_independent(S_bits) or not _is_maximal(S_bits):
            continue
        
        size = S_bits.bit_count()
        if size < best_size:
            best_size = size
            best_bits = S_bits
            if best_size <= 10:
                break
    
    # Phase 2: Backtracking search to find smaller sets
    if time.time() < deadline - 30.0:
        bt_bits, bt_size = _backtracking_search(deadline - 30.0, initial_size_limit=best_size)
        if bt_bits is not None:
            if _verify_independent(bt_bits) and _is_maximal(bt_bits):
                if bt_size < best_size:
                    best_size = bt_size
                    best_bits = bt_bits
    
    # Phase 3: Local search polish on the best solution
    if best_bits is not None and time.time() < deadline - 10.0:
        polish_deadline = min(deadline - 5.0, start_time + 285.0)
        polished = _local_search_polish(best_bits, max_iters=50, time_limit=polish_deadline)
        
        if _verify_independent(polished) and _is_maximal(polished):
            if polished.bit_count() < best_size:
                best_size = polished.bit_count()
                best_bits = polished
    
    # Phase 4: More greedy restarts if time remains
    if time.time() < deadline - 15.0:
        for seed in range(100, 200):
            if time.time() > deadline - 15.0:
                break
            S_bits = _construct_greedy_coverage(seed)
            if not _verify_independent(S_bits) or not _is_maximal(S_bits):
                continue
            size = S_bits.bit_count()
            if size < best_size:
                best_size = size
                best_bits = S_bits
                if best_size <= 8:
                    break
    
    # Phase 5: Final polish
    if best_bits is not None and time.time() < deadline - 5.0:
        polished = _local_search_polish(best_bits, max_iters=30, time_limit=deadline - 3.0)
        if _verify_independent(polished) and _is_maximal(polished):
            if polished.bit_count() < best_size:
                best_size = polished.bit_count()
                best_bits = polished
    
    if best_bits is None:
        best_bits = _construct_greedy_coverage(0)
    
    result = _bits_to_list(best_bits)
    return [POINTS[i] for i in result]

# EVOLVE-BLOCK-END
