"""Rewritten program: Best-Improvement 2-opt with O(1) delta evaluation via precomputed comparability bitsets."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

N = 12
M = 40
TOTAL_SUBSETS = 1 << N

# Precompute comparability as bitsets for O(1) delta evaluation
# For each mask i, comp_bits[i] is an integer where bit j is set iff i and j are comparable (i != j, one is strict subset of other)
_COMP_BITS = [0] * TOTAL_SUBSETS

def _precompute_comparability():
    """Precompute comparability bitsets."""
    for i in range(TOTAL_SUBSETS):
        mi = i
        bits = 0
        for j in range(TOTAL_SUBSETS):
            if i == j:
                continue
            mj = j
            if (mi & mj) == mi or (mi & mj) == mj:
                bits |= (1 << j)
        _COMP_BITS[i] = bits

# Precompute at module level
_precompute_comparability()

def _score_solution(selected_set):
    """Count comparable pairs in a set of masks."""
    count = 0
    sel = list(selected_set)
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            if (_COMP_BITS[sel[i]] >> sel[j]) & 1:
                count += 1
    return count

def _initial_solution_chain():
    """Generate initial solution using multiple maximal chains."""
    # Build all chains: a chain is a sequence of subsets where each is a strict subset of the next
    # We want to select 40 subsets that maximize comparable pairs
    # Strategy: pick several long chains and combine them
    
    # First, let's find good chains by greedily building them
    # A chain from empty set to full set has 13 elements (sizes 0 through 12)
    # We can have multiple chains that share some elements
    
    # Better approach: use a greedy method
    # Start with empty set, repeatedly add the mask that maximizes comparable pairs with current set
    selected = set()
    remaining = set(range(TOTAL_SUBSETS))
    
    # Precompute for greedy: for each mask, how many comparable masks are in current set
    # We'll just iterate
    for _ in range(M):
        best_mask = None
        best_gain = -1
        for m in remaining:
            # Count how many in selected are comparable with m
            gain = 0
            for s in selected:
                if (_COMP_BITS[m] >> s) & 1:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = m
        if best_mask is None:
            break
        selected.add(best_mask)
        remaining.discard(best_mask)
    
    return selected

def _initial_solution_greedy():
    """Greedy construction: repeatedly add mask with max comparable neighbors in current set."""
    selected = set()
    remaining = set(range(TOTAL_SUBSETS))
    
    for _ in range(M):
        best_mask = None
        best_gain = -1
        for m in remaining:
            gain = 0
            for s in selected:
                if (_COMP_BITS[m] >> s) & 1:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = m
        if best_mask is None:
            break
        selected.add(best_mask)
        remaining.discard(best_mask)
    
    return selected

def _best_improvement_2opt(selected_set, time_limit=60.0):
    """
    Best-Improvement 2-opt: try all (remove one, add one) swaps, keep the best.
    Use O(1) delta evaluation.
    
    For a swap: remove r from S, add a to S (a not in S).
    New score = old_score - (comparable(r, S\{r})) + (comparable(a, S\{r}))
    = old_score - popcount(_COMP_BITS[r] & bits(S\{r})) + popcount(_COMP_BITS[a] & bits(S\{r}))
    
    We maintain a bitset of selected masks for fast intersection.
    """
    start_time = time.time()
    selected = set(selected_set)
    score = _score_solution(selected)
    
    # Maintain bitset of selected
    sel_bits = 0
    for s in selected:
        sel_bits |= (1 << s)
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        best_delta = 0
        best_swap = None
        
        # Try all pairs (r in S, a not in S)
        # This is |S| * (TOTAL_SUBSETS - |S|) = 40 * 4056 = 162240 evaluations
        # Each is O(1) with bit operations
        for r in selected:
            r_bits = _COMP_BITS[r]
            # Comparable count of r with S\{r}
            r_comp_with_rest = bin(r_bits & (sel_bits & ~(1 << r))).count('1')
            
            for a in range(TOTAL_SUBSETS):
                if a in selected:
                    continue
                a_bits = _COMP_BITS[a]
                # Comparable count of a with S\{r}
                a_comp_with_rest = bin(a_bits & (sel_bits & ~(1 << r))).count('1')
                
                delta = a_comp_with_rest - r_comp_with_rest
                if delta > best_delta:
                    best_delta = delta
                    best_swap = (r, a)
        
        if best_swap is not None:
            r, a = best_swap
            selected.discard(r)
            selected.add(a)
            sel_bits = (sel_bits & ~(1 << r)) | (1 << a)
            score += best_delta
            improved = True
    
    return selected, score

def _random_restart_2opt(selected_set, time_limit=60.0, n_restarts=3):
    """Multiple restarts with perturbation + best-improvement 2-opt."""
    start_time = time.time()
    best_selected = set(selected_set)
    best_score = _score_solution(best_selected)
    rng = random.Random(42)
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Perturb: random swap of k elements
        perturbed = set(selected_set)
        k = min(5, len(perturbed) // 2)
        for _ in range(k):
            if len(perturbed) < 2:
                break
            items = list(perturbed)
            r = rng.choice(items)
            a = rng.randrange(TOTAL_SUBSETS)
            if a not in perturbed:
                perturbed.discard(r)
                perturbed.add(a)
        
        # Run 2-opt on perturbed
        time_left = time_limit - (time.time() - start_time) - 2.0
        if time_left < 1.0:
            break
        improved, sc = _best_improvement_2opt(perturbed, time_limit=time_left)
        
        if sc > best_score:
            best_score = sc
            best_selected = improved
    
    return best_selected, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Generate initial solution
    selected = _initial_solution_greedy()
    score = _score_solution(selected)
    
    # Run best-improvement 2-opt
    time_left = deadline - time.time() - 3.0
    if time_left > 5.0:
        selected, score = _best_improvement_2opt(selected, time_limit=time_left)
    
    # If time remains, do a random restart
    if time.time() < deadline - 10.0:
        time_left2 = deadline - time.time() - 5.0
        if time_left2 > 5.0:
            selected2, score2 = _random_restart_2opt(selected, time_limit=time_left2, n_restarts=2)
            if score2 > score:
                selected, score = selected2, score2
    
    # Convert to list of lists of ints
    result = []
    for mask in selected:
        s = []
        for b in range(N):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
    
    # Ensure exactly 40 distinct subsets
    assert len(set(tuple(s) for s in result)) == 40, "Not 40 distinct subsets"
    assert len(result) == 40, "Not 40 subsets"
    
    return result
# EVOLVE-BLOCK-END
