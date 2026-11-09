"""Spectral-guided constructive search for the 3-sum non-divisibility task."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

N = 192
MAX_K = 48
DEADLINE_S = 85.0  # safety margin under 90s kill

def _check_set(S):
    """Return True if no element in S divides the sum of any 3 distinct other elements in S."""
    S_list = sorted(S)
    n = len(S_list)
    # Precompute all 3-sums of distinct elements
    # For each a in S, check if a divides any sum of 3 distinct other elements
    for i, a in enumerate(S_list):
        others = [x for j, x in enumerate(S_list) if j != i]
        # Check all triples from others
        for tri in combinations(others, 3):
            if sum(tri) % a == 0:
                return False
    return True

def _check_addition(S_set, candidate):
    """Check if adding candidate to S_set (as a set of ints) is valid.
    Only need to check:
    1. candidate does not divide sum of any 3 distinct elements in S_set
    2. No element in S_set divides sum of candidate + any 2 distinct elements in S_set
    3. No element in S_set divides sum of any 3 distinct other elements in S_set (already valid)
    """
    S_list = sorted(S_set)
    # Condition 1: candidate doesn't divide sum of any 3 distinct elements in S_set
    for tri in combinations(S_list, 3):
        if sum(tri) % candidate == 0:
            return False
    # Condition 2: for each a in S_set, a doesn't divide (candidate + b + c) for distinct b,c in S_set
    for a in S_list:
        for bc in combinations(S_list, 2):
            s = candidate + bc[0] + bc[1]
            if s % a == 0:
                return False
    return True

def _greedy_build(order, max_size=MAX_K):
    """Greedy construction following given order."""
    selected = set()
    for x in order:
        if len(selected) >= max_size:
            break
        if _check_addition(selected, x):
            selected.add(x)
    return selected

def _local_search(S, deadline):
    """Local search with swaps and 1-out-2-in moves."""
    S_set = set(S)
    best_size = len(S_set)
    
    while time.time() < deadline:
        improved = False
        S_list = sorted(S_set)
        
        # Try swaps: remove one, add one
        for i, a in enumerate(S_list):
            remaining = [x for x in S_list if x != a]
            for cand in range(1, N + 1):
                if cand in S_set:
                    continue
                # Check if replacing a with cand is valid
                new_set = set(remaining)
                new_set.add(cand)
                # Only need to check interactions involving cand
                # cand doesn't divide sum of any 3 in remaining
                valid = True
                for tri in combinations(remaining, 3):
                    if sum(tri) % cand == 0:
                        valid = False
                        break
                if valid:
                    # Each b in remaining doesn't divide (cand + c + d) for distinct c,d in remaining
                    for b in remaining:
                        for cd in combinations(remaining, 2):
                            if cd[0] == b or cd[1] == b:
                                continue
                            if (cand + cd[0] + cd[1]) % b == 0:
                                valid = False
                                break
                        if not valid:
                            break
                if valid and len(new_set) > best_size:
                    S_set = new_set
                    best_size = len(S_set)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # Try 1-out-2-in
            S_list = sorted(S_set)
            for a in S_list:
                remaining = [x for x in S_list if x != a]
                for cand1 in range(1, N + 1):
                    if cand1 in S_set:
                        continue
                    for cand2 in range(cand1 + 1, N + 1):
                        if cand2 in S_set:
                            continue
                        new_set = set(remaining)
                        new_set.add(cand1)
                        new_set.add(cand2)
                        if len(new_set) <= best_size:
                            continue
                        # Full check
                        if _check_set(new_set):
                            S_set = new_set
                            best_size = len(new_set)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        if not improved:
            break
    return S_set

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_S
    
    # Spectral-inspired ordering: prioritize numbers with fewer divisibility conflicts
    # Compute a conflict score for each number
    conflict_scores = np.zeros(N + 1)
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            if a == b:
                continue
            for c in range(1, N + 1):
                if a == c or b == c:
                    continue
                if (a + b + c) % a == 0 or (a + b + c) % b == 0 or (a + b + c) % c == 0:
                    conflict_scores[a] += 1
                    conflict_scores[b] += 1
                    conflict_scores[c] += 1
    
    # Sort by ascending conflict score (fewer conflicts first)
    base_order = sorted(range(1, N + 1), key=lambda x: conflict_scores[x])
    
    # Start with greedy on base order
    best_set = _greedy_build(base_order)
    best_size = len(best_set)
    
    # Try multiple randomized perturbations of the order
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Create a perturbed order
        if rng.random() < 0.5:
            # Shuffle a portion
            order = base_order[:]
            idx = rng.randint(0, len(order) - 2)
            length = rng.randint(5, 30)
            order[idx:idx+length] = rng.sample(order[idx:idx+length], min(length, len(order)-idx))
        else:
            # Reverse a random segment
            order = base_order[:]
            i = rng.randint(0, len(order) - 2)
            j = rng.randint(i + 1, len(order) - 1)
            order[i:j+1] = reversed(order[i:j+1])
        
        candidate_set = _greedy_build(order)
        if len(candidate_set) > best_size:
            best_set = candidate_set
            best_size = len(candidate_set)
        
        # Local search on the best set found so far
        if time.time() < deadline - 2:
            improved_set = _local_search(best_set, deadline - 1)
            if len(improved_set) > best_size:
                best_set = improved_set
                best_size = len(best_set)
        
        # If we hit max, return
        if best_size >= MAX_K:
            break
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
