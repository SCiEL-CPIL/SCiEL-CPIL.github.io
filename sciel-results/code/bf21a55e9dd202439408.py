"""Rewritten program: optimize reciprocal sum over subset of [2,256] with lcm constraint."""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm
from fractions import Fraction
import itertools

DEADLINE = 80  # seconds, safety margin before 90s kill

def _lcm(a, b):
    return a * b // gcd(a, b)

def _valid_set(S):
    """Check if all distinct pairs in S have lcm > 256."""
    S = list(S)
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            if _lcm(S[i], S[j]) <= 256:
                return False
    return True

def _score(S):
    """Exact reciprocal sum as Fraction."""
    s = Fraction(0, 1)
    for x in S:
        s += Fraction(1, x)
    return s

def _greedy_construct():
    """Greedy by descending 1/x (i.e., ascending x), adding if compatible."""
    selected = []
    for x in range(2, 257):
        if all(_lcm(x, y) > 256 for y in selected):
            selected.append(x)
    return selected

def _greedy_by_weight():
    """Greedy: sort by 1/x descending (smallest x first), add if compatible."""
    # This is the same as _greedy_construct since we iterate 2,3,4,...
    # Let's try a different order: sort by 1/x descending means smallest x first
    # Actually let's try sorting by x ascending (which gives highest 1/x first)
    selected = []
    for x in range(2, 257):
        if all(_lcm(x, y) > 256 for y in selected):
            selected.append(x)
    return selected

def _local_search(S, max_iters=50000):
    """Improve set via add/remove/swap moves."""
    S = set(S)
    best = _score(S)
    improved = True
    while improved and time.time() < DEADLINE - 5:
        improved = False
        # Try adding elements
        for x in range(2, 257):
            if x not in S:
                if all(_lcm(x, y) > 256 for y in S):
                    newS = S | {x}
                    newScore = _score(newS)
                    if newScore > best:
                        S = newS
                        best = newScore
                        improved = True
        # Try swapping: remove one, add another
        if time.time() >= DEADLINE - 5:
            break
        for x in list(S):
            for y in range(2, 257):
                if y not in S:
                    newS = (S - {x}) | {y}
                    if _valid_set(newS):
                        newScore = _score(newS)
                        if newScore > best:
                            S = newS
                            best = newScore
                            improved = True
    return list(S), best

def _local_search2(S, max_rounds=200):
    """More aggressive local search with multiple passes."""
    S = set(S)
    best = _score(S)
    for _ in range(max_rounds):
        if time.time() >= DEADLINE - 5:
            break
        improved = False
        # Add moves
        for x in range(2, 257):
            if x not in S:
                if all(_lcm(x, y) > 256 for y in S):
                    newS = S | {x}
                    newScore = _score(newS)
                    if newScore > best:
                        S = newS
                        best = newScore
                        improved = True
        if not improved:
            break
    return list(S), best

def _try_small_pivot(small, others):
    """Given a small number, find best set of compatible large numbers."""
    # Filter others to those compatible with small
    compatible = [x for x in others if _lcm(small, x) > 256]
    # Now find max reciprocal sum subset of compatible with pairwise lcm > 256
    # Greedy: sort by 1/x descending
    selected = [small]
    for x in compatible:
        if all(_lcm(x, y) > 256 for y in selected):
            selected.append(x)
    return selected

def construct():
    start_time = time.time()
    
    # Initial greedy solution
    best_set = _greedy_construct()
    best_score = _score(best_set)
    
    # Try local search on initial
    if time.time() < DEADLINE - 10:
        ls_set, ls_score = _local_search(best_set)
        if ls_score > best_score:
            best_set = ls_set
            best_score = ls_score
    
    # Try different starting points
    if time.time() < DEADLINE - 10:
        # Try with each small number as forced inclusion
        for small in range(2, 20):
            if time.time() >= DEADLINE - 5:
                break
            others = list(range(2, 257))
            others.remove(small)
            cand = _try_small_pivot(small, others)
            cand_score = _score(cand)
            if cand_score > best_score:
                best_set = cand
                best_score = cand_score
            
            # Local search from this candidate
            if time.time() < DEADLINE - 5:
                ls_set, ls_score = _local_search2(cand, max_rounds=50)
                if ls_score > best_score:
                    best_set = ls_set
                    best_score = ls_score
    
    # Try greedy in different orders
    if time.time() < DEADLINE - 10:
        # Greedy by descending x (largest first)
        selected = []
        for x in range(256, 1, -1):
            if all(_lcm(x, y) > 256 for y in selected):
                selected.append(x)
        if selected:
            sc = _score(selected)
            if sc > best_score:
                best_set = selected
                best_score = sc
        
        # Greedy by descending x, then local search
        if time.time() < DEADLINE - 5:
            ls_set, ls_score = _local_search2(selected, max_rounds=50)
            if ls_score > best_score:
                best_set = ls_set
                best_score = ls_score
    
    # Final local search on best
    if time.time() < DEADLINE - 3:
        ls_set, ls_score = _local_search(best_set, max_iters=100000)
        if ls_score > best_score:
            best_set = ls_set
            best_score = ls_score
    
    # Verify final answer
    if not _valid_set(best_set):
        # Fallback to greedy
        best_set = _greedy_construct()
    
    return best_set
# EVOLVE-BLOCK-END
