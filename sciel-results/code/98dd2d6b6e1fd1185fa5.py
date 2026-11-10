"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import random
import heapq


def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_isosceles(p, a, b):
    d1 = _dist2(p, a)
    d2 = _dist2(p, b)
    d3 = _dist2(a, b)
    return d1 == d2 or d1 == d3 or d2 == d3


def _is_compatible(p, S):
    """Check if p can be added to set S without creating an isosceles triangle."""
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if _is_isosceles(p, S[i], S[j]):
                return False
    return True


def _greedy_with_blocking(points, S, random_state):
    """
    Greedy construction that prioritizes points which block the most other points.
    """
    # Precompute conflict graph? 
    # For N=16, 256 points. 
    # A point p conflicts with a pair (a,b) if {p,a,b} is isosceles.
    # We want to pick p that makes many other points q "un-addable".
    # q is un-addable if it conflicts with some pair in S.
    
    # Efficient check:
    # Maintain a set of "available" points.
    # Initially all points are available.
    # When we pick p, we remove p and all points q that conflict with p and some other point in S?
    # No, q conflicts with S if there exist a,b in S such that {q,a,b} is isosceles.
    # This is dynamic.
    
    # Simpler heuristic:
    # 1. Start with empty S.
    # 2. Calculate for each candidate p, how many other candidates q it "blocks".
    #    q is blocked by p if adding p to S makes q un-addable.
    #    This depends on current S.
    
    # Let's try a simpler scoring:
    # Score(p) = number of points q such that {p, q, a} is isosceles for some a in S?
    # Or just random with restarts.
    
    # Given time, let's stick to randomized greedy but with more trials and a better initial ordering.
    # We can sort points by a heuristic: points with higher "connectivity" in the isosceles graph might be better?
    
    # Let's just do many randomized trials.
    # But we can improve the randomization by using a "temperature" or weighted random choice.
    
    # For now, return the chosen list from the outer loop's logic.
    # This function is a placeholder for the inner logic if we refactor.
    pass


def construct():
    points = [(x, y) for x in range(N) for y in range(N)]
    random.seed(42)
    
    best_set = None
    best_size = float('inf')
    
    # Increase trials significantly
    num_trials = 50
    
    for trial in range(num_trials):
        # Randomize the order of points
        shuffled_points = points[:]
        random.shuffle(shuffled_points)
        
        chosen = []
        # Use a set for faster lookup if needed, but list is fine for N=16
        
        for p in shuffled_points:
            if _is_compatible(p, chosen):
                chosen.append(p)
        
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = chosen[:]
            
            # Early stopping if we find a very small set?
            # Hard to know the lower bound.
    
    if best_set is None:
        return []
    
    return best_set
# EVOLVE-BLOCK-END
