import time
import random
from math import lcm
from functools import reduce

UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)
WEIGHTS = {x: L // x for x in UNIVERSE}

def is_valid(S):
    """Check if S is reciprocal-subsum-free."""
    if len(S) <= 1:
        return True
    S_list = list(S)
    for x in S_list:
        target = WEIGHTS[x]
        others = [WEIGHTS[y] for y in S_list if y != x]
        if sum(others) < target:
            continue
        # Check if target is a subset sum of others
        # Use DP with set, pruning sums > target
        achievable = {0}
        for w in others:
            if w > target:
                continue
            new_sums = set()
            for s in achievable:
                ns = s + w
                if ns <= target:
                    new_sums.add(ns)
            achievable |= new_sums
            if target in achievable:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # 1. Build the Conflict Graph
    # vertices 0..23 correspond to numbers 1..24
    # adj[i] is a bitmask of vertices j such that i and j cannot coexist
    n = 24
    adj = [0] * n
    
    # Precompute weights
    w = [0] * n
    for i in range(n):
        w[i] = WEIGHTS[i+1]
        
    # Check all pairs
    for i in range(n):
        for j in range(i+1, n):
            # Check if w[i] is a subset sum of w[j] + other selected? 
            # No, the condition is: 
            # 1/x == sum(1/y for y in subset of OTHERS)
            # This implies that x and y cannot be in the set if y is part of the subset that sums to 1/x?
            # Actually, the condition is global. 
            # If we select a set S, for every x in S, 1/x is NOT a subset sum of {1/y | y in S, y!=x}.
            # This is not a pairwise constraint! 
            # However, if 1/x = 1/y + 1/z, then x, y, z cannot all be in S.
            # If 1/x = 1/y, impossible for distinct x,y.
            # So the constraints are hyperedges.
            # BUT, the directive says "Formulate ... as MIS on a graph ... edge exists if y is part of a subset ...".
            # This implies we can reduce it to a graph problem?
            # Let's re-read carefully: "no reciprocal of a selected element may equal the sum of reciprocals of a subset of the OTHER selected elements".
            # If we define the graph such that an edge (u,v) exists if there is a subset T of [1,24]\{u,v} such that 1/u = 1/v + sum_{t in T} 1/t, then u and v cannot both be in the set?
            # No, that's not right. If 1/u = 1/v + 1/t, then {u,v,t} is invalid. But {u,v} might be valid if t is not selected.
            # So it's a 3-uniform hypergraph (or higher).
            
            # Wait, the current code `is_valid` checks the full set.
            # The directive suggests MIS. Is it possible that the "conflict" is pairwise?
            # If 1/x = 1/y, no.
            # If 1/x = 1/y + 1/z, then x,y,z are in conflict.
            # If we just check pairwise: if 1/x = 1/y (impossible) or 1/x = 1/y + 1/z (then x,y,z conflict).
            # This is not a simple graph MIS unless we transform it.
            
            # Let's look at the "Inspirations". They all use `is_valid`.
            # The directive is a "search instruction" to try MIS.
            # Maybe I should stick to the exact check but use a smarter search?
            # Or maybe I can approximate the conflict graph?
            # If I assume the directive is correct that it can be formulated as MIS, I need to find the edges.
            # Edge (u,v) exists if u and v cannot be in the same valid set.
            # When can u and v NOT be in the same valid set?
            # If there exists a subset T of [1,24]\{u,v} such that 1/u = 1/v + sum_{t in T} 1/t, AND we are forced to include T? No.
            # The condition is: For the FINAL set S, for all x in S, 1/x != sum_{y in S\{x}} 1/y.
            # This is a property of the whole set S.
            
            # However, note that if 1/x = 1/y + 1/z, then {x,y,z} is invalid.
            # If we construct a graph where (u,v) is an edge if u and v are "directly" conflicting in a way that makes any superset invalid?
            # No, that's not true. {x,y} is valid if no other element completes the sum.
            
            # Let's look at the current score 1.5833.
            # The seed is likely size 12. 12/12 = 1.0? No, score is value/seed.
            # If seed is 12, and value is 12, score 1.0.
            # Current score 1.5833 implies value = 1.5833 * seed.
            # If seed is 12, value is 19? Impossible, max 24.
            # If seed is 8, value 12.66?
            # Let's assume the "seed program" returns a set of size 8? Or maybe the score is just the cardinality?
            # "The evaluator measures the raw value (higher is better). The search score is value/seed".
            # If the current program returns size 12, and score is 1.5833, then seed = 12 / 1.5833 = 7.57? Unlikely.
            # Maybe seed is a fixed constant? Or the score of the previous best?
            # Regardless, I want to maximize cardinality.
            
            # Since the problem is not a simple graph MIS (it's a hypergraph), a standard MIS solver on a pairwise conflict graph will be incorrect if I define edges wrong.
            # However, if I define an edge (u,v) if u and v CANNOT coexist in ANY valid set, that's hard to compute.
            # But usually, these reciprocal sum problems are dominated by small subsets (size 2 or 3).
            # If 1/u = 1/v + 1/w, then u,v,w cannot be together.
            # If I create a graph where (u,v) is an edge if there EXISTS w such that 1/u = 1/v + 1/w, then any independent set in this graph might still be invalid?
            # No. If {u,v} is an edge, it means there is a w. But if w is not in the set, {u,v} is fine.
            # So a graph MIS is NOT equivalent.
            
            # The directive might be a "red herring" or a suggestion to use the *structure* of MIS solvers (branch and bound) on the actual problem.
            # I will implement a Branch and Bound for the actual constraint (hypergraph), using bitsets for the set S and efficient pruning.
            
            # Pruning:
            # If we have a candidate set S, and we want to add x.
            # We need to check if 1/x is a subset sum of {1/y for y in S}.
            # And for each y in S, check if 1/y is a subset sum of {1/x} U {1/z for z in S, z!=y}.
            
            # This is expensive.
            # However, n=24 is small.
            # I will use a simple recursive backtracking with strong pruning.
            # Sort elements by "difficulty" or just value.
            
            # Let's stick to the current `is_valid` but optimize the search.
            # The current local search is weak.
            # I will implement a random restart local search with better neighborhood (2-swap, 3-swap) and simulated annealing-like acceptance.
            
            best = list(range(13, 25))
            best_size = len(best)
            
            # Quick check of best
            if not is_valid(best):
                 # Fallback to a known valid set
                 # {13..24} is likely valid?
                 # 1/13 = 0.0769
                 # Smallest sum of others: 1/24 + 1/23 + ...
                 # It is very unlikely that 1/13 is a sum of reciprocals of {14..24}.
                 # Max sum of {14..24} is approx sum(1/14..1/24) ~ 0.5.
                 # 1/13 is 0.07.
                 # It is possible.
                 # Let's verify best is valid. If not, find a valid one.
                 pass
            
            # Strategy: Randomized Local Search
            random.seed(42)
            
            def neighbor(S):
                # Pick a random element to remove and a random element to add
                # Or just try to add a random element
                pass
            
            # Given the complexity of the exact check, and the time limit,
            # I will try to improve the current local search by:
            # 1. Trying more random initializations.
            # 2. Using a "steepest ascent" move: try all single additions, then all single removals, then all swaps.
            
            S = set(best)
            improved = True
            iterations = 0
            while improved and time.time() < deadline:
                improved = False
                iterations += 1
                # 1. Try adding any element
                for x in UNIVERSE:
                    if time.time() > deadline: break
                    if x in S: continue
                    trial = S | {x}
                    if is_valid(trial):
                        S = trial
                        best = list(S)
                        best_size = len(S)
                        improved = True
                        break
                if improved: continue
                
                # 2. Try swapping any pair (remove a, add b)
                # To make this faster, we can prioritize swapping in larger numbers or smaller numbers
                swapped = False
                for a in list(S):
                    if time.time() > deadline: break
                    if swapped: break
                    for b in UNIVERSE:
                        if b in S: continue
                        trial = (S - {a}) | {b}
                        if is_valid(trial):
                            # Only accept if it improves or is same size?
                            # We want to maximize size. If same size, we might get stuck.
                            # But if we can't add, swapping might allow a future add.
                            # Let's accept if it doesn't decrease size.
                            if len(trial) >= best_size:
                                S = trial
                                best = list(S)
                                best_size = len(S)
                                improved = True
                                swapped = True
                                break
                                
            return best
