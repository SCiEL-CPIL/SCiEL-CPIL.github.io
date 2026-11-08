import time
import math
import numpy as np
from itertools import combinations
import sys

# Use pulp if available, otherwise fallback to a smarter heuristic
try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False

def compute_score(U):
    """
    Computes the lower bound exponent: 1 + log(|U-U|/|U+U|)/log(2 max U + 1)
    """
    if len(U) < 2:
        return 0
    U = sorted(set(U))
    maxU = U[-1]
    if maxU == 0:
        return 0
    
    # Efficiently compute sum and difference sets
    sum_set = set()
    diff_set = set()
    
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
            
    ratio = len(diff_set) / len(sum_set)
    if ratio <= 0:
        return 0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * maxU + 1)
    if log_denom <= 0:
        return 0
    return 1 + log_ratio / log_denom

def solve_milp_block_size(n):
    """
    Attempts to find a set of size n (including 0) that maximizes the ratio 
    |U-U|/|U+U| within a bounded range, effectively finding a "dense" optimal block.
    Since the score depends on log(maxU), we want a block with high ratio and small maxU 
    to be used as a base, or just a block that inherently has a good structure.
    
    Actually, for the exponent to be high, we need log(ratio)/log(2M+1) to be close to 1.
    This implies ratio ~ 2M+1.
    |U-U| <= 2M+1.
    So we want |U-U| to be as large as possible (close to 2M+1) and |U+U| as small as possible.
    
    Let's use MILP to maximize |U-U| - |U+U| or similar proxy for small N.
    """
    if not HAS_PULP:
        return None

    # We search for a set of size n.
    # Max element M is bounded. Let's try M up to 20 for n up to 8.
    max_M = 30 
    # Variables: x_i for i in 0..max_M
    prob = pulp.LpProblem("MaxDiffRatio", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", range(max_M + 1), cat="Binary")
    
    # Constraint: x_0 = 1
    prob += x[0] == 1
    
    # Constraint: Sum x_i = n
    prob += pulp.lpSum(x[i] for i in range(max_M + 1)) == n
    
    # Objective: We want to maximize |U-U| / |U+U|.
    # This is hard directly. Let's maximize |U-U| - lambda * |U+U|?
    # Or just maximize |U-U| subject to |U+U| <= K?
    # Let's try maximizing |U-U| first, as it's bounded by 2M+1.
    
    # Sum set variables: s_v for v in 0..2*max_M
    s = pulp.LpVariable.dicts("s", range(2 * max_M + 1), cat="Binary")
    # Diff set variables: d_v for v in -max_M..max_M
    d = pulp.LpVariable.dicts("d", range(-max_M, max_M + 1), cat="Binary")
    
    # Constraints linking x to s and d
    # s_v = 1 iff exists a,b in U such that a+b=v
    # d_v = 1 iff exists a,b in U such that a-b=v
    
    # This formulation is complex. Let's simplify.
    # For small n, we can just enumerate subsets of a small range and score them.
    # MILP might be overkill or slow if not formulated perfectly.
    # Let's stick to a smarter heuristic: "Greedy with Lookahead" or "Simulated Annealing".
    
    return None

def heuristic_construct():
    """
    Structural change: Use Simulated Annealing to find a set U that optimizes the score.
    Unlike the previous greedy/random approach, SA allows for non-improving moves to escape local optima.
    We will optimize the set composition directly.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial set: Powers of 2 is a good start, but let's try a mix.
    # Start with a small dense set and expand?
    # Or start with a random sparse set?
    
    # Let's try to construct a set that approximates a "difference basis" but is sparse.
    # A known good structure for large |U-U|/|U+U| is related to Sidon sequences or similar,
    # but powers of 2 are very sparse.
    
    # Let's use SA to optimize the set of integers.
    # State: List of integers U.
    # Move: Add, remove, or replace an element.
    
    # To make SA effective, we need a fast score.
    # For N ~ 20-50, O(N^2) is fine.
    
    best_U = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    best_score = compute_score(best_U)
    
    current_U = best_U[:]
    current_score = best_score
    
    # Temperature schedule
    T = 0.5
    T_min = 0.01
    cooling = 0.99
    
    max_iter = 5000
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Generate neighbor
        U = current_U[:]
        move_type = np.random.choice(['add', 'remove', 'replace'])
        
        if move_type == 'add' and len(U) < 100:
            # Add a number near the max or random
            if np.random.random() < 0.5:
                # Geometric growth step
                new_val = max(U) * 2 + np.random.randint(-max(U)//10, max(U)//10 + 1)
            else:
                new_val = np.random.randint(1, max(U) * 2 + 10)
            if new_val not in U:
                U.append(new_val)
        elif move_type == 'remove' and len(U) > 2:
            # Don't remove 0
            idx = np.random.randint(1, len(U))
            U.pop(idx)
        elif move_type == 'replace' and len(U) > 2:
            idx = np.random.randint(1, len(U))
            old_val = U[idx]
            if np.random.random() < 0.5:
                new_val = max(U) * 2 + np.random.randint(-max(U)//10, max(U)//10 + 1)
            else:
                new_val = np.random.randint(1, max(U) * 2 + 10)
            if new_val not in U:
                U[idx] = new_val
        
        if len(U) < 2:
            continue
            
        new_score = compute_score(U)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0:
            current_U = U
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_U = U[:]
        else:
            # Accept with probability exp(delta / T)
            if np.exp(delta / T) > np.random.random():
                current_U = U
                current_score = new_score
        
        T *= cooling
        if T < T_min:
            break
            
    return best_U

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # 1. Try the heuristic (Simulated Annealing)
    # This explores the space of sets more thoroughly than greedy.
    U_sa = heuristic_construct()
    
    # 2. Also keep the powers of 2 baseline as a fallback/comparison
    U_pow2 = [2**i for i in range(18)]
    U_pow2.insert(0, 0)
    
    score_sa = compute_score(U_sa)
    score_pow2 = compute_score(U_pow2)
    
    if score_sa >= score_pow2:
        return U_sa
    else:
        return U_pow2

if __name__ == "__main__":
    print(construct())
