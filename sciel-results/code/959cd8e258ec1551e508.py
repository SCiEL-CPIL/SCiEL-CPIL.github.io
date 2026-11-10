# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def szemeredi_trotter_incidences():
    """
    Construct a set of 54 points and 27 lines in the integer plane to maximize
    point-line incidences.
    
    Strategy: Use a 7x8 grid of points (56 total), remove 2 points to get 54.
    We select 7 vertical lines, 8 horizontal lines, 6 slope-1 diagonals, and
    6 slope-1 diagonals (total 27 lines). We optimize the choice of which
    6 b-values (for y-x=b) and 6 s-values (for x+y=s) to use, and which 2
    points to remove, to maximize total incidences.
    
    For a fixed choice of B (set of 6 b-values) and S (set of 6 s-values),
    the total incidences I = I_full - w(p1) - w(p2) where w(p) is the number
    of lines containing p in the full grid. We want to minimize w(p1)+w(p2)
    over all pairs of distinct points.
    """
    # Generate full 7x8 grid
    xs = np.arange(1, 8)
    ys = np.arange(1, 9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    full_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    n_full = len(full_points)  # 56
    
    # Pre-compute b and s for each point
    b_vals = full_points[:, 1] - full_points[:, 0]  # y - x
    s_vals = full_points[:, 0] + full_points[:, 1]  # x + y
    
    # All possible b values: y-x ranges from 1-8=-7 to 7-1=6, so -7..6
    all_b = list(range(-7, 7))
    # All possible s values: x+y ranges from 1+1=2 to 7+8=15, so 2..15
    all_s = list(range(2, 16))
    
    # Pre-compute incidence counts for each b and s in the full grid
    b_count = {}
    for b in all_b:
        b_count[b] = int(np.sum(b_vals == b))
    
    s_count = {}
    for s in all_s:
        s_count[s] = int(np.sum(s_vals == s))
    
    # For each point, pre-compute its b and s
    # w(p) = 2 (V and H) + 1[b_p in B] + 1[s_p in S]
    # So minimizing w(p1)+w(p2) is equivalent to minimizing 1[b_p1 in B]+1[b_p2 in B]+1[s_p1 in S]+1[s_p2 in S]
    
    best_score = -1
    best_points = None
    best_lines = None
    
    # Search over all combinations of 6 b-values from all_b, and 6 s-values from all_s
    # all_b has 14 elements, C(14,6) = 3003
    # all_s has 14 elements, C(14,6) = 3003
    # Total: 3003 * 3003 = ~9 million. For each, we need to find min w(p1)+w(p2).
    # This might be too slow in Python. Let's optimize.
    
    # For a given (B, S), w(p) = 2 + 1[b_p in B] + 1[s_p in S]
    # Let f(p) = 1[b_p in B] + 1[s_p in S]. f(p) in {0, 1, 2}.
    # We want min over pairs of distinct points of f(p1) + f(p2).
    # Let n0 = #{p: f(p)=0}, n1 = #{p: f(p)=1}, n2 = #{p: f(p)=2}.
    # n0 + n1 + n2 = 56.
    # Min f(p1)+f(p2):
    #   If n0 >= 2: min = 0
    #   If n0 == 1: min = 0 + min(1 if n1>0 else 2) = 1 if n1>0 else 2
    #   If n0 == 0: min = 2 if n1 >= 2 else (3 if n1 >= 1 else 4)
    
    # I_full = sum of all line incidences in full grid
    # = 7*8 (V) + 8*7 (H) + sum_{b in B} b_count[b] + sum_{s in S} s_count[s]
    # = 56 + 56 + sum_b + sum_s = 112 + sum_b + sum_s
    
    # I = I_full - (w(p1) + w(p2)) = I_full - (4 + f(p1) + f(p2))
    # So I = I_full - 4 - min(f(p1)+f(p2))
    
    # To maximize I, we want to maximize I_full - min(f(p1)+f(p2))
    # = 112 + sum_b + sum_s - 4 - min(f(p1)+f(p2))
    # = 108 + sum_b + sum_s - min(f(p1)+f(p2))
    
    # So we want to maximize: sum_b + sum_s - min(f(p1)+f(p2))
    
    # Let's iterate over combinations of B and S.
    # For efficiency, pre-compute for each point its b and s.
    
    best_val = -1
    best_B = None
    best_S = None
    best_min_f_sum = None
    
    for B in combinations(all_b, 6):
        B_set = set(B)
        sum_b = sum(b_count[b] for b in B)
        
        for S in combinations(all_s, 6):
            S_set = set(S)
            sum_s = sum(s_count[s] for s in S)
            
            # Compute f(p) for each point
            # f(p) = 1[b_p in B_set] + 1[s_p in S_set]
            f = np.zeros(n_full, dtype=np.int8)
            for i in range(n_full):
                f[i] = (1 if b_vals[i] in B_set else 0) + (1 if s_vals[i] in S_set else 0)
            
            n0 = int(np.sum(f == 0))
            n1 = int(np.sum(f == 1))
            n2 = int(np.sum(f == 2))
            
            if n0 >= 2:
                min_f_sum = 0
            elif n0 == 1:
                min_f_sum = 1 if n1 > 0 else 2
            else:  # n0 == 0
                if n1 >= 2:
                    min_f_sum = 2
                elif n1 >= 1:
                    min_f_sum = 3
                else:
                    min_f_sum = 4
            
            val = sum_b + sum_s - min_f_sum
            
            if val > best_val:
                best_val = val
                best_B = B
                best_S = S
                best_min_f_sum = min_f_sum
    
    # Now find the optimal pair of points to remove for the best (B, S)
    B_set = set(best_B)
    S_set = set(best_S)
    
    f = np.zeros(n_full, dtype=np.int8)
    for i in range(n_full):
        f[i] = (1 if b_vals[i] in B_set else 0) + (1 if s_vals[i] in S_set else 0)
    
    # Find pair with minimum f(p1)+f(p2)
    best_pair_score = 999
    best_i = 0
    best_j = 1
    for i in range(n_full):
        for j in range(i + 1, n_full):
            s = f[i] + f[j]
            if s < best_pair_score:
                best_pair_score = s
                best_i = i
                best_j = j
    
    # Construct points
    mask = np.ones(n_full, dtype=bool)
    mask[best_i] = False
    mask[best_j] = False
    points = full_points[mask]
    
    # Construct lines
    lines = []
    # 8 horizontal lines: y = c => 0*x + 1*y - c = 0
    for y_val in ys:
        lines.append((0, 1, -y_val))
    # 7 vertical lines: x = c => 1*x + 0*y - c = 0
    for x_val in xs:
        lines.append((1, 0, -x_val))
    # 6 slope-1 diagonals: y - x = b => x - y + b = 0 => A=1, B=-1, C=b
    for b in best_B:
        lines.append((1, -1, b))
    # 6 slope-1 diagonals: x + y = s => x + y - s = 0 => A=1, B=1, C=-s
    for s in best_S:
        lines.append((1, 1, -s))
    
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines

# EVOLVE-BLOCK-END
