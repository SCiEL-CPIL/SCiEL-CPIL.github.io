import itertools
import sys

def construct(v, k, t):
    # Precomputation
    points = range(v)
    t_subs = list(itertools.combinations(points, t))
    sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    idx_to_sub = t_subs
    n_t_subs = len(t_subs)
    
    # Map (t-1)-subset and point x not in R to index of R U {x}
    # R is a tuple of length t-1
    map_R_x_to_idx = {}
    for R in itertools.combinations(points, t-1):
        R_set = set(R)
        map_R_x_to_idx[R] = {}
        for x in points:
            if x not in R_set:
                S = tuple(sorted(R + (x,)))
                map_R_x_to_idx[R][x] = sub_to_idx[S]
                
    # List of indices of t-subsets containing each point
    subs_by_point = [[] for _ in range(v)]
    for i, S in enumerate(t_subs):
        for x in S:
            subs_by_point[x].append(i)
            
    uncovered = set(range(n_t_subs))
    blocks = []
    
    while uncovered:
        B = []
        while len(B) < k:
            if len(B) >= t - 1:
                best_x = -1
                max_gain = -1
                # Generate all (t-1) subsets of B
                R_list = list(itertools.combinations(B, t-1))
                B_set = set(B)
                
                for x in points:
                    if x in B_set:
                        continue
                    gain = 0
                    for R in R_list:
                        # R is a tuple of t-1 elements from B
                        # S = R U {x}
                        # Check if S is in uncovered
                        idx_S = map_R_x_to_idx[R][x]
                        if idx_S in uncovered:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_x = x
                if best_x == -1:
                    # Should not happen if uncovered is not empty and k > t
                    # But if no gain, pick any
                    for x in points:
                        if x not in B_set:
                            best_x = x
                            break
            else:
                # len(B) < t-1
                # Pick x from the first uncovered subset to ensure progress
                # Or pick x with max frequency in uncovered
                # For speed, just pick from first uncovered
                first_idx = next(iter(uncovered))
                target_S = idx_to_sub[first_idx]
                B_set = set(B)
                candidates = [x for x in target_S if x not in B_set]
                if candidates:
                    best_x = candidates[0]
                else:
                    # Fallback
                    for x in points:
                        if x not in B_set:
                            best_x = x
                            break
            
            B.append(best_x)
            B.sort()
            B_set = set(B)
            
            # Update uncovered
            # Only check subsets containing best_x
            to_remove = []
            for idx in subs_by_point[best_x]:
                if idx in uncovered:
                    S = idx_to_sub[idx]
                    # Check if S is subset of B
                    # Since S contains best_x and was not covered before,
                    # it is covered now iff S \ {best_x} is subset of B \ {best_x}
                    # Which is equivalent to S subset B
                    if set(S).issubset(B_set):
                        to_remove.append(idx)
            for idx in to_remove:
                uncovered.remove(idx)
                
        blocks.append(B)
        
    return blocks
