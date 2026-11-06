import itertools
import random

N = 16
POINTS = [(x, y) for x in range(N) for y in range(N)]
NUM_POINTS = len(POINTS)
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}

# Precompute isosceles relationships
# pair_to_qs: dict mapping (idx_i, idx_j) to set of idx_k such that {i,j,k} is isosceles
pair_to_qs = {}

def dist2(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Precomputation
# Iterate all pairs (i, j)
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        p_i = POINTS[i]
        p_j = POINTS[j]
        d_ij = dist2(p_i, p_j)
        
        blocked = set()
        for k in range(NUM_POINTS):
            if k == i or k == j:
                continue
            p_k = POINTS[k]
            d_ik = dist2(p_i, p_k)
            d_jk = dist2(p_j, p_k)
            
            # Check if isosceles
            if d_ik == d_jk or d_ik == d_ij or d_jk == d_ij:
                blocked.add(k)
        
        pair_to_qs[(i, j)] = blocked

def construct():
    S = set()
    
    while True:
        # Find unblocked points
        unblocked = set()
        for q in range(NUM_POINTS):
            if q in S:
                continue
            
            is_blocked = False
            # Check all pairs in S
            # Optimization: if len(S) < 2, no pairs, so not blocked
            if len(S) >= 2:
                s_list = list(S)
                for idx_a in range(len(s_list)):
                    for idx_b in range(idx_a + 1, len(s_list)):
                        a = s_list[idx_a]
                        b = s_list[idx_b]
                        key = (a, b) if a < b else (b, a)
                        if q in pair_to_qs[key]:
                            is_blocked = True
                            break
                    if is_blocked:
                        break
            
            if not is_blocked:
                unblocked.add(q)
                
        if not unblocked:
            return [POINTS[q] for q in S]
        
        # Select best candidate from unblocked
        best_p = None
        best_score = -1
        
        for p in unblocked:
            # Calculate how many unblocked points p will block
            # New pairs are (p, r) for r in S
            new_blocked = set()
            if S:
                for r in S:
                    key = (p, r) if p < r else (r, p)
                    blocked_by_pair = pair_to_qs[key]
                    # Intersect with unblocked, excluding p itself (though p is in S now, so it's not in unblocked for next iter, but here we count future blocked)
                    # We want to know how many q in unblocked \ {p} will become blocked
                    # A q is blocked if q is in blocked_by_pair
                    for q in blocked_by_pair:
                        if q in unblocked and q != p:
                            new_blocked.add(q)
            
            score = len(new_blocked)
            
            if score > best_score:
                best_score = score
                best_p = p
                
        S.add(best_p)
        
    # This should not be reached
    return [POINTS[q] for q in S]
