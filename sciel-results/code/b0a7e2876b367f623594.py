import random
import sys

N = 16
NUM_POINTS = N * N
POINTS = [(x, y) for x in range(N) for y in range(N)]
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}

def dist2(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Precompute isosceles triples
# iso_third[(i, j)] = set of k such that {i, j, k} is isosceles
iso_third = {}

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
            
            if d_ik == d_jk or d_ik == d_ij or d_jk == d_ij:
                blocked.add(k)
        
        iso_third[(i, j)] = blocked
        iso_third[(j, i)] = blocked

def get_blocked_set(S):
    """
    Returns the set of points that are blocked by S.
    A point u is blocked if there exist a, b in S such that {a, b, u} is isosceles.
    """
    blocked = set()
    s_list = list(S)
    for idx_a, a in enumerate(s_list):
        for b in s_list[idx_a+1:]:
            key = (a, b) if a < b else (b, a)
            blocked.update(iso_third[key])
    return blocked

def is_maximal(S):
    """
    Checks if S is a maximal independent set.
    S is maximal if no point can be added to S without violating independence.
    """
    blocked = get_blocked_set(S)
    for u in range(NUM_POINTS):
        if u not in S and u not in blocked:
            return False
    return True

def greedy_construct():
    S = set()
    blocked = [False] * NUM_POINTS
    
    while True:
        candidates = [q for q in range(NUM_POINTS) if q not in S and not blocked[q]]
        if not candidates:
            break
        
        best_p = None
        best_score = -1
        
        for p in candidates:
            # Count how many currently unblocked points p would block
            new_blocked_count = 0
            for r in S:
                key = (p, r) if p < r else (r, p)
                for q in iso_third[key]:
                    if q != p and q not in S and not blocked[q]:
                        new_blocked_count += 1
            
            if new_blocked_count > best_score:
                best_score = new_blocked_count
                best_p = p
        
        if best_p is None:
            break
        
        S.add(best_p)
        # Update blocked
        for r in S:
            if r == best_p:
                continue
            key = (best_p, r) if best_p < r else (r, best_p)
            for q in iso_third[key]:
                if q not in S:
                    blocked[q] = True
                    
    return S

def local_search(S):
    S = set(S)
    improved = True
    # Limit iterations to avoid timeout
    max_iter = 1000
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        s_list = list(S)
        random.shuffle(s_list)
        
        for p in s_list:
            if p not in S:
                continue
            
            # 1-Opt: Try removing p
            S_new = S - {p}
            if is_maximal(S_new):
                S = S_new
                improved = True
                break
            
            # 2-Opt: Try removing p and v, adding u
            # First, find points that would become unblocked if p is removed
            # This is slightly complex. Let's just try a few random u's from the unblocked set of S_new
            
            # Get unblocked points for S_new
            blocked_new = get_blocked_set(S_new)
            unblocked = [u for u in range(NUM_POINTS) if u not in S_new and u not in blocked_new]
            
            if not unblocked:
                # Should have been caught by 1-Opt
                continue
            
            # Try a few candidates for u
            random.shuffle(unblocked)
            for u in unblocked[:5]: # Limit to 5 candidates for speed
                S_mid = S_new | {u}
                
                # Now try to remove v from S_mid (v != u)
                # We want to see if we can remove another point to maintain maximality
                # This reduces size by 1 net (remove 2, add 1)
                
                # To speed up, we can check if removing any v from S_mid \ {u} results in a maximal set
                # This is O(|S|) checks, each O(|S|^2)
                # Total O(|S|^3) per u.
                # |S| is ~40. 40^3 = 64000. Times 5 u's = 320k ops.
                # Times 40 p's = 12M ops.
                # Times 1000 iterations = 12B ops. Too slow.
                
                # Optimization: Only try 2-Opt if 1-Opt fails and we are stuck.
                # Or limit 2-Opt frequency.
                
                # Let's try a cheaper 2-Opt:
                # Remove p. If not maximal, we MUST add something.
                # If we add u, we have S_mid.
                # Can we remove v from S_mid?
                # Only if v is "redundant".
                
                # For now, skip 2-Opt in the main loop to ensure speed, 
                # or do it very sparsely.
                
                pass

    return S

def construct():
    # Initialize random seed for reproducibility in local search shuffle
    random.seed(42)
    
    # 1. Greedy Construction
    S = greedy_construct()
    
    # 2. Local Search
    S = local_search(S)
    
    # Convert to list of points
    return [POINTS[q] for q in S]
