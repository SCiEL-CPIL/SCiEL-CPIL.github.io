import time
import random
import itertools
import numpy as np

N = 16
DEADLINE_SECONDS = 290
NUM_POINTS = N * N
ALL_MASK = (1 << NUM_POINTS) - 1

# Precompute points
POINTS = [(x, y) for x in range(N) for y in range(N)]
POINT_TO_IDX = {p: i for i, p in enumerate(POINTS)}

# Precompute squared distances
DIST = np.zeros((NUM_POINTS, NUM_POINTS), dtype=np.int32)
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        dx = POINTS[i][0] - POINTS[j][0]
        dy = POINTS[i][1] - POINTS[j][1]
        d = dx * dx + dy * dy
        DIST[i, j] = d
        DIST[j, i] = d

# Precompute IsoMask
# IsoMask[i][j] is a bitmask of k such that {i,j,k} is isosceles
ISO_MASK = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]

for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        mask = 0
        for k in range(NUM_POINTS):
            if k == i or k == j:
                continue
            # Check if {i,j,k} is isosceles
            d_ij = DIST[i, j]
            d_ik = DIST[i, k]
            d_jk = DIST[j, k]
            
            if d_ij == d_ik or d_ij == d_jk or d_ik == d_jk:
                mask |= (1 << k)
        
        ISO_MASK[i][j] = mask
        ISO_MASK[j][i] = mask

def construct():
    start_time = time.time()
    
    # Iterate K from 1 to NUM_POINTS
    # We can start with a lower bound if we have one, but 1 is safe.
    # Heuristic: Start from a known good estimate? 
    # Let's just start from 1.
    
    # To avoid deep recursion limits, we might need to increase recursion limit
    # or use an iterative approach, but N=16, K is likely < 20, so depth is small.
    
    for K in range(1, NUM_POINTS + 1):
        if time.time() - start_time > DEADLINE_SECONDS:
            break
            
        # DFS
        # S: list of indices
        # ForbiddenMask: bitmask of points that cannot be added
        # UsedMask: bitmask of points in S
        
        # Optimization: We can use a heuristic to order the points initially?
        # The DFS itself selects points based on coverage.
        
        # We need a way to stop DFS if it takes too long for a specific K.
        # We can use a global counter or time check inside DFS.
        
        node_count = 0
        max_nodes = 50000 # Limit nodes per K to prevent hanging on a single K
        
        def dfs(S, ForbiddenMask, UsedMask, depth):
            nonlocal node_count
            node_count += 1
            if node_count > max_nodes:
                return None
            
            if time.time() - start_time > DEADLINE_SECONDS:
                return None
                
            # Check if maximal
            # Uncovered = AllMask & ~(ForbiddenMask | UsedMask)
            if (ALL_MASK & ~(ForbiddenMask | UsedMask)) == 0:
                return S
            
            if depth == K:
                return None
                
            # Available points
            Available = ALL_MASK & ~(ForbiddenMask | UsedMask)
            if Available == 0:
                return None
                
            # Get indices of available points
            # This is a bit slow if done every time. 
            # But K is small, so S is small.
            
            # Heuristic: Pick points that cover the most uncovered points
            # Uncovered points are in Available.
            # For each p in Available, calculate how many points in Available it would forbid.
            
            candidates = []
            # Iterate through set bits in Available
            avail_bits = Available
            while avail_bits:
                # Get lowest set bit
                lsb = avail_bits & -avail_bits
                p = lsb.bit_length() - 1
                avail_bits -= lsb
                
                # Calculate new forbidden
                new_forb = 0
                for a in S:
                    new_forb |= ISO_MASK[p][a]
                
                # Coverage: points in Available that become forbidden or are p
                # p is in Available.
                # New forbidden points that are in Available: new_forb & Available
                # Plus p itself.
                coverage = (new_forb & Available) | (1 << p)
                score = coverage.bit_count()
                
                candidates.append((score, p))
                
            # Sort by score descending
            candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Try top candidates
            # To add randomness/variety, we could shuffle top candidates with same score
            # or just take top 1-5.
            
            # Limit branching factor to avoid explosion
            branch_factor = 5
            for score, p in candidates[:branch_factor]:
                new_used = UsedMask | (1 << p)
                new_forb = ForbiddenMask
                for a in S:
                    new_forb |= ISO_MASK[p][a]
                
                res = dfs(S + [p], new_forb, new_used, depth + 1)
                if res:
                    return res
                    
            return None

        # Start DFS
        # Initial S is empty, Forbidden is 0, Used is 0
        res = dfs([], 0, 0, 0)
        
        if res:
            # Verify
            # Check if res is independent and maximal
            # Independent: No isosceles triples
            # Maximal: No point can be added
            
            # Convert to points
            return [POINTS[i] for i in res]
            
    # Fallback if no solution found within time
    # Return a simple greedy set
    return _fallback_greedy()

def _fallback_greedy():
    # Simple greedy to ensure we return *something* valid
    S = []
    used = 0
    for i in range(NUM_POINTS):
        if (used >> i) & 1:
            continue
        # Check if i can be added
        ok = True
        for j in S:
            for k in S:
                if j == k: continue
                # Check if {i,j,k} is isosceles
                if (ISO_MASK[j][k] >> i) & 1:
                    ok = False
                    break
            if not ok: break
        if ok:
            S.append(i)
            used |= (1 << i)
            
    # Make it maximal? 
    # The greedy above doesn't ensure maximality.
    # We need to add points if possible.
    # Actually, the task requires a *maximal* set.
    # If the B&B fails, we should return a valid maximal set.
    
    # Let's do a proper greedy maximal construction
    S = []
    forbidden = 0
    used = 0
    available = ALL_MASK
    
    while available:
        # Pick any available point
        # Heuristic: pick one that forbids the most?
        # Just pick first one for simplicity in fallback
        lsb = available & -available
        p = lsb.bit_length() - 1
        
        S.append(p)
        used |= (1 << p)
        
        # Update forbidden
        for a in S[:-1]:
            forbidden |= ISO_MASK[p][a]
            
        available = ALL_MASK & ~(used | forbidden)
        
    return [POINTS[i] for i in S]
