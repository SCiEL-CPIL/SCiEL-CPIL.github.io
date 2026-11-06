import numpy as np
import random
import itertools

N = 16
POINTS = list(itertools.product(range(N), repeat=2))
NUM_POINTS = len(POINTS)

# Precompute squared distances
dist = np.zeros((NUM_POINTS, NUM_POINTS), dtype=int)
for i in range(NUM_POINTS):
    for j in range(i + 1, NUM_POINTS):
        d = (POINTS[i][0] - POINTS[j][0])**2 + (POINTS[i][1] - POINTS[j][1])**2
        dist[i][j] = d
        dist[j][i] = d

# Precompute iso_mask[p][r] as a bitmask of q such that (p, r, q) is isosceles
# q is in [0, 256)
iso_mask = [[0] * NUM_POINTS for _ in range(NUM_POINTS)]
for p in range(NUM_POINTS):
    for r in range(NUM_POINTS):
        if p == r:
            continue
        mask = 0
        for q in range(NUM_POINTS):
            if q == p or q == r:
                continue
            if dist[p][r] == dist[p][q] or dist[p][r] == dist[r][q] or dist[p][q] == dist[r][q]:
                mask |= (1 << q)
        iso_mask[p][r] = mask

def construct():
    best_set = None
    best_size = float('inf')
    
    # Number of restarts
    # Each restart is fast due to bitmasks
    for restart in range(500):
        S = []
        S_set = set()
        candidates_mask = (1 << NUM_POINTS) - 1
        
        while candidates_mask:
            # Find p in candidates that maximizes invalidation
            # invalidation_count(p) = popcount( candidates_mask & (Union of iso_mask[p][r] for r in S) )
            
            best_p = -1
            best_count = -1
            
            # Iterate through candidates
            # Get list of candidate indices
            # To avoid generating list every time, we can iterate bits, but N=16 is small, list is fine.
            cand_list = [i for i in range(NUM_POINTS) if (candidates_mask >> i) & 1]
            
            if not cand_list:
                break
                
            # If S is empty, any p is valid. Invalidation is 0 for all?
            # If S is empty, adding p doesn't invalidate any q (no pairs in S).
            # So we pick randomly? Or pick p that has highest potential?
            # If S is empty, we can just pick a random p to break symmetry.
            if not S:
                p = random.choice(cand_list)
            else:
                for p in cand_list:
                    # Compute union of iso_mask[p][r] for all r in S
                    union_mask = 0
                    for r in S:
                        union_mask |= iso_mask[p][r]
                    
                    # Count how many candidates are invalidated
                    # Note: p itself is in candidates, but p is not invalidated by p.
                    # The mask might contain p? No, we excluded p in precomputation.
                    # The mask might contain r? No, we excluded r? 
                    # Wait, in precomputation: if q == p or q == r, continue. 
                    # So iso_mask[p][r] does not contain p or r.
                    # So p is not in the invalidation mask.
                    
                    invalidated = candidates_mask & union_mask
                    count = bin(invalidated).count('1')
                    
                    # Add some randomness to avoid getting stuck in local optima of the heuristic
                    # We can pick the best, or pick randomly from top-k?
                    # Let's stick to best for now, maybe add slight noise.
                    if count > best_count:
                        best_count = count
                        best_p = p
                        # Tie-breaking: random
                        # If count == best_count, we might want to randomize
                        # But finding the first max is fine, or we can collect all max.
                        # For simplicity, take the first max found.
                    
                    # Early stop optimization? No, need to check all to find max.
                    
                p = best_p
                
            # Add p to S
            S.append(p)
            S_set.add(p)
            
            # Update candidates_mask
            # Remove p
            candidates_mask &= ~(1 << p)
            
            # Remove any q that is invalidated by p
            # q is invalidated if exists r in S (including p? No, r in S_old) such that is_isosceles(p, r, q)
            # Wait, the condition for q to be invalid in S_new = S_old U {p} is:
            # exists a, b in S_new such that is_isosceles(q, a, b).
            # Since S_old was valid, no pair in S_old invalidates q.
            # So q is invalid iff exists r in S_old such that is_isosceles(q, p, r).
            # This is exactly what iso_mask[p][r] gives us (symmetric in p, r, q? 
            # is_isosceles(p, r, q) is symmetric. Yes.)
            
            # So we need union of iso_mask[p][r] for r in S_old.
            # Note: S_old is S before adding p.
            # In the loop above, we computed this union for the best_p.
            # We can reuse it? 
            # The `union_mask` variable is local to the `for p in cand_list` loop.
            # We should store the best union mask or recompute.
            # Recomputing is O(|S|) which is small.
            
            invalidation_mask = 0
            for r in S:
                if r != p:
                    invalidation_mask |= iso_mask[p][r]
            
            candidates_mask &= ~invalidation_mask
            
        # End while candidates
        
        # Check if S is maximal
        # By construction, candidates_mask is 0, so no point can be added.
        # So S is maximal.
        # Check if S is no-isosceles?
        # By construction, we only added p if it didn't invalidate itself?
        # Wait, we didn't explicitly check if p is valid w.r.t S_old.
        # We assumed "all candidates are valid w.r.t S".
        # Is this true?
        # Initially, all are valid.
        # When we add p, we remove all q that are invalid w.r.t S_new.
        # So any remaining candidate q is valid w.r.t S_new.
        # So yes, any p picked from candidates is valid to add.
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
        # If we find a very small set, we can stop?
        # Theoretical minimum is unknown, but let's run all restarts.
        
    if best_set is None:
        # Fallback
        return [POINTS[0]]
        
    return [POINTS[i] for i in best_set]
