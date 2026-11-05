"""Minimal maximal no-isosceles set on [0, 16)^2 using greedy construction and local search."""
import random
import time
import numpy as np

N = 16
TOTAL = N * N

def _point_idx(p):
    return p[0] * N + p[1]

def _idx_point(i):
    return (i // N, i % N)

# Precompute iso_third[i][j][k] = True if {i, j, k} is isosceles
# Shape: (TOTAL, TOTAL, TOTAL)
# Memory: 256^3 * 1 byte = 16 MB. Acceptable.
def _precompute():
    pts = [_idx_point(i) for i in range(TOTAL)]
    # Using a 3D boolean array
    iso = np.zeros((TOTAL, TOTAL, TOTAL), dtype=bool)
    for i in range(TOTAL):
        a = pts[i]
        for j in range(TOTAL):
            if i == j:
                continue
            b = pts[j]
            da = (a[0]-b[0])**2 + (a[1]-b[1])**2
            for k in range(TOTAL):
                if k == i or k == j:
                    continue
                c = pts[k]
                db = (a[0]-c[0])**2 + (a[1]-c[1])**2
                dc = (b[0]-c[0])**2 + (b[1]-c[1])**2
                if da == db or da == dc or db == dc:
                    iso[i, j, k] = True
    return iso

_iso_third = _precompute()

def _greedy(order):
    """
    Greedily construct a maximal no-isosceles set given an order of points.
    Returns (chosen_indices, blocked_mask)
    """
    chosen = []
    blocked = np.zeros(TOTAL, dtype=bool)
    
    for p in order:
        if blocked[p]:
            continue
        
        # If we add p, which points become blocked?
        # A point c is blocked by p (given current chosen S) if there exists q in S 
        # such that {p, q, c} is isosceles.
        # iso_third[p] is a 2D array [q][c].
        # We need c such that exists q in chosen: iso_third[p][q][c] is True.
        
        if chosen:
            # Mask for chosen points
            cmask = np.zeros(TOTAL, dtype=bool)
            cmask[chosen] = True
            
            # sub[q, c] = iso_third[p][q][c]
            # We want c where any q in chosen has iso_third[p][q][c] == True
            # row = sub[cmask] gives a 2D array of shape (n_chosen, TOTAL)
            row = _iso_third[p][cmask]
            if row.size > 0:
                new_blocks = row.any(axis=0)
                blocked |= new_blocks
        
        # p itself is now "used" (blocked from being added again, though it's in chosen)
        blocked[p] = True
        chosen.append(p)
        
    return chosen, blocked

def _is_maximal(chosen, blocked):
    """
    Check if the set is maximal.
    Maximal means no unchosen point is unblocked.
    """
    if not chosen:
        return TOTAL == 0 # Trivial case, but N=16 so not empty
        
    cmask = np.zeros(TOTAL, dtype=bool)
    cmask[chosen] = True
    unchosen = ~cmask
    
    # If any unchosen point is not blocked, we can add it.
    # So maximal iff all unchosen points are blocked.
    return not (blocked[unchosen] == False).any()

def _local_search(chosen, rng, max_iters=50, deadline=None):
    """
    Attempt to reduce the size of the maximal set by removing points and re-filling.
    """
    best = list(chosen)
    best_size = len(best)
    
    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
            
        improved = False
        # Shuffle order of removal attempts
        indices = list(range(len(best)))
        rng.shuffle(indices)
        
        for idx in indices:
            if deadline and time.time() > deadline:
                break
                
            v = best[idx]
            # Try removing v
            candidate = [p for j, p in enumerate(best) if j != idx]
            
            # Check if candidate is maximal?
            # We need to compute the blocked mask for candidate.
            # This is expensive if done from scratch. 
            # However, we can re-run greedy on the remaining unblocked points?
            # No, removing a point might unblock others.
            
            # Strategy: 
            # 1. Compute blocked mask for 'candidate' from scratch? 
            #    This is O(|candidate|^2) roughly. 
            #    With |candidate| ~ 15, it's fast.
            
            # Let's build the blocked mask for 'candidate'
            cand_blocked = np.zeros(TOTAL, dtype=bool)
            cand_chosen = []
            
            # We don't need to re-greedy the whole thing, just check if candidate is valid (it is, subset of valid)
            # and then check maximality.
            # To check maximality, we need to know if any point p not in candidate can be added.
            # p can be added if for all a, b in candidate, {p, a, b} is NOT isosceles.
            # i.e., p is NOT in Union_{a,b in candidate} Iso(a,b).
            
            # Efficient check for maximality of 'candidate':
            # For each p not in candidate:
            #   Check if exists a, b in candidate such that _iso_third[p][a][b] is True.
            
            cand_set = set(candidate)
            is_max = True
            # We can optimize this by precomputing the "blocking set" of candidate
            # Block set B = { p | exists a,b in candidate, iso[p][a][b] }
            # If B contains all points not in candidate, it's maximal.
            
            # Let's compute B efficiently.
            # B is the union of iso_third[p] restricted to pairs in candidate?
            # Actually, for a fixed p, p is blocked if _iso_third[p][candidate][:, candidate].any()
            
            # Let's just iterate through all points p not in candidate
            for p in range(TOTAL):
                if p in cand_set:
                    continue
                # Check if p is blocked by candidate
                # sub = _iso_third[p] # shape (TOTAL, TOTAL)
                # We need to know if any [a][b] with a,b in candidate is True
                # mask_cand = np.zeros(TOTAL, dtype=bool); mask_cand[candidate]=True
                # sub_masked = sub[mask_cand][:, mask_cand]
                # if sub_masked.any(): p is blocked
                
                # Optimization: Precompute a mask for candidate once per iteration?
                # No, candidate changes in each removal attempt.
                # But candidate is small (~10-20).
                
                # Let's use the vectorized check
                # This is a bit slow if done for every p in every iteration.
                # TOTAL=256. Candidate size ~15.
                # 256 * (15*15 checks) = ~57k ops per removal attempt.
                # 15 removal attempts per iteration. 50 iterations.
                # Total ~ 40M ops. Might be too slow in Python loops.
                
                # Alternative: Re-greedy approach for filling.
                # If we remove v, we might need to add new points to maintain maximality.
                # Let's try to reconstruct a maximal set from 'candidate' using greedy on the remaining points.
                # This guarantees maximality (if greedy covers all) and validity.
                
                # Let's try this:
                # 1. Start with 'candidate'
                # 2. Find all points not in candidate and not blocked by candidate.
                # 3. Greedily add them in random order until maximal.
                
                # First, determine which points are currently "free" (not blocked by candidate)
                # A point p is free if it is not in candidate AND not blocked by candidate.
                # p is blocked by candidate if exists a,b in candidate: iso[p][a][b]
                
                # Let's compute the blocked mask for 'candidate'
                # We can do this by iterating through candidate points and adding their conflicts?
                # No, conflicts are pairwise.
                
                # Let's just use the slow check but optimized with numpy for the candidate set.
                # Create a boolean mask for candidate
                c_mask = np.zeros(TOTAL, dtype=bool)
                c_mask[candidate] = True
                
                # For each p not in candidate, check if _iso_third[p][c_mask][:, c_mask].any()
                # This is 256 numpy operations of shape (15,15).
                
                free_points = []
                for p in range(TOTAL):
                    if c_mask[p]:
                        continue
                    # Check if p is blocked
                    # sub = _iso_third[p]
                    # masked_sub = sub[c_mask][:, c_mask]
                    # if not masked_sub.any(): p is free
                    # This is slow in Python loop.
                    
                    # Vectorize the check for all p at once?
                    # _iso_third is (256, 256, 256).
                    # We want a mask 'blocked_by_cand' of shape (256,)
                    # blocked_by_cand[p] = _iso_third[p][c_mask][:, c_mask].any()
                    
                    # We can do this:
                    # iso_p = _iso_third[:, c_mask, c_mask] # Shape (256, 15, 15)
                    # blocked_by_cand = iso_p.any(axis=(1, 2))
                    
                    # This is much faster!
                    # Let's do this outside the p loop.
                    break # Break to do the vectorized version
                    
                # Vectorized Maximality Check and Free Point Identification
                # iso_restricted = _iso_third[:, c_mask, c_mask] # (256, k, k) where k=len(candidate)
                # blocked_by_cand = iso_restricted.any(axis=(1, 2)) # (256,)
                
                iso_restricted = _iso_third[:, c_mask, c_mask]
                blocked_by_cand = iso_restricted.any(axis=(1, 2))
                
                # Free points are those not in candidate and not blocked
                # free_mask = ~c_mask & ~blocked_by_cand
                free_mask = ~c_mask & ~blocked_by_cand
                free_points = np.where(free_mask)[0].tolist()
                
                if not free_points:
                    # Candidate is already maximal
                    if len(candidate) < best_size:
                        best = candidate
                        best_size = len(best)
                        improved = True
                    continue
                
                # Greedily fill from free_points
                rng.shuffle(free_points)
                new_chosen = list(candidate)
                new_blocked = blocked_by_cand.copy()
                # Note: new_blocked currently reflects blocking by 'candidate'.
                # As we add points, we need to update blocking.
                
                # We can reuse the _greedy logic but starting with 'candidate'
                # But _greedy takes an order and builds from scratch.
                # Let's write a specialized fill function.
                
                # Current state: new_chosen, new_blocked
                # We need to add points from free_points to new_chosen.
                # When we add p, we must update new_blocked.
                # p blocks c if exists q in new_chosen: iso[p][q][c]
                
                for p in free_points:
                    if new_blocked[p]:
                        continue
                    # Add p
                    # Update new_blocked
                    if new_chosen:
                        cmask_new = np.zeros(TOTAL, dtype=bool)
                        cmask_new[new_chosen] = True
                        row = _iso_third[p][cmask_new]
                        if row.size > 0:
                            new_blocked |= row.any(axis=0)
                    new_blocked[p] = True
                    new_chosen.append(p)
                    
                    # Check if we have achieved maximality?
                    # We can check periodically or at the end.
                    # For now, just finish the loop.
                    
                # Check if the resulting set is maximal
                # It should be, because we added all possible points from the initial free set?
                # No, adding a point might block another free point, but we process in order.
                # If we process all free_points, and some become blocked, they are skipped.
                # At the end, are there any unblocked points left?
                # The set of "free" points was defined relative to 'candidate'.
                # As we add points, the set of blocked points grows.
                # It is possible that a point not in 'free_points' (i.e., blocked by candidate) 
                # becomes unblocked? No, adding points only adds constraints (blocks more points).
                # So any point blocked by candidate remains blocked.
                # Any point not blocked by candidate was in free_points.
                # We tried to add them. If we didn't add them, it's because they were blocked by a previously added point.
                # So at the end, all points are either in new_chosen or blocked by new_chosen.
                # Thus new_chosen is maximal.
                
                if len(new_chosen) < best_size:
                    best = new_chosen
                    best_size = len(best)
                    improved = True
                    
        if not improved:
            break
            
    return best

def construct():
    all_idx = list(range(TOTAL))
    best = None
    best_size = TOTAL + 1
    start = time.time()
    deadline = start + 280  # Safety margin before 330s kill

    random.seed(42)
    rng = random.Random(42)
    
    # Phase 1: Multi-start greedy to find a good initial solution
    for attempt in range(5000):
        if time.time() > deadline:
            break
        order = all_idx[:]
        rng.shuffle(order)
        chosen, blocked = _greedy(order)
        if _is_maximal(chosen, blocked):
            if len(chosen) < best_size:
                best_size = len(chosen)
                best = chosen
                # If we find a very good one, we can break early or continue to local search
                if best_size <= 12:
                    break

    if best is None:
        # Fallback to deterministic if random failed (unlikely)
        order = all_idx[:]
        chosen, blocked = _greedy(order)
        if _is_maximal(chosen, blocked):
            best = chosen
            best_size = len(best)
        else:
            # Should not happen for greedy on full grid? 
            # Greedy always produces a maximal set? 
            # Yes, because we add a point if it's not blocked. 
            # At the end, every unadded point is blocked. 
            # So it is maximal.
            best = chosen
            best_size = len(best)

    # Phase 2: Local search to minimize size
    if best is not None:
        # Run local search multiple times with different random seeds/orders
        for ls_iter in range(5):
            if time.time() > deadline - 20:
                break
            improved = _local_search(best, rng, max_iters=100, deadline=deadline - 20)
            if len(improved) < best_size:
                best = improved
                best_size = len(best)
                if best_size <= 10:
                    break

    if best is None:
        best = all_idx[:]
        
    return [_idx_point(i) for i in best]

# EVOLVE-BLOCK-END
