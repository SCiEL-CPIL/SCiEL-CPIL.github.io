"""Maximum admissible tuple construction via optimized residue banning.

Implements a greedy strategy to select residue classes to ban for primes p <= M
such that the total number of banned points is minimized. This maximizes the
size of the remaining admissible tuple.
"""

import numpy as np

N = 5000


def _primes_up_to(limit):
    """Return list of primes up to limit using sieve."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(sieve[p * p::p]))
    return [p for p in range(2, limit + 1) if sieve[p]]


def construct():
    primes = _primes_up_to(N)
    
    # We want to find the largest admissible tuple.
    # Let's try to construct a tuple of size at least M.
    # We ban residues for primes p <= M.
    # Then we verify if the resulting set is admissible (i.e., misses residues for all p <= |S|).
    # If not, we ban more.
    
    # Start with a high estimate for M.
    # Based on previous analysis, M ~ 450-500 is a good starting point.
    # Let's try to maximize M.
    
    best_result = []
    
    # We can binary search or linear search for the max M that works.
    # But the "works" condition depends on the greedy choices.
    # Let's just try a range of M values from high to low, and return the first one that yields a valid admissible set.
    # Actually, a higher M means we ban for more primes, which usually reduces the set size.
    # But banning for more primes might allow us to keep the set larger if the overlaps are good?
    # No, banning for more primes always adds constraints.
    # However, if we set M too low, we might end up with a set of size K > M, which is invalid unless we check and fix.
    
    # Strategy:
    # 1. Pick a target M.
    # 2. Greedily ban residues for p <= M to minimize removed points.
    # 3. Check if the remaining set S is admissible.
    #    - If yes, return S.
    #    - If no, find the smallest prime p <= |S| that is "full" (all residues present).
    #      Ban a residue for p (minimizing new removals).
    #      Repeat until admissible.
    # 4. If the final size is very small, try a lower M?
    #    Actually, starting with a higher M forces us to ban for more primes, which might be suboptimal if the overlaps are poor.
    #    Starting with a lower M might leave us with a larger set that is already admissible for the larger primes?
    #    No, if p > M, we haven't banned it. If p <= |S|, we MUST ban it.
    #    So the final set will have bans for all p <= |S| regardless of initial M.
    #    The difference is the *order* and *choice* of bans.
    #    Banning small primes first (which have large classes) allows us to pick large prime residues that fall into the already-banned holes.
    #    So we should ALWAYS ban small primes first.
    
    # Therefore, the optimal strategy is:
    # 1. Iterate primes p in ascending order.
    # 2. For each p, if p <= current_estimate_of_size, ban the best residue.
    # 3. Update size.
    
    # Let's refine:
    # We don't know the final size.
    # But we know that if we ban for p, we remove ~N/p points.
    # Let's simulate the process.
    
    banned = np.zeros(N + 1, dtype=bool)
    current_size = N + 1
    
    # We will process primes in ascending order.
    # For each prime p, we decide whether to ban a residue.
    # We should ban if p <= current_size.
    # If we ban, we choose the residue that minimizes the number of NEWLY banned points.
    
    for p in primes:
        if p > current_size:
            break
        
        # Find the residue r that minimizes new bans
        best_r = -1
        min_new_bans = N + 1
        
        # Precompute indices for this prime?
        # Or just loop r.
        for r in range(p):
            # Points x such that x % p == r
            # Check how many are not already banned
            # Vectorized:
            # mask = (np.arange(N+1) % p == r) & (~banned)
            # count = mask.sum()
            
            # To be faster, we can precompute the list of points for each residue?
            # Or use modulo on the array.
            # N=5000, p is small initially.
            # 5000 operations is fast.
            
            # Optimization:
            # We can just check the specific points.
            # But numpy is fine.
            
            # Create mask for x % p == r
            # This is O(N).
            # Doing this for all r in [0, p-1] is O(N*p).
            # Sum of p for primes up to 500 is roughly 500^2 / (2 ln 500) ~ 100k.
            # 100k * 5000 = 500M operations. Might be slow in Python loops, but numpy vectorized is C speed.
            # However, creating the array `np.arange(N+1) % p` for each r is wasteful.
            
            # Better:
            # residues = np.arange(N+1) % p
            # Then for each r, count = (~banned)[residues == r].sum()
            
            pass # Placeholder
        
        # Let's implement the efficient version inside the loop
        
        residues = np.arange(N + 1) % p
        not_banned = ~banned
        
        # For each r, count new bans
        # We can use np.bincount on the residues of not_banned points?
        # Indices of not banned points:
        # nb_indices = np.where(not_banned)[0]
        # nb_residues = nb_indices % p
        # counts = np.bincount(nb_residues, minlength=p)
        # This gives the number of not-banned points for each residue.
        
        nb_indices = np.where(not_banned)[0]
        if len(nb_indices) == 0:
            break
        nb_residues = nb_indices % p
        counts = np.bincount(nb_residues, minlength=p)
        
        # We want to pick r with min counts[r]
        best_r = np.argmin(counts)
        new_bans = counts[best_r]
        
        if new_bans == 0:
            # All points with this residue are already banned?
            # Then banning it costs 0.
            # But we still need to mark it as "banned" for the logic?
            # Actually, if counts[best_r] is 0, it means all points x % p == best_r are already banned.
            # So the residue class is effectively empty in the remaining set.
            # We don't need to update `banned` array as no new points are added.
            # But we must ensure the logic holds.
            pass
        else:
            # Update banned array
            # Mark all x % p == best_r as banned
            # We can use the mask
            mask = (np.arange(N + 1) % p == best_r)
            banned |= mask
            
        current_size = (~banned).sum()
        
    # After this loop, we have banned residues for all p <= current_size (at the time of processing).
    # But current_size decreases.
    # We need to check if the final set is admissible.
    # The set is S = ~banned.
    # Size K = current_size.
    # Check for all p <= K, is there a missing residue?
    
    final_set_indices = np.where(~banned)[0]
    K = len(final_set_indices)
    
    # Check admissibility
    # For each prime p <= K, check if all residues are present in final_set_indices.
    # If all present, we need to ban one.
    
    # We might need to iterate again if we missed some primes because K dropped below p during the first pass?
    # In the first pass, we broke if p > current_size.
    # If current_size dropped, we skipped larger primes.
    # But if K is the final size, we need to check primes up to K.
    # Since K <= initial_size, and we processed primes up to initial_size (mostly), we might have skipped some if size dropped fast.
    # Let's do a verification pass.
    
    is_admissible = True
    # We need to check primes p <= K
    primes_to_check = [p for p in primes if p <= K]
    
    for p in primes_to_check:
        # Check if any residue is missing
        # Get residues of final_set_indices mod p
        if len(final_set_indices) == 0:
            break
        res = final_set_indices % p
        # Check if all 0..p-1 are in res
        # Use a set or boolean array
        present = np.zeros(p, dtype=bool)
        present[res] = True
        if present.all():
            # All residues present. We must ban one.
            # Choose the one with fewest points in final_set_indices
            counts = np.bincount(res, minlength=p)
            r_to_ban = np.argmin(counts)
            
            # Ban this residue
            # Update banned array
            mask = (np.arange(N + 1) % p == r_to_ban)
            banned |= mask
            
            # Update final_set_indices
            final_set_indices = final_set_indices[~(final_set_indices % p == r_to_ban)]
            K = len(final_set_indices)
            
            # Since K decreased, we might need to re-check primes?
            # The condition is p <= K.
            # If K drops, we might no longer need to check larger primes.
            # But we are iterating primes in ascending order.
            # If we ban for p, K drops.
            # We continue to the next prime q > p.
            # If q > new K, we can stop.
            
            # We need to be careful: if K drops, some primes we already checked might now be > K?
            # No, we are checking p <= K.
            # If we ban for p, K becomes smaller.
            # The primes we have already checked were <= old K.
            # They are still <= new K? No, new K < old K.
            # So some primes we checked might now be > K.
            # But if they were checked, we ensured they had a missing residue.
            # Removing points cannot create a new residue class. It can only empty them.
            # So if a prime p had a missing residue before, it still has it (or more).
            # So previously fixed primes remain fixed.
            # We only need to check primes that were NOT checked (because they were > old K) but are now <= new K?
            # Impossible, new K < old K.
            # So we only need to check primes in (new K, old K]? No.
            # We need to check primes <= new K.
            # We have already checked primes <= old K.
            # So we have checked all primes <= new K.
            # Wait, did we?
            # In the verification loop, we iterate `primes_to_check` which was computed based on initial K.
            # If K drops, `primes_to_check` contains primes > new K.
            # We should skip those.
            
            # So the loop `for p in primes_to_check` is fine, but we should break if p > K.
            
    return final_set_indices.tolist()
