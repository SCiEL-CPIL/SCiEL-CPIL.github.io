import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    def disjoint(r1, m1, r2, m2):
        """Check if residue classes x ≡ r1 (mod m1) and x ≡ r2 (mod m2) are disjoint."""
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0
    
    def is_valid(r, m, assigned):
        """Check if (r, m) is disjoint from all assigned classes."""
        for (r2, m2) in assigned:
            if not disjoint(r, m, r2, m2):
                return False
        return True
    
    def greedy_order(order, assigned, remaining_candidates):
        """Greedy: add classes in the given order if possible."""
        assigned = list(assigned)
        for m in order:
            if m in remaining_candidates:
                # Find a valid residue for m
                # Prefer residues that maximize "distance" from existing classes?
                # Or just pick the first valid one.
                # To be more strategic, we can try to pick a residue that is
                # incompatible with as many *future* candidates as possible?
                # That's expensive. Let's stick to first valid for speed, 
                # but maybe randomize the start point for diversity.
                
                # Optimization: pre-compute valid residues? No, depends on assigned.
                # m is at most 128, so iterating r is cheap.
                
                # Heuristic: try residues in a pseudo-random order or just 0..m-1
                # Actually, the specific residue matters.
                # Let's try all valid residues? No, branching factor explodes.
                # Let's pick the first valid residue.
                # To improve, we can shuffle the candidate residues if multiple are valid.
                # But usually, if gcd(m, m2) is small, many residues work.
                
                # Let's try to pick a residue that is "maximally distinct"
                # Simple heuristic: pick r such that (r - r2) % g is maximized?
                # Or just pick 0 if valid?
                
                # Let's just pick the first valid residue found.
                # But we can randomize the search order of residues to break ties.
                residues = list(range(m))
                random.shuffle(residues)
                
                found = None
                for r in residues:
                    if is_valid(r, m, assigned):
                        found = (r, m)
                        break
                
                if found:
                    assigned.append(found)
                    remaining_candidates.remove(m)
        
        return assigned, remaining_candidates

    def local_search(assigned, remaining_candidates):
        """
        Try to improve the set by swapping or adding.
        1. Try to add any remaining candidate.
        2. Try to swap an existing class with a remaining one to allow more additions?
        """
        assigned = list(assigned)
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any remaining candidate
            for m in list(remaining_candidates):
                # Find valid residue
                residues = list(range(m))
                random.shuffle(residues)
                found = False
                for r in residues:
                    if is_valid(r, m, assigned):
                        assigned.append((r, m))
                        remaining_candidates.remove(m)
                        improved = True
                        found = True
                        break
                if found:
                    break # Restart the loop with the new state
            
            # 2. Try to swap
            # For each assigned class, remove it, and try to add two remaining classes?
            # This is complex. Let's stick to simple addition first.
            # If no additions, try a simple swap: remove one assigned, add one remaining.
            # This doesn't increase count, but might unblock future additions?
            # Actually, if we can't add anything, the set is maximal w.r.t single additions.
            # To increase count, we need to find a set of swaps that increases total count.
            
            # Let's try a "kick" move:
            # Pick a random assigned class A.
            # Remove A.
            # Try to add a random remaining class B.
            # If B is added, try to add another remaining class C.
            # If C is added, we have net +1 (removed A, added B and C).
            
            if len(assigned) > 0 and len(remaining_candidates) > 1:
                for _ in range(10): # Limit kicks
                    if time.time() > deadline:
                        return assigned, remaining_candidates
                    
                    idx = random.randint(0, len(assigned) - 1)
                    a_class = assigned[idx]
                    assigned.pop(idx)
                    
                    # Try to add two classes from remaining
                    added = []
                    for m in list(remaining_candidates):
                        if len(added) == 2:
                            break
                        residues = list(range(m))
                        random.shuffle(residues)
                        for r in residues:
                            if is_valid(r, m, assigned + added):
                                added.append((r, m))
                                break
                    
                    if len(added) == 2:
                        # Net gain: -1 (a_class) + 2 (added) = +1
                        assigned.extend(added)
                        for a in added:
                            remaining_candidates.remove(a[1])
                        improved = True
                        break
                    else:
                        # Restore a_class if we didn't improve
                        assigned.append(a_class)
                        # Note: 'added' classes were not committed to assigned in the check,
                        # so no need to remove them from remaining if we fail?
                        # Wait, in the check `is_valid(r, m, assigned + added)`,
                        # 'added' is a local list. 'remaining_candidates' was not modified.
                        # So if we fail, we just restore a_class.
                        pass
        
        return assigned, remaining_candidates

    # Main Search
    all_moduli = list(range(2, 129))
    
    best_assigned = []
    best_remaining = set(all_moduli)
    
    # 1. Greedy with different orderings
    # Order by increasing modulus? Decreasing? Random?
    orderings = [
        sorted(all_moduli),
        sorted(all_moduli, reverse=True),
        all_moduli[:],
    ]
    random.shuffle(orderings[2])
    
    for _ in range(20):
        random_order = all_moduli[:]
        random.shuffle(random_order)
        orderings.append(random_order)
    
    for order in orderings:
        if time.time() > deadline:
            break
        
        assigned, remaining = greedy_order(order, [], set(all_moduli))
        assigned, remaining = local_search(assigned, remaining)
        
        if len(assigned) > len(best_assigned):
            best_assigned = assigned
            best_remaining = remaining
    
    # 2. Local search from the best found
    best_assigned, best_remaining = local_search(best_assigned, best_remaining)
    
    # 3. Try more local search iterations
    # The local_search above does a limited number of kicks.
    # Let's run a more aggressive local search on the best result.
    # We can also try to restart local search from the best result multiple times.
    
    for _ in range(5):
        if time.time() > deadline:
            break
        # Create a copy to allow destructive local search if we track changes?
        # My local_search modifies copies.
        # Let's just call it again, it might find different moves due to randomization.
        assigned, remaining = local_search(best_assigned, best_remaining)
        if len(assigned) > len(best_assigned):
            best_assigned = assigned
            best_remaining = remaining
        else:
            break
    
    # Format output: list of [residue, modulus]
    result = [[r, m] for r, m in best_assigned]
    
    # Fallback if empty (should not happen)
    if not result:
        result = [[0, 2]]
        
    return result

# Quick sanity check for disjointness
# def verify(result):
#     for i in range(len(result)):
#         for j in range(i+1, len(result)):
#             r1, m1 = result[i]
#             r2, m2 = result[j]
#             if not disjoint(r1, m1, r2, m2):
#                 print(f"Conflict: {r1} mod {m1} and {r2} mod {m2}")
#                 return False
#     return True

# construct()
