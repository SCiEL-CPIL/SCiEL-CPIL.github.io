"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time

def _is_sunflower_triple(a, b, c):
    """Check if three distinct 4-sets form a sunflower (all pairwise intersections equal)."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _can_add(family, s):
    """Check if set s can be added to family without creating a sunflower triple."""
    for i in range(len(family)):
        for j in range(i + 1, len(family)):
            if _is_sunflower_triple(family[i], family[j], s):
                return False
    return True

def _greedy_random(all_sets, max_time=55.0):
    """Greedy algorithm with random ordering to find a large sunflower-free family."""
    start = time.time()
    family = []
    remaining = list(all_sets)
    random.shuffle(remaining)
    
    while remaining and (time.time() - start) < max_time:
        s = remaining.pop()
        if _can_add(family, s):
            family.append(s)
    
    return family

def _local_search(family, all_sets, max_time=55.0):
    """Try to improve the family by swapping out sets for better ones."""
    start = time.time()
    family = list(family)
    family_set = set(frozenset(s) for s in family)
    
    # Build index: for each set in family, store it
    # Try to add random sets not in family
    candidates = [s for s in all_sets if frozenset(s) not in family_set]
    random.shuffle(candidates)
    
    improved = True
    while improved and (time.time() - start) < max_time:
        improved = False
        for s in candidates:
            if (time.time() - start) >= max_time:
                break
            fs = frozenset(s)
            if fs in family_set:
                continue
            # Try adding s
            if _can_add(family, s):
                family.append(s)
                family_set.add(fs)
                improved = True
                break
            # Try swapping: remove one family member, add s
            # Only try a few swaps to keep it fast
            for idx in range(min(3, len(family))):
                old = family[idx]
                # Check if removing old and adding s is valid
                new_family = family[:idx] + family[idx+1:]
                if _can_add(new_family, s):
                    # Verify no sunflower in new_family (should be fine since we removed one)
                    # But need to check that new_family itself is valid
                    valid = True
                    nf = new_family
                    for i2 in range(len(nf)):
                        for j2 in range(i2+1, len(nf)):
                            # We already know new_family minus old is valid from before
                            pass
                    # Actually, since family was valid, removing one keeps it valid
                    family = new_family
                    family.append(s)
                    family_set.remove(frozenset(old))
                    family_set.add(fs)
                    improved = True
                    break
            if improved:
                break
    
    return family

def construct():
    start_time = time.time()
    deadline = 55.0  # seconds
    
    # Generate all 4-element subsets of [1,16]
    all_sets = [frozenset(combo) for combo in itertools.combinations(range(1, 17), 4)]
    
    # Try multiple random greedy passes and keep the best
    best_family = []
    random.seed(42)
    
    num_passes = 0
    while time.time() - start_time < deadline:
        num_passes += 1
        family = _greedy_random(all_sets, max_time=min(20.0, deadline - (time.time() - start_time)))
        if len(family) > len(best_family):
            best_family = family
        
        # If we've found a large family, do local search
        if len(best_family) >= 50 and time.time() - start_time < deadline - 10:
            improved = _local_search(best_family, all_sets, max_time=min(10.0, deadline - (time.time() - start_time)))
            if len(improved) > len(best_family):
                best_family = improved
    
    # Convert to sorted lists for output
    result = [sorted(s) for s in best_family]
    return result
# EVOLVE-BLOCK-END
