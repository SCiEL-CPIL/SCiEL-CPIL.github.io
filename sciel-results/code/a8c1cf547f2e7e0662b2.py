"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Find a maximum-size Sidon set (all pairwise sums a+b for a<=b distinct)
    within [1, 284] that must contain {1, 2, 5, 11}.
    """
    start_time = time.time()
    deadline = 110  # safety margin before 120s limit
    
    prescribed = [1, 2, 5, 11]
    
    def get_sums(a):
        """Compute set of all pairwise sums a+b for a<=b in sorted list a."""
        sums = set()
        n = len(a)
        for i in range(n):
            for j in range(i, n):
                sums.add(a[i] + a[j])
        return sums
    
    def can_add(a, sums, x):
        """Check if x can be added to a (with its current sums) without conflict."""
        # New sums introduced by x: x + y for all y in a, plus 2*x
        for y in a:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        # Also need to check that x+y for different y are distinct among themselves
        # and distinct from 2*x. Since x+y > x+z if y>z (x fixed, y>z), they are distinct.
        # 2*x vs x+y: 2*x = x+y => x=y, but x is new and not in a, so 2*x != x+y for y in a.
        # So we only need to check against existing sums.
        return True
    
    def greedy_extend(a, sums, candidates):
        """Greedily add candidates in order if possible."""
        a = list(a)
        sums = set(sums)
        for x in candidates:
            if x in a:
                continue
            if can_add(a, sums, x):
                a.append(x)
                a.sort()
                # Update sums
                for y in a[:-1]:  # all existing elements before adding x
                    sums.add(x + y)
                sums.add(2 * x)
        return a, sums
    
    def local_search(a, max_iters=5000):
        """
        Local search: try to improve by swapping or adding.
        Returns improved set or the same set.
        """
        a = sorted(a)
        sums = get_sums(a)
        n = len(a)
        used = set(a)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding a new element
            for x in range(1, 285):
                if x in used:
                    continue
                if can_add(a, sums, x):
                    a.append(x)
                    a.sort()
                    used.add(x)
                    for y in a[:-1]:
                        sums.add(x + y)
                    sums.add(2 * x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swapping: remove one element, add another
            # This is more expensive, so do it less frequently
            if random.random() < 0.3:
                # Pick a random element to remove
                idx = random.randrange(len(a))
                removed = a[idx]
                a2 = a[:idx] + a[idx+1:]
                sums2 = get_sums(a2)
                used2 = set(a2)
                
                # Try to add a new element
                added = False
                for x in range(1, 285):
                    if x in used2:
                        continue
                    if can_add(a2, sums2, x):
                        a2.append(x)
                        a2.sort()
                        a = a2
                        sums = sums2
                        for y in a[:-1]:
                            sums.add(x + y)
                        sums.add(2 * x)
                        used = set(a)
                        added = True
                        break
                
                if added:
                    continue
            
            # No improvement found in this iteration, break to avoid infinite loop
            break
        
        return a
    
    # Initial greedy construction
    # Start with prescribed elements
    a = list(prescribed)
    a.sort()
    sums = get_sums(a)
    used = set(a)
    
    # Greedily add elements from 12 to 284
    for x in range(12, 285):
        if x in used:
            continue
        if can_add(a, sums, x):
            a.append(x)
            a.sort()
            used.add(x)
            for y in a[:-1]:
                sums.add(x + y)
            sums.add(2 * x)
    
    best_size = len(a)
    best_a = list(a)
    
    # Multiple random restarts with different orders
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Create a random order of candidates
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        
        # Filter to only those not in prescribed
        candidates = [x for x in candidates if x not in used]
        
        a_restart, sums_restart = greedy_extend(prescribed, get_sums(prescribed), candidates)
        
        if len(a_restart) > best_size:
            best_size = len(a_restart)
            best_a = list(a_restart)
        
        # Local search on the best found so far
        a_ls = local_search(best_a, max_iters=2000)
        if len(a_ls) > best_size:
            best_size = len(a_ls)
            best_a = list(a_ls)
        
        # Also try local search on the restart result
        a_ls2 = local_search(a_restart, max_iters=1000)
        if len(a_ls2) > best_size:
            best_size = len(a_ls2)
            best_a = list(a_ls2)
    
    # Final local search on the best
    if time.time() < deadline - 5:
        a_final = local_search(best_a, max_iters=10000)
        if len(a_final) > best_size:
            best_a = list(a_final)
    
    return best_a
# EVOLVE-BLOCK-END
