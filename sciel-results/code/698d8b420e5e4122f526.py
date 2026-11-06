<<<<<<< SEARCH
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random
import time

def construct():
    """
    Construct a tournament where every 4 vertices have a common dominator.
    We try to find a tournament on n=67 vertices (smallest known).
    If we can't find one within time, we fall back to a known valid construction
    (e.g., Paley on 83 or similar) if we can verify it quickly, or just return
    the best effort. However, the task requires a VALID tournament.
    
    Strategy:
    1. Use a randomized local search (simulated annealing or hill climbing) 
       with a sampled cost function to find a valid tournament on n=67.
    2. The cost is the number of 4-sets without a common dominator.
    3. To speed up, we sample a subset of 4-sets for the cost function during optimization.
    4. Once cost is 0 on the sample, we verify the full set of 4-sets.
    5. If time runs out, we return a known valid tournament (Paley 83) to ensure score > 0.
    """
    
    # Time limit for the search part
    start_time = time.time()
    time_limit = 100.0  # Leave 20s for verification and fallback
    
    # Target n: smallest known is 67
    n_target = 67
    
    # 1. Initialize a random tournament
    # Orientation: 0 if i->j, 1 if j->i. Stored in a matrix A[i,j] = 1 if i->j else 0.
    # Wait, the output format is: orientation[k] corresponds to pair (i,j) with i<j.
    # 0 means i -> j. 1 means j -> i.
    # Let's use a matrix M where M[i,j] = 1 if i beats j (i->j), 0 otherwise.
    # M[i,j] = 1 - M[j,i] for i != j.
    
    M = np.zeros((n_target, n_target), dtype=np.uint8)
    # Fill upper triangle randomly
    for i in range(n_target):
        for j in range(i+1, n_target):
            M[i,j] = 1 if random.random() < 0.5 else 0
            M[j,i] = 1 - M[i,j]
    
    # Precompute 4-sets? No, too many (67 choose 4 = ~700,000).
    # We need a fast cost function.
    # Cost of a 4-set {a,b,c,d}: 
    # A vertex v is a dominator if v->u for all u in the other 3.
    # We want at least one such vertex.
    
    # Let's define a helper to check if a specific 4-set is valid.
    # But calling this 10,000 times per step is slow in Python.
    # We can vectorize the check for a batch of 4-sets or use numpy tricks.
    
    # Alternative: 
    # Instead of checking all 4-sets, we can check for "bad" configurations.
    # A 4-set is bad if no vertex dominates the other 3.
    
    # Let's try a greedy construction first?
    # Or use the known structure?
    # Since we must produce a VALID tournament, and finding a 67-vertex one is hard,
    # let's see if we can use a known construction or a simpler search.
    
    # Actually, the prompt says "Minimize n". The seed uses 83.
    # If we can't beat 83, we should return 83 to get 1.0.
    # But we can try to get 67.
    
    # Let's implement a fast local search.
    # Move: flip the direction of one edge (i,j).
    # Cost change: only 4-sets containing both i and j are affected.
    # Number of such 4-sets is (n-2) choose 2. For n=67, that's 65*64/2 = 2080.
    # This is manageable.
    
    # We will maintain the full cost? No, full cost is sum over all 4-sets.
    # We can maintain a counter `bad_count` of how many 4-sets are bad.
    # When we flip edge (i,j), we update bad_count by checking the 2080 affected 4-sets.
    
    # Let's initialize the bad_count.
    # This requires checking all 4-sets initially. ~700k checks.
    # Each check: 4 vertices. Check if any dominates the other 3.
    # We can precompute for each vertex v, the set of vertices it dominates?
    # Or just compute on the fly with numpy?
    
    # Let's try to vectorize the initial cost calculation.
    # For each 4-tuple (a,b,c,d), check:
    # M[a,b]&M[a,c]&M[a,d] OR M[b,a]&M[b,c]&M[b,d] OR ...
    
    # This is too slow in pure Python loops for 700k sets * 100 steps.
    
    # Better idea:
    # Use the Paley tournament on 67? 67 is prime.
    # Paley tournament on p=67: i->j iff (j-i) is a quadratic residue mod 67.
    # Let's check if Paley(67) is valid for k=4.
    # Paley tournaments are highly symmetric. 
    # It is known that Paley(p) has the property that every subset of size (p+1)/2 has a common dominator? No.
    # Let's just test Paley(67) first. It's fast to construct.
    
    def check_paley(p):
        squares = {x*x % p for x in range(1, p)}
        # Construct matrix
        M_p = np.zeros((p, p), dtype=np.uint8)
        for i in range(p):
            for j in range(p):
                if i != j:
                    M_p[i,j] = 1 if (j-i) % p in squares else 0
        
        # Check validity
        # Sample some 4-sets to see if it's likely valid.
        # If it's not valid, we can't use it.
        
        # Let's check a few random 4-sets
        for _ in range(100):
            s = random.sample(range(p), 4)
            a,b,c,d = s
            # Check if any dominates
            valid = False
            for v in s:
                others = [x for x in s if x != v]
                if all(M_p[v][u] == 1 for u in others):
                    valid = True
                    break
            if not valid:
                return False # Likely not valid, or this sample is bad
        
        # If 100 samples pass, do a full check?
        # Full check is expensive. But if it fails, we fall back to 83.
        # Let's assume if it passes 100 random samples, it might be valid?
        # Actually, for k=4, the density of bad sets in a random tournament is low, but in a structured one it's either 0 or high.
        # If Paley(67) is not the solution, it will have many bad sets.
        
        # Let's do a more thorough check on a subset of 1000 sets.
        for _ in range(1000):
            s = random.sample(range(p), 4)
            a,b,c,d = s
            valid = False
            for v in s:
                others = [x for x in s if x != v]
                if all(M_p[v][u] == 1 for u in others):
                    valid = True
                    break
            if not valid:
                return False
        return True

    # Test Paley 67
    if check_paley(67):
        # If valid, construct it
        p = 67
        squares = {x*x % p for x in range(1, p)}
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                orientation.append(0 if (j-i) % p in squares else 1)
        return {'n': p, 'orientation': orientation}
        
    # Test Paley 73? 79? 83 is known to work.
    # Let's try 73.
    if check_paley(73):
        p = 73
        squares = {x*x % p for x in range(1, p)}
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                orientation.append(0 if (j-i) % p in squares else 1)
        return {'n': p, 'orientation': orientation}
        
    # Test Paley 79?
    if check_paley(79):
        p = 79
        squares = {x*x % p for x in range(1, p)}
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                orientation.append(0 if (j-i) % p in squares else 1)
        return {'n': p, 'orientation': orientation}
        
    # If none of the small Paley tournaments work, we fall back to 83.
    # 83 is known to be valid (from the seed program).
    p = 83
    squares = {x*x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i+1, p):
            orientation.append(0 if (j-i) % p in squares else 1)
    return {'n': p, 'orientation': orientation}

# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
import numpy as np
import random
import time
import itertools

def construct():
    """
    Find a valid tournament where every 4-set has a common dominator.
    Uses local search with efficient edge-flip cost updates.
    Falls back to Paley(83) if search fails within time limit.
    """
    start_time = time.time()
    time_limit = 100.0
    
    n = 67
    
    # Initialize with Paley(67) as starting point
    p = n
    squares = set((x*x) % p for x in range(1, p))
    M = np.zeros((p, p), dtype=np.int8)
    for i in range(p):
        for j in range(p):
            if i != j:
                M[i, j] = 1 if ((j - i) % p) in squares else 0
    
    def count_bad_sets(M, n):
        """Count the number of 4-sets without a common dominator."""
        bad_count = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        # Check if any of a,b,c,d dominates the other three
                        good = False
                        # a dominates b,c,d?
                        if M[a,b] and M[a,c] and M[a,d]:
                            good = True
                        # b dominates a,c,d?
                        elif M[b,a] and M[b,c] and M[b,d]:
                            good = True
                        # c dominates a,b,d?
                        elif M[c,a] and M[c,b] and M[c,d]:
                            good = True
                        # d dominates a,b,c?
                        elif M[d,a] and M[d,b] and M[d,c]:
                            good = True
                        if not good:
                            bad_count += 1
        return bad_count
    
    def is_set_good(M, a, b, c, d):
        """Check if 4-set {a,b,c,d} has a common dominator."""
        if M[a,b] and M[a,c] and M[a,d]:
            return True
        if M[b,a] and M[b,c] and M[b,d]:
            return True
        if M[c,a] and M[c,b] and M[c,d]:
            return True
        if M[d,a] and M[d,b] and M[d,c]:
            return True
        return False
    
    # Initial bad count
    initial_bad = count_bad_sets(M, n)
    
    if initial_bad == 0:
        # Paley(67) is valid!
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                # 0 means i->j, 1 means j->i
                # M[i,j]=1 means i->j, so orientation bit is 0
                # M[i,j]=0 means j->i, so orientation bit is 1
                orientation.append(0 if M[i,j] == 1 else 1)
        return {'n': p, 'orientation': orientation}
    
    # Local search: flip edges to reduce bad count
    bad_count = initial_bad
    best_bad = initial_bad
    best_M = M.copy()
    
    max_iterations = 50000
    stuck_count = 0
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random edge to flip
        i = random.randint(0, n-1)
        j = random.randint(0, n-1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        # Flip edge (i,j)
        M[i, j] = 1 - M[i, j]
        M[j, i] = 1 - M[j, i]
        
        # Recompute bad count only for 4-sets containing both i and j
        # Old contribution of 4-sets containing i and j
        old_contrib = 0
        new_contrib = 0
        
        for k in range(n):
            if k == i or k == j:
                continue
            for l in range(k+1, n):
                if l == i or l == j:
                    continue
                # 4-set {i, j, k, l}
                if not is_set_good(M, i, j, k, l):
                    new_contrib += 1
        
        # We need the old contribution, but we already flipped.
        # Let's flip back temporarily to compute old, then flip forward.
        # This is inefficient. Better: compute delta.
        
        # Actually, let's just recompute the full bad count periodically,
        # and for now, do a simple hill climb with occasional full recompute.
        
        # Flip back
        M[i, j] = 1 - M[i, j]
        M[j, i] = 1 - M[j, i]
        
        # Compute old contribution
        old_contrib = 0
        for k in range(n):
            if k == i or k == j:
                continue
            for l in range(k+1, n):
                if l == i or l == j:
                    continue
                if not is_set_good(M, i, j, k, l):
                    old_contrib += 1
        
        # Now flip forward and compute new contribution
        M[i, j] = 1 - M[i, j]
        M[j, i] = 1 - M[j, i]
        
        new_contrib = 0
        for k in range(n):
            if k == i or k == j:
                continue
            for l in range(k+1, n):
                if l == i or l == j:
                    continue
                if not is_set_good(M, i, j, k, l):
                    new_contrib += 1
        
        delta = new_contrib - old_contrib
        
        if delta < 0:
            # Accept the flip
            bad_count += delta
            stuck_count = 0
            if bad_count < best_bad:
                best_bad = bad_count
                best_M = M.copy()
        else:
            # Reject the flip, undo
            M[i, j] = 1 - M[i, j]
            M[j, i] = 1 - M[j, i]
            stuck_count += 1
        
        if bad_count == 0:
            break
        
        # Periodic full recompute to correct drift
        if iteration % 100 == 0:
            bad_count = count_bad_sets(M, n)
            if bad_count < best_bad:
                best_bad = bad_count
                best_M = M.copy()
    
    # Check if we found a valid solution
    if best_bad == 0:
        M = best_M
        orientation = []
        for i in range(p):
            for j in range(i+1, p):
                orientation.append(0 if M[i,j] == 1 else 1)
        return {'n': p, 'orientation': orientation}
    
    # Fallback: use Paley(83) which is known to be valid
    p = 83
    squares = set((x*x) % p for x in range(1, p))
    orientation = []
    for i in range(p):
        for j in range(i+1, p):
            orientation.append(0 if ((j-i) % p) in squares else 1)
    return {'n': p, 'orientation': orientation}

# EVOLVE-BLOCK-END
>>>>>>> REPLACE
