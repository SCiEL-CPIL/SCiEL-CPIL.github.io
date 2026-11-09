import time
import random
from itertools import combinations

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def largest_sidon_subset_size(S):
    S = sorted(S)
    n = len(S)
    # Try from largest k down
    for k in range(n, 0, -1):
        for combo in combinations(S, k):
            if is_sidon(combo):
                return k
    return 1

def count_sum_collisions(S):
    """Count how many pair sums are duplicated (higher = more collisions = harder for Sidon)"""
    sums = {}
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                sums[s] += 1
            else:
                sums[s] = 1
    # Total collisions = sum of (count-1) for each sum that appears more than once
    collisions = sum(v - 1 for v in sums.values() if v > 1)
    return collisions

def build_collision_graph():
    """Build adjacency matrix where A[i][j] = number of shared pair sums between i and j.
    For each integer x in [0,128], we consider all pairs (x, y) and (x, z) where y != z,
    and count how many times x+y == x+z (impossible) or more generally how many
    collisions x creates with other elements.
    
    Actually, let's think differently. For two integers a and b, they 'collide' if
    there exist c, d such that a+c = b+d. This is equivalent to a-b = d-c.
    So the collision weight between a and b is the number of pairs (c,d) with a-b = d-c.
    
    Simpler: For each integer x, define its 'collision profile' as the set of all sums
    x+y for y in [0,128], y != x. Two integers a,b have high collision if their
    collision profiles overlap a lot.
    """
    n = 129
    # collision_matrix[i][j] = number of y such that i+y == j+z for some z, or more simply:
    # For fixed i,j, the number of pairs (y,z) with y!=i, z!=j, y!=z such that i+y = j+z
    # This is complex. Let's use a simpler proxy:
    # collision_matrix[i][j] = |{y : y != i, y != j, i+y == j+(y + (i-j))} |
    # i+y = j+z => z = i+y-j. So for each y != i,j, check if z = i+y-j is in [0,128] and z != j and z != i
    # Actually let's just count: for each pair (i,j), how many y exist such that i+y = j+z for valid z
    
    # Even simpler and faster: use the fact that two numbers a,b are "close to colliding"
    # if |a-b| is small (since a-b = d-c means many (d,c) pairs exist for small differences)
    # The number of (d,c) pairs with d-c = k is (129-|k|) for |k| < 129
    # So collision_weight(a,b) = 129 - |a-b| (for a != b), and for a==b it's the diagonal
    
    # But this ignores the actual set structure. Let's use a different approach:
    # Build a graph where edge weight between a and b = number of "witness" pairs
    # that make a+b appear in multiple ways. 
    
    # Most practical: just use |a-b| as a proxy for collision potential.
    # Small differences => more collisions.
    collision_matrix = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            # Number of (c,d) with c-d = i-j (or j-i)
            diff = abs(i - j)
            if diff < n:
                collision_matrix[i][j] = n - diff
                collision_matrix[j][i] = n - diff
        collision_matrix[i][i] = 0  # No self-collision in this metric
    
    return collision_matrix

def spectral_centrality_select(k, top_n=12):
    """Use power iteration on the collision graph to find top-k eigenvector centrality nodes."""
    n = 129
    # Build the collision matrix
    A = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            diff = abs(i - j)
            w = n - diff  # collision weight
            A[i][j] = w
            A[j][i] = w
    
    # Power iteration for eigenvector centrality
    # Initialize with random vector
    random.seed(42)
    v = [random.random() for _ in range(n)]
    
    for iteration in range(100):
        new_v = [0.0]*n
        for i in range(n):
            s = 0.0
            for j in range(n):
                s += A[i][j] * v[j]
            new_v[i] = s
        # Normalize
        norm = sum(x*x for x in new_v) ** 0.5
        if norm > 0:
            new_v = [x/norm for x in new_v]
        v = new_v
    
    # Select top-k nodes by centrality
    indexed = sorted(range(n), key=lambda i: v[i], reverse=True)
    return indexed[:top_n]

def greedy_collision_build():
    """Greedy construction that at each step picks the element that maximizes
    total collision with already-selected elements."""
    n = 129
    # Precompute collision weights
    coll_w = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            diff = abs(i - j)
            w = n - diff
            coll_w[i][j] = w
            coll_w[j][i] = w
    
    current = set()
    # Start with a random element
    current.add(random.randint(0, 128))
    
    for step in range(11):
        best_candidate = None
        best_score = -1
        for c in range(n):
            if c in current:
                continue
            # Score = total collision with current set
            score = sum(coll_w[c][x] for x in current)
            if score > best_score:
                best_score = score
                best_candidate = c
        if best_candidate is not None:
            current.add(best_candidate)
    
    return sorted(current)

def local_search_improve(S, deadline):
    """Try single-element replacements to reduce largest Sidon subset size."""
    S = S[:]
    n = len(S)
    current_val = largest_sidon_subset_size(S)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            old = S[i]
            # Try replacing with nearby values first
            for offset in range(1, 129):
                for new in [old + offset, old - offset]:
                    if new < 0 or new > 128:
                        continue
                    if new == old or new in S:
                        continue
                    trial = S[:]
                    trial[i] = new
                    val = largest_sidon_subset_size(trial)
                    if val < current_val:
                        current_val = val
                        S = trial
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return S, current_val

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    best_set = None
    best_val = 99
    
    # Strategy 1: Spectral centrality based selection
    if time.time() < deadline - 30:
        # Get top-12 by eigenvector centrality
        spectral_set = spectral_centrality_select(12)
        val = largest_sidon_subset_size(spectral_set)
        if val < best_val:
            best_val = val
            best_set = spectral_set[:]
        
        # Local search from spectral set
        if best_set is not None and time.time() < deadline - 10:
            improved_set, improved_val = local_search_improve(best_set, deadline)
            if improved_val < best_val:
                best_val = improved_val
                best_set = improved_set
    
    # Strategy 2: Greedy collision-maximizing construction with multiple restarts
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        current = greedy_collision_build()
        if len(current) == 12:
            val = largest_sidon_subset_size(current)
            if val < best_val:
                best_val = val
                best_set = current[:]
        
        # Local search from this point
        if best_set is not None and time.time() < deadline - 5:
            improved_set, improved_val = local_search_improve(best_set, deadline)
            if improved_val < best_val:
                best_val = improved_val
                best_set = improved_set
    
    # Strategy 3: Random restarts with greedy collision building
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        
        current = set()
        current.add(random.randint(0, 128))
        
        # Precompute collision weights
        n = 129
        coll_w = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                diff = abs(i - j)
                w = n - diff
                coll_w[i][j] = w
                coll_w[j][i] = w
        
        for step in range(11):
            if time.time() > deadline - 5:
                break
            best_candidate = None
            best_score = -1
            # Sample candidates
            candidates = random.sample(range(n), min(40, n - len(current)))
            for c in candidates:
                if c in current:
                    continue
                score = sum(coll_w[c][x] for x in current)
                if score > best_score:
                    best_score = score
                    best_candidate = c
            if best_candidate is not None:
                current.add(best_candidate)
        
        if len(current) == 12:
            S = sorted(current)
            val = largest_sidon_subset_size(S)
            if val < best_val:
                best_val = val
                best_set = S[:]
    
    # Final local search polish
    if best_set is not None and time.time() < deadline - 2:
        improved_set, improved_val = local_search_improve(best_set, deadline)
        if improved_val < best_val:
            best_val = improved_val
            best_set = improved_set
    
    # Fallback
    if best_set is None:
        best_set = list(range(12))
        best_val = largest_sidon_subset_size(best_set)
    
    return best_set
