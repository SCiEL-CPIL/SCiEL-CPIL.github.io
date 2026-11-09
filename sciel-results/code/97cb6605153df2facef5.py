<<<<<<< SEARCH
def _greedy_construct(seed, deadline):
    """
    Construct an admissible tuple using greedy residue removal.
    Returns the list of points.
    """
    random.seed(seed)
    active = np.ones(RANGE, dtype=bool)
    x_indices = np.arange(RANGE)
    current_size = RANGE
    
    # Precompute residues for speed? No, just use numpy.
    
    for p in PRIMES:
        if time.time() > deadline:
            break
        if p > current_size:
            break
        
        # Count active elements in each residue class mod p
        # Only consider active elements
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
            
        residues = active_indices % p
        if len(residues) == 0:
            break
        
        # Count occurrences of each residue
        counts = np.bincount(residues, minlength=p)
        
        min_count = np.min(counts)
        
        if min_count == 0:
            # Already admissible for this prime
            continue
        
        # Find all classes with min count
        candidate_classes = np.where(counts == min_count)[0]
        
        # Randomly pick one
        r = candidate_classes[random.randint(0, len(candidate_classes) - 1)]
        
        # Deactivate elements in this class
        # We need to deactivate only the ACTIVE ones in this class
        # mask = (x_indices % p) == r AND active
        mask = (x_indices % p == r) & active
        current_size -= np.sum(mask)
        active = active & ~mask
        
        if current_size == 0:
            break
    
    return np.where(active)[0].tolist()
=======
def _min_degree_greedy(seed, deadline):
    """
    Construct an admissible tuple using min-degree greedy on a conflict graph.
    Two numbers x, y conflict if (x-y) % p == 0 for some prime p <= K.
    We greedily select the vertex with minimum degree among remaining vertices.
    """
    random.seed(seed)
    
    # Conflict threshold: primes up to K
    K = 100
    conflict_primes = [p for p in PRIMES if p <= K]
    
    # Build adjacency: for each x in [0, N], find which y's conflict with x
    # Two numbers x, y conflict if there exists p in conflict_primes such that x ≡ y (mod p)
    # This means x and y are in the same residue class mod p for some p.
    
    # For efficiency, we'll track for each x which residue classes it occupies
    # and which other numbers share those classes.
    
    # Precompute residue class membership for each prime
    # residue_class[p][r] = set of x in [0, N] such that x % p == r
    residue_class = {}
    for p in conflict_primes:
        residue_class[p] = [set() for _ in range(p)]
        for x in range(RANGE):
            residue_class[p][x % p].add(x)
    
    # Now, for each x, compute its neighbors (conflicting vertices)
    # neighbor[x] = set of y != x such that x and y conflict
    
    # This could be expensive, so let's use a different approach:
    # Instead of building the full graph, we'll use a greedy selection that
    # maintains a "degree" count for each remaining vertex.
    
    # Initialize: all vertices are available
    available = set(range(RANGE))
    selected = set()
    
    # For each available vertex, compute its current degree (number of conflicts with other available vertices)
    # degree[x] = |{y in available, y != x, x conflicts with y}|
    
    # To compute this efficiently, for each x, we can count how many y in available
    # share a residue class with x for any p in conflict_primes.
    
    # But this is still O(N^2) in the worst case. Let's use a smarter approach.
    
    # Alternative: Instead of computing exact degrees, we can use a heuristic
    # that approximates the degree based on residue class sizes.
    
    # For now, let's try a simpler approach: randomly select vertices while
    # maintaining admissibility, but prioritize vertices that are "less constrained."
    
    # Actually, let's just implement a straightforward min-degree greedy:
    # 1. Start with all vertices available.
    # 2. Repeatedly select the vertex with minimum degree among available vertices.
    # 3. Remove it and all its neighbors from available.
    # 4. Continue until no vertices remain.
    
    # But we want the LARGEST independent set, so we should select vertices
    # that don't conflict with each other. The min-degree greedy for maximum
    # independent set selects the vertex with minimum degree, adds it to the
    # independent set, and removes it and its neighbors from the graph.
    
    # Let's implement this.
    
    # For efficiency, we'll precompute for each x the set of y that conflict with x.
    # But this is O(N^2) memory, which is too much for N=5000.
    
    # Instead, we'll compute degrees on the fly using residue classes.
    
    # Let's try a different approach: use the greedy residue removal but with
    # a different tie-breaking strategy that mimics min-degree.
    
    # Actually, let's just implement a simple version:
    # - Start with all numbers available.
    # - For each prime p <= K, we need to ensure not all residue classes are filled.
    # - Greedily remove the residue class with the most elements (to minimize removals).
    
    # This is similar to the existing _greedy_construct but with a different strategy.
    
    # Let's try: for each prime, remove the residue class with the MOST elements.
    # This is the opposite of the current approach (which removes the MIN).
    
    active = np.ones(RANGE, dtype=bool)
    x_indices = np.arange(RANGE)
    current_size = RANGE
    
    for p in conflict_primes:
        if time.time() > deadline:
            break
        if p > current_size:
            break
        
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
            
        residues = active_indices % p
        if len(residues) == 0:
            break
        
        counts = np.bincount(residues, minlength=p)
        
        # Remove the class with the MAX count (opposite of current)
        max_count = np.max(counts)
        
        if max_count == 0:
            continue
        
        candidate_classes = np.where(counts == max_count)[0]
        
        # Randomly pick one among the max
        r = candidate_classes[random.randint(0, len(candidate_classes) - 1)]
        
        mask = (x_indices % p == r) & active
        current_size -= np.sum(mask)
        active = active & ~mask
        
        if current_size == 0:
            break
    
    return np.where(active)[0].tolist()
>>>>>>> REPLACE

