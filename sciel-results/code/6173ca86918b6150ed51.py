"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes a heuristic matrix based on a data-driven co-occurrence prior.
    
    Strategy:
    1. Run a large number of First-Fit-Decreasing (FFD) simulations. FFD is a
       strong lower-bound heuristic for bin packing.
    2. In each simulation, record which items are placed in the same bin.
    3. Construct a co-occurrence matrix C where C[i, j] is the count of times
       item i and item j appeared in the same bin across all simulations.
    4. Normalize and transform this count into an attractiveness score.
       - If C[i, j] > 0, the pair is frequently seen together in good solutions.
       - We use a log-scale or power-scale to amplify frequent co-occurrences
         while keeping rare ones low but non-zero.
       - Pairs that never co-occur get a very low baseline (1e-9).
       
    This approach shifts from a local geometric heuristic (gap size) to a
    global structural heuristic learned from the problem instance's optimal
    structure, effectively teaching the ACO which items "belong" together.
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Leave 50s margin for ACO and overhead
    
    n = len(demands)
    demands = np.asarray(demands, dtype=float)
    
    # We will run multiple FFD simulations.
    # To make it "randomized" enough to capture different valid packings,
    # we can slightly perturb the sorting order for ties or just run standard FFD.
    # Standard FFD is deterministic. To get a distribution, we can:
    # 1. Run FFD on the original.
    # 2. Run FFD on reversed? No, FFD requires sorting.
    # 3. Better: Use a "Randomized FFD" where ties in size are broken randomly,
    #    or items with similar sizes are shuffled.
    #    However, for 120 items, running 1000-5000 simulations is fast.
    
    # Let's implement a fast loop for FFD simulations.
    # We need to count co-occurrences.
    # C is an n x n matrix.
    C = np.zeros((n, n), dtype=np.int32)
    
    # Pre-sort indices by demand descending.
    # In case of ties, we can randomize the order of indices with equal demand
    # in each simulation to generate different valid FFD packings.
    sorted_indices = np.argsort(-demands)
    
    # Group indices by demand value to allow random shuffling within groups
    # This is an optimization, but simple random permutation of ties is enough.
    # Actually, simple FFD is deterministic. 
    # To get a "prior", we can simulate the ACO's own behavior? No, that's circular.
    # Let's stick to Randomized FFD:
    # 1. Sort by size desc.
    # 2. For items with the same size, shuffle their relative order.
    
    num_sims = 2000  # Start with a reasonable number, increase if time permits
    
    # Check if we have time to do more
    # Each FFD run is O(N^2) roughly. 120^2 * 2000 = ~28M ops. Very fast in numpy/python loops.
    
    # To speed up, we can do this in a loop.
    # We need to track which bin each item goes into.
    
    for sim in range(num_sims):
        # Check time
        if time.time() > deadline:
            break
            
        # Create a shuffled version of indices where ties are broken randomly
        # Simple approach: sort by (-demand, random_noise)
        # This ensures items of different sizes are strictly ordered, 
        # but items of same size are random.
        noise = np.random.rand(n)
        # Sort by demand descending, then noise
        # We want stable sort on demand, but random on noise?
        # argsort is not stable by default in a way that mixes them easily.
        # Let's just use a tuple sort.
        sort_keys = np.lexsort((noise, -demands))
        
        # FFD Simulation
        bins = []  # List of lists of item indices
        bin_loads = [] # List of current loads
        
        for idx in sort_keys:
            d = demands[idx]
            placed = False
            for b in range(len(bins)):
                if bin_loads[b] + d <= capacity:
                    # Place item idx in bin b
                    bins[b].append(idx)
                    bin_loads[b] += d
                    placed = True
                    break
            
            if not placed:
                bins.append([idx])
                bin_loads.append(d)
        
        # Record co-occurrences
        for b in bins:
            if len(b) > 1:
                # All pairs in this bin co-occur
                for i in range(len(b)):
                    for j in range(i+1, len(b)):
                        u = b[i]
                        v = b[j]
                        C[u, v] += 1
                        C[v, u] += 1
                        
    # Transform C to heuristic eta
    # C contains counts. 
    # We want high eta for high C.
    # If C[i,j] == 0, eta should be very low (1e-9).
    # If C[i,j] > 0, eta should be scaled.
    
    # Normalization:
    # Let max_C = max(C).
    # eta[i,j] = (C[i,j] / max_C) ** alpha ?
    # Or eta[i,j] = 1 + C[i,j] ?
    
    # The ACO probability is proportional to pheromone * heuristic.
    # Pheromone starts at 1.
    # If heuristic is 1 for all valid pairs, it's uniform.
    # If heuristic is 1e-9 for invalid, it's effectively banned.
    
    # We want to strongly encourage pairs that FFD likes.
    # Let's use a power law to amplify the difference.
    max_c = np.max(C)
    if max_c == 0:
        # Fallback if no co-occurrences found (shouldn't happen)
        eta = np.full((n, n), 1e-9)
        valid = (demands[:, None] + demands[None, :]) <= capacity
        eta[valid] = 1.0
        np.fill_diagonal(eta, 1e-9)
        return eta
    
    # Base heuristic for all items
    # We only care about valid pairs (sum <= capacity).
    valid_mask = (demands[:, None] + demands[None, :]) <= capacity
    
    # Initialize eta with a very small value
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # For valid pairs, assign score based on C
    # Score = (C[i,j] + 1) ** 2 ? 
    # If C=0, score=1. If C=10, score=121.
    # This gives a baseline of 1 for valid pairs that FFD didn't pair,
    # and boosts those it did.
    
    # However, if C[i,j] is 0 for a valid pair, it means FFD *never* put them together.
    # This might be because they are incompatible (too big together) or just not optimal.
    # But if they fit, they *can* be together.
    # FFD is greedy. It might miss some good pairs.
    # But FFD is a good proxy for "good" packing structure.
    
    # Let's try: eta[i,j] = 1.0 if C[i,j] == 0 else (C[i,j] * 10.0)
    # This creates a strong preference for co-occurring pairs.
    
    scores = np.ones_like(C, dtype=float)
    non_zero = C > 0
    scores[non_zero] = C[non_zero] * 5.0 + 1.0  # Boost co-occurrences
    
    # Apply only to valid pairs
    eta[valid_mask] = scores[valid_mask]
    
    # Ensure diagonal is low
    np.fill_diagonal(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
