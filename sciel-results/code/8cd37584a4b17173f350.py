"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    start_time = time.time()
    deadline = start_time + 105.0

    # Degree
    degree = adj.sum(axis=1).astype(np.float64)

    # Strategy: Simulate greedy selection to estimate each vertex's "independent-set contribution".
    # For each vertex, we estimate: if picked now, how many future candidates does it block?
    # Vertices that block fewer (low-degree, in sparse regions) should be picked first.
    # But also: vertices in dense clusters where many neighbors also have high degree
    # are "wasteful" to pick because their neighbors would also be blocked anyway.
    #
    # Key insight: simulate the greedy process and track for each vertex:
    #   - "exclusion_cost": number of still-eligible vertices it would exclude if picked
    #   - "inclusion_value": how many vertices could still be added to MIS after picking it
    #
    # We do a single greedy pass with dynamic scoring to estimate per-vertex value.

    # Compute: for each vertex, the sum of degrees of its neighbors (weighted exclusion cost)
    # If vertex v is picked, it excludes all its neighbors. The "cost" is higher if
    # neighbors have high degree (they'd have excluded many others too).
    # But the "benefit" of picking v is 1 (v itself is in the set).
    # So net benefit = 1 - (opportunity cost of excluded neighbors)
    
    # Opportunity cost of neighbor u: if u were picked instead, it would add 1 but exclude deg(u) others.
    # So picking v over u costs us: the chance that u (and its non-overlapping neighbors) would be in MIS.
    
    # Simpler proxy: 
    # score(v) = -degree(v) + alpha * (sum of degrees of neighbors of v) / degree(v)
    # This rewards vertices whose neighbors are high-degree (those neighbors are "expensive" to lose)
    # Wait, that's backwards. Let me think again.
    
    # If v has low degree: picking v is cheap (few exclusions).
    # If v's neighbors have HIGH degree: those neighbors would be expensive to pick, so losing them is not great.
    #   Actually this means: if neighbors are high-degree, picking v is relatively better because
    #   those neighbors would have caused more damage if picked.
    # If v's neighbors have LOW degree: those neighbors are cheap to pick, so losing them is bad.
    
    # So: prefer vertices where neighbors are high-degree (opportunity cost of exclusion is high)
    # AND where v itself has low degree (few exclusions).
    
    # Let's use: score = -degree(v) + beta * mean_degree_of_neighbors(v)
    # This says: low degree is good, but if your neighbors are high-degree, that offsets the penalty
    # because those neighbors would have been "wasteful" picks anyway.
    
    # Even better: simulate a few rounds of greedy to get dynamic estimates.
    
    # Dynamic simulation approach:
    # Run greedy with a simple rule (pick lowest degree first), track when each vertex
    # becomes "eligible" (no selected neighbor yet). The order of eligibility gives
    # a dynamic priority: vertices that become eligible early in a sparse region
    # are more valuable.
    
    # But we need a single static score. Let's use a multi-round simulation:
    # In each round, pick the vertex with lowest "effective degree" (degree among unblocked vertices).
    # Track for each vertex: at what "phase" it gets picked or blocked.
    # Vertices picked in early phases (when many candidates remain) are more valuable.
    
    # Actually, the simplest effective approach from the inspiration:
    # "Dynamic exclusion simulation captures interference patterns that static degree misses;
    #  vertices in low-density regions become eligible earlier and get higher priority"
    
    # Let's do: simulate greedy (lowest-degree-first) and assign score based on
    # the "round" at which each vertex is either selected or first blocked.
    # Earlier selection = higher score.
    
    eligible = np.ones(n, dtype=bool)
    selected = np.zeros(n, dtype=bool)
    round_index = np.full(n, -1.0, dtype=np.float64)
    
    # We'll simulate up to a limited number of steps for speed
    max_steps = min(n, 5000)  # cap for safety
    step = 0
    
    while step < max_steps and time.time() < deadline:
        # Find eligible vertices
        elig_idx = np.where(eligible)[0]
        if len(elig_idx) == 0:
            break
        
        # Among eligible, pick the one with lowest degree among still-eligible neighbors
        # Approximate: use original degree as proxy (recomputing each step is O(n^2))
        # For speed, use original degree
        best = elig_idx[np.argmin(degree[elig_idx])]
        
        # Select best
        selected[best] = True
        eligible[best] = False
        round_index[best] = float(step)
        
        # Block neighbors
        neighbors = np.where(adj[best])[0]
        for nb in neighbors:
            if eligible[nb]:
                eligible[nb] = False
                round_index[nb] = float(step)  # blocked at this step
        
        step += 1
        
        # Early exit if time is tight
        if step % 100 == 0 and time.time() > deadline - 15.0:
            break
    
    # For vertices never touched (shouldn't happen in connected graphs, but safety):
    unvisited = np.where(round_index < 0)[0]
    if len(unvisited) > 0:
        round_index[unvisited] = float(max_steps)
    
    # Normalize round_index: lower round = picked earlier = higher priority
    # Score should be HIGHER for vertices picked earlier
    # So score = -normalized_round_index
    
    rmin = round_index.min()
    rmax = round_index.max()
    if rmax > rmin:
        round_norm = (round_index - rmin) / (rmax - rmin)
    else:
        round_norm = np.zeros_like(round_index)
    
    # Combine with degree: among vertices at similar round, prefer lower degree
    dmin = degree.min()
    dmax = degree.max()
    if dmax > dmin:
        degree_norm = (degree - dmin) / (dmax - dmin)
    else:
        degree_norm = np.zeros_like(degree)
    
    # Primary signal: dynamic round (earlier is better)
    # Secondary: lower degree is better
    score = -1.0 * round_norm - 0.3 * degree_norm
    
    return score
# EVOLVE-BLOCK-END
