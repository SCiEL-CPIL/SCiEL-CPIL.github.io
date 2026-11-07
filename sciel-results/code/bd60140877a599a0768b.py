"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

N = P['n']
MIN_V = P['minimum_vertices']
edges = P['edges']

# Precompute adjacency bitmask for each vertex
adj = [0] * N
for u, v in edges:
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

# Precompute full-degree (degree in the whole graph) for each vertex
full_deg = [bin(adj[v]).count('1') for v in range(N)]

# Precompute popcount for all 2^17 masks
popcount = [0] * (1 << N)
for m in range(1, 1 << N):
    popcount[m] = popcount[m >> 1] + (m & 1)

# Precompute degree-in-subset for each vertex and each subset mask
# deg_in_subset[v][mask] = number of neighbors of v that are in mask
deg_in_subset = [[0] * (1 << N) for _ in range(N)]
for v in range(N):
    a = adj[v]
    # For each mask, count bits in (a & mask)
    # Efficient: iterate masks
    for mask in range(1 << N):
        deg_in_subset[v][mask] = popcount[a & mask]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin under 90s
    
    # Best solution tracking
    best_count = 0
    best_mask = 0
    best_vertices = None
    
    # Upper bound: max possible distinct degrees is min(popcount(mask), N)
    # But more tightly: distinct degrees <= number of vertices in subset
    
    # Branch and bound over all subsets with popcount >= MIN_V
    # Use recursive approach: build subset bit by bit
    
    # Actually, with 2^17 = 131072 masks, brute force with precomputation is fast.
    # Let's just iterate all masks with popcount >= 8.
    
    # Precompute all valid masks (popcount >= 8)
    valid_masks = []
    for mask in range(1 << N):
        if popcount[mask] >= MIN_V:
            valid_masks.append(mask)
    
    # For each valid mask, compute distinct degrees
    # deg_in_subset[v][mask_without_v] gives degree of v within mask (since v's neighbors in mask are neighbors in mask excluding v, but we need neighbors IN mask, and v is in mask)
    # Actually deg_in_subset[v][mask] counts neighbors of v that are in mask. Since v is not its own neighbor, this is exactly the degree of v in the induced subgraph on mask (when v is in mask).
    
    for mask in valid_masks:
        # Time check
        if time.time() > deadline:
            break
        
        # Compute distinct degrees
        # For each vertex v in mask, degree = deg_in_subset[v][mask]
        # We need the set of these degrees
        degs = set()
        m = mask
        while m:
            v = (m & -m).bit_length() - 1
            degs.add(deg_in_subset[v][mask])
            m &= m - 1
        
        cnt = len(degs)
        if cnt > best_count:
            best_count = cnt
            best_mask = mask
            # Extract vertices
            verts = []
            m = mask
            while m:
                v = (m & -m).bit_length() - 1
                verts.append(v)
                m &= m - 1
            best_vertices = verts
    
    if best_vertices is None:
        # Fallback: first MIN_V vertices
        best_vertices = list(range(MIN_V))
    
    # Ensure types are correct: list of Python ints
    return [int(v) for v in best_vertices]

# EVOLVE-BLOCK-END
