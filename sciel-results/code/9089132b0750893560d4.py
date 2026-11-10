import numpy as np

def ring_routing() -> np.ndarray:
    # Node count
    N = 10
    
    # Demands: (source, target, weight)
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2)
    ]
    
    D = len(demands)
    
    # Pre-compute clockwise and counter-clockwise edge usage matrices
    # A_cw[i, e] = 1 if demand i uses edge e clockwise
    # A_ccw[i, e] = 1 if demand i uses edge e counter-clockwise
    
    A_cw = np.zeros((D, N), dtype=np.float64)
    A_ccw = np.zeros((D, N), dtype=np.float64)
    weights = np.zeros(D, dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        # Clockwise path: s, s+1, ..., t-1 (mod N)
        # Edges are (0,1), (1,2), ..., (N-1, 0)
        # Edge e connects node e and (e+1)%N
        # If s == t, path is empty? No, demands usually s != t.
        # If s < t, clockwise edges are s, s+1, ..., t-1
        # If s > t, clockwise edges are s, s+1, ..., N-1, 0, ..., t-1
        # Counter-clockwise is the complement.
        
        # Determine clockwise edges
        cw_edges = []
        if s == t:
            # This case is not in the provided demands, but handle just in case
            # Usually a loop demand might use all edges or 0 edges depending on definition.
            # Given the problem statement "Clockwise from s to t uses edges s, ..., t-1",
            # if s==t, the set is empty? Or all? 
            # Standard ring routing: s!=t.
            pass
        else:
            current = s
            while current != t:
                cw_edges.append(current)
                current = (current + 1) % N
        
        for e in cw_edges:
            A_cw[i, e] = 1.0
        
        # Counter-clockwise edges are all edges NOT in cw_edges
        # We can compute A_ccw as 1 - A_cw for the specific edges involved?
        # No, A_cw has 0s for edges not used.
        # But wait, does a demand use ALL other edges?
        # "counter-clockwise uses all other ring edges"
        # So yes, A_ccw[i, e] = 1 if A_cw[i, e] == 0.
        # However, we must be careful. The union of CW and CCW paths is all edges.
        A_ccw[i, :] = 1.0 - A_cw[i, :]
        
        # Sanity check: sum of edges used should be consistent
        # For a simple path, sum(A_cw) + sum(A_ccw) = N
        
    # Weights vector W
    W = weights
    
    # Load contribution if all routed CW: L_cw = W @ A_cw
    # L_cw is a vector of length N
    L_cw = W @ A_cw
    
    # Difference matrix D = A_ccw - A_cw
    # If we flip demand i from CW to CCW, the change in load is W[i] * (A_ccw[i] - A_cw[i])
    # Let r be a binary vector of length D, where r[i]=1 means CCW, r[i]=0 means CW.
    # Load = L_cw + sum_{i} r[i] * W[i] * (A_ccw[i] - A_cw[i])
    # Let M[i, e] = W[i] * (A_ccw[i, e] - A_cw[i, e])
    # Then Load = L_cw + r @ M
    
    M = (W[:, np.newaxis]) * (A_ccw - A_cw)
    
    # Generate all 2^D routing combinations
    # D=15, so 32768 combinations
    # We can use np.indices or bit manipulation
    # Efficient way:
    # Create an array of all integers from 0 to 2^D - 1
    # Convert to binary
    
    # Method 1: np.arange and bit unpacking
    num_combos = 1 << D
    # This creates an array of shape (num_combos, D)
    # We want row i to be the binary representation of i
    
    # Using np.unpackbits is tricky with endianness and padding.
    # Let's use a simpler loop or vectorized bit ops.
    
    # Vectorized bit extraction:
    # Powers of 2
    powers = 1 << np.arange(D)
    # All integers
    ints = np.arange(num_combos)
    # Broadcasting: (num_combos, 1) // (1, D) % 2
    # This works because we are looking at each bit position
    AllR = (ints[:, np.newaxis] // powers) % 2
    AllR = AllR.astype(np.float64)
    
    # Compute all loads: (num_combos, N)
    # Load = L_cw[None, :] + AllR @ M
    AllLoads = L_cw[None, :] + AllR @ M
    
    # Find max load for each combination
    MaxLoads = np.max(AllLoads, axis=1)
    
    # Find the combination with the minimum max load
    best_idx = np.argmin(MaxLoads)
    best_routing = AllR[best_idx]
    
    # The task requires an array of 15 numbers.
    # Entry >= 0.5 routes CCW, else CW.
    # best_routing contains 0s and 1s.
    # Return it as is.
    
    return best_routing

# EVOLVE-BLOCK-END
