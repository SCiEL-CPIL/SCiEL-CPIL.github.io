"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def _score_bitset(adj_bits, n=16):
    """
    Compute max triangle degree sum using bitsets for speed.
    adj_bits: list of integers, bit j set if edge (i,j) exists.
    """
    deg = [bin(b).count('1') for b in adj_bits]
    max_sum = 0
    for i in range(n):
        for j in range(i + 1, n):
            if (adj_bits[i] >> j) & 1:
                # Common neighbors
                common = adj_bits[i] & adj_bits[j]
                # Only consider k > j to avoid duplicates
                common &= ~((1 << (j + 1)) - 1)
                while common:
                    # Get lowest set bit
                    lsb = common & -common
                    k = lsb.bit_length() - 1
                    s = deg[i] + deg[j] + deg[k]
                    if s > max_sum:
                        max_sum = s
                    common ^= lsb
    return max_sum

def _local_search(n=16, target_edges=65, time_limit=100, seed=None):
    """
    Fast local search using bitsets.
    Moves: Remove one edge, add one non-edge.
    """
    if seed is not None:
        random.seed(seed)
    
    start_time = time.time()
    
    # Initialize: K_{8,8} + 1 random internal edge
    part_a = list(range(8))
    part_b = list(range(8, 16))
    random.shuffle(part_a)
    random.shuffle(part_b)
    
    adj_bits = [0] * n
    for u in part_a:
        for v in part_b:
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
    
    # Add one internal edge in A
    u1, u2 = part_a[0], part_a[1]
    adj_bits[u1] |= (1 << u2)
    adj_bits[u2] |= (1 << u1)
    
    current_score = _score_bitset(adj_bits, n)
    best_score = current_score
    best_adj_bits = adj_bits[:]
    
    # Precompute list of edges and non-edges for faster sampling?
    # For n=16, total pairs = 120.
    # We can just iterate.
    
    # Temperature
    T = 5.0
    cooling = 0.999
    
    # Counters
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() - start_time < time_limit:
        if no_improve > max_no_improve:
            # Restart with slight perturbation or break
            # Let's just continue, maybe lower T
            T = max(0.1, T * 0.5)
            no_improve = 0
            max_no_improve = 5000
            
        # Pick a random edge to remove
        # Generate random pair
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
        
        if not ((adj_bits[u] >> v) & 1):
            continue # Not an edge
        
        # Pick a random non-edge to add
        found = False
        for _ in range(20):
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            if (a, b) == (u, v):
                continue
            if (adj_bits[a] >> b) & 1:
                continue # Already an edge
            
            # Perform swap
            adj_bits[u] ^= (1 << v)
            adj_bits[v] ^= (1 << u)
            adj_bits[a] ^= (1 << b)
            adj_bits[b] ^= (1 << a)
            
            new_score = _score_bitset(adj_bits, n)
            
            if new_score < current_score or random.random() < math.exp(-(new_score - current_score) / T):
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_adj_bits = adj_bits[:]
                    no_improve = 0
                else:
                    no_improve += 1
                found = True
                break
            else:
                # Revert
                adj_bits[u] ^= (1 << v)
                adj_bits[v] ^= (1 << u)
                adj_bits[a] ^= (1 << b)
                adj_bits[b] ^= (1 << a)
        
        T = max(0.1, T * cooling)
        
    # Convert best_adj_bits back to edge list
    best_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj_bits[i] >> j) & 1:
                best_edges.append([i, j])
                
    return best_edges, best_score

def construct():
    """
    Construct a graph on 16 vertices with 65 edges minimizing the max triangle degree sum.
    """
    n = 16
    target_edges = 65
    
    best_score = float('inf')
    best_edges = None
    
    # Run multiple local searches with different seeds
    # Time limit per run: 10s. Total 6 runs = 60s.
    # We have 120s limit.
    for seed in range(10):
        try:
            edges_opt, score_opt = _local_search(n, target_edges, time_limit=10, seed=seed)
            if score_opt < best_score:
                best_score = score_opt
                best_edges = edges_opt
        except Exception:
            continue
    
    # Fallback if all failed (should not happen)
    if best_edges is None:
        # K_{8,8} + (0,1)
        best_edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        best_score = _score_bitset(
            [sum(1 for v in range(8,16) if u==v or True) for u in range(8)] + [0]*8 # Dummy, just to be safe
            # Actually, let's just compute score properly for fallback
        )
        # Recalculate fallback score properly
        adj_bits_fallback = [0]*16
        for u in range(8):
            for v in range(8, 16):
                adj_bits_fallback[u] |= (1 << v)
                adj_bits_fallback[v] |= (1 << u)
        adj_bits_fallback[0] |= (1 << 1)
        adj_bits_fallback[1] |= (1 << 0)
        best_score = _score_bitset(adj_bits_fallback, 16)
        
    return {'n': n, 'edges': best_edges}
# EVOLVE-BLOCK-END
