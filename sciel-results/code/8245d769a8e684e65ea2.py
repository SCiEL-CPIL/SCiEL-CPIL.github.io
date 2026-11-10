"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import math

def construct():
    """Construct a graph with no regular induced subgraph on 6+ vertices, maximizing n."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Target sizes to try, in descending order
    target_ns = [21, 20]
    
    best_n = 0
    best_edges = []
    
    # Precompute combinations for n=20 and n=21 to avoid overhead in the loop
    # We will generate them on the fly per n to save memory if n changes, 
    # but since we only try 21 and 20, we can precompute for max_n=21? 
    # Actually, itertools.combinations is fast enough if we just iterate.
    # But we need masks. Let's precompute masks for all 6-subsets for n=21.
    
    # Helper to check if a subset (represented as a bitmask and a list of vertices) is regular
    def is_regular_subset(subset_mask, subset_vertices, adj_bits):
        # Calculate degrees of vertices in the subset
        degs = []
        for v in subset_vertices:
            # Count neighbors of v that are also in the subset
            # adj_bits[v] is a bitmask of all neighbors
            # & subset_mask keeps only those in the subset
            # bin(x).count('1') is popcount
            degs.append(bin(adj_bits[v] & subset_mask).count('1'))
        
        # Check if all degrees are equal
        if not degs:
            return False
        d0 = degs[0]
        for d in degs[1:]:
            if d != d0:
                return False
        return True

    def count_bad_6_subsets(adj_bits, n):
        count = 0
        # Iterate over all 6-subsets
        for subset in itertools.combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_regular_subset(mask, subset, adj_bits):
                count += 1
        return count

    def check_validity(adj_bits, n):
        # Check all subsets of size >= 6
        # First check 6s
        for subset in itertools.combinations(range(n), 6):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_regular_subset(mask, subset, adj_bits):
                return False
        
        # Check 7s
        for subset in itertools.combinations(range(n), 7):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_regular_subset(mask, subset, adj_bits):
                return False
        
        # Check 8s
        for subset in itertools.combinations(range(n), 8):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_regular_subset(mask, subset, adj_bits):
                return False
        
        # Check 9s
        for subset in itertools.combinations(range(n), 9):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_regular_subset(mask, subset, adj_bits):
                return False
        
        # Check 10s
        for subset in itertools.combinations(range(n), 10):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            if is_regular_subset(mask, subset, adj_bits):
                return False
                
        # For n=20/21, sizes 11+ are less likely to be the bottleneck if 6-10 are clean, 
        # but we should check them if time allows. 
        # However, if a graph has no regular 6-subset, it's "sparse" in regularity.
        # Let's check up to n.
        for size in range(11, n + 1):
            for subset in itertools.combinations(range(n), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                if is_regular_subset(mask, subset, adj_bits):
                    return False
        return True

    for n in target_ns:
        if time.time() > deadline:
            break
            
        found = False
        
        # Simulated Annealing parameters
        # We want to minimize the number of bad 6-subsets.
        # Initial temperature high, cool down.
        
        # Try a few restarts with different initial random seeds/graphs
        for restart in range(5):
            if time.time() > deadline:
                break
            
            # Initialize random graph
            adj_bits = [0] * n
            # Random edges
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
                        
            current_energy = count_bad_6_subsets(adj_bits, n)
            
            if current_energy == 0:
                if check_validity(adj_bits, n):
                    # Extract edges
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj_bits[i] & (1 << j):
                                edges.append([i, j])
                    best_n = n
                    best_edges = edges
                    found = True
                    break
            
            # Annealing loop
            T = 1.0
            cooling = 0.995
            min_T = 0.01
            max_steps = 2000 # Limit steps per restart
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                    
                # Pick a random edge to flip
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                    
                # Flip edge (u, v)
                # Remove
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                
                # Calculate new energy
                # Optimization: Instead of recounting all 6-subsets, 
                # we could update only those affected by (u,v).
                # But for simplicity and robustness, let's recount if n is small?
                # No, recounting 38k subsets every step is too slow (2000 steps * 38k * ~10us = 760s).
                # We MUST use incremental update or a smarter heuristic.
                
                # Let's use a heuristic: 
                # The change in energy depends on 6-subsets containing u or v.
                # Number of such subsets: 
                # Subsets containing u: C(n-1, 5)
                # Subsets containing v: C(n-1, 5)
                # Subsets containing both: C(n-2, 4)
                # Total ~ 2*C(19,5) - C(18,4) ~ 2*11628 - 3060 ~ 20k.
                # Still too many to check every step if we do full popcount.
                
                # Alternative: Just check if the specific subsets containing u or v changed status.
                # This is complex.
                
                # Let's try a simpler local search first: 
                # Find ALL bad 6-subsets. Pick one. Flip an edge in it.
                # This is what the original code did, but with bitmask speed.
                
                # Let's switch back to local search but with bitmask speed.
                # 1. Find all bad 6-subsets.
                bad_subs = []
                for subset in itertools.combinations(range(n), 6):
                    mask = 0
                    for v in subset:
                        mask |= (1 << v)
                    if is_regular_subset(mask, subset, adj_bits):
                        bad_subs.append(subset)
                
                if not bad_subs:
                    # Check larger subsets
                    if check_validity(adj_bits, n):
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if adj_bits[i] & (1 << j):
                                    edges.append([i, j])
                        best_n = n
                        best_edges = edges
                        found = True
                        break
                    else:
                        # Bad larger subset found, but we don't know which.
                        # Randomly flip an edge to try to break it.
                        u = random.randint(0, n - 1)
                        v = random.randint(0, n - 1)
                        if u == v: continue
                        if u > v: u, v = v, u
                        adj_bits[u] ^= (1 << v)
                        adj_bits[v] ^= (1 << u)
                        continue
                
                # Pick a random bad subset
                subset = random.choice(bad_subs)
                # Pick a random edge in it to flip
                u, v = random.sample(list(subset), 2)
                
                # Flip
                adj_bits[u] ^= (1 << v)
                adj_bits[v] ^= (1 << u)
                
                # If we found a valid graph in the "not bad_subs" branch, we break.
                # If we flipped an edge, we continue to the next iteration of local search.
                
            if found:
                break
                
        if found:
            break
            
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
        
    # Fallback: Return a small valid graph
    # Empty graph on 5 vertices is valid (no subset of size 6)
    return {'n': 5, 'edges': []}

# EVOLVE-BLOCK-END
