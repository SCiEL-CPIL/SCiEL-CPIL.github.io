import time
import sys
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for safety
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Build the conflict graph using bitsets
    # Vertex v (1..256) maps to bit (v-1)
    # conflicts_mask[v] has bit j set if vertex (j+1) conflicts with vertex v
    conflicts_mask = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflicts_mask[i] |= (1 << (j - 1))
                conflicts_mask[j] |= (1 << (i - 1))
    
    def greedy_randomized(remaining_mask, rng):
        """
        Randomized greedy MIS using bitsets.
        remaining_mask: bitmask of available vertices (bit v-1 set means vertex v is available)
        Returns a list of vertices in the independent set.
        """
        solution = []
        while remaining_mask:
            # Find vertex with minimum degree in remaining subgraph
            # For speed, sample a subset or use a fast heuristic
            # Get list of available vertices
            available = []
            m = remaining_mask
            while m:
                bit = (m & -m).bit_length() - 1
                available.append(bit + 1)
                m &= m - 1
            
            if not available:
                break
            
            # Compute degree of each available vertex in the remaining subgraph
            best_v = None
            best_deg = float('inf')
            candidates = []
            
            for v in available:
                # Count neighbors of v that are in remaining_mask
                deg = bin(conflicts_mask[v] & remaining_mask).count('1')
                if deg < best_deg:
                    best_deg = deg
                    candidates = [v]
                elif deg == best_deg:
                    candidates.append(v)
            
            # Random choice among min-degree candidates
            v = rng.choice(candidates)
            solution.append(v)
            # Remove v and its neighbors from remaining
            remaining_mask &= ~(1 << (v - 1))
            remaining_mask &= ~conflicts_mask[v]
        
        return solution
    
    # Initial deterministic greedy
    remaining_mask = (1 << N) - 2  # bits 1..256 set (vertex 1 is bit 0, but we skip vertex 0)
    # Actually, vertices are 1..256, so bits 0..255
    remaining_mask = (1 << N) - 1  # bits 0..255 set
    
    # First pass: deterministic (always pick first min-degree)
    best_sol = greedy_randomized(remaining_mask, random.Random(42))
    best_size = len(best_sol)
    best_sol_set = set(best_sol)
    
    # Randomized restarts
    rng = random.Random()
    restart_count = 0
    while True:
        if time.time() > deadline:
            break
        
        remaining_mask = (1 << N) - 1
        sol = greedy_randomized(remaining_mask, rng)
        restart_count += 1
        
        if len(sol) > best_size:
            best_size = len(sol)
            best_sol = sol
            best_sol_set = set(sol)
        
        # Try to improve via 1-swap local search on the best solution
        if len(sol) == best_size and best_size > 0:
            # Try swapping each vertex in best_sol with a vertex outside
            # This is expensive, so only do it occasionally
            if restart_count % 10 == 0 and time.time() < deadline - 5:
                improved = False
                for idx, v in enumerate(best_sol):
                    if time.time() > deadline - 2:
                        break
                    # Remove v from solution
                    temp_set = set(best_sol)
                    temp_set.remove(v)
                    # Find a vertex not in temp_set that doesn't conflict with temp_set
                    for u in range(1, N + 1):
                        if u in temp_set:
                            continue
                        # Check if u conflicts with any in temp_set
                        conflict = False
                        for w in temp_set:
                            if conflicts_mask[u] & (1 << (w - 1)):
                                conflict = True
                                break
                        if not conflict:
                            # We can swap v with u, but size stays the same
                            # Only useful if we can then add another vertex
                            # Skip for now - just track best
                            break
    
    return best_sol
# EVOLVE-BLOCK-END
