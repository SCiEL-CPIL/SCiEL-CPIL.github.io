"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from math import gcd

def construct():
    """
    Find a large subset of [1, 512] such that lcm(a, b) <= 512 for every pair (a, b).
    
    Approach:
    1. Model the problem as finding a Maximum Independent Set (MIS) in a conflict graph.
       - Nodes: integers 1 to 512.
       - Edge between a, b if lcm(a, b) > 512.
    2. Use a randomized greedy heuristic to find a large independent set.
       - Run multiple iterations with random node orderings.
       - In each iteration, greedily add nodes to the set if they don't conflict with already added nodes.
       - Keep the best result found.
    3. Also try a deterministic min-degree greedy as a baseline.
    """
    N = 512
    
    # Build conflict graph
    # conflicts[i] is a set of indices j such that lcm(i+1, j+1) > 512
    conflicts = [set() for _ in range(N)]
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g = gcd(a, b)
            lcm = a * b // g
            if lcm > N:
                conflicts[a-1].add(b-1)
                conflicts[b-1].add(a-1)
    
    def greedy_independent_set(order):
        """
        Greedy MIS: iterate through nodes in 'order'.
        If a node is not yet removed (not in IS and not conflicting with IS), add it.
        """
        in_is = [False] * N
        iset = []
        # To speed up removal, we can maintain a 'blocked' set or just check conflicts against IS.
        # Checking conflicts against IS is O(deg) per check.
        # Alternative: maintain a set of 'available' nodes.
        available = set(range(N))
        
        for node in order:
            if node in available:
                iset.append(node)
                # Remove node and all its neighbors from available
                # neighbors are in conflicts[node]
                # We need to remove from 'available'
                to_remove = {node}
                to_remove.update(conflicts[node])
                available -= to_remove
        
        return iset

    # Deterministic Min-Degree Greedy (from seed, optimized slightly)
    def min_degree_greedy():
        remaining = set(range(N))
        iset = []
        # Pre-compute degrees? No, degrees change.
        while remaining:
            min_deg = float('inf')
            best_node = -1
            for node in remaining:
                deg = len(conflicts[node] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_node = node
                    if deg == 0:
                        break
            iset.append(best_node)
            remaining.discard(best_node)
            # Remove neighbors
            for neighbor in conflicts[best_node]:
                if neighbor in remaining:
                    remaining.discard(neighbor)
        return iset

    best_iset = min_degree_greedy()
    best_size = len(best_iset)
    
    # Randomized Greedy
    # Use a fixed seed for reproducibility if needed, but random is fine
    random.seed(42)
    num_iterations = 1000
    
    for _ in range(num_iterations):
        order = list(range(N))
        random.shuffle(order)
        iset = greedy_independent_set(order)
        if len(iset) > best_size:
            best_size = len(iset)
            best_iset = iset
    
    # Convert indices back to values (1-based)
    return [x + 1 for x in best_iset]

# EVOLVE-BLOCK-END
