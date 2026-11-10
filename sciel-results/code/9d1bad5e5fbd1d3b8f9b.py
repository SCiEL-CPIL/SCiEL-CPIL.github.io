"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Constructs a graph consisting of two disjoint copies of K_{3,3}.
    K_{3,3} is a 3-regular bipartite graph with girth 4.
    The simple cycles in K_{3,3} have lengths 4 and 6 only.
    Thus, the union of two disjoint K_{3,3} has distinct cycle lengths {4, 6}.
    Number of distinct cycle lengths = 2.
    
    This is optimal for cubic graphs on 12 vertices as connected alternatives
    (like the 3x4 grid or hexagonal prism) typically introduce 8-cycles or more,
    increasing the count of distinct lengths to 3 or more.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    edges = []
    
    # First K_{3,3} on vertices 0..5
    # Part A: 0, 1, 2
    # Part B: 3, 4, 5
    for u in [0, 1, 2]:
        for v in [3, 4, 5]:
            edges.append([u, v])
            
    # Second K_{3,3} on vertices 6..11
    # Part A: 6, 7, 8
    # Part B: 9, 10, 11
    for u in [6, 7, 8]:
        for v in [9, 10, 11]:
            edges.append([u, v])
            
    # Check deadline - this construction is O(1) so we'll never hit it,
    # but we include the guard per the directive
    if time.time() > deadline:
        # This should never happen for this O(1) construction
        pass
    
    # Ensure types are Python int
    return [[int(u), int(v)] for u, v in edges]
# EVOLVE-BLOCK-END
