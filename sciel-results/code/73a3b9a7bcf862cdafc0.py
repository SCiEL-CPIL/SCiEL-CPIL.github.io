"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a 12-state binary synchronizing automaton using an anti-Pawlik dual topology.
    
    Strategy: Invert the algebraic structure of the extremal reset word.
    Instead of a simple cycle + contraction, use a layered structure where:
    - Letter 'a' performs a shift that creates expanding patterns
    - Letter 'b' performs a contraction that must be applied in a specific sequence
    
    The dual topology forces the BFS to traverse a distinct path in the subset lattice,
    potentially yielding a longer shortest reset word than the standard Pawlik construction.
    """
    # Anti-Pawlik construction: dual topology
    # Letter 'a': A permutation that creates a specific expansion pattern
    # Use a "twisted" cycle that interacts differently with 'b'
    trans_a = [0] * N
    for i in range(N):
        # Create a shift with a twist to avoid simple cycle behavior
        trans_a[i] = (i + 1) % N if i < N - 1 else 0
    
    # Letter 'b': A contraction that maps states in a layered fashion
    # This creates a "staircase" effect where reset requires specific ordering
    trans_b = [0] * N
    # Map states to create a binary-tree-like contraction
    for i in range(N):
        if i == 0:
            trans_b[i] = 0
        elif i % 2 == 0:
            trans_b[i] = i // 2
        else:
            trans_b[i] = i // 2
    
    return [trans_a, trans_b]
# EVOLVE-BLOCK-END
