# EVOLVE-BLOCK-START
import itertools
import time

def construct():
    n = 16
    # The current design is the complete bipartite graph K_{8,8}.
    # This graph has 64 edges, diameter 2, and is edge-critical for diameter 2.
    # Adding any intra-part edge creates a triangle, making the opposite edge non-essential.
    # Thus, 64 is the maximum possible edge count for this family.
    
    # Deadline guard: since the construction is O(1), we check time only for safety.
    start_time = time.time()
    
    half = n // 2
    set_a = range(half)
    set_b = range(half, n)
    
    edges = [[int(u), int(v)] for u, v in itertools.product(set_a, set_b)]
    
    # Safety margin check (though unnecessary for O(1) work)
    if time.time() - start_time > 50:
        # This should never happen, but if it does, return what we have
        return edges
        
    return edges
# EVOLVE-BLOCK-END
