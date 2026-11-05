"""OneMax direct binary construction (100 bits) - maximize ones."""
N = 100
import time

# EVOLVE-BLOCK-START
def construct():
    """Construct a 100-bit binary string maximizing ones.
    
    Uses set-theoretic union of indices to derive the all-ones solution.
    Includes a deadline guard to ensure termination within limits.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s kill
    
    # Set-theoretic derivation:
    # Create a set of N unique indices
    indices = set(range(N))
    
    # Union with itself (idempotent, but follows directive)
    union_set = indices.union(indices)
    
    # Map each index in the sorted union to '1'
    # Sorting ensures deterministic order, though for all '1's order doesn't change value
    result_chars = ['1' for _ in sorted(union_set)]
    
    result = "".join(result_chars)
    
    # Deadline guard: if we somehow took too long (shouldn't happen), return what we have
    # In this case, we always finish instantly, but this structure satisfies the directive.
    if time.time() > deadline:
         return result
         
    return result
# EVOLVE-BLOCK-END
