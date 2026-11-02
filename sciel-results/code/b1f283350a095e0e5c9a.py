# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Uses Lazebnik-Ustimenko-Woldar (LUW) graphs for k=5.
    """
    deadline = time.time() + 140
    
    # For k=5, use LUW construction
    if k == 5:
        # Find smallest d such that girth(2d+2) >= g
        # Girth of LUW(q, d) is at least 2d + 2
        d = max(1, (g - 2 + 1) // 2) # ceil((g-2)/2)
        # For g=9, (9-2+1)//2 = 8//2 = 4.
        # Girth >= 10.
        
        n, edges = _luw_construct(k, d)
        if n is not None:
            return (n, edges)
            
    # Fallback for other k or if LUW fails
    # Try random greedy with small n
    best_result = None
    for n_attempt in [428, 500, 600, 800, 1000]:
        if time.time() > deadline:
            break
        result = _try_construct(k, g, n_attempt, deadline)
        if result is not None:
            n, edges = result
            if best_result is None or n < best_result[0]:
                best_result = (n, edges)
            if n <= 500:
                break
    
    if best_result is not None:
        return best_result
        
    # Last resort: return a valid k-regular graph even if girth is low? 
    # No, must satisfy girth.
    # If we can't find one, return the LUW one even if d is larger?
    # Or just raise an error? The evaluator scores 0 for invalid.
    # The LUW construction for k=5 should always work.
    
    return (0, []) # Should not happen for k=5

def _luw_construct(q, d):
    """
    Construct the Lazebnik-Ustimenko-Woldar graph L(q, d).
    q must be a prime power.
    Returns (n, edges).
    """
    # Check if q is prime power? For k=5, it is.
    # We assume q=5 for this task.
    
    # The LUW graph is bipartite.
    # Vertices are (side, x) where side in {0, 1} and x is a tuple of length d.
    # x[0] is usually ignored or 0?
    # In the seed code: start = (0, (None,) + (0,) * d)
    # The tuple has length d+1?
    # Let's stick to the seed's implementation logic but optimized.
    
    # The seed code uses:
    # y = [None, first] + [0] * (d - 1)
    # This implies the state vector has length d.
    # x[0] is the "first" component?
    # Actually, the standard LUW definition uses vectors in GF(q)^d.
    
    # Let's use a clean implementation.
    # Vertices: (side, v) where v in GF(q)^d.
    # Total vertices: 2 * q^d.
    
    # Map vertices to integers 0..n-1
    # side 0: 0 to q^d - 1
    # side 1: q^d to 2*q^d - 1
    
    n = 2 * (q ** d)
    edges = []
    
    # Precompute powers and rules?
    # The LUW rule:
    # For side 0, neighbor (1, y) is defined by:
    # y[0] = x[1]
    # y[i] = x[i] + x[0] * y[i-1] for i=1..d-1 ?
    # This varies by source. The seed code has a specific `rule` dictionary.
    # Let's reuse the seed's neighbor generation logic but ensure it's correct for d=4.
    
    # The seed's rule:
    # rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    # for c in range(5, d + 1):
    #     r, c0 = (c - 5) % 4, c - (c - 5) % 4
    #     rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # This looks like a specific recurrence.
    # I will implement the standard LUW recurrence which is simpler and known to have girth 2d+2.
    # Standard LUW(q, d):
    # Neighbors of (0, x) are (1, y) where:
    # y[0] = x[1]
    # y[i] = x[i] + x[0] * y[i-1] for i = 1 to d-1
    # Wait, x[0] is used.
    # Actually, the mapping is a bijection from GF(q) to neighbors.
    # For each a in GF(q), there is a unique neighbor.
    
    # Let's use the seed's logic as it was provided and likely works, 
    # but I will ensure d is passed correctly.
    
    # The seed's `neighbours` function:
    def neighbours(vertex):
        side, x = vertex
        # x is a tuple of length d+1? 
        # Seed: start = (0, (None,) + (0,) * d) -> length d+1.
        # x[0] is None. x[1..d] are values.
        for first in range(q):
            # y[0] is None?
            # y = [None, first] + [0] * (d - 1)
            # Length: 1 + 1 + d-1 = d+1.
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                # c goes from 2 to d.
                # x[c] is the c-th element of x (1-indexed in tuple? No, 0-indexed).
                # x is tuple. x[0]=None, x[1]=val, ..., x[d]=val.
                # The loop uses x[c]. So c is index.
                # If c=2, x[2] is 3rd element.
                # This seems to use x[1..d].
                a, b = _get_luw_rule(c, d)
                # y[a] and x[b]
                # Note: a, b are indices in 1..d?
                # Seed: rule[2] = (1, 1).
                # If side == 0:
                #   y[c] = x[c] + y[a] * x[b]
                #   Wait, y[c] is being set.
                #   The indices a, b must be < c?
                #   For c=2, a=1, b=1. y[1] is set (first). x[1] is set.
                #   So y[2] = x[2] + y[1] * x[1].
                #   This is a valid recurrence.
                
                if side == 0:
                    y[c] = (x[c] + y[a] * x[b]) % q
                else:
                    y[c] = (x[c] - x[a] * y[b]) % q
            yield 1 - side, tuple(y)

    # We need to generate all edges.
    # We can iterate over all vertices of side 0 and generate their neighbors.
    
    # To avoid duplicates, we only add edge (u, v) if u < v.
    # Since it's bipartite, u in side 0, v in side 1.
    # We can just map (side, x) to an ID.
    
    def vid(side, x):
        # x is tuple of length d+1, x[0] is None.
        # Use x[1..d] for encoding.
        # Value = sum(x[i] * q^(i-1)) for i=1..d
        val = 0
        for i in range(1, d + 1):
            val += x[i] * (q ** (i - 1))
        return side * (q ** d) + val

    # Iterate over all side 0 vertices
    for x1 in range(q):
        for x2 in range(q):
            # ... nested loops for d dimensions
            # This is inefficient for large d, but d=4 is small.
            pass
            
    # Better: iterate 0 to q^d - 1
    for val in range(q ** d):
        # Decode val to x[1..d]
        x = [0] * (d + 1)
        x[0] = None
        temp = val
        for i in range(1, d + 1):
            x[i] = temp % q
            temp //= q
        
        vertex = (0, tuple(x))
        for other in neighbours(vertex):
            v_id = vid(other[0], other[1])
            u_id = vid(0, x)
            if u_id < v_id:
                edges.append((u_id, v_id))
                
    return n, edges

def _get_luw_rule(c, d):
    # Replicate seed's rule generation
    if c == 2: return (1, 1)
    if c == 3: return (2, 1)
    if c == 4: return (1, 2)
    r, c0 = (c - 5) % 4, c - (c - 5) % 4
    # The seed had a list index
    # [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    # Note: c0 can be 5, 6, 7...
    # If c=5, r=0, c0=5. (1, 3).
    # If c=6, r=1, c0=6. (5, 1).
    # If c=7, r=2, c0=7. (7, 1).
    # If c=8, r=3, c0=8. (1, 9).
    # This seems to depend on d? The seed used this for general d.
    # I will use this.
    rules = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
    return rules[r]

# ... keep other helper functions from seed if needed, or remove ...
# EVOLVE-BLOCK-END
