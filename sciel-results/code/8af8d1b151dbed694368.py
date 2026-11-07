"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 40.0  # Safety margin for 90s limit
    
    # 1. Generate 14 distinct positive integers <= 10000.
    # We use a Sidon set (powers of 2) to ensure that any 35 edges have exactly 35 distinct sums.
    # This sets a lower bound on the "distinct sums" part of the max objective to 35.
    # The objective is max(distinct_sums, distinct_products).
    # Since distinct_sums >= 35, the objective is at least 35.
    # If we can find a set of 35 edges such that distinct_products <= 35, the score is 35.
    # If distinct_products > 35, the score is distinct_products.
    
    values = [2**i for i in range(14)]
    
    # Verify constraints
    assert len(values) == 14
    assert len(set(values)) == 14
    assert all(isinstance(v, int) for v in values)
    assert all(1 <= v <= 10000 for v in values)
    
    # 2. Select 35 edges to minimize distinct products.
    # In a Sidon set of powers of 2, v[u]*v[v] = 2^(u+v).
    # The product is determined by the sum of the indices u+v.
    # Indices are 0..13.
    # Possible index sums range from 0+1=1 to 12+13=25.
    # There are 25 distinct index sums.
    # Therefore, there are at most 25 distinct products.
    # Since 25 < 35, the number of distinct products will be at most 25.
    # The number of distinct sums is exactly 35 (because it's a Sidon set).
    # So max(distinct_sums, distinct_products) = max(35, <=25) = 35.
    
    # Can we do better?
    # If we use a non-Sidon set, distinct sums might be > 35.
    # If we use a set where products collide more, maybe distinct products < 25?
    # But distinct sums would likely increase.
    # The theoretical minimum for distinct sums is 35 (one per edge).
    # So the score cannot be less than 35.
    # Thus, 35 is the optimal score.
    
    # We just need to pick any 35 edges. The specific choice doesn't change the distinct sum count (always 35).
    # It might change the distinct product count, but since max possible distinct products for this value set is 25,
    # and 25 < 35, the max will always be 35 regardless of which 35 edges we pick (as long as we pick 35 edges).
    # Wait, does the set of 35 edges matter for the product count?
    # Yes, if we pick edges that cover all 25 index sums, distinct products = 25.
    # If we pick edges that cover only 10 index sums, distinct products = 10.
    # In either case, max(35, k) = 35.
    # So any 35 edges work.
    
    # Let's just pick the first 35 combinations.
    edges = [list(e) for e in list(combinations(range(14), 2))[:35]]
    
    # Verify edges
    assert len(edges) == 35
    assert all(isinstance(e, list) for e in edges)
    assert all(len(e) == 2 for e in edges)
    assert all(isinstance(i, int) for e in edges for i in e)
    assert all(0 <= i < 14 for e in edges for i in e)
    # Check no duplicates
    edge_tuples = set(tuple(sorted(e)) for e in edges)
    assert len(edge_tuples) == 35
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
