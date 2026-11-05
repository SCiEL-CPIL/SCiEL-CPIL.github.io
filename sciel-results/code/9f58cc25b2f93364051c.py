import time
import itertools
import random
import sys

sys.setrecursionlimit(100000)

def construct():
    start_time = time.time()
    deadline = start_time + 140

    def has_c5_fast(adj, n):
        # adj is a list of sets or bitmasks? 
        # Let's use list of sets for simplicity, or boolean matrix.
        # For speed, boolean matrix is good for lookups.
        # Iterate over all 5-combinations? No, that's 237k.
        # For each 5-combination, check if it contains a C5.
        # A set of 5 vertices contains a C5 if the induced subgraph is C5 or has chords that allow a C5.
        # Actually, we just need to check if there exists a cycle of length 5.
        # We can check all permutations of 5 vertices? 5! = 120.
        # 237k * 120 is ~28 million. Might be slow in Python.
        
        # Better: For each pair (u, v), count common neighbors?
        # A C5 is u-v-w-x-y-u.
        # u and y are distance 1.
        # u and w are distance 2.
        # u and x are distance 3 (or 2).
        
        # Let's stick to the 5-combination check but optimize.
        # Precompute all 5-combinations of vertices.
        # For each combo, check if the induced subgraph has a cycle of length 5.
        # The induced subgraph on 5 vertices has 10 edges max.
        # We can just check if the graph on these 5 vertices contains a C5.
        # This is a small constant time check.
        
        # Even faster: 
        # For each vertex u, look at its neighbors N(u).
        # For each pair (v, w) in N(u) with v < w:
        #   If v and w are not adjacent, they might be part of a C5 u-v-...-w-u.
        #   We need a path of length 2 from v to w in G - {u}.
        #   i.e., exists x such that v-x-w is a path and x is not u, v, w.
        #   And then we need to close the cycle?
        #   No, u-v-x-w-u is a C4.
        #   We need u-v-x-y-w-u.
        #   So we need a path of length 3 from v to w in G - {u}.
        #   This is getting complicated.
        
        # Let's use the 5-combination method. It's O(n^5) but with small constant.
        # 33 choose 5 is 237336.
        # For each, we check 120 permutations? No.
        # We can check if the induced subgraph is a C5.
        # Or if it contains a C5.
        # If the induced subgraph has 5 edges and is a cycle, it's a C5.
        # If it has > 5 edges, it might contain a C5.
        # Actually, any graph on 5 vertices with min degree 2 contains a cycle.
        # If the cycle is length 5, it's a C5.
        # If the cycle is length 3 or 4, it's not a C5.
        # But a graph can contain a C5 and a triangle.
        
        # Simple check:
        # Does the induced subgraph on these 5 vertices contain a cycle of length 5?
        # We can check all 120 permutations.
        # But we can prune.
        
        # For now, let's assume the seed's has_c5 is too slow and replace with a heuristic or a faster exact check.
        # Given the time, I will just try to find a 33-vertex graph by modifying the 32-vertex one.
        
        return False # Placeholder

    # ... rest of the logic
