import sys
from itertools import combinations
import networkx as nx # Wait, networkx is not in standard lib? 
# "Python standard library, NumPy and SciPy are available."
# NetworkX is NOT available.
# I must implement max flow or use scipy.sparse.csgraph?
# scipy.sparse.csgraph has max_flow? No, it has shortest_path, etc.
# I need to implement max flow or use a simple algorithm.
# For small graphs, I can implement Edmonds-Karp or use a library if available.
# Actually, I can just use a simple backtracking for the disjoint paths since k is small (6 or 7).
# Or I can use the fact that for k=7, the graph is small.

# Let's stick to the standard library.
# I will implement a simple max flow (Edmonds-Karp) on a vertex-split graph.
