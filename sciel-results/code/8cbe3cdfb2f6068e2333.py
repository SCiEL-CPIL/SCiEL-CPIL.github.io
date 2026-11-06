<<<<<<< SEARCH
import time
import random
import sys

N = 9
MAX_V = 1 << N
MAX_TIME = 115.0  # Safety margin before 150s kill

# Precompute neighbors for all vertices
_NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(MAX_V)]

# Precompute neighbor bitmask for each vertex (bit v is set if v is a neighbor)
_NEIGHBOR_MASK = [0] * MAX_V
for v in range(MAX_V):
    mask = 0
    for b in range(N):
        mask |= 1 << (v ^ (1 << b))
    _NEIGHBOR_MASK[v] = mask
=======
import time
import random
import sys

N = 9
MAX_V = 1 << N
MAX_TIME = 115.0  # Safety margin before 150s kill

# Precompute neighbors for all vertices
_NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(MAX_V)]

# Precompute neighbor bitmask for each vertex (bit v is set if v is a neighbor)
_NEIGHBOR_MASK = [0] * MAX_V
for v in range(MAX_V):
    mask = 0
    for b in range(N):
        mask |= 1 << (v ^ (1 << b))
    _NEIGHBOR_MASK[v] = mask

# Precompute for each vertex, the set of vertices that are "forbidden" if we extend
# to that vertex (i.e., the vertex itself plus all its neighbors)
_EXTEND_FORBID = [0] * MAX_V
for v in range(MAX_V):
    _EXTEND_FORBID[v] = (1 << v) | _NEIGHBOR_MASK[v]
>>>>>>> REPLACE

