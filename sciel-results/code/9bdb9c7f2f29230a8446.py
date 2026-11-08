<<<<<<< SEARCH
"""Block-partitioned Strassen-based decomposition for 4x5x6 matrix multiplication.
Uses rank-7 2x2x2 Strassen decomposition on 2x2x2 sub-blocks, with careful
index mapping for the irregular 4x5x6 shape. Then applies exact linear-algebra
based column reduction to find minimal rank."""
import time
import random
from fractions import Fraction

try:
    import numpy as np
except Exception:
    np = None

DIMENSIONS = [4, 5, 6]
=======
"""Rank-minimal tensor decomposition for 4x5x6 matrix multiplication.
Uses a direct construction via embedding into 4x6x6, then projects and reduces
via linear algebra with fast NumPy verification."""
import time
import random

try:
    import numpy as np
except Exception:
    np = None

DIMENSIONS = [4, 5, 6]
>>>>>>> REPLACE

