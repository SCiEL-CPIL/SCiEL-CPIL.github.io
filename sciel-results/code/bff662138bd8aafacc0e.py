<<<<<<< SEARCH
"""
Circle packing in unit square using Delaunay-based tight packing.
Centers are optimized; radii are derived from the local geometry
(half the minimum distance to any other center or to a boundary).
This enforces tight packing and reduces the effective search space.
"""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 115.0  # Safety margin from 120s limit (evaluator kills at 150s, but keep margin)
=======
"""
Circle packing in unit square using force-directed repulsion with
independent radius variables and Delaunay-based local refinement.
Structural change: radii are independent decision variables with
explicit non-overlap constraints, not derived from center geometry.
"""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 110.0  # Safety margin from 120s limit
>>>>>>> REPLACE

