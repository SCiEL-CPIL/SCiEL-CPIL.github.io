<<<<<<< SEARCH
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2.0])

# Pre-compute triangle indices for efficiency
tri_indices = list(combinations(range(N), 3))
tri_indices_arr = np.array(tri_indices)
=======
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2.0])

# Pre-compute triangle indices
_tri_i, _tri_j, _tri_k = [], [], []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            _tri_i.append(i)
            _tri_j.append(j)
            _tri_k.append(k)
_tri_i = np.array(_tri_i)
_tri_j = np.array(_tri_j)
_tri_k = np.array(_tri_k)
_num_tris = len(_tri_i)

# Barycentric coordinate basis
# Point = b0*V0 + b1*V1 + b2*V2 where b0+b1+b2=1, b_i >= 0
# We parameterize with b1, b2 (b0 = 1-b1-b2)
# x = b1*1 + b2*0.5, y = b2*sqrt(3)/2
# So from (b1, b2): x = b1 + 0.5*b2, y = sqrt(3)/2 * b2
# Inverse: b2 = 2y/sqrt(3), b1 = x - 0.5*b2 = x - y/sqrt(3)
# b0 = 1 - b1 - b2 = 1 - x + y/sqrt(3) - 2y/sqrt(3) = 1 - x - y/sqrt(3)

def _bary_to_cart(b):
    """Convert barycentric coords (N,3) to cartesian (N,2)."""
    b1 = b[:, 1]
    b2 = b[:, 2]
    x = b1 + 0.5 * b2
    y = (np.sqrt(3) / 2.0) * b2
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """Convert cartesian (N,2) to barycentric (N,3)."""
    x = pts[:, 0]
    y = pts[:, 1]
    b2 = 2.0 * y / np.sqrt(3)
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _all_areas(pts):
    """Compute areas of all C(N,3) triangles. Returns (num_tris,) array."""
    p1 = pts[_tri_i]
    p2 = pts[_tri_j]
    p3 = pts[_tri_k]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    dx1 = p2[:, 0] - p1[:, 0]
    dy1 = p2[:, 1] - p1[:, 1]
    dx2 = p3[:, 0] - p1[:, 0]
    dy2 = p3[:, 1] - p1[:, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area(pts):
    """Minimum triangle area."""
    return np.min(_all_areas(pts))

def _soft_min_area(pts, temperature=5.0):
    """
    Smooth approximation of min area using log-sum-exp of reciprocal areas.
    min(a_i) ≈ -T * log(sum(exp(-a_i/T)))  for small T
    We want to maximize min area, so minimize -min area.
    -min(a_i) ≈ T * log(sum(exp(-a_i/T)))
    """
    areas = _all_areas(pts)
    # Avoid log(0)
    areas = np.maximum(areas, 1e-15)
    neg_areas = -areas / temperature
    # Numerical stability: subtract max
    max_neg = np.max(neg_areas)
    log_sum = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    return temperature * log_sum
>>>>>>> REPLACE

