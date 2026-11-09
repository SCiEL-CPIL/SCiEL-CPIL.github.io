import numpy as np
from collections import Counter
import itertools

def score(points):
    pts = np.array(points)
    n = len(pts)
    areas = []
    # Vectorized calculation of areas for all triples is O(n^3) which is 30^3 = 27000.
    # This is very fast.
    for i in range(n-2):
        for j in range(i+1, n-1):
            for k in range(j+1, n):
                x1, y1 = pts[i]
                x2, y2 = pts[j]
                x3, y3 = pts[k]
                det = abs(x1*(y2-y3) - y1*(x2-x3) + x2*y3 - x3*y2)
                if det > 0:
                    areas.append(det)
    if not areas:
        return 0
    counts = Counter(areas)
    return max(counts.values())
