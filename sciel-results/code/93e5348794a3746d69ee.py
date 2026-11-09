# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that their maximum norm is smaller than their minimum pairwise distance, aiming to maximize the number of points.

    Returns:
        points: np.ndarray of shape (num_points,11)
    """
    d = 11
    deadline = time.time() + 600
    
    # Try the 2*E8 + 2*D3 lattice construction
    # E8 lattice points with norm squared 8:
    # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) with even number of nonzeros
    # Type 2: (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even number of plus signs
    # But we need integer coordinates, so let's use scaled versions
    
    # For integer lattice, use points from Z^11
    # Let's try a simple construction: all points with norm squared = 8
    
    # Points with exactly two nonzero coordinates of ±2:
    # (±2, ±2, 0, ..., 0) in all positions
    points_list = []
    
    # Add all points with two ±2's and rest zeros
    for i in range(d):
        for j in range(i + 1, d):
            for si in [1, -1]:
                for sj in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = 2 * si
                    p[j] = 2 * sj
                    points_list.append(p)
    
    # Also add points with one coordinate ±4
    for i in range(d):
        for s in [1, -1]:
            p = np.zeros(d, dtype=np.int64)
            p[i] = 4 * s
            points_list.append(p)
    
    # Also add points with four coordinates ±2
    # This might be too many, let's check
    # Actually, let's just use the two-coordinate ones first
    
    points = np.array(points_list, dtype=np.int64)
    
    # Verify the constraint
    k = len(points)
    if k < 2:
        return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
    
    # Check no zero vector
    if np.any(np.all(points == 0, axis=1)):
        return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)
    
    # Compute norms
    norms = np.sum(points * points, axis=1)
    max_norm = np.max(norms)
    
    # Check minimum pairwise distance
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = points[i] - points[j]
            dist_sq = np.dot(diff, diff)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
            if min_dist_sq < max_norm:
                break
        if min_dist_sq < max_norm:
            break
    
    if min_dist_sq >= max_norm:
        return points
    else:
        # Fallback to simple solution
        return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)


# EVOLVE-BLOCK-END
