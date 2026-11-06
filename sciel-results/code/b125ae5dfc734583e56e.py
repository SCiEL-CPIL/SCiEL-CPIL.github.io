"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import laplacian, connected_components
from scipy.sparse import csc_matrix


def priority(adjacency):
    """
    Assign a score to each vertex such that a greedy decoder picking vertices
    in descending score order (skipping neighbors of already-picked vertices)
    yields a large independent set.
    
    Strategy:
    1. Compute degree of each vertex.
    2. Compute a 'local density' or clustering-based penalty: vertices that are
       part of dense substructures (high clustering coefficient or high sum of
       neighbor degrees) should be deprioritized because picking them blocks
       many other vertices.
    3. Combine: low degree and low 'blocking potential' get high scores.
    
    Specifically, we use:
    - degree: d[i]
    - neighbor_degree_sum: sum of degrees of neighbors of i (call it nds[i])
      This approximates how many vertices i 'blocks' if picked.
    - We want to pick vertices with low d and low nds.
    
    Score = -(d[i] * alpha + nds[i] * beta)
    But to make it more nuanced, we can use a normalized version.
    
    Let's compute:
    d = adjacency.sum(axis=1)
    nds = adjacency @ d  (sum of degrees of neighbors)
    
    Then score = -(d + 0.5 * nds / max_degree)  or similar.
    
    Actually, a known heuristic for MaxIS is to pick vertices with minimum
    degree first (greedy). The current program does that. To improve, we can
    consider the "effective" degree after accounting for the structure.
    
    Another idea: use the eigenvector centrality of the complement graph, or
    simply use a combination of degree and clustering.
    
    Let's try: score = - (d[i] + 0.1 * (adjacency @ d)[i])
    This penalizes vertices that have neighbors with high degrees (i.e., picking
    such a vertex blocks many high-degree vertices, which is bad because those
    high-degree vertices would have blocked even more if they were picked, but
    actually we want to avoid picking vertices that are adjacent to many vertices
    that are themselves adjacent to many vertices).
    
    Wait, let's think differently. The greedy algorithm picks the highest score
    vertex, removes it and its neighbors. So we want the first picked vertex to
    have as few neighbors as possible (low degree). Then, among the remaining
    vertices, we want the next picked vertex to also have low degree in the
    remaining graph. But since we can't simulate the greedy process in the
    priority function (it must be a static score), we need a static approximation.
    
    A better static heuristic: use the "minimum degree" but adjusted for the
    local structure. For example, vertices in sparse parts of the graph should
    be picked earlier.
    
    Let's compute:
    d = degree
    nds = sum of neighbor degrees
    
    A vertex with low d and low nds is in a sparse region.
    
    Score = -(d[i] * 1.0 + nds[i] * 0.1)
    
    Let's test this idea. We'll normalize nds by the max degree to keep it in
    a reasonable range.
    
    Actually, let's use a simpler and more robust approach: 
    score = - (d[i] + 0.5 * (adjacency @ d)[i] / (max_d + 1))
    
    This should work better than just -d.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    d = adj.sum(axis=1)
    max_d = d.max()
    if max_d == 0:
        return np.zeros(len(d))
    
    # Sum of degrees of neighbors
    nds = adj @ d
    
    # Normalize nds to be on a similar scale as d
    nds_norm = nds / (max_d + 1)
    
    # Score: lower d and lower nds is better
    score = -(d + 0.5 * nds_norm)
    
    return score
# EVOLVE-BLOCK-END
