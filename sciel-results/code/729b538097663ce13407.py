"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    # Convert to numpy if needed
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # Degree of each vertex
    degrees = adj.sum(axis=1)
    
    # For each vertex, compute the average degree of its non-neighbors
    # This is a heuristic: vertices whose non-neighbors have low degree
    # are "safer" to pick because their neighbors are constrained,
    # and non-neighbors are likely to be picked too.
    
    # For each vertex i, non-neighbors are vertices j where adj[i,j] == 0
    # We want mean degree of non-neighbors.
    
    # Sum of degrees of non-neighbors for vertex i = total_degree_sum - degrees[i] - sum of degrees of neighbors of i
    total_degree_sum = degrees.sum()
    
    # Sum of degrees of neighbors of i: this is adj[i,:] dot degrees
    neighbor_degree_sums = adj @ degrees
    
    # Sum of degrees of non-neighbors (excluding self)
    non_neighbor_degree_sums = total_degree_sum - degrees - neighbor_degree_sums
    
    # Count of non-neighbors (excluding self)
    non_neighbor_counts = (n - 1) - degrees.astype(int)
    
    # Avoid division by zero
    non_neighbor_counts_safe = np.where(non_neighbor_counts > 0, non_neighbor_counts, 1)
    
    # Mean degree of non-neighbors
    mean_non_neighbor_degree = non_neighbor_degree_sums / non_neighbor_counts_safe
    
    # We want vertices with LOW mean non-neighbor degree to have HIGH priority
    # (because their non-neighbors are "less central" and thus more likely to be in the independent set)
    # Actually, let me think again: if a vertex's neighbors have high degree, picking it blocks many potential picks.
    # So we want vertices whose neighbors have LOW degree.
    
    # Let's use: priority = -mean degree of neighbors
    # But also consider the vertex's own degree: lower degree is better.
    
    # Combine: lower degree is better, and neighbors with lower degree are better.
    
    # Mean degree of neighbors
    neighbor_counts = degrees.astype(int)
    neighbor_counts_safe = np.where(neighbor_counts > 0, neighbor_counts, 1)
    mean_neighbor_degree = neighbor_degree_sums / neighbor_counts_safe
    
    # Priority: lower own degree and lower mean neighbor degree
    # Normalize to combine
    score = -degrees - mean_neighbor_degree
    
    return score
# EVOLVE-BLOCK-END
