"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance
    # Add epsilon to avoid division by zero
    base_heur = 1.0 / (distance_matrix + 1e-9)
    
    # Generate a reference set of edges using a capacity-aware Nearest Neighbor heuristic
    # This provides a "good" set of edges to boost in the heuristic matrix
    visited = np.zeros(n, dtype=bool)
    visited[0] = True  # Depot is always visited/started
    current = 0
    remaining_cap = capacity
    ref_edges = set()
    
    # We need to visit all customers (1 to n-1)
    unvisited_count = n - 1
    
    while unvisited_count > 0:
        if current == 0:
            # At depot, find nearest customer that fits
            best_j = -1
            best_dist = float('inf')
            for j in range(1, n):
                if not visited[j]:
                    if demands[j] <= remaining_cap:
                        d = distance_matrix[current, j]
                        if d < best_dist:
                            best_dist = d
                            best_j = j
            if best_j == -1:
                # Should not happen if total demand fits, but handle gracefully
                break
            ref_edges.add((0, best_j))
            ref_edges.add((best_j, 0))
            visited[best_j] = True
            current = best_j
            remaining_cap -= demands[current]
            unvisited_count -= 1
        else:
            # At customer, find nearest unvisited customer that fits
            best_j = -1
            best_dist = float('inf')
            for j in range(1, n):
                if not visited[j]:
                    if demands[j] <= remaining_cap:
                        d = distance_matrix[current, j]
                        if d < best_dist:
                            best_dist = d
                            best_j = j
            
            if best_j != -1:
                ref_edges.add((current, best_j))
                ref_edges.add((best_j, current))
                visited[best_j] = True
                current = best_j
                remaining_cap -= demands[current]
                unvisited_count -= 1
            else:
                # No customer fits, return to depot
                ref_edges.add((current, 0))
                ref_edges.add((0, current))
                current = 0
                remaining_cap = capacity

    # Boost the heuristic values for edges in the reference solution
    # A boost factor of 5.0 balances guidance with exploration
    boost_factor = 5.0
    for i, j in ref_edges:
        base_heur[i, j] *= boost_factor
        base_heur[j, i] *= boost_factor

    # Ensure no non-positive or non-finite values (as per contract, though 1/d is always positive)
    # The contract says: "Heuristic entries that are not positive become 1e-9"
    base_heur[~np.isfinite(base_heur)] = 1e-9
    base_heur[base_heur <= 0] = 1e-9

    return base_heur
# EVOLVE-BLOCK-END
