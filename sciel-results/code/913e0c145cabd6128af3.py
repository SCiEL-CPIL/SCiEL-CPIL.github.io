"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    
    # Base heuristic: Inverse distance
    # Using a power of 1.0 keeps it simple, but 1.2-1.5 is often better for ACO.
    # Let's stick with 1.0 for the base and let the boosting do the work.
    power = 1.0
    eps = 1e-9
    base_heur = 1.0 / (distance_matrix + eps) ** power
    
    # Generate a reference set of edges using a Nearest Neighbor (NN) Heuristic
    # This approach constructs routes greedily by picking the nearest feasible customer.
    
    unvisited = set(range(1, n))
    ref_edges = set()
    
    # We run the NN construction multiple times to get a more robust set of "good" edges?
    # Or just once? The standard NN creates a set of tours.
    # Let's create one set of tours using NN.
    
    while unvisited:
        current = 0
        remaining_cap = capacity
        
        # Start a new route from depot
        while True:
            # Find the nearest unvisited customer that fits in remaining capacity
            best_next = -1
            best_dist = float('inf')
            
            for j in unvisited:
                if demands[j] <= remaining_cap:
                    d = distance_matrix[current, j]
                    if d < best_dist:
                        best_dist = d
                        best_next = j
                        
            if best_next == -1:
                # No customer fits, return to depot (conceptually)
                break
                
            # Add edge current -> best_next
            ref_edges.add((current, best_next))
            ref_edges.add((best_next, current)) # Symmetric boost
            
            # Update state
            remaining_cap -= demands[best_next]
            unvisited.remove(best_next)
            current = best_next
            
            if not unvisited:
                break
                
            # If remaining cap is very small, we might still find a customer, 
            # but if none fits, the loop breaks.
            
    # Boost the heuristic values for edges in the reference solution
    # A moderate boost is usually safe.
    boost_factor = 3.0
    for i, j in ref_edges:
        base_heur[i, j] *= boost_factor
        base_heur[j, i] *= boost_factor

    # Ensure no non-positive or non-finite values
    base_heur[~np.isfinite(base_heur)] = 1e-9
    base_heur[base_heur <= 0] = 1e-9

    return base_heur
# EVOLVE-BLOCK-END
