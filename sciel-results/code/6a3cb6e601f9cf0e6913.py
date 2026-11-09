def feasible_assignment(demands, capacities):
    n_customers = len(demands)
    n_facilities = len(capacities)
    assignment = np.full(n_customers, -1, dtype=int)
    load = np.zeros(n_facilities, dtype=np.int64)
    
    # Sort customers by demand descending
    order = np.argsort(-demands)
    
    for i in order:
        # Try to place customer i in the facility with the most remaining capacity
        # that can fit it. This is "Best Fit" which is often better for packing.
        best_f = -1
        best_rem = -1
        for f in range(n_facilities):
            rem = capacities[f] - load[f]
            if rem >= demands[i]:
                if rem < best_rem or best_f == -1: # Actually Best Fit wants smallest rem that fits
                    # Wait, Best Fit: smallest remaining space that fits.
                    # If best_rem is -1, any fit is better.
                    if best_f == -1 or rem < best_rem:
                        best_rem = rem
                        best_f = f
        
        if best_f != -1:
            assignment[i] = best_f
            load[best_f] += demands[i]
        else:
            # If no facility fits, try any facility with space (shouldn't happen if total cap > total demand)
            # But if fragmentation is severe, we might need to swap.
            # For now, if FFD fails, we return invalid.
            return None, None
            
    return assignment, load
