import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 75.0
    
    # Known good solution: D=42
    # Denoms: 2, 3, 7, 42 -> Sum=1
    # Replace 42 with 43 -> Sum = 1 - 1/(42*43)
    best_deficit = Fraction(1, 1806)
    best_denoms = [2, 3, 7, 43]
    
    # Search for better D
    # We want to find D in [47, 2] such that 1 - 1/D is sum of 1/d for d in [2, D-1]
    # If found, new deficit is 1/(D*(D+1))
    
    def find_subset_summing_to(target, max_den_exclusive):
        # max_den_exclusive means denominators are in [2, max_den_exclusive-1]
        # Returns list of denominators or None
        reachable = {Fraction(0): []}
        # We iterate d from 2 to max_den_exclusive - 1
        for d in range(2, max_den_exclusive):
            new_reachable = dict(reachable)
            for s, denoms in reachable.items():
                new_s = s + Fraction(1, d)
                if new_s <= target:
                    # If we already have this sum, we might want to keep the one with fewer terms? 
                    # Or doesn't matter.
                    if new_s not in new_reachable:
                        new_reachable[new_s] = denoms + [d]
            reachable = new_reachable
            if target in reachable:
                return reachable[target]
        return None

    for D in range(47, 1, -1):
        if time.time() - start_time > DEADLINE:
            break
            
        target = Fraction(D - 1, D)
        # We need subset of {2, ..., D-1} summing to target
        subset = find_subset_summing_to(target, D)
        
        if subset is not None:
            # Check if this D gives a better deficit
            # Deficit is 1 / (D * (D + 1))
            new_deficit = Fraction(1, D * (D + 1))
            if new_deficit < best_deficit:
                # Construct the final denominators
                # Remove D from subset (it's not there, subset is from 2..D-1)
                # Add D+1
                # Wait, the subset sums to 1 - 1/D.
                # So subset + [D] sums to 1.
                # We want subset + [D+1].
                final_denoms = subset + [D + 1]
                best_deficit = new_deficit
                best_denoms = final_denoms
                # Since we are iterating D downwards, the first one found is the best (largest D)
                # So we can break?
                # Yes, larger D means smaller deficit 1/(D(D+1)).
                break
                
    # Convert denominators to indices
    best_indices = [d - 2 for d in best_denoms]
    
    return best_indices
