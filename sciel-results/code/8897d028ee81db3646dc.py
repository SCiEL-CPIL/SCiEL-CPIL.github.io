import time
import random
import sys

def construct():
    n = 12
    m = 40
    deadline = time.time() + 50.0
    
    # Precompute comparability: for any two masks, is one a strict subset of the other?
    # We'll work with 12-bit masks representing subsets of {0,...,11}
    
    def count_comparable(family):
        """Count unordered pairs where one set is a strict subset of the other."""
        count = 0
        for i in range(len(family)):
            for j in range(i + 1, len(family)):
                mi = family[i]
                mj = family[j]
                if (mi & mj) == mi and mi != mj:
                    count += 1
                elif (mi & mj) == mj and mi != mj:
                    count += 1
        return count
    
    def random_family():
        """Generate a random family of m distinct subsets."""
        all_masks = list(range(1 << n))
        random.shuffle(all_masks)
        return all_masks[:m]
    
    def structured_family():
        """Generate a family biased toward comparable sets (chains and near-chains)."""
        family = []
        # Start with a random chain
        current = 0
        while len(family) < m and current < (1 << n) - 1:
            family.append(current)
            # Add a random element to extend the chain
            available = [b for b in range(n) if not (current >> b) & 1]
            if available:
                current |= (1 << random.choice(available))
            else:
                break
        # Fill remaining with random sets
        while len(family) < m:
            mask = random.randint(0, (1 << n) - 1)
            if mask not in family:
                family.append(mask)
        return family
    
    # Start with a structured initial solution
    best_family = structured_family()
    best_score = count_comparable(best_family)
    current_family = best_family[:]
    current_score = best_score
    
    # Simulated annealing
    T = 100.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        # Generate a neighbor: swap one set in the family with a random new set
        pos = random.randint(0, m - 1)
        new_mask = random.randint(0, (1 << n) - 1)
        
        if new_mask in current_family:
            T *= cooling
            continue
        
        old_mask = current_family[pos]
        current_family[pos] = new_mask
        
        # Compute delta score efficiently
        # Remove old_mask's contributions, add new_mask's contributions
        delta = 0
        for i in range(m):
            if i == pos:
                continue
            mi = current_family[i]
            
            # Check old_mask vs mi
            if (old_mask & mi) == old_mask and old_mask != mi:
                delta -= 1
            elif (old_mask & mi) == mi and old_mask != mi:
                delta -= 1
            
            # Check new_mask vs mi
            if (new_mask & mi) == new_mask and new_mask != mi:
                delta += 1
            elif (new_mask & mi) == mi and new_mask != mi:
                delta += 1
        
        new_score = current_score + delta
        
        if new_score > current_score or random.random() < pow(2.718281828, delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_family = current_family[:]
        else:
            current_family[pos] = old_mask
        
        T *= cooling
    
    # Convert best family to list of lists
    result = []
    for mask in best_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result
