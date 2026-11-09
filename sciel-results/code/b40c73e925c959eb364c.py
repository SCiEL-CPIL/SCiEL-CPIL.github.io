"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools

def construct():
    n = 12
    total_masks = 1 << n
    all_masks = list(range(total_masks))
    
    # Precompute subset/superset relations? No, just compute on the fly.
    # We represent sets as bitmasks for speed.
    
    # Initialization: Choose 40 sets from the middle levels (sizes 5, 6, 7)
    # to have a good starting point with high potential for comparability.
    # Level 6 has C(12,6) = 924 sets.
    # Let's pick randomly from levels 5, 6, 7.
    candidates = []
    for size in [5, 6, 7]:
        for mask in all_masks:
            if bin(mask).count('1') == size:
                candidates.append(mask)
                
    random.seed(42) # Fixed seed for reproducibility
    initial_family = random.sample(candidates, 40)
    
    def score_family(family):
        count = 0
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                a = family[i]
                b = family[j]
                if (a & b) == a or (a & b) == b:
                    count += 1
        return count
    
    current_family = initial_family
    current_score = score_family(current_family)
    
    # Local Search
    # We will try to replace each element in the family with any other element
    # that gives a higher score.
    improved = True
    max_iterations = 1000
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(len(current_family)):
            old_set = current_family[i]
            best_gain = 0
            best_new_set = -1
            
            # Calculate current contribution of old_set to the score
            # Contribution = number of other sets in family that are comparable with old_set
            # Let's just re-evaluate the whole score for each candidate to be safe, 
            # or optimize by calculating delta.
            # Delta calculation:
            # Remove old_set. Score drops by C_old (comparables with others).
            # Add new_set. Score increases by C_new (comparables with others in family\{old_set}).
            # Net change = C_new - C_old.
            
            C_old = 0
            for j in range(len(current_family)):
                if i == j: continue
                a = old_set
                b = current_family[j]
                if (a & b) == a or (a & b) == b:
                    C_old += 1
            
            for new_set in all_masks:
                if new_set == old_set:
                    continue
                if new_set in current_family:
                    continue
                
                C_new = 0
                for j in range(len(current_family)):
                    if i == j: continue
                    a = new_set
                    b = current_family[j]
                    if (a & b) == a or (a & b) == b:
                        C_new += 1
                
                gain = C_new - C_old
                if gain > best_gain:
                    best_gain = gain
                    best_new_set = new_set
            
            if best_new_set != -1:
                current_family[i] = best_new_set
                current_score += best_gain
                improved = True
    
    # Convert masks to lists of integers
    result = []
    for m in current_family:
        s = []
        for i in range(n):
            if (m >> i) & 1:
                s.append(i)
        result.append(s)
        
    return result

# EVOLVE-BLOCK-END
