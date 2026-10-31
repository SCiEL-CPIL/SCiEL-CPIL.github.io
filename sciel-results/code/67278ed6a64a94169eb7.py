"""Cap set priority: target central, diverse vectors to avoid boundary APs."""
import time

# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Priority based on product of distances to all-zeros and all-twos vectors,
    plus a diversity bonus for having all three coordinate values present.
    
    Vectors with many 1s and balanced 0s/2s are prioritized because they're
    geometrically central and less likely to form 3-term APs with boundary vectors.
    """
    count_0 = 0
    count_1 = 0
    count_2 = 0
    for v in vector:
        if v == 0:
            count_0 += 1
        elif v == 1:
            count_1 += 1
        else:
            count_2 += 1
    
    # Product of distances to all-zeros and all-twos
    dist_to_zero = count_1 + 2 * count_2
    dist_to_two = 2 * count_0 + count_1
    center_score = dist_to_zero * dist_to_two
    
    # Diversity bonus: reward having all three values present
    diversity_bonus = 0.0
    if count_0 > 0 and count_1 > 0 and count_2 > 0:
        # Scale bonus by how balanced the distribution is
        min_count = min(count_0, count_1, count_2)
        diversity_bonus = min_count * 2.0
    elif count_1 > 0 and (count_0 > 0 or count_2 > 0):
        # Partial bonus for having 1s and at least one other type
        diversity_bonus = 0.5 * min(count_0 + count_2, count_1)
    
    return center_score + diversity_bonus
# EVOLVE-BLOCK-END
