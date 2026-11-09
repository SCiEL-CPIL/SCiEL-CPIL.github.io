# EVOLVE-BLOCK-START
def priority(vector, n):
    """
    Assign priority to vectors in {0,1,2}^n to maximize cap set size via greedy insertion.
    
    Strategy:
    - Use a geometric heuristic based on the sum of coordinates.
    - Vectors with sums far from the average (n) are at the "corners" or edges of the distribution.
    - In the center (sum ~ n), the density of points is highest, and lines are more likely to be formed.
    - By prioritizing "extreme" vectors first, we fill the periphery, which might be less prone to immediate line conflicts, or we structure the set in a way that leaves the center for later, potentially allowing a larger total set.
    - This is a different approach from simple coordinate counting (which is linear in counts) as it uses a quadratic/non-linear measure of the vector's position relative to the centroid.
    """
    s = sum(vector)
    # Average sum is n (since each coord is 0,1,2, avg is 1, so n*1 = n)
    # We want higher priority for larger distance from n.
    # Using squared distance to emphasize extremes.
    return (s - n) ** 2
# EVOLVE-BLOCK-END
