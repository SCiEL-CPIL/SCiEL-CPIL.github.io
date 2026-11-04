import math

def construct():
    # Lattice vectors for unit hexagons (side 1)
    # Distance between centers is sqrt(3)
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 1.5)
    
    sqrt3 = math.sqrt(3)
    u = (sqrt3, 0.0)
    v = (sqrt3 / 2.0, 1.5)
    
    # Function to get Cartesian coordinates from lattice indices (i, j)
    def lattice_to_cart(i, j):
        x = i * u[0] + j * v[0]
        y = i * u[1] + j * v[1]
        return (x, y)

    # 7 central hexagons (i, j) with i^2 - ij + j^2 <= 1
    # (0,0), (1,0), (0,1), (-1,0), (0,-1), (1,-1), (-1,1)
    centers_indices = [
        (0, 0),
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),
        (1, -1),
        (-1, 1)
    ]
    
    # 4 additional hexagons from the next shell
    # We need 4 more.
    # Candidates at distance 3 (i^2 - ij + j^2 = 3)
    # (1, 2), (2, 1), (-1, -2), (-2, -1), (1, -1) is dist sqrt(3)? No (1,-1) is 1-(-1)+1=3? No 1 - (-1) + 1 = 3. Wait.
    # i=1, j=-1: 1 - (1)(-1) + (-1)^2 = 1 + 1 + 1 = 3. Yes.
    # So (1, -1) is dist 3?
    # Let's re-evaluate dist for (1, -1).
    # x = 1*sqrt3 + (-1)*sqrt3/2 = sqrt3/2.
    # y = 0 + (-1)*1.5 = -1.5.
    # dist^2 = 3/4 + 2.25 = 0.75 + 2.25 = 3.0.
    # dist = sqrt(3).
    # So (1, -1) is in Ring 1?
    # My Ring 1 list: (1,0), (0,1), (-1,0), (0,-1), (1,-1), (-1,1).
    # Yes, (1,-1) is in Ring 1.
    
    # So the 6 points at dist sqrt(3) are:
    # (1,0), (0,1), (-1,0), (0,-1), (1,-1), (-1,1).
    
    # The points at dist 3 (i^2 - ij + j^2 = 3) are:
    # (1, 2): 1 - 2 + 4 = 3.
    # (2, 1): 4 - 2 + 1 = 3.
    # (-1, -2): 1 - 2 + 4 = 3.
    # (-2, -1): 4 - 2 + 1 = 3.
    # (1, -1) was dist sqrt(3).
    # What about (2, -1)? 4 - (-2) + 1 = 7. No.
    # What about (0, 2)? 0 - 0 + 4 = 4. Dist 2*sqrt(3).
    # What about (2, 0)? 4. Dist 2*sqrt(3).
    
    # So the 6 points at dist 3 are:
    # (1, 2), (2, 1), (-1, -2), (-2, -1) ... and two more?
    # (1, 2) and (2, 1) are in the first quadrant.
    # Symmetry:
    # (-1, -2), (-2, -1) in third.
    # What about second and fourth?
    # (-1, 1) is dist sqrt(3).
    # (-2, 1)? 4 - (-2) + 1 = 7.
    # (-1, 2)? 1 - (-2) + 4 = 7.
    # It seems there are only 4 points at dist 3?
    # Let's check (1, 2) and (2, 1).
    # Are there points like (0, ?) no.
    # The number of solutions to x^2 - xy + y^2 = 3 is 6?
    # (1, 2), (2, 1), (-1, -2), (-2, -1).
    # (1, -1) gave 3?
    # 1 - (1)(-1) + (-1)^2 = 1 + 1 + 1 = 3.
    # So (1, -1) IS dist 3?
    # But I listed (1, -1) in Ring 1.
    # Let's re-calc dist for (1, -1).
    # x = sqrt3 - sqrt3/2 = sqrt3/2.
    # y = 0 - 1.5 = -1.5.
    # x^2 + y^2 = 3/4 + 9/4 = 12/4 = 3.
    # So dist is sqrt(3).
    # Why did I think it was dist 3?
    # Ah, the formula $i^2 - ij + j^2$ gives $dist^2 / 3$.
    # For (1, -1): $1 - (-1) + 1 = 3$. So $dist^2 = 3 \times 3 = 9$?
    # No.
    # $x = i \sqrt{3} + j \frac{\sqrt{3}}{2}$.
    # $y = j \frac{3}{2}$.
    # $x^2 + y^2 = 3 (i + j/2)^2 + \frac{9}{4} j^2 = 3 (i^2 + ij + j^2/4) + \frac{9}{4} j^2 = 3i^2 + 3ij + \frac{3}{4}j^2 + \frac{9}{4}j^2 = 3i^2 + 3ij + 3j^2 = 3(i^2 + ij + j^2)$.
    # Wait, my lattice definition:
    # $u = (\sqrt{3}, 0)$. $v = (\frac{\sqrt{3}}{2}, \frac{3}{2})$.
    # This corresponds to $60^\circ$ angle?
    # Dot product $u \cdot v = \sqrt{3} \cdot \frac{\sqrt{3}}{2} + 0 = 1.5$.
    # $|u|= \sqrt{3}$. $|v| = \sqrt{3/4 + 9/4} = \sqrt{3}$.
    # $\cos \theta = 1.5 / 3 = 0.5$. $\theta = 60^\circ$.
    # Distance squared: $|i u + j v|^2 = i^2 |u|^2 + j^2 |v|^2 + 2ij (u \cdot v) = 3i^2 + 3j^2 + 2ij(1.5) = 3(i^2 + j^2 + ij)$.
    # So $dist^2 = 3(i^2 + ij + j^2)$.
    
    # For (1, -1): $1 + 1 - 1 = 1$. $dist^2 = 3$. $dist = \sqrt{3}$. Correct.
    # For (1, 2): $1 + 2 + 2 = 5$. $dist^2 = 15$. $dist = \sqrt{15} \approx 3.87$.
    # For (2, 1): $4 + 1 + 2 = 7$. $dist^2 = 21$.
    # For (1, 1): $1 + 1 + 1 = 3$. $dist^2 = 9$. $dist = 3$.
    
    # So the points at dist 3 are those with $i^2 + ij + j^2 = 3$.
    # Solutions:
    # (1, 1), (-1, -1)
    # (2, -1): $4 - 2 + 1 = 3$. Yes.
    # (-2, 1): $4 - 2 + 1 = 3$. Yes.
    # (1, -2): $1 - 2 + 4 = 3$. Yes.
    # (-1, 2): $1 - 2 + 4 = 3$. Yes.
    
    # So the 6 points at dist 3 are:
    # (1, 1), (-1, -1), (2, -1), (-2, 1), (1, -2), (-1, 2).
    
    # We need 4 of these.
    # Let's pick:
    add_indices = [
        (1, 1),
        (-1, -1),
        (2, -1),
        (-2, 1)
    ]
    
    all_indices = centers_indices + add_indices
    
    inner = []
    for i, j in all_indices:
        x, y = lattice_to_cart(i, j)
        inner.append([x, y, 0.0])
        
    # Calculate required side length
    # Max distance from center to any vertex of inner hexagons
    max_r = 0
    for x, y, _ in inner:
        r_center = math.sqrt(x*x + y*y)
        # The hexagon has radius 1. Max extent is r_center + 1
        # But we need to check against the outer hexagon boundary
        # For simplicity, use max_r + 1 as a lower bound for circumradius S
        # Outer hexagon circumradius is S.
        # So S >= max_r + 1
        max_r = max(max_r, r_center + 1)
        
    # Add a small margin for safety
    side = max_r + 0.01
    
    return inner, (0.0, 0.0), side, 0.0
