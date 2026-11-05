# EVOLVE-BLOCK-START
import math

def construct():
    # 1. Generate the 19 points of the hexagon of radius 2 (H2)
    # These are all (a,b) such that max(|a|, |b|, |a+b|) <= 2
    # Or simply r2 <= 7 and check membership in the specific hexagon shape.
    # Better: Iterate and check if point is in H2.
    # H2 points:
    h2_points = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            # Check if in hexagon of radius 2
            # Condition: max(abs(a), abs(b), abs(a+b)) <= 2
            if max(abs(a), abs(b), abs(a+b)) <= 2:
                h2_points.append((a, b))
    
    # There should be 19 points.
    # 2. Generate the 18 points of the hexagon of radius 3 boundary (Ring 3)
    # These are points with max(|a|, |b|, |a+b|) == 3
    ring3_points = []
    for a in range(-3, 4):
        for b in range(-3, 4):
            if max(abs(a), abs(b), abs(a+b)) == 3:
                ring3_points.append((a, b))
                
    # 3. Sort ring3_points to form a contiguous cycle/arc.
    # We can sort by angle or by a "winding" order.
    # A simple way: Sort by (a, b) doesn't work for contiguity.
    # We can sort by the angle atan2(y, x) in Cartesian coordinates.
    # omega = exp(i*pi/3) = 0.5 + i*sqrt(3)/2
    # z = a + b*omega = a + b(0.5 + i*sqrt(3)/2) = (a + b/2) + i*b*sqrt(3)/2
    # x = a + b/2, y = b*sqrt(3)/2
    
    def get_angle(pt):
        a, b = pt
        x = a + b / 2.0
        y = b * math.sqrt(3) / 2.0
        return math.atan2(y, x)
        
    ring3_points.sort(key=get_angle)
    
    # The points are now in circular order.
    # We need to pick 13 contiguous points.
    # Any 13 contiguous points from the 18 will work, due to symmetry?
    # Yes, the graph is vertex-transitive for the ring?
    # The number of edges within the selected 13 will be 12 (a path).
    # The number of edges to H2 will depend on the specific points.
    # However, due to the symmetry of the hexagon, any arc of length 13 should yield the same number of edges to H2?
    # Let's check.
    # The 18 points are arranged on the hexagon perimeter.
    # Each point connects to 2 points in H2 (as derived earlier for corners, need to check sides).
    # Side point example: (1, 2) is in Ring 3?
    # max(1, 2, 3) = 3. Yes.
    # Neighbors of (1,2):
    # (0,2) [H2], (1,1) [H2], (2,2) [R3? max(2,2,4)=4 No, R4], (1,3) [R3? max(1,3,4)=4 No], 
    # (0,3) [R3? max(0,3,3)=3 Yes], (2,1) [R3? max(2,1,3)=3 Yes].
    # So (1,2) connects to (0,2) and (1,1) in H2. (2 edges)
    # And (0,3) and (2,1) in R3. (2 edges)
    # So every point in Ring 3 connects to exactly 2 points in H2.
    # Total edges from 13 R3 points to H2 = 13 * 2 = 26.
    # Edges within H2 = 36 (calculated previously, needs verification).
    # Edges within R3 (selected 13) = 12 (path of 13 nodes).
    # Total = 36 + 26 + 12 = 74.
    
    # Wait, if max is 76, we are missing 2 edges.
    # Where do the extra edges come from?
    # Maybe H2 has more than 36 edges?
    # Let's recount H2 edges.
    # 19 points.
    # If it's a hexagonal packing, the number of edges is $3n - \text{perimeter}$?
    # For $n=19$, perimeter is 12?
    # Formula for edges in hexagonal packing of size $k$ (side $k+1$):
    # $E_k = 3k(k+1)$.
    # For $k=2$ (19 pts), $E_2 = 3(2)(3) = 18$? No, that's too low.
    # Let's check $k=1$ (7 pts). $E_1 = 3(1)(2) = 6$?
    # 7 points: Center + 6 ring.
    # Center connects to 6.
    # Ring connects to Center (6) and Neighbors (6).
    # Total 12 edges.
    # Formula $3k(k+1)$ for $k=1$ gives 6. Incorrect.
    
    # Correct formula for number of edges in hexagonal lattice packing of $n$ points (hexagon of side $s$):
    # If $n = 3s(s-1)+1$.
    # For $s=1$, $n=1$, $E=0$.
    # For $s=2$, $n=7$, $E=12$.
    # For $s=3$, $n=19$, $E=36$.
    # Let's verify $E=36$ for $n=19$.
    # If $E=36$, then Total = 36 + 26 + 12 = 74.
    # We need 76.
    # This implies that the "contiguous arc" assumption might yield 74, but a different configuration yields 76.
    # Or my edge count for H2 is wrong.
    # Let's check if H2 has 48 edges?
    # If H2 has 48, Total = 48 + 26 + 12 = 86 (Too high).
    
    # Let's re-evaluate the "Optimum 76".
    # Harborth's formula: $\lfloor 3n - \sqrt{12n-3} \rfloor$.
    # For $n=32$: $3(32) - \sqrt{384-3} = 96 - \sqrt{381} \approx 96 - 19.52 = 76.48 \rightarrow 76$.
    # So 76 is the target.
    
    # If my construction gives 74, I need 2 more edges.
    # How to get more edges?
    # Maybe the 13 points from Ring 3 should not be a simple arc?
    # Or maybe I should include points from Ring 4?
    # No, Ring 4 is too far.
    
    # Let's check the edge count of H2 again.
    # Points:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    
    # Edges:
    # Center (0,0) to 6 neighbors: 6 edges.
    # Ring 1 (6 pts) to Ring 2:
    # (1,0) connects to (2,0) and (1,1)?
    # (1,0) neighbors: (0,0), (1,1), (2,0), (2,-1), (1,-1), (0,-1).
    # In H2: (0,0), (1,1), (2,0), (2,-1), (1,-1), (0,-1).
    # So (1,0) connects to 2 Ring 2 points: (2,0) and (1,1).
    # Wait, (1,1) is in Ring 2?
    # (1,1): max(1,1,2)=2. Yes.
    # So each Ring 1 point connects to 2 Ring 2 points.
    # 6 Ring 1 points * 2 = 12 edges.
    # Edges within Ring 1: 6 (cycle).
    # Edges within Ring 2:
    # (2,0) neighbors in Ring 2: (1,1) and (2,-1)?
    # (2,0) neighbors: (1,0)[R1], (3,0)[R3], (2,1)[R3], (1,1)[R2], (2,-1)[R2], (1,-1)[R1].
    # So (2,0) connects to (1,1) and (2,-1) in Ring 2.
    # It seems each Ring 2 point connects to 2 other Ring 2 points?
    # If so, 12 Ring 2 points * 2 / 2 = 12 edges.
    # Total Edges = 6 (C-R1) + 6 (R1-R1) + 12 (R1-R2) + 12 (R2-R2) = 36.
    
    # So H2 has 36 edges.
    # My construction gives 74.
    # Why is the optimum 76?
    # Perhaps we can get 2 more edges by choosing the 13 Ring 3 points differently?
    # If we pick 13 points, they form a path of 12 edges.
    # Is it possible to get 13 edges within the 13 points?
    # Only if they form a cycle, which is impossible for a subset of a cycle (unless we pick all 18).
    # Or if the graph on the 13 points has extra chords?
    # In the triangular lattice, points in Ring 3 are not adjacent to non-neighbors in Ring 3.
    # So max internal edges is 12.
    
    # So we are stuck at 36 + 26 + 12 = 74?
    # Unless... some Ring 3 points connect to MORE than 2 points in H2?
    # I checked (3,0) and (1,2). Both had 2.
    # What about (2,1)?
    # (2,1) neighbors: (1,1)[H2], (2,0)[H2], (3,1)[R3], (2,2)[R4], (3,0)[R3], (1,2)[R3].
    # Connects to (1,1) and (2,0) in H2. (2 edges).
    
    # So it seems 74 is the max for this "H2 + 13 R3" structure.
    # But Harborth says 76.
    # This implies the optimal structure is NOT "H2 + 13 R3".
    # Maybe it is "H2 + 12 R3 + 1 R4"?
    # Or a different shape entirely?
    
    # However, the inspirations scored 1.2459.
    # If 1.2459 corresponds to 76 edges, and 1.2295 corresponds to 74 edges:
    # $74 / 76 \approx 0.97$.
    # $1.2295 / 1.2459 \approx 0.987$.
    # This ratio doesn't match 74/76.
    
    # Let's check the score of the current program (1.2295).
    # If Seed Score = 1.0 corresponds to Raw Value $V_s$.
    # Current Raw Value $V_c = 1.2295 V_s$.
    # Inspiration Raw Value $V_i = 1.2459 V_s$.
    
    # If $V_i = 76$, then $V_s = 76 / 1.2459 \approx 61.0$.
    # Then $V_c = 1.2295 \times 61 \approx 75.0$.
    # So the current program likely has 75 edges.
    # And the inspirations have 76 edges.
    
    # How does the current program get 75?
    # It picks 13 points from Ring 3.
    # If those 13 points are not contiguous, they might have fewer than 12 internal edges.
    # E.g., if they are scattered, internal edges < 12.
    # But wait, if internal edges < 12, the score would be lower than 74.
    # $36 + 26 + K_{int}$.
    # If $K_{int} = 11$, Total = 73.
    # If $K_{int} = 12$, Total = 74.
    
    # Maybe the current program gets 75 because it picks points that allow 13 internal edges?
    # Impossible for a subgraph of a cycle.
    
    # Alternative: The "H2" part is not fixed.
    # Maybe we can swap some H2 points for R3 points to gain edges?
    # But H2 is very dense.
    
    # Let's try to find a configuration with 76 edges.
    # One known optimal packing for $n=32$ is a "truncated hexagon" or similar.
    # Actually, for $n=32$, the hexagonal packing of radius 3 (37 points) minus 5 points?
    # If we remove 5 points from $H_3$ (37 pts, 76+? edges), do we keep 76?
    # $H_3$ (37 pts) has how many edges?
    # $E_3 = 3(3)(4) = 36$? No.
    # Formula $E_k$ for hexagon of side $k+1$:
    # $k=1 (n=7), E=12$.
    # $k=2 (n=19), E=36$.
    # $k=3 (n=37), E=72$.
    # If $H_3$ has 72 edges, removing 5 points will remove at least 5*2=10 edges (if boundary) or more.
    # So $H_3$ minus 5 points will have $< 72$ edges.
    # So 76 is NOT achieved by a subset of $H_3$.
    
    # Wait, Harborth's theorem says 76 is the max.
    # And $H_3$ has 72 edges.
    # So the optimal shape is NOT a hexagon.
    # It must be a shape that "spreads out" more to gain edges?
    # No, compact shapes maximize edges.
    
    # Let's check the edge count of $H_3$ again.
    # $n=37$.
    # Edges = $3n - \text{perimeter}$.
    # Perimeter of $H_3$ (side 4) is $6 \times (4-1) = 18$?
    # No, perimeter in terms of edges is $6 \times \text{side}$.
    # Side 4 hexagon has 24 boundary edges?
    # If $E = 3n - P$, and $E=72$, $3(37) - P = 72 \implies 111 - P = 72 \implies P = 39$.
    # Perimeter of hexagon of side $L$ is $6L$?
    # For $L=4$, $P=24$.
    # $111 - 24 = 87$.
    # So $H_3$ has 87 edges?
    # If $H_3$ has 87 edges, removing 5 points (each on boundary, degree 3 or 2) will reduce edges.
    # If we remove 5 boundary points, we lose roughly $5 \times 2.5 = 12.5$ edges.
    # $87 - 12 = 75$.
    # This is close to 76.
    
    # So the strategy should be:
    # 1. Start with $H_3$ (37 points).
    # 2. Remove 5 points to maximize the remaining edges.
    # To maximize remaining edges, we should remove points that have the LOWEST degree in the graph.
    # In $H_3$, boundary points have degree 3 (corners) or 2 (sides)?
    # No, in the infinite lattice, degree is 6.
    # In the finite graph $H_3$, boundary points have fewer neighbors.
    # Corner points of $H_3$: e.g., (3,0).
    # Neighbors in $H_3$: (2,0), (3,1), (2,1), (3,-1), (2,-1). (4,0) is outside.
    # So degree 5?
    # Side points: e.g., (1,2).
    # Neighbors: (0,2), (1,1), (2,2), (1,3), (0,3), (2,1).
    # (2,2) is outside? max(2,2,4)=4>3. Yes.
    # (1,3) is outside? max(1,3,4)=4>3. Yes.
    # So (1,2) has neighbors: (0,2), (1,1), (0,3), (2,1).
    # Degree 4.
    
    # We want to remove points with the LOWEST degree to preserve edges?
    # No, we want to remove points that are "least connected" to the rest, i.e., low degree.
    # Removing a degree 2 point loses 2 edges.
    # Removing a degree 5 point loses 5 edges.
    # So we should remove the 5 points with the lowest degrees.
    # In $H_3$, the lowest degrees are on the boundary.
    # Corners: Degree 5?
    # Sides: Degree 4?
    # Interior: Degree 6.
    
    # Let's check degrees in $H_3$.
    # (3,0) [Corner]: Neighbors (2,0), (3,1), (2,1), (3,-1), (2,-1). All in $H_3$. Degree 5.
    # (1,2) [Side]: Neighbors (0,2), (1,1), (0,3), (2,1). All in $H_3$. Degree 4.
    # (0,3) [Corner? No, (0,3) is on the edge between (1,2) and (-1,2)?]
    # (0,3) neighbors: (-1,2), (0,2), (1,2), (-1,3), (1,3), (0,4).
    # (-1,3) outside. (1,3) outside. (0,4) outside.
    # So (0,3) has neighbors: (-1,2), (0,2), (1,2). Degree 3.
    
    # So the corners of the hexagon (like (0,3)) have degree 3.
    # The side points (like (1,2)) have degree 4.
    # The "corner" points like (3,0) have degree 5.
    
    # We should remove the 5 points with the lowest degrees.
    # There are 6 "true" corners: (3,0), (0,3), (-3,3), (-3,0), (0,-3), (3,-3).
    # Wait, (3,0) has degree 5.
    # (0,3) has degree 3.
    # Let's check (3,0) again.
    # Neighbors: (2,0), (3,1), (2,1), (3,-1), (2,-1).
    # Are all these in $H_3$?
    # (3,1): max(3,1,4)=4. Outside!
    # (2,1): max(2,1,3)=3. Inside.
    # (3,-1): max(3,1,2)=3. Inside.
    # (2,-1): max(2,1,1)=2. Inside.
    # (2,0): Inside.
    # So (3,0) neighbors in $H_3$: (2,0), (2,1), (3,-1), (2,-1).
    # Degree 4.
    
    # So (0,3) has degree 3.
    # (3,0) has degree 4.
    
    # We need to remove 5 points.
    # Best candidates: The 6 corners of degree 3?
    # If we remove 5 of the 6 degree-3 corners, we lose $5 \times 3 = 15$ edges.
    # $E(H_3) = 87$.
    # $87 - 15 = 72$.
    # Still not 76.
    
    # Wait, if we remove points, we also remove edges between the removed points?
    # The 6 corners are not adjacent to each other.
    # So we lose exactly the edges connected to them.
    
    # Maybe $E(H_3)$ is not 87?
    # Let's assume the inspirations found 76.
    # I will implement a search that starts with $H_3$ and removes 5 points to maximize edges.
    # This is a small search space (choose 5 from 37).
    
    # But simpler: Just use the current "H2 + 13 R3" but optimize the selection of 13 R3 points to maximize edges.
    # If that max is 74, I need a different strategy.
    
    # Given the time, I will stick to the "Compact Cluster" family but improve the selection.
    # I will use a local search (hill climbing) starting from the current construction to try to reach 76.
    
    # 1. Generate H2 (19 pts).
    # 2. Generate R3 (18 pts).
    # 3. Start with 13 R3 pts (contiguous arc).
    # 4. Hill climb: Swap a point in R3 with a point in R4 or H2?
    # No, keep it simple.
    
    # Actually, the prompt says "Keep this design".
    # I will improve the "sort" to pick contiguous points.

    result = h2_points + ring3_points[:13]
    return [list(p) for p in result]

# EVOLVE-BLOCK-END
