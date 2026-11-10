"""Program for the no-isosceles set task using a greedy randomized construction."""
import time
import random
import numpy as np
from collections import defaultdict

N = 64
DEADLINE_MARGIN = 5.0  # seconds to spare before the 330s limit

def construct():
    """
    Construct a set of points in [0, N)^2 with no three forming an isosceles triangle.
    
    Strategy: Greedy randomized construction.
    1. Maintain a set of selected points.
    2. For each candidate point (randomly ordered), check if adding it creates an isosceles triangle with any two existing points.
    3. An isosceles triangle with vertices A, B, C exists if:
       - |AB| == |AC| (B and C equidistant from A)
       - |BA| == |BC| (A and C equidistant from B)
       - |CA| == |CB| (A and B equidistant from C)
    4. To check efficiently, for a new point P, for each existing point Q, we need to ensure no other existing point R is at the same distance from Q as P is.
       Wait, that's not quite right. 
       
       Let's re-evaluate the condition.
       We have a set S. We want to add P.
       For any two points A, B in S, adding P creates an isosceles triangle if:
       - dist(P, A) == dist(P, B)  -> P is on the perpendicular bisector of AB
       - dist(A, P) == dist(A, B)  -> P is on the circle centered at A with radius dist(A,B)
       - dist(B, P) == dist(A, B)  -> P is on the circle centered at B with radius dist(A,B)
       
       Actually, the simplest way to check if adding P to S creates an isosceles triangle is:
       For every pair (A, B) in S, check if dist(P,A) == dist(P,B) OR dist(A,P) == dist(A,B) OR dist(B,P) == dist(A,B).
       This is O(k^2) per insertion where k is the size of S. With k ~ 100, k^2 = 10000. Total candidates ~ 4096. 4096 * 10000 = 40M operations. Might be too slow in Python.
       
       Alternative check:
       Maintain for each point A in S, a set of distances to other points in S.
       When adding P, for each A in S:
         d = dist(P, A)
         If d is already in the distance set of A, then there exists B in S such that dist(A,B) == d.
         Then dist(P,A) == dist(B,A), so P, A, B form an isosceles triangle with apex A.
         So we must check: for each A in S, is dist(P,A) in distances[A]?
         
       This is O(k) per insertion, where k is the size of S.
       Total work: sum_{k=1}^{K} N^2 * k = N^2 * K(K+1)/2.
       If K=100, N^2=4096, 4096 * 5050 = ~20M. This is feasible in Python if optimized, or maybe tight.
       
       We can optimize by pre-computing distances or using numpy.
       But let's try the O(k) check first.
       
       We also need to handle the case where P, A, B form an isosceles triangle with apex P or apex B.
       Wait, the condition "dist(P,A) in distances[A]" covers apex A.
       What about apex P? dist(A,B) == dist(A,P) is the same as dist(A,P) == dist(A,B).
       If dist(A,P) is in distances[A], it means there is some B such that dist(A,B) == dist(A,P).
       This covers apex A.
       Does it cover apex P?
       If dist(A,B) == dist(P,B), then dist(P,B) is in distances[B].
       So if we check for all A in S: "is dist(P,A) in distances[A]?", we cover:
       - Apex A: dist(P,A) == dist(B,A) for some B.
       - Apex P: dist(A,B) == dist(P,B) -> dist(P,B) == dist(A,B). This means dist(P,B) is in distances[B].
         So when we check A=B, we see if dist(P,B) is in distances[B].
       - Apex B: dist(A,P) == dist(B,P). This means dist(A,P) is in distances[B]? No.
         dist(A,P) == dist(B,P) means P is equidistant from A and B.
         This is not directly covered by "dist(P,A) in distances[A]".
         If dist(A,P) == dist(B,P), then dist(A,P) = d.
         Is d in distances[A]? Not necessarily.
         Is d in distances[B]? Not necessarily.
         
       Let's re-read the condition for isosceles triangle.
       Triangle PAB is isosceles if:
       1. PA == PB (apex P)
       2. AP == AB (apex A)
       3. BP == BA (apex B)
       
       My check "for each A in S, is dist(P,A) in distances[A]?" covers:
       - Case 2: AP == AB. Here B is the point in S such that dist(A,B) = dist(P,A).
       - Case 3: BP == BA. Here A is the point in S such that dist(B,A) = dist(P,B).
         When we iterate A over S, we check dist(P,A) in distances[A].
         If we let A be the apex B from the triangle notation, we check dist(P,B) in distances[B].
         If dist(P,B) == dist(A,B), then dist(P,B) is in distances[B]. So case 3 is covered.
       
       What about Case 1: PA == PB?
       This means P is on the perpendicular bisector of AB.
       This is NOT covered by the distance set check.
       
       So I need to also check if P is equidistant from any pair (A,B) in S.
       This requires checking all pairs (A,B) in S, which is O(k^2).
       
       However, note that if PA == PB, then for the point A, dist(P,A) = d.
       For the point B, dist(P,B) = d.
       So d is in distances[A] (if there is another point C with dist(A,C)=d) and d is in distances[B].
       But if there is no such C, d is not in distances[A].
       
       So the distance set check is insufficient.
       
       We must check for PA == PB explicitly.
       This means for each pair (A,B) in S, check if dist(P,A) == dist(P,B).
       This is O(k^2).
       
       Total complexity O(N^2 * K^2).
       If K=50, N^2=4096, 4096 * 2500 = 10M. Feasible.
       If K=100, 4096 * 10000 = 40M. Might be tight in 300s.
       
       Let's implement the O(k^2) check and see.
       
    """
    start_time = time.time()
    
    # Precompute all grid points
    grid_points = [(x, y) for x in range(N) for y in range(N)]
    
    # Shuffle for randomness
    random.shuffle(grid_points)
    
    selected = []
    # distances[A] is a set of squared distances from A to all other selected points
    distances = {}
    
    for p in grid_points:
        # Check deadline
        if time.time() - start_time > 330 - DEADLINE_MARGIN:
            break
            
        px, py = p
        valid = True
        
        # Check 1: For each A in selected, is there a B in selected such that dist(P,A) == dist(B,A)?
        # This is equivalent to: is dist(P,A)^2 in distances[A]?
        for a in selected:
            ax, ay = a
            d2 = (px - ax)**2 + (py - ay)**2
            if d2 in distances[a]:
                valid = False
                break
        
        if not valid:
            continue
            
        # Check 2: For each pair (A,B) in selected, is dist(P,A) == dist(P,B)?
        # This is equivalent to: is P on the perpendicular bisector of AB?
        # We can check this by comparing squared distances.
        k = len(selected)
        for i in range(k):
            ax, ay = selected[i]
            dax2 = (px - ax)**2 + (py - ay)**2
            for j in range(i+1, k):
                bx, by = selected[j]
                dbx2 = (px - bx)**2 + (py - by)**2
                if dax2 == dbx2:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            selected.append(p)
            # Update distances
            distances[p] = set()
            for a in selected[:-1]:  # all previously selected
                ax, ay = a
                d2 = (px - ax)**2 + (py - ay)**2
                distances[p].add(d2)
                distances[a].add(d2)
            
            # Optional: early stop if we think we can't do better? No, keep going.
    
    return selected

# Test locally
if __name__ == "__main__":
    import time
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Found {len(res)} points in {end-start:.2f}s")
    # Verify no isosceles triangles
    n = len(res)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = res[i], res[j], res[k]
                d12 = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
                d23 = (p2[0]-p3[0])**2 + (p2[1]-p3[1])**2
                d13 = (p1[0]-p3[0])**2 + (p1[1]-p3[1])**2
                if d12 == d23 or d23 == d13 or d13 == d12:
                    print(f"Isosceles found: {p1}, {p2}, {p3}")
                    exit(1)
    print("Valid!")
