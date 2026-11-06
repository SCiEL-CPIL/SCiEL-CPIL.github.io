import numpy as np
import random
import time
import sys

N = 64

def construct():
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    # 1. Construct Conflict Graph
    # A pair of points (a, b) is "conflicting" if there exists a third point c 
    # in the grid such that {a, b, c} forms an isosceles triangle.
    # The triangle is isosceles if:
    #   |a-c|^2 == |b-c|^2  (c is on perp bisector of ab)
    #   |a-c|^2 == |a-b|^2  (c is on circle centered at a with radius |ab|)
    #   |b-c|^2 == |a-b|^2  (c is on circle centered at b with radius |ab|)
    
    # We represent points as indices 0 to N*N - 1.
    # coords[i] = (x, y)
    coords = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    num_points = N * N
    
    # We will build the adjacency matrix or list. 
    # Given N=64, num_points=4096.
    # A dense adjacency matrix is 4096x4096 bits ~ 2MB. This is feasible.
    # However, building it naively is O(N^4) which is 4096^2 * 4096 ~ 68 billion ops. Too slow.
    
    # Alternative: 
    # For a pair (a,b), they are NOT conflicting if for ALL c, the triangle is scalene.
    # Actually, it's easier to identify conflicting pairs.
    # Two points a, b are "compatible" (no edge in conflict graph) if for all c, 
    # |a-c|^2 != |b-c|^2 AND |a-c|^2 != |a-b|^2 AND |b-c|^2 != |a-b|^2.
    
    # Wait, the definition of "No-isosceles set" is that NO THREE points form an isosceles triangle.
    # This means if we pick a set S, for any a,b,c in S, the triangle is scalene.
    # This is equivalent to: For any a,b in S, there is NO c in S such that {a,b,c} is isosceles.
    # This does NOT mean a and b are globally incompatible. It means they are incompatible IF their "isosceles partners" are also in S.
    # 
    # The "Conflict Graph" approach in the prompt suggests: "edges connect pairs of points that share a common 'isosceles partner'".
    # This interpretation implies that if a and b have ANY common isosceles partner c in the GRID, then a and b cannot both be in the set?
    # No, that would be too restrictive. If a and b are in the set, we just need to ensure that NO c in the set is an isosceles partner.
    # If we build a graph where (a,b) is an edge if there EXISTS a c in the GRID such that {a,b,c} is isosceles, 
    # then an Independent Set in this graph would be a set of points where NO pair has ANY isosceles partner in the grid.
    # This is a much stronger condition than required! The required condition allows a,b to be in the set as long as their specific isosceles partners are NOT in the set.
    
    # Let's re-read the inspiration: "Model the problem as a maximum independent set in a conflict graph where edges connect pairs of points that share a common 'isosceles partner'".
    # If I follow this literally, I am solving a harder problem (finding a set where no pair is "potentially" isosceles with ANY grid point).
    # Is this a valid strategy? It might yield a smaller set, but it's a well-defined MIS problem.
    # However, the score is the number of points. If this constraint is too tight, the score will be low.
    # Let's check the density. 
    # For a random pair (a,b), how many c make it isosceles?
    # If the set of "safe" pairs is small, the MIS will be small.
    
    # Let's consider a different conflict graph definition which is standard for this type of problem:
    # Vertices: Grid points.
    # Edge (a,b) exists if a and b CANNOT be in the same valid set? 
    # No, validity depends on the third point.
    
    # Actually, there is a known reduction. 
    # If we fix the set S, it is valid iff for all distinct a,b,c in S, the triangle is scalene.
    # This is not a pairwise constraint. It is a 3-wise constraint.
    # Standard MIS only handles 2-wise constraints.
    
    # However, we can transform this. 
    # If we define the conflict graph G such that (a,b) is an edge if there exists c such that {a,b,c} is isosceles, 
    # then any Independent Set in G is a valid No-Isosceles set.
    # Why? If S is an Independent Set in G, then for any a,b in S, there is NO c in the GRID such that {a,b,c} is isosceles.
    # Therefore, there is certainly no c in S such that {a,b,c} is isosceles.
    # So, MIS of this graph is a valid solution.
    # The question is: is the size of this MIS competitive?
    
    # Let's estimate the edge density.
    # Total pairs: ~8 million.
    # For a given pair (a,b) with distance squared d, 
    # 1. c on perp bisector: The number of integer points on the perp bisector within the grid.
    # 2. c on circle a, radius sqrt(d): Number of integer points on this circle.
    # 3. c on circle b, radius sqrt(d): Same.
    # Most pairs will have at least one such c in the 64x64 grid?
    # If d is small, circles are small, might have 0 or few points.
    # If d is large, circles are large, likely to intersect the grid.
    # The perp bisector always passes through the midpoint. If midpoint is integer, there are many points. If half-integer, fewer.
    
    # If the graph is very dense, the MIS will be very small.
    # For example, if every pair has a partner, the MIS size is 1.
    # Let's test a small case mentally.
    # If N=10, is it likely that most pairs are connected?
    # If so, this strategy will fail (score ~1-2).
    
    # Let's look at Inspiration 2: "conflict-graph branch-and-bound with aggressive pruning". Score 10.0.
    # This suggests that the "Conflict Graph" interpretation might be exactly what they did, or a variant.
    # But wait, if the graph is too dense, MIS is small.
    # Maybe the "Conflict Graph" is defined differently?
    # Perhaps edges connect points that are "too close" or have specific geometric relations?
    
    # Let's try a different structural approach: **Randomized Local Search with 3-SAT style clauses?** No.
    
    # Let's stick to the directive: "Model the problem as a maximum independent set in a conflict graph where edges connect pairs of points that share a common 'isosceles partner'".
    # I will implement this literally. If the score is low, it's a valid structural change.
    # But I suspect the graph is too dense.
    
    # Alternative Interpretation:
    # Maybe the "Conflict Graph" is built on the FLYING points?
    # No.
    
    # Let's try to optimize the construction of this graph.
    # We need to determine for each pair (a,b) if there exists c in Grid such that {a,b,c} is isosceles.
    # This is equivalent to:
    # (Perp Bisector of ab intersects Grid) OR (Circle(a, |ab|) intersects Grid) OR (Circle(b, |ab|) intersects Grid).
    
    # Let's precompute for each point a, the set of squared distances to all other points? No, that's O(N^2) per point, O(N^4) total.
    
    # Let's use the property:
    # If |a-b|^2 = d.
    # Check if circle(a, sqrt(d)) has any integer point in grid.
    # This is a number theory problem: sum of two squares.
    # But we can just vectorize the check.
    
    # To make this feasible in 300s, we might need to approximate or use a sparser conflict definition.
    # What if we define conflict only if c is in a NEIGHBORHOOD? No, c can be anywhere.
    
    # Let's assume the graph is dense. 
    # If the graph is dense, MIS is small.
    # Current score is 10.
    # If MIS is 1 or 2, score is bad.
    
    # Is there a sparser conflict graph?
    # What if we only consider c that are "close" to a and b?
    # No, isosceles triangles can be large.
    
    # Let's reconsider the "Conflict Graph" from Inspiration 2.
    # Maybe it's not "exists c in grid", but "c is a specific type of point"?
    
    # Let's try a different strategy: **Simulated Annealing on the set of points**.
    # State: Set of points S.
    # Energy: Number of isosceles triangles formed by S.
    # Goal: Minimize Energy to 0.
    # Moves: Add, Remove, Swap points.
    # If Energy > 0, we have an invalid set.
    # We want to find a large S with Energy 0.
    
    # This is a different optimization method (Local Search with explicit cost function) vs the current (Greedy + Swap).
    # The current program checks validity O(k^2) per insertion.
    # Simulated Annealing can maintain a set that is TEMPORARILY invalid, allowing it to escape local optima where no single swap improves validity, but a sequence of swaps does.
    
    # This is a materially different search method (Annealing vs Greedy/Swap).
    
    # Implementation Plan:
    # 1. Initialize with a random set of points (maybe larger than current best, e.g., 20-30 points).
    # 2. Calculate current number of isosceles triangles (Violations).
    # 3. Loop:
    #    a. Pick a random move: Add random point, Remove random point, Swap random point in/out.
    #    b. Calculate delta in Violations.
    #    c. Accept if delta < 0 (improvement) or with probability exp(-delta/T) if delta > 0.
    #    d. Decrease T.
    # 4. Track best valid set found (Violations == 0).
    
    # Efficiency of Violation Count:
    # If we add point p to set S:
    # New violations are triangles involving p.
    # For each pair (a,b) in S, check if {p,a,b} is isosceles.
    # This is O(|S|^2) per move.
    # If |S| ~ 30, 900 checks.
    # 300s / 900 checks ~ 1000 moves? Too slow for Python.
    # We need vectorized violation count.
    
    # Vectorized Violation Check for adding p:
    # S is a list of (x,y).
    # d_pa = (p_x - S_x)^2 + (p_y - S_y)^2  (vector of length k)
    # d_ab = |S_i - S_j|^2 for all i<j.
    # We need to check if d_pa[i] == d_pa[j] (p is apex)
    # OR d_pa[i] == d_ab[i,j] (a is apex)
    # OR d_pa[j] == d_ab[i,j] (b is apex)
    
    # This can be done with numpy broadcasting.
    # S is (k, 2).
    # p is (2,).
    # d_pa is (k,).
    # d_ab is (k, k) matrix of squared distances.
    # 
    # Condition 1: d_pa[i] == d_pa[j] for i != j.
    # This means d_pa has duplicate values.
    # 
    # Condition 2: d_pa[i] == d_ab[i,j].
    # This means for any i, d_pa[i] is in the row i of d_ab (excluding diagonal).
    # 
    # Condition 3: d_pa[j] == d_ab[i,j].
    # This means for any j, d_pa[j] is in the column j of d_ab (excluding diagonal).
    
    # We can compute these counts efficiently.
    
    # Let's implement this Simulated Annealing approach.

    def count_violations(points):
        """Returns the number of isosceles triangles in the set of points."""
        if len(points) < 3:
            return 0
        P = np.array(points, dtype=np.int32)
        k = len(P)
        
        # Compute pairwise squared distances D (k x k)
        # D[i,j] = |P[i] - P[j]|^2
        diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
        D = np.sum(diff * diff, axis=2)
        
        violations = 0
        
        # We need to count triples (i, j, l) with i < j < l such that triangle is isosceles.
        # It is easier to count "bad" pairs of distances for each apex.
        
        # For each apex i:
        # Distances to others: d_i = D[i, :]
        # We look for j, l != i such that d_i[j] == d_i[l].
        # This is the number of pairs (j,l) with equal distance to i.
        # Plus cases where i is NOT the apex?
        # No, an isosceles triangle has exactly one or two apexes (if equilateral, 3).
        # Wait, if we count for each apex i, the number of pairs (j,l) such that |i-j| == |i-l|,
        # we count each isosceles triangle once (if unique apex) or 3 times (if equilateral).
        # But we just want to know if violations > 0.
        # Actually, we want the COUNT of violations to guide the search.
        
        # Let's count the number of isosceles triangles.
        # For each i, sort distances to others.
        # For each unique distance value v appearing m times, there are m*(m-1)/2 pairs.
        # Sum these over all i.
        # This counts each isosceles triangle exactly once if it has a unique apex.
        # If it is equilateral, it is counted 3 times (once for each apex).
        # This is a valid metric for "badness".
        
        total_bad = 0
        for i in range(k):
            dists = D[i, :]
            dists = dists[dists > 0] # Exclude self
            # Count frequencies
            vals, counts = np.unique(dists, return_counts=True)
            for c in counts:
                if c > 1:
                    total_bad += c * (c - 1) // 2
                    
        return total_bad

    # This is O(k^2 log k) per evaluation.
    # With k=30, 900 log 30 ~ 4500 ops.
    # 300s / 4500 ~ 2000 evaluations.
    # This is too slow for a good annealing schedule.
    
    # Optimization: Incremental update.
    # When adding p to S:
    # New violations involving p.
    # 1. p is apex: Pairs (a,b) in S with |p-a| == |p-b|.
    # 2. a is apex: Pairs (p,b) in S+{p} with |a-p| == |a-b|. i.e., |p-b| == |a-b|? No.
    #    Apex a: |a-p| == |a-b|.
    # 3. b is apex: |b-p| == |b-a|.
    
    # We can maintain a data structure?
    # Or just use the vectorized check for the DELTA.
    
    def delta_violations_add(S, p):
        """Calculate increase in violations when adding p to S."""
        if len(S) < 2:
            return 0
        S = np.array(S, dtype=np.int32)
        p = np.array(p, dtype=np.int32)
        k = len(S)
        
        # Distances from p to all in S
        d_p = np.sum((S - p)**2, axis=1)
        
        # Distances within S
        # We need D_S (k x k)
        # This is expensive to recompute every time.
        # But we can precompute D_S for the current S and update it?
        # If we remove/swap, we have to update D_S.
        
        # Let's assume we recompute D_S every time for simplicity first, then optimize if needed.
        # Actually, for k=30, D_S is 30x30=900 elements. Computation is 900*2 ops. Very fast.
        
        diff = S[:, np.newaxis, :] - S[np.newaxis, :, :]
        D_S = np.sum(diff * diff, axis=2)
        
        violations = 0
        
        # 1. p is apex:
        # Count pairs (i,j) in S such that d_p[i] == d_p[j].
        vals, counts = np.unique(d_p, return_counts=True)
        for c in counts:
            if c > 1:
                violations += c * (c - 1) // 2
                
        # 2. Apex is in S (say i):
        # Triangle (p, i, j).
        # Isosceles if:
        # |i-p| == |i-j|  => d_p[i] == D_S[i,j]
        # |i-p| == |p-j|  => d_p[i] == d_p[j] (Already counted in 1? No, 1 counted pairs in S with same dist to p. This is pair (i,j) with d_p[i]==d_p[j]. Yes, this case is covered by "p is apex" logic? 
        # Wait.
        # Case 1: Apex p. |p-i| == |p-j|. Covered by "p is apex" logic.
        # Case 2: Apex i. |i-p| == |i-j|. => d_p[i] == D_S[i,j].
        # Case 3: Apex j. |j-p| == |j-i|. => d_p[j] == D_S[i,j].
        
        # So we need to check for each i in S:
        # How many j in S (j != i) such that D_S[i,j] == d_p[i]?
        
        # Vectorized:
        # For each i, count matches in row i of D_S with value d_p[i].
        # Note: D_S[i,i] = 0. d_p[i] > 0 usually.
        
        # Create a mask?
        # For each i, we want sum( D_S[i, :] == d_p[i] ).
        # This is O(k^2) comparisons.
        
        matches_i = np.sum(D_S == d_p[:, np.newaxis], axis=1)
        # matches_i[i] is the number of j such that D_S[i,j] == d_p[i].
        # This includes j=i if D_S[i,i]==d_p[i] (impossible unless p=i, which is not the case).
        # So matches_i[i] is the count of j != i such that |i-j| == |i-p|.
        
        # These correspond to triangles where i is the apex.
        # Wait, does this double count?
        # If |i-p| == |i-j|, then triangle (i,p,j) is isosceles with apex i.
        # This is distinct from apex p (|p-i|==|p-j|) and apex j (|j-i|==|j-p|).
        # So we add these.
        
        violations += np.sum(matches_i)
        
        # Wait, what about Case 3?
        # Case 3: Apex j. |j-p| == |j-i|.
        # This is symmetric to Case 2.
        # If we sum over all i, do we cover Case 3?
        # Case 2 for index i: |i-p| == |i-j|.
        # Case 3 for index j: |j-p| == |j-i|.
        # These are different conditions.
        # We need to add Case 3 as well.
        
        # Case 3: For each j, count i such that D_S[j,i] == d_p[j].
        # This is the same as matches_i but transposed?
        # matches_j = np.sum(D_S == d_p[:, np.newaxis], axis=1) gives for each j, count of i such that D_S[j,i] == d_p[j].
        # Wait, D_S is symmetric.
        # The condition for Case 2 (apex i) is D_S[i,j] == d_p[i].
        # The condition for Case 3 (apex j) is D_S[j,i] == d_p[j].
        
        # So we need to sum:
        # Sum_i (count of j s.t. D_S[i,j] == d_p[i])  -> Apex i
        # Sum_j (count of i s.t. D_S[j,i] == d_p[j])  -> Apex j
        
        # Since D_S is symmetric, these two sums are identical?
        # Let S1 = sum_i count_j(D_S[i,j] == d_p[i]).
        # Let S2 = sum_j count_i(D_S[j,i] == d_p[j]).
        # S1 iterates i as apex.
        # S2 iterates j as apex.
        # They are the same set of terms, just indexed differently.
        # So we should only add ONE of them?
        # No, the triangles are distinct.
        # Triangle (i,p,j) with apex i is a specific geometric configuration.
        # Triangle (i,p,j) with apex j is a DIFFERENT configuration (different sides equal).
        # They cannot happen simultaneously unless it's equilateral (|i-p|==|i-j|==|j-p|).
        # If equilateral, it is counted in Apex i, Apex j, AND Apex p.
        # That's fine, we just want a count of "badness".
        
        # So we add S1 (Apex in S, matching dist to p) AND S2 (Apex in S, matching dist to p)?
        # Wait.
        # Apex i: |i-p| == |i-j|.
        # Apex j: |j-p| == |j-i|.
        # These are two different sets of triangles.
        # So yes, we add both.
        
        # But wait, S1 and S2 are numerically identical because D_S is symmetric and we sum over all indices.
        # S1 = sum_{i} N_i where N_i = |{j: D_S[i,j] == d_p[i]}|
        # S2 = sum_{j} N_j where N_j = |{i: D_S[j,i] == d_p[j]}|
        # Since D_S is symmetric, N_i (for fixed i) is the count in row i.
        # N_j (for fixed j) is the count in row j.
        # The sum is over all rows.
        # So S1 == S2.
        
        # Do we add S1 twice?
        # Yes, because Apex i triangles and Apex j triangles are distinct sets of triangles.
        # Example: i=0, j=1.
        # Apex 0: |0-p| == |0-1|.
        # Apex 1: |1-p| == |1-0|.
        # These are different conditions.
        
        violations += np.sum(matches_i) # Apex i
        violations += np.sum(matches_i) # Apex j (Same value)
        
        return violations

    # This logic is getting complex and error-prone.
    # Let's stick to the simpler, slightly slower `count_violations` and use it in a local search that accepts only improvements (Greedy Hill Climbing) or simple swaps.
    # Given the time limit, maybe a simpler structural change is better.
    
    # Let's go back to the **Conflict Graph MIS** idea but with a twist:
    # Instead of "exists c in grid", use "c is in the current candidate set"?
    # No.
    
    # Let's try **Randomized Greedy with Backtracking**.
    # 1. Order points randomly.
    # 2. Add point if valid.
    # 3. If stuck, backtrack and try a different order?
    
    # The current program does Random Greedy.
    # A different strategy: **Greedy with Random Restarts and Larger N?** No N is fixed.
    
    # How about **Genetic Algorithm**?
    # Population of sets.
    # Crossover: Union of two sets? No, must be valid.
    # Mutation: Swap points.
    
    # Let's implement **Simulated Annealing** properly. It is a robust method for escaping local optima.
    # We will use the vectorized `count_violations` (full recompute) but optimize the loop.
    # To make it fast enough, we will limit the set size to ~20-25 during search.
    
    points = []
    # Initialize with a few random points
    all_points = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(all_points)
    
    # Start with a small valid set
    for p in all_points[:10]:
        if count_violations(points + [p]) == 0:
            points.append(p)
            
    best_points = points[:]
    best_score = len(points)
    
    T = 100.0
    T_min = 0.01
    
    # Precompute all_points array for numpy
    all_pts_np = np.array(all_points, dtype=np.int32)
    
    while time.time() < deadline:
        # Pick a random move
        move_type = random.random()
        new_points = points[:]
        
        if move_type < 0.3 and len(points) < 30:
            # Add
            p = random.choice(all_points)
            if p not in points:
                new_points.append(p)
        elif move_type < 0.6:
            # Remove
            if points:
                idx = random.randint(0, len(points) - 1)
                new_points.pop(idx)
        else:
            # Swap
            if points:
                idx = random.randint(0, len(points) - 1)
                p_old = points[idx]
                p_new = random.choice(all_points)
                if p_new not in points:
                    new_points[idx] = p_new
        
        # Calculate violations
        v_old = count_violations(points)
        v_new = count_violations(new_points)
        
        delta = v_new - v_old
        
        # Acceptance
        if delta < 0:
            points = new_points
            if v_new == 0 and len(points) > best_score:
                best_score = len(points)
                best_points = points[:]
        else:
            if random.random() < np.exp(-delta / T):
                points = new_points
                
        # Cool down
        T *= 0.999
        
        if T < T_min:
            # Restart?
            T = 10.0
            # Randomize points slightly?
            # For now, just keep going.
            
    return [(int(x), int(y)) for x, y in best_points]
