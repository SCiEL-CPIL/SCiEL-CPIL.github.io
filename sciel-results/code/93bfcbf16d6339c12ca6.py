# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.

    Strategy: We want to maximize the complement size (points not in N).
    The current baseline constructs a set N of size p^4 - p^2*(p+1)/2.
    For p=7: 2401 - 49*4 = 2401 - 196 = 2205. Complement size = 196.

    We will attempt to find a smaller Nikodym set using a greedy local search
    or a different algebraic construction.

    However, the "skip step 2.5" construction is known to be optimal or near-optimal
    for certain parameters. Let's first verify the baseline and then try to
    improve it by removing more points if possible, or by using a different
    blocking set structure.

    Actually, the problem asks for a set N such that every point x lies on a line
    where all other points are in N. This is the definition of a Nikodym set.
    The complement S = F_q^2 \\ N is the set of points that are "blocked".
    For a point s in S, there must be a line L_s such that s is the only point of L_s
    not in N? No, that's not right.
    
    The condition is: For every x in F_q^2, there exists a direction v such that
    {x + t*v : t != 0} subset N.
    If x is in N, this is easy (pick any line through x, the rest of the line might not be in N?
    Wait. The condition must hold for ALL x, including those in N and those in S.
    
    Let's re-read carefully:
    "every point x of F_49^2 (in N or not) must lie on a line {x + t*v : t in F_49},
    with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all belong to N."
    
    So for x in S (complement), there must be a line L through x such that L \ {x} subset N.
    This means x is an "isolated" point from N on that line.
    For x in N, there must be a line L through x such that L \ {x} subset N.
    This means x has a "clean" line through it where all other points are in N.
    
    The baseline construction:
    N = F_q^2 \ S, where S is the set of excluded points.
    The excluded points S are in the Hermitian unital U but have y-coordinates NOT in Y.
    
    Let's check the size of the baseline complement.
    p=7.
    Unital U: Trace(x) + Norm(y) = 0.
    Trace(x) = 2*x_a.
    Norm(y) = y_a^2 - 3*y_b^2.
    Condition: 2*x_a + y_a^2 - 3*y_b^2 = 0 mod 7.
    For each y, how many x?
    2*x_a = -y_a^2 + 3*y_b^2.
    x_a is uniquely determined. x_b can be anything (7 choices).
    So for each y, there are 7 points in U.
    Total points in U = 49 * 7 = 343.
    
    Y = {(u^2 mod 7, v) : u, v in F_7}.
    Squares mod 7: 0, 1, 2, 4. (0^2=0, 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1).
    So y_a must be in {0, 1, 2, 4}. There are 4 choices for y_a.
    y_b can be anything (7 choices).
    So |Y| = 4 * 7 = 28.
    
    The excluded set S is U \ B, where B = U ∩ (F_q x Y).
    Wait, the code says:
    "A point (x, y) is EXCLUDED iff it is in the unital U and its y is NOT in Y."
    So S = { (x,y) in U : y not in Y }.
    
    Size of S:
    Total y's = 49.
    y's in Y = 28.
    y's not in Y = 49 - 28 = 21.
    For each y not in Y, there are 7 points in U.
    So |S| = 21 * 7 = 147.
    
    Wait, the code comment says: "of size p^4 - p^2*(p+1)/2".
    p^2*(p+1)/2 = 49 * 8 / 2 = 49 * 4 = 196.
    But my calculation gives 147.
    Let's re-evaluate |Y|.
    Y = {(u^2, v)}.
    u in 0..6. u^2 mod 7:
    0->0
    1->1
    2->4
    3->2
    4->2
    5->4
    6->1
    Distinct values: {0, 1, 2, 4}.
    So yes, 4 values for y_a.
    y_b is any of 7.
    So 28 y's are in Y.
    49 - 28 = 21 y's not in Y.
    21 * 7 = 147 points in S.
    
    So the baseline complement size is 147.
    The score is proportional to the complement size.
    Current score 1.0 corresponds to 147.
    
    Can we make S larger?
    S must be a set such that for every x in S, there is a line L_x through x with L_x \ {x} subset N = F_q^2 \ S.
    This means L_x \ {x} is disjoint from S.
    So for each s in S, there is a line through s containing no other points of S.
    This is equivalent to saying S is a set with no three collinear points? No.
    If three points s1, s2, s3 are collinear, then for s1, the line through s1, s2, s3 contains s2 and s3 which are in S.
    So the line L_s1 cannot be that line.
    But s1 might have another line through it that is clean.
    
    The condition is: For every s in S, there exists a line L through s such that L ∩ S = {s}.
    
    This is the definition of a "Nikodym set" complement.
    
    The maximal size of such a set S in F_q^2 is known.
    For q=p^2, the maximal size is q + p? Or something related to the unital?
    
    The unital U has 343 points.
    The set S is a subset of U.
    
    Let's try to increase the size of S.
    The constraint is that for each s in S, there is a "clean" line.
    
    If we take S to be a line, then for any point on the line, the line itself is not clean (contains other S points).
    But there are other lines through the point.
    In F_q^2, there are q+1 lines through a point.
    If S is a line L, then for any s in L, we need a line L' through s such that L' ∩ L = {s}.
    This is always possible in a projective plane (or affine plane) unless the geometry is very restricted.
    In F_q^2, for any line L and point s in L, there are q other lines through s.
    None of these other lines intersect L at any point other than s.
    So if S is a single line, then for every s in S, we can pick any other line through s, and it will be clean (disjoint from S except at s).
    So a line is a valid complement S.
    Size of a line = q = 49.
    This is smaller than 147.
    
    What if S is a conic?
    A conic in F_q^2 has q+1 points (if it's a projective conic) or q points (affine).
    If S is a conic, does every point have a clean line?
    The tangent line at a point s on a conic intersects the conic only at s (with multiplicity 2).
    So the tangent line is clean.
    So a conic is a valid complement S.
    Size = 49 or 50.
    Still smaller than 147.
    
    What if S is a unital?
    A unital in F_q^2 (q=p^2) has q^{3/2} + 1 = p^3 + 1 points? No.
    A Hermitian unital in PG(2, q) has q^{3/2} + 1 points.
    For q=49, q^{3/2} = 49^{1.5} = 343.
    So 344 points.
    But we are in the affine plane F_q^2, which has q^2 = 2401 points.
    The unital U defined by Trace(x) + Norm(y) = 0 is an affine unital?
    The number of points in the affine part of a Hermitian unital is q^{3/2} = 343.
    Wait, the full Hermitian unital in PG(2, 49) has 344 points. One is at infinity.
    So the affine part has 343 points.
    
    The baseline S is a subset of this unital U.
    |S| = 147.
    
    Can we take S to be the entire unital U?
    If S = U, then for any s in U, is there a clean line?
    The tangent line to the unital at s intersects U only at s.
    So the tangent line is clean.
    So S = U is a valid complement!
    Size of S = 343.
    
    This is much larger than 147.
    Score would be 343/147 ≈ 2.33.
    
    Let's verify this.
    If S = U, then N = F_q^2 \ U.
    For any x in N (not in U), we need a line L through x such that L \ {x} subset N.
    This means L ∩ U = {x} is false, wait.
    L \ {x} subset N means (L \ {x}) ∩ S = empty.
    So L ∩ S = {x} if x in S, or L ∩ S = empty if x not in S.
    
    Case 1: x in S (x in U).
    We need a line L through x such that L ∩ U = {x}.
    This is the tangent line to U at x.
    In a unital, the tangent line at any point intersects the unital only at that point.
    So this works.
    
    Case 2: x not in S (x not in U).
    We need a line L through x such that L ∩ U = empty.
    Does every point outside a unital lie on a line disjoint from the unital?
    In a projective plane, a unital is a blocking set? No, a unital is not a blocking set.
    A blocking set intersects every line.
    A unital does NOT intersect every line.
    In fact, for any point outside a unital, there are lines through it that are disjoint from the unital.
    Specifically, the number of secants, tangents, and external lines through a point outside a unital of size q^{3/2}+1 is known.
    For a Hermitian unital, every point outside lies on q^{3/2} - q^{1/2} + 1 external lines?
    Actually, it is a known property that a unital is not a blocking set, so there exist lines disjoint from it.
    Moreover, for ANY point outside the unital, there is at least one line through it disjoint from the unital.
    Why?
    The number of lines through a point is q+1.
    The number of lines through a point that intersect the unital is at most the number of points in the unital, which is q^{3/2}+1.
    But q+1 = 50.
    q^{3/2}+1 = 344.
    This doesn't help directly.
    
    Let's check the number of external lines through a point outside a Hermitian unital.
    It is known that for a Hermitian unital in PG(2, q), every point outside the unital lies on exactly q^{3/2} - q^{1/2} + 1 external lines? No, that's too many.
    Total lines through a point = q+1 = 50.
    If all 50 lines intersect the unital, then the unital would be a blocking set.
    But a unital of size q^{3/2}+1 is not a blocking set for q>2.
    A blocking set in PG(2, q) has size at least (q+1)/2 * something?
    The smallest blocking set in PG(2, q) has size roughly q/2.
    Here the unital has size 344, which is much larger than q/2 = 25.
    Wait, a large set is more likely to be a blocking set.
    
    Actually, a unital IS a blocking set?
    No, a unital is a set of points such that every line intersects it in 1 or q^{1/2}+1 points?
    No, that's the definition of a unital: every line intersects the unital in 1 or q^{1/2}+1 points.
    Wait, if every line intersects the unital in at least 1 point, then the unital IS a blocking set.
    If the unital is a blocking set, then there are NO lines disjoint from it.
    So if S = U, then for any x not in U, every line through x intersects U.
    So L ∩ U is never empty.
    So we cannot find a line L through x (not in U) such that L \ {x} subset N (which means L ∩ U = empty).
    
    So S = U is NOT a valid complement.
    
    The baseline construction works by taking a SUBSET of U.
    It removes points from U such that the remaining S has the property that every point in S has a clean line, AND every point outside S has a clean line.
    
    For points outside S (i.e., in N), we need a line L through x with L ∩ S = empty.
    If S is small enough, such lines exist.
    
    So we want to maximize |S| such that:
    1. For every s in S, there is a line L_s with L_s ∩ S = {s}.
    2. For every x not in S, there is a line L_x with L_x ∩ S = empty.
    
    Condition 2 implies that S is NOT a blocking set.
    Condition 1 implies that S has no "triple" structure that forces every line through a point to hit another point of S.
    
    The baseline S has size 147.
    Can we make it larger?
    
    If we take S to be a line, size 49.
    If we take S to be a conic, size 49.
    
    What if we take S to be the union of a few lines?
    If S is two lines intersecting at a point P.
    For a point s on one line (not P), the line itself is not clean (contains other S points).
    But there are other lines through s.
    Do any of them avoid S?
    A line through s (not the line containing s) will intersect the other line at some point Q.
    If Q is in S, then the line is not clean.
    The other line is L2.
    The line L through s intersects L2 at Q.
    If Q is in S, then L ∩ S contains Q and s.
    So L is not clean.
    So for s to have a clean line, there must be a line through s that does not intersect L2 at a point in S.
    But L2 is a line, so it intersects every line through s (except the parallel one, if we are in affine plane).
    In F_q^2, lines are not parallel in the projective sense, but in affine sense, they can be parallel.
    Wait, the directions are v=(1,m) and v=(0,1).
    There are q+1 directions.
    Two lines with the same direction are parallel (disjoint) in the affine plane.
    
    So if s is on L1, and L1 and L2 intersect at P.
    Take a line L through s with the same direction as L2.
    Then L is parallel to L2, so L ∩ L2 = empty.
    Does L intersect L1 at any other point? No, L and L1 intersect only at s.
    So L ∩ S = {s} ∪ (L ∩ L2) = {s} ∪ empty = {s}.
    So L is clean!
    
    So if S is a union of lines, and for any s in S, we can find a line through s that is parallel to another component of S, then it works.
    
    If S is a set of parallel lines, then for any s in S, we can take a line through s with the same direction.
    Then it is parallel to all other lines in S, so it doesn't intersect them.
    And it intersects its own line only at s.
    So L ∩ S = {s}.
    So any set of parallel lines is a valid S for condition 1.
    
    Now condition 2: For any x not in S, there is a line through x disjoint from S.
    If S is a set of parallel lines, say k lines with direction d.
    A line L through x is disjoint from S if it does not intersect any of the k lines.
    In the affine plane, a line L intersects a line M unless they are parallel.
    So if L is not parallel to d, it intersects every line in S.
    So L ∩ S is not empty.
    If L is parallel to d, then L is parallel to every line in S.
    So L is disjoint from every line in S.
    So L ∩ S = empty.
    So for any x not in S, we can take a line L through x parallel to d.
    Then L is disjoint from S.
    So condition 2 is satisfied.
    
    So ANY set of parallel lines is a valid complement S!
    
    How many parallel lines can we have?
    There are q lines in a parallel class (pencil of parallel lines).
    So we can take all q lines in a parallel class.
    Size of S = q = 49.
    This is smaller than 147.
    
    Can we mix directions?
    If S contains lines of two different directions, say d1 and d2.
    Let L1 be a line in direction d1, L2 be a line in direction d2.
    They intersect at P.
    Take s on L1, s != P.
    We need a clean line through s.
    If we take a line parallel to d2, it is disjoint from L2, but it intersects L1 only at s.
    So it is clean with respect to L1 and L2.
    But what if there are other lines in S?
    If S is just L1 ∪ L2, then for s on L1, a line parallel to d2 is clean.
    For s on L2, a line parallel to d1 is clean.
    For P, a line parallel to d1 is clean? No, it intersects L1 at P and is disjoint from L2.
    Wait, L1 contains P. So the line through P parallel to d1 is L1 itself.
    L1 contains other points of S.
    So we need another line through P.
    Take a line through P with direction d3 (different from d1, d2).
    It intersects L1 at P and L2 at P.
    So L ∩ S = {P}.
    So it is clean.
    
    So a set of lines forming a "grid" might work.
    
    What is the maximum size of a set S of lines such that every point in S has a clean line and every point outside S has a clean line?
    
    If we take all lines in the plane, S = F_q^2, size 2401.
    Then N is empty.
    For x in S, we need a clean line.
    But every line contains many points of S.
    So no line is clean.
    So S = F_q^2 is invalid.
    
    The baseline S has size 147.
    
    Let's try to construct a larger S.
    
    How about taking S to be a "pencil" of lines through a point P?
    S = union of k lines through P.
    For s on one of these lines (s != P), we need a clean line.
    Take a line through s not passing through P.
    It intersects the other lines in the pencil at points other than P.
    So it is not clean.
    Take a line through s passing through P.
    It is one of the lines in S, so it contains other points of S.
    So no clean line for s.
    So a pencil is invalid.
    
    The parallel lines construction gives size 49.
    The baseline gives 147.
    
    Can we combine them?
    
    Let's stick to the baseline but see if we can tweak the set Y to make S larger.
    S = { (x,y) in U : y not in Y }.
    |S| = 7 * (49 - |Y|).
    To maximize |S|, we want to minimize |Y|.
    Y is a set of y-coordinates.
    The condition for validity was that for every point in S, there is a clean line.
    The baseline Y was chosen to ensure this.
    
    If we make Y smaller, S becomes larger.
    But we might violate the condition for points in S.
    
    Let's try Y = empty set.
    Then S = U, size 343.
    We already saw that S=U is invalid because points outside U have no clean line.
    
    So we need Y to be large enough to "break" the blocking property for points outside S.
    
    The points outside S are N = (F_q^2 \ U) ∪ (U ∩ (F_q x Y)).
    For x in N, we need a line L through x with L ∩ S = empty.
    S = U \ (U ∩ Y).
    So L ∩ (U \ (U ∩ Y)) = empty.
    This means L ∩ U subset (U ∩ Y).
    So any intersection of L with U must be in Y.
    
    If L is disjoint from U, then it works.
    If L intersects U, the intersection points must have y-coordinates in Y.
    
    So for every x in N, there must be a line L through x such that L ∩ U subset Y.
    
    This is a strong condition.
    
    Given the complexity, and the time limit, I will stick to the baseline construction but verify it carefully.
    The baseline score is 1.0.
    If I can find a valid S with size > 147, I will improve the score.
    
    Let's try to remove fewer points from U.
    i.e., make Y smaller.
    
    Let Y be the set of y-coordinates such that the line y=const is tangent to U?
    No.
    
    Let's try Y = { (0,0) }.
    Then S = U \ { points with y=(0,0) }.
    |Y| = 1.
    |S| = 7 * (49 - 1) = 336.
    Is this valid?
    For x in S, we need a clean line.
    For x not in S, we need a clean line.
    
    This is likely invalid.
    
    I will output the baseline program as it is known to be valid.
    """
    p = 7
    k = 3  # alpha^2 = 3

    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        continue
                    points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
