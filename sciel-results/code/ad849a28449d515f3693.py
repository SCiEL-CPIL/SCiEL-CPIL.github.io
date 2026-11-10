# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    The task requires N to be a set of points such that for every point x in F_49^2,
    there exists a line L through x (with direction (1, m) or (0, 1)) such that
    (L \ {x}) is a subset of N.
    
    This is equivalent to: Let S = F_49^2 \ N.
    1. For every x in S, there is a line through x that intersects S only at x.
    2. For every x in N, there is a line through x that is disjoint from S.
    
    The metric is |S| = 2401 - |N|. We want to maximize |S|.
    
    We use a greedy approach to maximize |S|.
    We start with S = empty set.
    We iterate through all points in a specific order.
    For each point p, we tentatively add it to S.
    We check if the condition for the "dangerous" points is still satisfied.
    
    To make this efficient, we note that adding a point p to S only "breaks" the condition for:
    - Other points in S that might have only been covered by lines passing through p.
    - Points in N that might have only been covered by lines passing through p (which are now blocked).
    
    Actually, let's think about the condition for a point x to be "satisfied":
    x is satisfied if there exists a line L through x such that L \ {x} is disjoint from S.
    
    If we add p to S, any line L that contains p is no longer "disjoint from S" (unless we remove p from L, but L is fixed by direction).
    Specifically, if x is on a line L passing through p, then L is not valid for x anymore (if L previously was valid, it must have been disjoint from S, but now it contains p in S).
    Wait, if L is valid for x, it means L \ {x} has no points from S.
    If we add p to S:
    - If p == x, then x is in S. The condition for x in S is that there is a line L through x such that L \ {x} is disjoint from S. Adding x to S doesn't affect L \ {x} directly, but it might affect other points in S? No, the condition is local to x and the current S.
    - If p != x and p is on L, then L is no longer disjoint from S. So if L was the ONLY valid line for x, x becomes unsatisfied.
    
    So, adding p to S can only cause points x that lie on lines passing through p to become unsatisfied.
    Which points lie on lines passing through p?
    All points in F_49^2?
    Yes, for any x != p, there is a unique line through x and p.
    So adding p to S potentially invalidates the "disjointness" of the line xp for every other point x.
    But x might have other valid lines.
    
    A point x is satisfied if at least one line through x is disjoint from S.
    Let Valid(x, S) be the set of lines through x disjoint from S.
    x is satisfied if Valid(x, S) is not empty.
    
    When we add p to S:
    For any x != p, the line L(x, p) is no longer in Valid(x, S).
    If Valid(x, S_old) = {L(x, p)}, then Valid(x, S_new) = empty, and x becomes unsatisfied.
    
    So, to greedily add p:
    1. Check if all current x in S are still satisfied (they might have relied on lines not passing through p, or if they relied on a line passing through p, they are in trouble).
       Wait, if x is in S, and p is added to S.
       The line L(x, p) passes through p (which is in S). So L(x, p) is not disjoint from S.
       If x's only valid line was L(x, p), x becomes unsatisfied.
    2. Check if all current x in N are still satisfied.
       Similarly, if x in N relied on L(x, p), it becomes unsatisfied.
    
    This requires checking satisfaction for all x.
    There are 2401 points.
    For each point, there are 50 lines (49 slopes + vertical).
    Checking if a line is disjoint from S is O(|S|) or O(q) if we have a lookup.
    With a set lookup for S, checking if a line is disjoint is O(q) = 49.
    So checking satisfaction for one x is O(50 * 49) = 2450.
    Checking all x is 2401 * 2450 ~ 5.9M operations.
    
    If we do this for each candidate p (up to 2401 candidates), it's 2401 * 5.9M ~ 14B operations.
    This is too slow for Python.
    
    We need a smarter heuristic or a smaller search space.
    
    Let's look at the structure of the valid sets found so far.
    The previous score 1.3333 corresponds to |S| = 147.
    Seed score 1.0 corresponds to |S| = 110.25? No, score is value/seed.
    If seed value is V_seed, and new value is V_new, score = V_new / V_seed.
    If score is 1.3333, V_new = 1.3333 * V_seed.
    Let's assume the seed program produced a specific S.
    
    Alternative approach:
    The condition "for every x, there is a line L_x disjoint from S (or intersecting S only at x)" is very strong.
    It implies that S cannot be too "dense" in the direction of any line.
    Specifically, if a line L contains 2 points of S, say a and b.
    Then for any point x on L, the line L is not valid for x (if x != a, b) because it contains a, b in S.
    So x must have another valid line.
    
    Consider a "thin" set S.
    If S is an arc (no 3 collinear), then any line contains at most 2 points of S.
    If a line contains 2 points a, b of S.
    Then for any other point x on that line, the line is invalid.
    x must use another line.
    
    The maximum size of an arc in F_q^2 is q+1? No, q+1 is for projective.
    In affine F_q^2, the maximum arc size is q+1?
    Actually, a conic has q+1 points? No, a conic in F_q^2 has q+1 points if we count the point at infinity?
    In AG(2, q), it has q+1 or q points?
    Usually q+1 points.
    
    If we take S to be a conic (size q+1 = 50), is it a valid "excluded" set?
    For x in S: Any line through x intersects the conic in at most 1 other point (tangent -> 0, secant -> 1).
    So there are q-1 lines through x.
    1 line is tangent (intersects only at x).
    q-2 lines are secants (intersect at x and 1 other).
    So there is at least 1 valid line (the tangent).
    So points in S are satisfied.
    
    For x in N: We need a line through x disjoint from S.
    If S is a conic, does every point in N lie on a line disjoint from S?
    A line is disjoint from a conic if it doesn't intersect it.
    In F_q^2, lines can be tangent, secant, or exterior.
    Exterior lines are disjoint.
    Do exterior lines exist through every point not on the conic?
    Yes, generally.
    
    So S = Conic (size 50) should be valid.
    Score would be 50 / V_seed.
    
    Can we do better?
    What if S is the union of two disjoint conics?
    Size 100.
    If they are disjoint, no line intersects both?
    No, a line can intersect two conics.
    But if a line intersects Conic 1 at a and Conic 2 at b.
    Then for x on that line, the line contains a, b in S.
    So the line is invalid for x.
    
    If S is the Hermitian curve (size q+1=50? No, Hermitian curve in F_q^2 has q+1 points?
    Wait, earlier I calculated 343.
    Let's re-verify the size of the Hermitian curve in F_q^2.
    Equation: x^q + x + y^q + y = 0? No, that's for q=p.
    For q=p^2, Tr(x) + Nm(y) = 0.
    Tr(x) = x + x^q. Nm(y) = y y^q.
    This is a curve of degree q.
    In F_q^2, a line intersects it in 1 or q+1 points?
    Actually, it is known that any line in F_q^2 intersects the Hermitian curve in 1 or 2 points?
    No, that's for conics.
    The Hermitian curve is a unital in PG(2, q).
    It has q^2 + q + 1 points in PG(2, q).
    In AG(2, q), it has q^2 + q + 1 - (q+1) = q^2 points?
    If q=49, size is 49^2 = 2401? No, that's the whole plane.
    The number of points on the Hermitian curve in AG(2, q) is q^2 + 1?
    Let's check small q.
    q=4. Hermitian curve has 4^2+1 = 17 points in PG(2,4).
    In AG(2,4), it has 17 - 5 = 12 points?
    
    Actually, the "unital" in the affine plane might just be a specific set.
    Let's stick to the known valid construction from the prompt's history:
    The current program uses a subset of the "Unital" defined by 2*x_a + y_a^2 - k*y_b^2 = 0.
    Size of this set is 343.
    The current S is a subset of this, with size 147.
    
    My hypothesis:
    The maximum size of S is related to the size of a "blocking set" complement.
    If we can find a larger valid S, we improve the score.
    
    Let's try to use a greedy algorithm with a timeout.
    We will try to add points to S one by one, checking validity only for the affected points (those on lines through the new point).
    
    Algorithm:
    1. Initialize S = set().
    2. Precompute for each point x, the list of 50 lines passing through x.
    3. For each point p in a shuffled order (or sorted by some metric):
       a. Tentatively add p to S.
       b. Identify all points x that are on a line with p. (All points x != p).
       c. For each such x, check if x is still satisfied.
          x is satisfied if there is a line L through x such that L \ {x} is disjoint from S.
          Note: If x is in S, we check L \ {x} disjoint from S.
          If x is in N, we check L \ {x} disjoint from S.
       d. If all such x are satisfied, keep p. Else, remove p.
    
    This is still expensive because step (c) iterates over all 2400 points.
    However, we can prune.
    If we maintain a count of valid lines for each x, we can update it efficiently.
    
    Let's implement a fast version:
    - Maintain `valid_count[x]` = number of lines through x that are disjoint from S (or intersect S only at x if x in S).
    - Initially S is empty. All lines are disjoint. valid_count[x] = 50 for all x.
    - When we add p to S:
      - For each line L passing through p:
        - For each other point x on L:
          - L is no longer valid for x (unless x is the only point of S on L? No, L contains p in S. So L \ {x} contains p. So L is not disjoint from S. Thus L is invalid for x).
          - So valid_count[x] decreases by 1.
          - If valid_count[x] becomes 0, x is unsatisfied.
      - If any x becomes unsatisfied, we must revert the addition of p.
      - To revert, we increment valid_count[x] back by 1 for all affected x.
    
    This update is local to the lines passing through p.
    There are 50 lines through p.
    Each line has 49 other points.
    So 50 * 49 = 2450 points are affected.
    Updating their counts is O(2450).
    
    We iterate through 2401 candidate points.
    Total operations: 2401 * 2450 ~ 5.9M.
    This is very fast!
    
    Let's implement this greedy algorithm.
    
    Steps:
    1. Define the field arithmetic for F_49.
    2. Generate all 2401 points.
    3. Precompute lines: For each point x, list of 50 lines. Each line is represented by (slope, intercept) or similar.
       Actually, we just need the set of points on each line.
       Or better, for each point p, we need to know which other points are on the 50 lines through p.
       We can just generate these on the fly or precompute a list of neighbors.
       `neighbors[p]` = list of 2450 points (all other points).
       Wait, every other point is on exactly one line through p.
       So the set of affected points is always ALL other points.
       
       Ah, if we add p to S, it invalidates the line xp for EVERY x.
       So EVERY other point x loses 1 valid line (the line xp).
       So valid_count[x] decreases by 1 for ALL x != p.
       
       If valid_count[x] becomes 0 for any x, we must revert.
       
       This means that if we add too many points to S, eventually some x will have all 50 lines passing through a point in S.
       Specifically, x is unsatisfied if the set of points in S blocks all 50 lines through x.
       i.e., S is a blocking set for the lines through x.
       
       So the condition is: For all x, S does not block all lines through x.
       
       We want to maximize |S| such that for all x, S is not a blocking set for x.
       
       Greedy algorithm:
       Start with S = empty.
       For each p:
         Tentatively add p.
         For all x != p:
           valid_count[x] -= 1
         If any valid_count[x] == 0:
           Revert (add p back to valid_count[x] for all x != p)
         Else:
           Keep p.
       
       This is O(|S| * 2400) if we do it naively?
       No, each step is O(2400).
       Total steps 2401.
       Total operations 2401 * 2400 ~ 5.8M.
       This is fast.
       
       Let's implement this.
    """
    p_prime = 7
    q = 49
    k = 3
    
    # Points are indices 0..2400
    # Point i corresponds to (xa, xb, ya, yb)
    # Index = ((xa * 7 + xb) * 7 + ya) * 7 + yb
    
    def idx_to_coords(i):
        yb = i % 7
        ya = (i // 7) % 7
        xb = (i // 49) % 7
        xa = (i // 343) % 7
        return (xa, xb, ya, yb)

    def coords_to_idx(xa, xb, ya, yb):
        return ((xa * 7 + xb) * 7 + ya) * 7 + yb

    # Precompute lines through each point
    # line[p] is a list of 2450 indices (all other points)
    # Actually, we just need to decrement valid_count for all other points.
    # So we don't even need to store lines explicitly for the update logic,
    # just the fact that every other point is affected.
    
    # valid_count[x] = 50 initially.
    
    valid_count = np.full(2401, 50, dtype=np.int32)
    in_S = np.zeros(2401, dtype=bool)
    
    # We want to maximize |S|.
    # The order of adding points matters.
    # Let's try adding points in a random order, or sorted by some property.
    # Random order might be good to avoid local minima?
    # Or maybe sorted by "degree" or something?
    # Let's try random order with a fixed seed for reproducibility.
    
    rng = np.random.RandomState(42)
    candidates = np.arange(2401)
    rng.shuffle(candidates)
    
    start_time = time.time()
    deadline = start_time + 500  # Leave 100s for safety
    
    for p_idx in candidates:
        if time.time() > deadline:
            break
        
        # Tentatively add p_idx to S
        # This decreases valid_count[x] by 1 for all x != p_idx
        
        # We need to check if any valid_count[x] becomes 0.
        # Instead of updating all and checking, we can check if min(valid_count[all != p_idx]) > 0 after decrement.
        # But we have to update anyway if we keep it.
        
        # Optimization: 
        # If min(valid_count) > 1, then decrementing all (except p_idx) won't make any 0.
        # min_valid = np.min(np.delete(valid_count, p_idx))
        # This is O(N).
        
        # Let's just do the update and check.
        # Update: valid_count -= 1 for all x != p_idx.
        # We can do:
        #   valid_count -= 1
        #   valid_count[p_idx] += 1  # Restore p_idx
        
        # Check:
        #   if np.min(valid_count) == 0:
        #       revert
        #   else:
        #       in_S[p_idx] = True
        
        # However, we must be careful: p_idx itself is in S.
        # The condition for p_idx (which is in S) is that there is a line through p_idx disjoint from S.
        # The lines through p_idx are the same as the lines through any other point.
        # valid_count[p_idx] tracks the number of lines through p_idx disjoint from S.
        # When we add p_idx to S, does valid_count[p_idx] change?
        # The lines through p_idx now contain p_idx (in S).
        # For x = p_idx, the condition is L \ {p_idx} disjoint from S.
        # Before adding p_idx, L was disjoint from S (old S).
        # After adding p_idx, L \ {p_idx} is still disjoint from S (new S) because p_idx is removed from the check.
        # Wait.
        # If L is a line through p_idx.
        # Old condition: L disjoint from S_old.
        # New condition: L \ {p_idx} disjoint from S_new.
        # S_new = S_old U {p_idx}.
        # L \ {p_idx} contains no points from S_old (since L was disjoint).
        # L \ {p_idx} does not contain p_idx.
        # So L \ {p_idx} is disjoint from S_new.
        # So valid_count[p_idx] should NOT decrease when we add p_idx.
        
        # What about other points x?
        # For x != p_idx, the line L_xp contains p_idx.
        # Old condition: L_xp disjoint from S_old.
        # New condition: L_xp \ {x} disjoint from S_new.
        # L_xp \ {x} contains p_idx.
        # p_idx is in S_new.
        # So L_xp \ {x} is NOT disjoint from S_new.
        # So L_xp is no longer valid for x.
        # So valid_count[x] decreases by 1 for all x != p_idx.
        
        # So the update is:
        # 1. valid_count[x] -= 1 for all x != p_idx.
        # 2. Check if min(valid_count) > 0. (Note: valid_count[p_idx] is unchanged by the step 1 logic if we are careful).
        
        # Let's refine the update:
        # Save old_min? No.
        
        # Perform update:
        # valid_count -= 1
        # valid_count[p_idx] += 1  # Because it wasn't supposed to change
        
        # Check:
        # if np.min(valid_count) > 0:
        #     in_S[p_idx] = True
        # else:
        #     revert
        #     valid_count += 1
        #     valid_count[p_idx] -= 1
        
        # One nuance: What if p_idx was already in S? We only add once.
        
        # Let's implement this.
        
        valid_count -= 1
        valid_count[p_idx] += 1
        
        if np.min(valid_count) > 0:
            in_S[p_idx] = True
        else:
            # Revert
            valid_count += 1
            valid_count[p_idx] -= 1
    
    # Construct the result N
    # N is the set of points NOT in S.
    # The task asks for an array of points in N.
    
    n_points = np.sum(~in_S)
    
    # Create the array
    # We need to return an array of shape (k, 2, 2)
    # where k is the number of points in N.
    
    result_list = []
    for i in range(2401):
        if not in_S[i]:
            xa, xb, ya, yb = idx_to_coords(i)
            result_list.append([
                [xa, xb],
                [ya, yb]
            ])
    
    if len(result_list) == 0:
        # Should not happen, but just in case
        return np.empty((0, 2, 2), dtype=np.int64)
        
    result = np.array(result_list, dtype=np.int64)
    
    return result

# EVOLVE-BLOCK-END
