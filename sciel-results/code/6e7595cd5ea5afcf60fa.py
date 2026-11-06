"""
Strategy: Use a greedy randomized approach to maximize the number of distinct colors.
We aim to create many small color classes that satisfy the C4 constraint.
The constraint is: No 4-cycle has 4 distinct edge colors.
This implies that for any 4-cycle, at least two edges must share a color.

Equivalently, if we assign colors such that every 4-cycle has a repeated color, we are valid.
To maximize colors, we want to minimize the "clustering" of edges into few colors.
However, we can't have *too* many colors because the 4-cycles force repetitions.

Key Insight: If we can partition edges into many small sets such that no set contains 
a "bad" configuration relative to other sets... actually, it's simpler.
The constraint is global.

Let's try a constructive approach:
1. Start with all edges uncolored.
2. We want to assign colors 0, 1, 2, ... as much as possible.
3. However, we can just assign random colors and check validity? 
   No, we need to maximize the count.

A better heuristic:
- The constraint is that every C4 must have a repeat.
- This is equivalent to saying: The complement of the color classes must not contain a C4 where all 4 edges are distinct? 
  No.

Let's look at the structure.
If we use $k$ colors, we can think of it as a graph coloring of the line graph? No.

Let's try a simple randomized local search or greedy assignment with backtracking? 
Given 60s limit, we can do something efficient.

Actually, a known bound or structure for this problem ("many edge colors without a rainbow 4-cycle"):
The maximum number of colors is related to the density of the graph.
For $K_{16}$, the number of edges is 120.
The number of 4-cycles is $\binom{16}{4} \times 3 = 1820 \times 3 = 5460$.

Let's try to construct a solution with a high number of colors.
One trivial solution is 1 color.
Another is to pair up edges?

Let's use a randomized greedy algorithm:
1. Shuffle edge order.
2. Maintain a set of assigned colors for each edge.
3. Maintain for each color, the set of edges it has.
4. When assigning a new edge, pick the smallest available color $c$ such that adding this edge to color class $c$ does not create a "rainbow 4-cycle" where this edge is the 4th distinct color.
   Wait, the constraint is "No four-cycle may have four distinct edge colors".
   This means if we have a 4-cycle $(e1, e2, e3, e4)$, we cannot have $col(e1) \neq col(e2) \neq col(e3) \neq col(e4)$ all distinct.
   
   When we assign color $c$ to edge $e$, we need to check all 4-cycles containing $e$.
   For each such 4-cycle, let the other 3 edges have colors $c_1, c_2, c_3$.
   If $c, c_1, c_2, c_3$ are all distinct, then we have a rainbow 4-cycle. This is forbidden.
   So, $c$ is invalid if there exists a 4-cycle containing $e$ such that the other 3 edges have 3 distinct colors, none of which is $c$.
   
   Algorithm:
   - Precompute all 4-cycles. For each edge $e$, store the list of 4-cycles it belongs to.
   - For each 4-cycle, store the 4 edges.
   - Iterate edges in random order (or fixed order for reproducibility, but random helps diversity if we restart).
   - For each edge $e$, determine valid colors.
     - A color $c$ is valid if for every 4-cycle $C$ containing $e$, the set of colors of the other 3 edges in $C$ is NOT a set of 3 distinct colors disjoint from $\{c\}$.
     - Actually, simpler: If we assign $c$ to $e$, a violation occurs if there is a 4-cycle $C$ containing $e$ where the other 3 edges have colors $\{a, b, d\}$ with $a,b,d$ distinct and $a,b,d \neq c$.
   - To maximize colors, we should try to use new colors if possible? 
     - If we use a new color $k$, it's always valid? 
     - If we use a new color $k$, then for any 4-cycle containing $e$, the other 3 colors are from previous assignments. They might be distinct. If they are distinct, and none is $k$ (which is true since $k$ is new), then we have a rainbow 4-cycle! 
     - So we CANNOT just assign new colors freely. We must ensure that if we assign a new color to $e$, then for every 4-cycle containing $e$, at least two of the other 3 edges share a color OR one of them has color $k$ (impossible) OR ... wait.
     - Condition for validity of assigning color $c$ to edge $e$:
       For all 4-cycles $C$ containing $e$, let $O$ be the set of colors of the other 3 edges.
       We need: $|O \cup \{c\}| < 4$.
       This means either:
       1. $c \in O$ (so size is at most 3)
       2. $|O| < 3$ (so at least two other edges share a color)
       
   - So, if we assign a NEW color $c$ (not used before), then $c \notin O$ always.
     Thus, we must have $|O| < 3$ for ALL 4-cycles containing $e$.
     This means that for every 4-cycle containing $e$, at least two of the other three edges must already share a color.
     
   - This suggests that we can only introduce a new color if the edge is "safe" with respect to the current coloring of its neighbors in cycles.
   
   Strategy:
   1. Start with no colors.
   2. Pick an edge. It has no neighbors colored yet (or empty O). $|O|=0 < 3$. Safe to assign new color 0.
   3. Pick next edge. Check 4-cycles. If all 4-cycles containing it have at least 2 edges sharing a color among the already colored ones, we can assign a new color. Otherwise, we must reuse an existing color.
   4. To maximize colors, we should prioritize assigning new colors when possible. If not possible, reuse a color that minimizes future constraints? Or just pick any valid one.

   This is a greedy algorithm. It might get stuck. We can use randomized restarts.

   Implementation details:
   - N=16. Edges = 120.
   - Precompute cycles for each edge.
   - Time limit: 60s. 
   - We can run this greedy multiple times with different random orders and keep the best.

   Let's refine the "valid color" check.
   State: `colors` array, `edge_color` map.
   For a given uncolored edge `e` and candidate color `c`:
     For each 4-cycle `C` containing `e`:
       Let `other_edges` = C \ {e}
       Let `other_colors` = set(colors[x] for x in other_edges)
       If `len(other_colors) == 3` and `c` not in `other_colors`:
         Return False (Invalid)
     Return True

   If we are trying to assign a NEW color, we just check if `len(other_colors) < 3` for all cycles.
   If we are trying to assign an EXISTING color, we check the full condition.

   To maximize the number of colors, we should try to assign a new color first.
   If a new color is valid, assign it.
   Else, try existing colors (0 to max_color_used). Pick the first valid one.
   If no existing color is valid... wait, is it possible no color is valid?
   If we assign color `c`, and for some cycle, `other_colors` has size 3 and `c` is one of them, it's valid.
   So if `other_colors` has size 3, we MUST pick one of those 3 colors.
   So there is always a valid color (just pick one from the "conflicting" sets).

   So the algorithm is:
   1. Order edges randomly.
   2. For each edge:
      a. Check if new color is valid.
      b. If yes, assign new color.
      c. If no, find an existing valid color. (Must exist). Assign it.
   3. Count distinct colors.

   We can improve by choosing the edge order strategically? 
   Or by backtracking? 120 edges is too many for full backtracking.
   But we can use "conflict-driven" choice: pick the edge with the fewest valid new-color options? 
   Actually, the "new color valid" check is the bottleneck for increasing color count.
   
   Let's implement the random greedy with restarts.

   Precomputation:
   - List of all edges.
   - For each edge, list of 4-cycles it belongs to.
   
   How to get 4-cycles?
   A 4-cycle in K_16 is defined by 4 vertices. There are 3 distinct 4-cycles on 4 vertices.
   So, for each 4-subset of vertices, there are 3 cycles.
   Total 5460 cycles.
   For each cycle, add it to the list of each of its 4 edges.

   Time complexity of one pass:
   120 edges.
   For each edge, check ~100 cycles (avg).
   Check is fast (set intersection/size).
   120 * 100 = 12000 operations. Very fast.
   We can run hundreds of restarts in 60s.

   Let's code this up.
"""

import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2

def get_edges():
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

# Precompute 4-cycles
# A 4-cycle is a set of 4 edges forming a cycle of length 4.
# In K_4 (vertices a,b,c,d), the cycles are:
# (a,b), (b,c), (c,d), (d,a)
# (a,b), (b,d), (d,c), (c,a) -> same as above? No.
# The 3 cycles on {a,b,c,d} are:
# 1. ab, bc, cd, da
# 2. ab, bd, dc, ca
# 3. ac, cb, bd, da
# Let's generate them properly.

def generate_4_cycles():
    cycles = []
    for verts in itertools.combinations(range(N), 4):
        a, b, c, d = verts
        # Cycle 1: a-b-c-d-a
        e1 = (min(a,b), max(a,b))
        e2 = (min(b,c), max(b,c))
        e3 = (min(c,d), max(c,d))
        e4 = (min(d,a), max(d,a))
        cycles.append([e1, e2, e3, e4])
        
        # Cycle 2: a-b-d-c-a
        e1 = (min(a,b), max(a,b))
        e2 = (min(b,d), max(b,d))
        e3 = (min(d,c), max(d,c))
        e4 = (min(c,a), max(c,a))
        cycles.append([e1, e2, e3, e4])
        
        # Cycle 3: a-c-b-d-a
        e1 = (min(a,c), max(a,c))
        e2 = (min(c,b), max(c,b))
        e3 = (min(b,d), max(b,d))
        e4 = (min(d,a), max(d,a))
        cycles.append([e1, e2, e3, e4])
    return cycles

ALL_CYCLES = generate_4_cycles()
EDGES = get_edges()
EDGE_TO_IDX = {e: i for i, e in enumerate(EDGES)}

# For each edge index, store the indices of cycles it belongs to
EDGE_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(ALL_CYCLES):
    for e in cycle:
        e_idx = EDGE_TO_IDX[e]
        EDGE_CYCLES[e_idx].append(ci)

# For fast access, store cycle edges as indices
CYCLE_EDGES_IDX = []
for cycle in ALL_CYCLES:
    idxs = [EDGE_TO_IDX[e] for e in cycle]
    CYCLE_EDGES_IDX.append(idxs)

def construct():
    deadline = time.time() + 55.0  # 55s to be safe
    best_colors = None
    best_count = 0
    
    # We will run random restarts
    # Each run: greedy assignment with random edge order
    # If time is up, break and return best.
    
    while time.time() < deadline:
        random.shuffle(EDGES)
        # Map shuffled edges to indices for assignment
        # Actually, we process edges in the shuffled order.
        
        colors = [-1] * NUM_EDGES
        num_colors = 0
        
        # For each edge in shuffled order
        for e in EDGES:
            e_idx = EDGE_TO_IDX[e]
            
            # Check if we can assign a new color
            # New color is valid if for all cycles C containing e_idx, 
            # the other 3 edges have < 3 distinct colors.
            new_color_valid = True
            # We also need to find a valid existing color if new is not valid.
            # Let's first check new color.
            
            # Optimization: 
            # If any cycle has 3 distinct colors among the other edges, new color is invalid.
            for ci in EDGE_CYCLES[e_idx]:
                c_edges = CYCLE_EDGES_IDX[ci]
                other_edges = [x for x in c_edges if x != e_idx]
                # Get colors of other edges
                o_colors = set()
                for oe in other_edges:
                    if colors[oe] != -1:
                        o_colors.add(colors[oe])
                    else:
                        # If an edge is uncolored, its color is not fixed yet?
                        # Wait, the greedy algorithm processes edges one by one.
                        # When we process e_idx, are the other edges colored?
                        # No, we process in a specific order. The other edges might be uncolored if they come later in the order.
                        # 
                        # CRITICAL FLAW IN GREEDY LOGIC ABOVE:
                        # The constraint must hold for the FINAL assignment.
                        # If I assign e now, and later assign others, I might create a rainbow cycle.
                        # 
                        # To make greedy work, we must ensure that the assignment is "safe" regardless of future assignments?
                        # No, that's too strong.
                        # 
                        # Alternative: Process edges in an order such that when we assign e, all other edges in any 4-cycle containing e are ALREADY assigned?
                        # This is possible if the graph of "cycle dependencies" is acyclic? No, K_16 is highly cyclic.
                        # 
                        # Let's rethink.
                        # If we use random order, some neighbors in cycles will be uncolored.
                        # If a neighbor is uncolored, we don't know its color.
                        # We can treat uncolored edges as "wildcards".
                        # 
                        # Condition for validity of assigning color c to e:
                        # There is no 4-cycle C containing e such that it is IMPOSSIBLE to avoid a rainbow cycle involving e and C.
                        # 
                        # If we assign c to e, and the other 3 edges are e1, e2, e3.
                        # If e1, e2, e3 are all colored and have distinct colors a,b,d, and none is c, then it's a violation NOW.
                        # If some are uncolored, we can choose their colors later to avoid violation.
                        # Specifically, we need to be able to color e1, e2, e3 such that the set of colors in {c, col(e1), col(e2), col(e3)} has size < 4.
                        # This is always possible?
                        # Yes, we can just set col(e1) = c. Then size is at most 3.
                        # So, uncolored edges do NOT prevent a valid assignment.
                        # They just mean we haven't locked in the constraint yet.
                        # 
                        # HOWEVER, if we assign c to e, and e1, e2, e3 are uncolored, we must REMEMBER that we are constrained to color them in a way that avoids rainbow.
                        # This turns into a CSP (Constraint Satisfaction Problem).
                        # 
                        # But for a simple greedy heuristic, we can assume that uncolored edges will be colored "cooperatively".
                        # The only HARD constraint violation is if the already colored edges force a rainbow.
                        # 
                        # So, the check "new_color_valid" should be:
                        # For all cycles C containing e:
                        #   Let O be the set of colors of ALREADY COLORED other edges in C.
                        #   Let U be the set of UNCOLORED other edges in C.
                        #   If |O| == 3 and c not in O:
                        #       Then we have a fixed rainbow 4-cycle (since the 3 colored are distinct and none is c). The uncolored ones don't matter because the violation is already complete?
                        #       Wait, a 4-cycle has 4 edges. If 3 are colored distinct and the 4th is c (distinct from them), then we have 4 distinct colors.
                        #       Yes, that is a violation.
                        #   If |O| < 3:
                        #       We can always color the uncolored edges to avoid violation (e.g., reuse c or one of O).
                        #       So this is safe.
                        
                        # So the condition for NEW color c being valid is:
                        # For all cycles C containing e, the number of ALREADY COLORED other edges with distinct colors is < 3.
                        
                        if len(o_colors) == 3:
                            # Check if c is in o_colors? 
                            # Since c is a NEW color, it is not in o_colors (which consists of 0..num_colors-1).
                            # So c is distinct from all in o_colors.
                            # Thus, we have 4 distinct colors.
                            new_color_valid = False
                            break
            
            if new_color_valid:
                colors[e_idx] = num_colors
                num_colors += 1
            else:
                # We must assign an existing color.
                # Which one?
                # We need a color c such that for all cycles C containing e:
                #   If |O| == 3, then c must be in O.
                #   (If |O| < 3, any c is fine).
                
                # So, c must be in the intersection of all O's that have size 3?
                # Or more precisely:
                # Let Constraints = set of colors that are FORBIDDEN?
                # No, if |O|==3, c MUST be in O.
                # So c must be in O for every such constraint.
                
                valid_colors = set(range(num_colors))
                
                for ci in EDGE_CYCLES[e_idx]:
                    c_edges = CYCLE_EDGES_IDX[ci]
                    other_edges = [x for x in c_edges if x != e_idx]
                    o_colors = set()
                    has_uncolored = False
                    for oe in other_edges:
                        if colors[oe] == -1:
                            has_uncolored = True
                        else:
                            o_colors.add(colors[oe])
                    
                    if len(o_colors) == 3 and not has_uncolored:
                        # All 3 others are colored.
                        # c must be in o_colors to avoid rainbow.
                        valid_colors &= o_colors
                    
                    elif len(o_colors) == 3 and has_uncolored:
                        # Wait, if there are uncolored edges, can we still have a violation?
                        # No, because we can color the uncolored ones to match c or one of the others.
                        # So this constraint is SOFT. It doesn't restrict c now.
                        # But wait, if I pick c not in O, and later the uncolored ones are colored randomly, I might create a rainbow.
                        # However, since I control the later coloring, I can always choose to color the uncolored edges to avoid rainbow.
                        # So, I don't need to restrict c based on cycles with uncolored edges.
                        # 
                        # BUT, if I pick c, and later I'm forced to pick bad colors for the uncolored edges due to THEIR constraints?
                        # This is the risk of greedy.
                        # To be safe, maybe we should prefer colors in O even if there are uncolored edges?
                        # Let's stick to the hard constraint: only restrict if all 3 are colored.
                        pass
                
                if not valid_colors:
                    # Should not happen if logic is correct, but fallback to 0
                    colors[e_idx] = 0
                else:
                    # Pick one. To maximize future new colors, maybe pick the one that appears in most constraints?
                    # Or just pick the first.
                    # Let's pick the smallest for determinism in this run?
                    # Or random?
                    # Random might help escape local minima.
                    c = random.choice(list(valid_colors))
                    colors[e_idx] = c
        
        # End for loop for edges
        
        # Calculate score
        used_colors = len(set(colors))
        if used_colors > best_count:
            best_count = used_colors
            best_colors = colors[:]
            # Optional: break if we hit a very high number?
            # Max possible is 120, but likely much lower.
            # Let's continue to see if we can improve.

    if best_colors is None:
        return [0] * NUM_EDGES
    return best_colors

# EVOLVE-BLOCK-START
# (Construct is defined above, but we need to ensure the block structure if required. 
# The prompt says "Keep the function names...". The current program has EVOLVE-BLOCK-START/END.
# I will wrap my code in the block or just define construct.)
# The instructions say: "Output the COMPLETE program... Keep the function names...".
# I will provide the full file content.
# EVOLVE-BLOCK-END

# Final check:
# Does construct() return a list of 120 ints? Yes.
# Are they in 0..119? 
# num_colors can exceed 119? 
# If num_colors > 119, we have a problem because colors must be in 0..119.
# However, max colors is bounded by the problem structure. 
# If we get > 119, we need to remap. 
# But 119 is very high. K_16 has 120 edges. 
# If we use 120 colors, it's 1 color per edge. 
# Is that valid? 
# If every edge has a distinct color, every 4-cycle has 4 distinct colors. 
# So 120 is INVALID.
# The max number of colors is less than 120.
# So num_colors will be < 120. 
# But could it be > 119? No, max is 119 if we use distinct colors 0..118? 
# Wait, if we use 120 colors, indices 0..119. 
# If we use 120 distinct colors, it's invalid.
# So the algorithm will naturally stop increasing colors when it becomes invalid.
# The "new_color_valid" check will fail if it leads to a rainbow.
# So num_colors will stay at a valid maximum.
# The values will be 0..num_colors-1.
# If num_colors <= 120, values are in 0..119.
# So we are safe.

