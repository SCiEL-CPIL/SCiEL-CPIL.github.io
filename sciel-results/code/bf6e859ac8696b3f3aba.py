# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    
    Strategy: Maximum coverage formulation via greedy line selection with local search.
    1. Formulate as selecting a set of lines to cover all 2401 points.
    2. Minimize the union of points in selected lines (which is N).
    3. Maximize the complement C = F_49^2 \ N.
    4. Use greedy set cover heuristic with randomized restarts and local search swaps.
    
    A point x is in N if it lies on at least one selected line.
    A point x is in C if it lies on no selected line.
    Validity: For every x in C, there must exist a line L through x such that L ∩ C = {x},
    which means L is NOT in the selected set (since L ∩ N = L \ {x} ⊆ N).
    Wait, the condition is: every point x of F_49^2 (in N or not) must lie on a line L,
    whose points other than x all belong to N.
    If x in C: L \ {x} ⊆ N. Since x not in N, L ∩ N = L \ {x}. So L is not "fully in N".
    If x in N: L \ {x} ⊆ N. So L \ {x} ⊆ N.
    
    Actually, the standard Nikodym set definition:
    For each point x, there is a line L_x through x such that L_x \ {x} ⊆ N.
    
    Let's re-evaluate the "line selection" idea.
    If we select a set of lines S, and define N = Union(S), then:
    For x in N: x is in some line L in S. Does L \ {x} ⊆ N? Yes, because L ⊆ Union(S) = N.
    So any x in N is satisfied by any line in S passing through it.
    For x in C: x is not in any line in S. We need a line L through x such that L \ {x} ⊆ N.
    This means L \ {x} ⊆ Union(S).
    So for every x in C, there must be a line L through x such that all other points on L are in Union(S).
    
    This is a different constraint. It's not just "cover all points".
    The "line selection" view is:
    We want to choose N (or C) to minimize |N|.
    Constraint: ∀x, ∃ line L through x s.t. L \ {x} ⊆ N.
    
    Let's stick to the proven greedy expansion of C, but improve the initialization and search.
    The inspiration 2 (score 1.5238) uses a baseline construction with optimized greedy improvement.
    The current program (score 1.2449) uses constraint propagation + greedy expansion.
    
    The key to high scores is likely in the *initial* C and the *order* of greedy expansion.
    Inspiration 2 starts with a specific C (unital points with non-square y-coords?) and then expands.
    
    Let's try a hybrid:
    1. Start with C = empty.
    2. Use a "potential" function to guide the addition of points to C.
    3. The potential of adding x to C is the number of points y that *lose* a witness line.
    4. We want to minimize this loss, or maximize the "safety" of x.
    
    Actually, the most effective strategy observed in similar problems is often:
    - Start with a large valid C (e.g., from a known construction).
    - Try to add more points to C (greedy).
    - Try to remove points from C to see if we can add more elsewhere (local search).
    
    Let's implement a robust greedy expansion with multiple random restarts and local search.
    
    Key optimization: Efficient update of witness counts.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # F_49 arithmetic
    def mul_f49(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1*a2 + k*b1*b2) % p, (a1*b2 + a2*b1) % p)
        
    def add_f49(p1, p2):
        return ((p1[0] + p2[0]) % p, (p1[1] + p2[1]) % p)
        
    def sub_f49(p1, p2):
        return ((p1[0] - p2[0]) % p, (p1[1] - p2[1]) % p)
        
    def neg_f49(p1):
        return ((-p1[0]) % p, (-p1[1]) % p)

    # All points as (xa, xb, ya, yb)
    all_points = [(xa, xb, ya, yb) for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    n_points = len(all_points)
    point_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # Precompute lines
    # Vertical lines: x = const
    vert_lines = []
    for xa in range(p):
        for xb in range(p):
            pts = [i for i, (x_a, x_b, y_a, y_b) in enumerate(all_points) 
                   if x_a == xa and x_b == xb]
            vert_lines.append(pts)
            
    # Slope lines: y = m*x + c
    slope_lines = []
    for ma in range(p):
        for mb in range(p):
            if ma == 0 and mb == 0:
                continue
            lines_dict = {}
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                mx = mul_f49((ma, mb), (xa, xb))
                c = sub_f49((ya, yb), mx)
                if c not in lines_dict:
                    lines_dict[c] = []
                lines_dict[c].append(i)
            for c, pts in lines_dict.items():
                slope_lines.append(pts)
                
    # Combine all lines
    all_lines = vert_lines + slope_lines
    n_lines = len(all_lines)
    
    # Map point to lines through it
    point_to_lines = [[] for _ in range(n_points)]
    for idx, line in enumerate(all_lines):
        for pt_idx in line:
            point_to_lines[pt_idx].append(idx)
            
    # Helper class to manage the state of C and witness counts
    class NikodymState:
        def __init__(self):
            self.in_C = [False] * n_points
            self.line_c_count = [0] * n_lines
            self.line_c_point = [-1] * n_lines  # if count is 1, store the point
            self.witness_N = [0] * n_points  # count of lines L s.t. L ∩ C = ∅
            self.witness_C = [0] * n_points  # count of lines L s.t. L ∩ C = {x}
            # We don't need the full sets, just counts for validity checks
            # But for local search, we might need to know *which* lines
            # Let's stick to counts for speed, and recompute if needed for complex moves
            
            # Initialize witness_N: all lines have count 0, so all points have n_lines as witness_N?
            # No, witness_N[x] is the number of lines L through x such that L ∩ C = ∅.
            # Initially C is empty, so all lines have count 0.
            # So witness_N[x] = len(point_to_lines[x]) for all x.
            for x in range(n_points):
                self.witness_N[x] = len(point_to_lines[x])
                
        def add_to_C(self, x_idx):
            """Add x_idx to C. Returns False if invalid."""
            if self.in_C[x_idx]:
                return True
            if self.witness_N[x_idx] == 0:
                return False
                
            # Check validity
            valid = True
            updates = []
            
            for l_idx in point_to_lines[x_idx]:
                count = self.line_c_count[l_idx]
                if count == 0:
                    # Line becomes count 1
                    # Check if any y in line \ {x_idx} has witness_N[y] == 1 and this is their only witness
                    # We don't track *which* lines are witnesses, only counts.
                    # This is a problem for exact validity check in greedy.
                    # We need to know if removing this witness from y leaves y with 0 witnesses.
                    # i.e., if witness_N[y] == 1, then y loses its last witness.
                    # But wait, if witness_N[y] == 1, it means there is exactly one line L' through y with count 0.
                    # If L_idx is that line, then y becomes invalid.
                    # So we just need to check if witness_N[y] == 1.
                    # But we don't know if L_idx is the one.
                    # Actually, if witness_N[y] == 1, and L_idx is a line through y with count 0,
                    # then L_idx MUST be the unique witness.
                    # So we just check witness_N[y] == 1.
                    for y_idx in all_lines[l_idx]:
                        if y_idx == x_idx:
                            continue
                        if self.witness_N[y_idx] == 1:
                            valid = False
                            break
                    if not valid:
                        break
                    updates.append((l_idx, '0_to_1'))
                elif count == 1:
                    y0 = self.line_c_point[l_idx]
                    if y0 != x_idx:
                        # Line becomes count 2
                        # y0 loses this witness from witness_C
                        if self.witness_C[y0] == 1:
                            valid = False
                            break
                        updates.append((l_idx, '1_to_2', y0))
                else:
                    pass
                    
            if not valid:
                return False
                
            # Apply changes
            self.in_C[x_idx] = True
            for update in updates:
                if update[1] == '0_to_1':
                    l_idx = update[0]
                    self.line_c_count[l_idx] = 1
                    self.line_c_point[l_idx] = x_idx
                    self.witness_C[x_idx] += 1
                    for y_idx in all_lines[l_idx]:
                        if y_idx == x_idx:
                            continue
                        self.witness_N[y_idx] -= 1
                elif update[1] == '1_to_2':
                    l_idx = update[0]
                    y0 = update[2]
                    self.line_c_count[l_idx] = 2
                    self.line_c_point[l_idx] = -1
                    self.witness_C[y0] -= 1
                    
            return True
            
        def remove_from_C(self, x_idx):
            """Remove x_idx from C. Returns False if invalid (shouldn't happen if we track correctly)."""
            if not self.in_C[x_idx]:
                return True
            
            # We need to reverse the updates.
            # This requires knowing which lines were affected.
            # We can recompute the effect.
            
            # For each line l through x_idx:
            # If count was 1 (and x_idx was the point): becomes 0
            #   witness_N[y] += 1 for all y in line \ {x_idx}
            #   witness_C[x_idx] -= 1
            # If count was 2 (and x_idx was one of the points? No, we don't track which 2 points)
            # This is the flaw of count-only tracking.
            # To support removal, we need to track the set of points in C for each line, or at least the count and be able to reverse.
            # Given the complexity, let's restrict local search to "add" moves only, 
            # or use a different state representation.
            
            # Alternative: Use a full state with sets for local search, but it's slow.
            # Let's stick to greedy "add only" for now, but with better initialization.
            # Or, implement a simple "undo" stack for the greedy phase.
            
            # For now, return True but do nothing, as we won't use removal in the greedy phase.
            # We will use random restarts instead.
            return True

    # Run multiple random restarts with different initializations
    best_complement_size = 0
    best_in_C = None
    
    # Initialization strategies:
    # 1. Empty C
    # 2. Unital points
    # 3. Random points
    
    for restart in range(20):  # 20 restarts
        if time.time() > deadline - 10:
            break
            
        state = NikodymState()
        
        # Seed C with some points
        if restart == 0:
            # Empty
            pass
        elif restart == 1:
            # Unital points
            for i, (xa, xb, ya, yb) in enumerate(all_points):
                if (2*xa + ya*ya - yb*yb*k) % p == 0:
                    state.add_to_C(i)
        else:
            # Random points
            for i in range(n_points):
                if np.random.rand() < 0.1:
                    state.add_to_C(i)
                    
        # Greedy expansion
        # Use a priority queue or simple loop
        # Simple loop is fine for 2401 points
        improved = True
        while improved:
            improved = False
            if time.time() > deadline - 5:
                break
                
            # Shuffle points for randomness
            order = list(range(n_points))
            np.random.shuffle(order)
            
            for x_idx in order:
                if state.in_C[x_idx]:
                    continue
                if state.witness_N[x_idx] == 0:
                    continue
                    
                if state.add_to_C(x_idx):
                    improved = True
                    
        # Calculate complement size
        complement_size = sum(1 for i in range(n_points) if state.in_C[i])
        if complement_size > best_complement_size:
            best_complement_size = complement_size
            best_in_C = state.in_C.copy()
            
    # Build result N = complement of C
    if best_in_C is None:
        # Fallback: all points in N
        best_in_C = [False] * n_points
        
    pts = [all_points[i] for i in range(n_points) if not best_in_C[i]]
    result = np.array([[[pt[0], pt[1]], [pt[2], pt[3]]] for pt in pts], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
