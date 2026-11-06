"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

_N = 256
# Precompute: for each pair (a,b) with a<b, the forbidden third element c = 2*b-a
# and also c = 2*a-b (if in range). We need to ensure no 3-term AP.
# A 3-term AP is {a, b, c} with a+c = 2*b (b is middle).
# So for any two selected elements x, y, we cannot select z such that
#   z = 2*y - x  (y is middle, x < y < z)
#   z = 2*x - y  (x is middle, y < x < z)  -- same thing just reordered
#   z = 2*(x+y) - ... no wait.
# Actually: three numbers a < b < c form a 3-AP iff a + c = 2*b.
# So if we have selected a and c, we cannot select b = (a+c)/2.
# If we have selected a and b (a<b), we cannot select c = 2*b - a.
# If we have selected b and c (b<c), we cannot select a = 2*b - c.

# Precompute conflicts: conflict[x] = set of y such that x and y cannot both be in the set
# Two elements x, y conflict if there exists z in [1,256] such that {x,y,z} is a 3-AP.
# That happens when:
#   - x + y is even and (x+y)/2 is in [1,256] and != x, != y  -> middle = (x+y)/2
#   - 2*y - x is in [1,256] and != x, != y  -> x is smallest, y is middle, 2y-x is largest
#   - 2*x - y is in [1,256] and != x, != y  -> y is smallest, x is middle, 2x-y is largest

def _build_conflict_graph():
    """Build adjacency: conflict[x] is a set of y values that conflict with x."""
    conflict = [set() for _ in range(_N + 1)]
    for x in range(1, _N + 1):
        for y in range(x + 1, _N + 1):
            # Check if x, y, z form a 3-AP for some z in [1, _N]
            # Case 1: x is smallest, y is middle -> z = 2*y - x
            z = 2 * y - x
            if 1 <= z <= _N and z != x and z != y:
                conflict[x].add(y)
                conflict[y].add(x)
                conflict[x].add(z)
                conflict[z].add(x)
                conflict[y].add(z)
                conflict[z].add(y)
            # Case 2: y is smallest, x is middle -> z = 2*x - y
            z = 2 * x - y
            if 1 <= z <= _N and z != x and z != y:
                conflict[x].add(y)
                conflict[y].add(x)
                conflict[x].add(z)
                conflict[z].add(x)
                conflict[y].add(z)
                conflict[z].add(y)
    return conflict

def _greedy_init(conflict):
    """Greedy: pick elements in order of 1/x (i.e., smallest first) if no conflict with already picked."""
    selected = []
    selected_set = set()
    for x in range(1, _N + 1):
        if not (selected_set & conflict[x]):
            selected.append(x)
            selected_set.add(x)
    return selected

def _weight(selected):
    """Compute exact weight as Fraction."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def _check_valid(selected):
    """Verify no 3-term AP exists in selected."""
    sel_set = set(selected)
    for a in selected:
        for b in selected:
            if a >= b:
                continue
            c = 2 * b - a
            if c in sel_set:
                return False
    return True

def _local_search(selected, conflict, deadline):
    """Try 1-swap and 1-add improvements until no improvement or deadline."""
    selected_set = set(selected)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        current_w = _weight(selected)
        
        # Try adding elements not in selected
        best_add = None
        best_add_gain = Fraction(0, 1)
        for x in range(1, _N + 1):
            if x not in selected_set:
                if not (selected_set & conflict[x]):
                    gain = Fraction(1, x)
                    if gain > best_add_gain:
                        best_add_gain = gain
                        best_add = x
        if best_add is not None:
            selected.append(best_add)
            selected_set.add(best_add)
            improved = True
            continue
        
        # Try 1-swap: remove one, add another
        best_swap = None
        best_swap_gain = Fraction(0, 1)
        for rem in selected:
            if time.time() > deadline:
                break
            new_set = selected_set - {rem}
            for add in range(1, _N + 1):
                if add in new_set:
                    continue
                if not (new_set & conflict[add]):
                    gain = Fraction(1, add) - Fraction(1, rem)
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap = (rem, add)
        if best_swap is not None:
            rem, add = best_swap
            selected.remove(rem)
            selected_set.discard(rem)
            selected.append(add)
            selected_set.add(add)
            improved = True
    
    return selected

def _branch_and_bound(conflict, deadline, time_budget):
    """Branch and bound for maximum weight independent set.
    
    Nodes: elements 1..256. We process in order of decreasing weight (smallest x first).
    State: (index, current_set, current_weight, upper_bound)
    Upper bound: current_weight + sum of 1/x for all remaining candidates that don't conflict.
    """
    # Order by weight descending: element 1 has highest weight
    order = list(range(1, _N + 1))
    
    best_solution = []
    best_weight = Fraction(0, 1)
    
    # Use iterative DFS with explicit stack to avoid recursion depth issues
    # Stack entries: (idx, selected_list, selected_set, current_weight)
    # We use a greedy upper bound: current_weight + sum of weights of all unprocessed
    # elements that are compatible with current selection.
    
    # Precompute suffix sums of 1/x for quick upper bound
    # suffix[i] = sum of 1/j for j in order[i:]
    suffix = [Fraction(0, 1)] * (_N + 2)
    for i in range(_N - 1, -1, -1):
        suffix[i] = suffix[i + 1] + Fraction(1, order[i])
    
    # Initial greedy solution
    init_selected = _greedy_init(conflict)
    init_weight = _weight(init_selected)
    best_solution = list(init_selected)
    best_weight = init_weight
    
    # DFS stack: (position_in_order, selected_set, current_weight, selected_list)
    # Position means: we've decided on elements order[0..pos-1], now deciding order[pos]
    stack = [(0, set(), Fraction(0, 1), [])]
    nodes_explored = 0
    
    while stack:
        if time.time() > deadline:
            break
        
        pos, sel_set, cur_w, sel_list = stack.pop()
        nodes_explored += 1
        
        # Upper bound: current weight + all remaining compatible weights
        # Quick bound: cur_w + suffix[pos] (ignoring conflicts, so it's an overestimate)
        if cur_w + suffix[pos] <= best_weight:
            continue
        
        if pos == _N:
            if cur_w > best_weight:
                best_weight = cur_w
                best_solution = list(sel_list)
            continue
        
        x = order[pos]
        
        # Option 1: include x (if compatible)
        if not (sel_set & conflict[x]):
            new_sel_set = sel_set | {x}
            new_cur_w = cur_w + Fraction(1, x)
            new_sel_list = sel_list + [x]
            # Upper bound check
            if new_cur_w + suffix[pos + 1] > best_weight:
                stack.append((pos + 1, new_sel_set, new_cur_w, new_sel_list))
        
        # Option 2: exclude x
        # Upper bound for exclusion
        if cur_w + suffix[pos + 1] > best_weight:
            stack.append((pos + 1, sel_set, cur_w, sel_list))
    
    return best_solution

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s safety margin
    
    conflict = _build_conflict_graph()
    
    # Phase 1: Greedy initialization
    selected = _greedy_init(conflict)
    
    # Phase 2: Local search refinement
    if time.time() < deadline - 2.0:
        selected = _local_search(selected, conflict, deadline - 2.0)
    
    # Phase 3: Branch and bound (if time remains)
    if time.time() < deadline - 1.0:
        bb_solution = _branch_and_bound(conflict, deadline - 1.0, deadline - 1.0)
        if _weight(bb_solution) > _weight(selected):
            selected = bb_solution
    
    # Final local search polish
    if time.time() < deadline - 0.5:
        selected = _local_search(selected, conflict, deadline - 0.5)
    
    # Verify validity
    if not _check_valid(selected):
        # Fallback to greedy
        selected = _greedy_init(conflict)
    
    if not selected:
        selected = [1]
    
    return selected
# EVOLVE-BLOCK-END
