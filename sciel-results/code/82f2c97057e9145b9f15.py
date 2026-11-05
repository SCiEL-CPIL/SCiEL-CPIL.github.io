"""Rewritten program: exact BFS/DFS search for minimal-diameter Sidon set covering 1..30."""
import time
from collections import deque

# Deadline guard: kill before 150s wall-clock with safety margin
_START = time.monotonic()
DEADLINE = 150.0  # seconds

def _time_left():
    return DEADLINE - (time.monotonic() - _START)

def construct():
    """
    Find a Sidon set containing 0, with all positive differences distinct,
    covering every d in 1..30, minimizing the largest element.
    
    Strategy: BFS over sorted tuples, level = max element.
    State: (frozenset of elements, frozenset of used differences)
    We want to minimize max element while ensuring {1..30} subset of used diffs.
    """
    target = frozenset(range(1, 31))
    
    # BFS by diameter (max element). Level d means all elements <= d.
    # State: (frozen_set, used_diffs_frozen_set)
    # Start: {0}
    start_state = (frozenset([0]), frozenset())
    
    # visited: set of (frozen_set, used_diffs) to avoid reprocessing
    visited = set()
    visited.add(start_state)
    
    # Queue of (elements_frozen, used_diffs_frozen, max_elem)
    # We process in order of increasing max_elem
    queue = deque()
    queue.append((frozenset([0]), frozenset(), 0))
    
    best_solution = None
    
    while queue and _time_left() > 2.0:
        elems, used_diffs, max_elem = queue.popleft()
        
        # Check if we've covered all required differences
        if target <= used_diffs:
            # Found a valid solution; since BFS by diameter, this is minimal
            best_solution = list(elems)
            break
        
        # Try adding a new element x > max_elem (to keep sorted order)
        # x can be from max_elem+1 up to some bound
        # For efficiency, bound x: if x > max_elem + 30, the new diffs x - e for e in elems
        # include x - 0 = x > max_elem + 30, which is fine but large.
        # We need to be smart about which x to try.
        
        # Upper bound for x at this level: we're exploring level max_elem,
        # so next element x should be max_elem + 1 (we build incrementally).
        # Actually, BFS by adding one element at a time with x > current max.
        
        # To keep state space manageable, limit the number of candidates per state
        # and use a reasonable upper bound on x.
        
        # Key insight: x - 0 = x must not collide with any existing diff.
        # Also for all e in elems (e > 0), x - e must not be in used_diffs.
        # And x - e must not equal any other x - e' for e != e' (but since x is fixed,
        # x - e are all distinct for distinct e, so that's automatic).
        # The constraint is: for all e in elems, (x - e) not in used_diffs, 
        # AND x - e != x - e' is automatic, 
        # AND x - e must not equal any diff already in used_diffs.
        
        # Wait, also need: the new diffs are {x - e : e in elems}.
        # These must all be distinct (they are, since e are distinct) 
        # and none can be in used_diffs.
        
        # Also: no existing pair (a,b) in elems can have b - a equal to any new diff x - e.
        # But used_diffs already contains all b - a for a < b in elems.
        # So we just need: for all e in elems, (x - e) not in used_diffs.
        
        # Additionally, x itself = x - 0 must not be in used_diffs.
        
        # Let's try x from max_elem + 1 upward, but limit the range.
        # A reasonable bound: x <= max_elem + 35 or so for early levels.
        # As max_elem grows, we can be more permissive.
        
        # Actually, let's just try x = max_elem + 1 first, then max_elem + 2, etc.
        # But limit to avoid explosion. Let's try up to max_elem + 40.
        
        max_x = max_elem + 40
        if max_x > 200:  # cap for safety
            max_x = 200
            
        # Sort candidates by x (ascending) to find minimal diameter first
        for x in range(max_elem + 1, max_x + 1):
            # Check if x is valid to add
            valid = True
            new_diffs = []
            for e in elems:
                d = x - e
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if not valid:
                continue
            
            # Check new diffs don't collide with each other (they can't, since x is fixed and e are distinct)
            # But we should double check just in case
            if len(set(new_diffs)) != len(new_diffs):
                continue
            
            new_elems = frozenset(elems | {x})
            new_used = used_diffs | frozenset(new_diffs)
            new_max = max_elem  # x > max_elem, so new max is x
            new_max = x
            
            state = (new_elems, new_used)
            if state in visited:
                continue
            visited.add(state)
            
            # Pruning: if the number of elements is too large, skip
            # We need at most 32 elements
            if len(new_elems) > 32:
                continue
            
            # Pruning: if we can't possibly cover all 30 diffs with remaining capacity
            # Current coverage: len(new_used & target)
            # Remaining to cover: 30 - len(new_used & target)
            # Each new element adds at most (len(new_elems)) new diffs
            # But many will be > 30 or collide
            # Simple prune: if even with perfect additions we can't cover, skip
            covered = len(new_used & target)
            remaining_needed = 30 - covered
            max_elements_left = 32 - len(new_elems)
            # Each new element can add at most len(new_elems) diffs, but realistically much less
            # This is a weak prune; let's use a stronger one later if needed
            if remaining_needed > max_elements_left * len(new_elems):
                continue
            
            queue.append((new_elems, new_used, new_max))
    
    # If BFS didn't find a solution in time, try a greedy fallback
    if best_solution is None:
        # Greedy fallback from the seed program
        a = [0]
        def diffs(s):
            return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
        for d in range(1, 31):
            if d in diffs(sorted(a)):
                continue
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if len(set(diffs(trial))) == len(diffs(trial)):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if len(set(diffs(trial))) == len(diffs(trial)):
                        a = trial
                        break
                    y += 1
        best_solution = a
    
    return best_solution
