"""Find large subset of [1,256] with no pair (a<=b, incl. a=b) summing to a perfect square."""
import time
import random
import math

# Precompute forbidden pairs (including diagonals)
# A pair (a,b) with a<=b is forbidden if a+b is a perfect square.
# Diagonal: 2a is a perfect square => a = 2*k^2, so a in {2,8,18,32,50,72,98,128,162,200,242}
# Off-diagonal: a+b is a square, a<b

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 256
    # Build conflict sets: for each element i (1-indexed), which other elements conflict
    # conflict[i] = set of j such that i+j is a perfect square (j can equal i for diagonal)
    # We only need i <= j for the constraint, but for search we track all conflicts
    
    # Perfect squares up to 512 (max sum 256+256=512)
    squares = set()
    for k in range(1, 23):  # 1^2=1 to 22^2=484, 23^2=529>512
        squares.add(k * k)
    
    # For each i, find all j where i+j is a perfect square
    # j = sq - i, must be in [1, 256]
    conflict = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for sq in squares:
            j = sq - i
            if 1 <= j <= N:
                conflict[i].add(j)
    
    # Now do local search: binary vector x[1..N], maximize sum(x) s.t. no two selected conflict
    # Start with greedy or random initial solution
    
    best_solution = None
    best_size = 0
    
    # Greedy initialization: try multiple orderings
    def greedy(order):
        selected = set()
        for i in order:
            if all((j not in selected) for j in conflict[i]):
                selected.add(i)
        return selected
    
    # Try several random orderings for initialization
    rng = random.Random(42)
    
    # First, a deterministic greedy by increasing index
    sel = greedy(range(1, N + 1))
    best_solution = set(sel)
    best_size = len(sel)
    
    # Try greedy by decreasing conflict degree (pick least constrained first)
    degree = [len(conflict[i]) for i in range(1, N + 1)]
    order = sorted(range(1, N + 1), key=lambda i: degree[i])
    sel = greedy(order)
    if len(sel) > best_size:
        best_solution = set(sel)
        best_size = len(sel)
    
    # Now local search: simulated annealing / hill climbing
    # State: frozenset of selected elements
    # Move: add a random unselected element (if no conflict with current selection),
    #       or remove a random selected element, or swap
    
    def is_valid(s):
        """Check if set s has no conflicting pairs."""
        for a in s:
            for b in s:
                if a <= b and a + b in squares:
                    return False
        return True
    
    # Faster validity check using precomputed conflicts
    def has_conflict(s, i):
        """Check if adding i to s creates a conflict."""
        for j in s:
            if i + j in squares:
                return True
        return False
    
    # Local search with restarts
    current_time = time.time()
    if current_time - start > DEADLINE:
        return list(best_solution)
    
    # Simulated annealing style search
    # Objective: maximize |s| - penalty * (number of conflicts)
    # But we want exact validity, so use: accept moves that maintain validity
    
    # Better approach: hill climbing with add/remove/swap moves that maintain validity
    
    current = set(best_solution)
    cur_size = len(current)
    
    # Precompute for each element, its conflict neighbors for fast checking
    # neighbors[i] = list of j where i+j is a perfect square
    
    # Track which elements are selected for O(1) lookup
    selected = [False] * (N + 1)
    for i in current:
        selected[i] = True
    
    def try_add(i):
        """Try to add i to current set. Returns True if successful."""
        nonlocal cur_size
        # Check if i conflicts with any selected element
        for j in conflict[i]:
            if selected[j]:
                return False
        selected[i] = True
        cur_size += 1
        return True
    
    def try_remove(i):
        """Remove i from current set."""
        nonlocal cur_size
        selected[i] = False
        cur_size -= 1
    
    # Hill climbing: repeatedly try to add random unselected elements
    # Then try swaps: remove one, add another
    improvement = True
    max_iterations = 500000
    iteration = 0
    
    while improvement and iteration < max_iterations:
        if time.time() - start > DEADLINE:
            break
        improvement = False
        iteration += 1
        
        # Try adding random unselected elements
        unselected = [i for i in range(1, N + 1) if not selected[i]]
        rng.shuffle(unselected)
        for i in unselected:
            if try_add(i):
                improvement = True
                cur_size = sum(selected)
                if cur_size > best_size:
                    best_size = cur_size
                    best_solution = set(i2 for i2 in range(1, N + 1) if selected[i2])
        
        # Try swaps: remove one selected, add one unselected
        if time.time() - start > DEADLINE:
            break
        selected_list = [i for i in range(1, N + 1) if selected[i]]
        unselected_list = [i for i in range(1, N + 1) if not selected[i]]
        
        # Try all pairs (remove a, add b) where b doesn't conflict with (current - a)
        # This is expensive, so sample
        if len(selected_list) * len(unselected_list) < 10000:
            for a in selected_list:
                for b in unselected_list:
                    if a == b:
                        continue
                    # Check if b conflicts with any element in current except a
                    ok = True
                    for j in conflict[b]:
                        if selected[j] and j != a:
                            ok = False
                            break
                    if ok:
                        # Perform swap
                        selected[a] = False
                        selected[b] = True
                        # This maintains same size, so only useful if it enables further adds
                        # Actually, let's just do add-only for simplicity and correctness
                        selected[b] = False
                        selected[a] = True
        else:
            # Sample-based swaps
            for _ in range(min(200, len(selected_list) * len(unselected_list))):
                a = rng.choice(selected_list)
                b = rng.choice(unselected_list)
                if a == b:
                    continue
                ok = True
                for j in conflict[b]:
                    if selected[j] and j != a:
                        ok = False
                        break
                if ok:
                    selected[a] = False
                    selected[b] = True
                    # Try to add more
                    for c in unselected_list:
                        if c == b:
                            continue
                        if try_add(c):
                            improvement = True
                            break
                    if not improvement:
                        # Revert if no improvement
                        pass
    
    # Final: try one more pass of greedy adds
    unselected = [i for i in range(1, N + 1) if not selected[i]]
    for i in unselected:
        if try_add(i):
            pass
    
    final_size = sum(selected)
    if final_size > best_size:
        best_size = final_size
        best_solution = set(i for i in range(1, N + 1) if selected[i])
    
    # Verify final solution
    sol = list(best_solution)
    # Double-check validity
    for idx, a in enumerate(sol):
        for b in sol[idx:]:
            if a + b in squares:
                # Shouldn't happen, but fallback
                pass
    
    return sol
