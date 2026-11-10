# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    n = 128
    numbers = list(range(1, n + 1))

    # 1. Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_table[i][j] = l
            lcm_table[j][i] = l

    # 2. Identify all bad triples: (a,b,c) with a<b<c where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))

    # 3. Build conflict structure: for each element x, store pairs (u,v) such that {x,u,v} is bad
    conflict = {i: set() for i in range(1, n + 1)}
    for a, b, c in bad_triples:
        conflict[a].add((b, c))
        conflict[b].add((a, c))
        conflict[c].add((a, b))

    def is_valid(S, x):
        """Check if x can be added to set S without forming a bad triple."""
        for u, v in conflict[x]:
            if u in S and v in S:
                return False
        return True

    def greedy_add(S):
        """Add all valid elements to S."""
        S = set(S)
        changed = True
        while changed:
            changed = False
            for x in numbers:
                if x not in S and is_valid(S, x):
                    S.add(x)
                    changed = True
        return S

    # 4. Branch and bound for maximum independent set in 3-uniform hypergraph
    # Key insight: an element x is "conflicting" with set S if there exists (u,v) in conflict[x]
    # with both u,v in S. We want max |S| such that no element is conflicting.
    
    # Precompute degree (number of conflicting pairs) for ordering
    degree = {i: len(conflict[i]) for i in range(1, n + 1)}
    
    # Order elements by decreasing degree for branching
    ordered_elements = sorted(numbers, key=lambda x: -degree[x])
    
    # For B&B, we process elements in order. At each step, we decide include or exclude.
    # Upper bound: current_size + number of remaining elements that could potentially be added.
    # A tighter bound: for remaining elements, count those not immediately conflicting.
    
    best_set = set()
    best_size = 0
    nodes_explored = 0
    max_nodes = 500000  # Limit to stay within time budget

    def branch_and_bound(idx, current_set, remaining_count):
        """
        idx: index in ordered_elements
        current_set: set of included elements so far
        remaining_count: upper bound on how many more can be added
        """
        nonlocal best_set, best_size, nodes_explored, deadline
        
        if time.time() > deadline - 2:
            return
        
        nodes_explored += 1
        if nodes_explored > max_nodes:
            return
        
        # Pruning: if current + remaining can't beat best, stop
        if len(current_set) + remaining_count <= best_size:
            return
        
        if idx >= len(ordered_elements):
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
            return
        
        x = ordered_elements[idx]
        
        # Option 1: Include x
        if is_valid(current_set, x):
            new_set = current_set | {x}
            # Compute tighter upper bound for remaining
            # Count how many of remaining elements are still potentially addable
            rem_bound = 0
            for j in range(idx + 1, len(ordered_elements)):
                y = ordered_elements[j]
                if is_valid(new_set, y):
                    rem_bound += 1
                else:
                    # Still might be addable after adding other elements? No, adding only makes it harder
                    # Actually, removing elements could make it valid, but we're only adding
                    pass
            # Conservative: just use remaining_count - 1
            branch_and_bound(idx + 1, new_set, remaining_count - 1)
        
        # Option 2: Exclude x
        branch_and_bound(idx + 1, current_set, remaining_count - 1)

    # Initialize with a good lower bound using greedy
    initial_greedy = greedy_add(set())
    best_set = set(initial_greedy)
    best_size = len(initial_greedy)
    
    # Also try greedy from different orderings
    random.seed(42)
    for _ in range(5):
        random.shuffle(numbers)
        temp = set()
        for x in numbers:
            if is_valid(temp, x):
                temp.add(x)
        if len(temp) > best_size:
            best_size = len(temp)
            best_set = set(temp)
    
    # Run B&B with initial upper bound = n
    # But first, compute a better upper bound
    # Upper bound: for each element, if it's in any bad triple, we might need to exclude at least one from each triple
    # Simple upper bound: n - minimum vertex cover of the hypergraph
    # For now, use n as upper bound and let pruning handle it
    
    if time.time() < deadline - 10:
        branch_and_bound(0, set(), n)
    
    # 5. Local search refinement on the best set found
    if time.time() < deadline - 5 and best_set:
        current = set(best_set)
        # Add phase
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            for x in numbers:
                if x not in current and is_valid(current, x):
                    current.add(x)
                    changed = True
        # Swap phase: try 1-for-1 swaps
        for _ in range(500):
            if time.time() > deadline - 3:
                break
            elems = list(current)
            if not elems:
                break
            out_elem = random.choice(elems)
            trial = set(current)
            trial.discard(out_elem)
            improved = False
            for in_elem in numbers:
                if in_elem in trial:
                    continue
                if is_valid(trial, in_elem):
                    trial.add(in_elem)
                    # Try to add more
                    local_changed = True
                    while local_changed and time.time() < deadline - 3:
                        local_changed = False
                        for y in numbers:
                            if y not in trial and is_valid(trial, y):
                                trial.add(y)
                                local_changed = True
                    if len(trial) > len(current):
                        current = trial
                        improved = True
                    break
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
    
    # 6. Also try the greedy-remove-then-add-back strategy
    if time.time() < deadline - 10:
        S_all = set(numbers)
        conflict_count = {i: len(conflict[i]) for i in range(1, n + 1)}
        order = sorted(numbers, key=lambda x: -conflict_count[x])
        
        S_remove = set(S_all)
        for x in order:
            if time.time() > deadline - 5:
                break
            needs_removal = False
            for u, v in conflict[x]:
                if u in S_remove and v in S_remove:
                    needs_removal = True
                    break
            if needs_removal:
                S_remove.discard(x)
        
        S_back = set(S_remove)
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            for x in numbers:
                if x not in S_back and is_valid(S_back, x):
                    S_back.add(x)
                    changed = True
        
        if len(S_back) > best_size:
            best_size = len(S_back)
            best_set = S_back
    
    # 7. Fallback: ensure non-empty
    if not best_set:
        best_set = {1}
    
    return sorted(list(best_set))

# EVOLVE-BLOCK-END
