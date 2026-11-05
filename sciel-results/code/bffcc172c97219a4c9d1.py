import time
import random

def construct():
    N = 512
    deadline = time.time() + 85  # Safety margin before 90s kill
    
    # Precompute all 4-APs as tuples (a, b, c, d) where a < b < c < d
    # and b-a = c-b = d-c
    all_aps = []
    for a in range(1, N - 2):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            all_aps.append((a, b, c, e))
    
    # For each element x, store indices of APs that contain x
    # And for each AP, store which elements are in it
    ap_by_element = {x: [] for x in range(1, N + 1)}
    for i, ap in enumerate(all_aps):
        for x in ap:
            ap_by_element[x].append(i)
    
    # Count of "violations" for each AP: how many of its 4 elements are in S
    # An AP is violated if all 4 are in S
    # We track for each AP how many of its elements are currently in S
    ap_count = [0] * len(all_aps)  # number of elements of this AP in S
    # Number of violated APs (count == 4)
    violated = 0
    
    # Current set as a boolean array for O(1) lookup
    in_set = [False] * (N + 1)
    S = set()
    
    def add_element(x):
        """Add x to S, update ap_count and violated count."""
        nonlocal violated
        in_set[x] = True
        S.add(x)
        for i in ap_by_element[x]:
            ap_count[i] += 1
            if ap_count[i] == 4:
                violated += 1
    
    def remove_element(x):
        """Remove x from S, update ap_count and violated count."""
        nonlocal violated
        in_set[x] = False
        S.remove(x)
        for i in ap_by_element[x]:
            ap_count[i] -= 1
            if ap_count[i] == 3:
                violated -= 1
    
    def is_valid(x):
        """Check if x can be added without creating a new violation."""
        # Adding x would make any AP containing x go from count 3 to count 4
        for i in ap_by_element[x]:
            if ap_count[i] == 3:
                return False
        return True
    
    def fill(S_current, order=None):
        """Greedy fill: add all valid elements in given order."""
        if order is None:
            order = list(range(1, N + 1))
        for x in order:
            if not in_set[x] and is_valid(x):
                add_element(x)
    
    # Phase 1: Initial greedy construction
    # Try multiple orders to find a good starting point
    best_set = set()
    best_size = 0
    
    for trial in range(20):
        if time.time() > deadline:
            break
        # Reset
        for x in list(S):
            remove_element(x)
        S.clear()
        in_set = [False] * (N + 1)
        ap_count = [0] * len(all_aps)
        violated = 0
        
        if trial == 0:
            order = list(range(1, N + 1))
        elif trial == 1:
            order = list(range(N, 0, -1))
        else:
            order = list(range(1, N + 1))
            random.shuffle(order)
        
        fill(S, order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Phase 2: Simulated Annealing with periodic fill
    # State: current set S
    # Moves: add a valid element, remove an element, or swap (remove one, add another)
    
    current_set = set(best_set)
    # Reset state to match current_set
    S.clear()
    in_set = [False] * (N + 1)
    ap_count = [0] * len(all_aps)
    violated = 0
    for x in current_set:
        add_element(x)
    
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.001
    cooling_rate = 0.995
    
    T = T_start
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 5:
        if T < T_end:
            T = T_end
            if no_improve_count > max_no_improve:
                break
        
        # Choose a random move
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to add a random element not in S
            candidates = [x for x in range(1, N + 1) if not in_set[x]]
            if not candidates:
                continue
            x = random.choice(candidates)
            if is_valid(x):
                add_element(x)
                delta_obj = 1  # size increased
                # Always accept size increases
                accept = True
            else:
                # Would create a violation; accept with probability exp(-1/T)
                delta_obj = 0
                accept = random.random() < (0.01 / T)
                if accept:
                    add_element(x)
        elif move_type < 0.6:
            # Try to remove a random element from S
            if not S:
                continue
            x = random.choice(list(S))
            remove_element(x)
            delta_obj = -1
            # Accept with probability exp(1/T) since we're losing size
            accept = random.random() < (1.0 / T)
            if not accept:
                add_element(x)  # revert
        else:
            # Swap: remove one, add another
            if not S:
                continue
            x_remove = random.choice(list(S))
            candidates = [x for x in range(1, N + 1) if not in_set[x]]
            if not candidates:
                continue
            x_add = random.choice(candidates)
            
            # Check if swap is valid: after removing x_remove and adding x_add, no new violations
            # This is complex; let's just do it and check
            remove_element(x_remove)
            if is_valid(x_add):
                add_element(x_add)
                delta_obj = 0  # size unchanged
                accept = True
            else:
                # Adding x_add would create a violation
                # Accept with small probability
                accept = random.random() < (0.005 / T)
                if accept:
                    add_element(x_add)
                else:
                    # Revert: add back x_remove
                    add_element(x_remove)
        
        # Check for improvement
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Periodic fill: try to add all currently valid elements
        if no_improve_count % 1000 == 0 and no_improve_count > 0:
            order = list(range(1, N + 1))
            random.shuffle(order)
            fill(S, order)
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
                no_improve_count = 0
        
        # Cool down
        T *= cooling_rate
    
    # Final fill pass
    order = list(range(1, N + 1))
    random.shuffle(order)
    fill(S, order)
    if len(S) > best_size:
        best_size = len(S)
        best_set = set(S)
    
    return sorted(best_set)
