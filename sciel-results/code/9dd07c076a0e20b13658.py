import random
import time

def construct():
    n = 256
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    # Precompute all 3-term APs: (a, b, c) where a < b < c and b-a = c-b
    all_3aps = []
    ap_to_elements = {}  # ap_index -> (a, b, c)
    elem_to_3aps = {i: [] for i in range(1, n + 1)}  # element -> list of ap_indices where it appears
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            c = 2 * b - a
            if c > b and c <= n:
                idx = len(all_3aps)
                all_3aps.append((a, b, c))
                ap_to_elements[idx] = (a, b, c)
                elem_to_3aps[a].append(idx)
                elem_to_3aps[b].append(idx)
                elem_to_3aps[c].append(idx)
    
    total_3aps = len(all_3aps)
    
    # Precompute all 4-term APs: (a, b, c, d) where a < b < c < d and b-a = c-b = d-c
    all_4aps = []
    ap4_to_elements = {}  # ap4_index -> (a, b, c, d)
    elem_to_4aps = {i: [] for i in range(1, n + 1)}  # element -> list of ap4_indices where it appears
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            d = b - a
            c = b + d
            e = c + d
            if e <= n:
                idx = len(all_4aps)
                all_4aps.append((a, b, c, e))
                ap4_to_elements[idx] = (a, b, c, e)
                elem_to_4aps[a].append(idx)
                elem_to_4aps[b].append(idx)
                elem_to_4aps[c].append(idx)
                elem_to_4aps[e].append(idx)
    
    total_4aps = len(all_4aps)
    
    def count_3aps_in_set(S):
        """Count 3-APs fully contained in set S."""
        S_set = set(S)
        count = 0
        for idx in range(total_3aps):
            a, b, c = ap_to_elements[idx]
            if a in S_set and b in S_set and c in S_set:
                count += 1
        return count
    
    def has_4ap_in_set(S):
        """Check if set S contains a 4-term AP."""
        S_set = set(S)
        for idx in range(total_4aps):
            a, b, c, d = ap4_to_elements[idx]
            if a in S_set and b in S_set and c in S_set and d in S_set:
                return True
        return False
    
    def delta_3ap_count_add(S_set, x):
        """Count how many 3-APs are completed by adding element x to S_set."""
        count = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = ap_to_elements[ap_idx]
            # Check if the other two elements are in S_set
            other = [y for y in (a, b, c) if y != x]
            if all(y in S_set for y in other):
                count += 1
        return count
    
    def delta_3ap_count_remove(S_set, x):
        """Count how many 3-APs are lost by removing element x from S_set."""
        count = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = ap_to_elements[ap_idx]
            other = [y for y in (a, b, c) if y != x]
            if all(y in S_set for y in other):
                count += 1
        return count
    
    def creates_4ap(S_set, x):
        """Check if adding x to S_set creates any 4-AP."""
        for ap4_idx in elem_to_4aps[x]:
            a, b, c, d = ap4_to_elements[ap4_idx]
            other = [y for y in (a, b, c, d) if y != x]
            if all(y in S_set for y in other):
                return True
        return False
    
    def breaks_4ap(S_set, x):
        """Check if removing x from S_set breaks any 4-AP (i.e., x is part of an active 4-AP)."""
        for ap4_idx in elem_to_4aps[x]:
            a, b, c, d = ap4_to_elements[ap4_idx]
            other = [y for y in (a, b, c, d) if y != x]
            if all(y in S_set for y in other):
                return True
        return False
    
    # Greedy construction: add elements in order, skip if it creates a 4-AP
    # Try multiple orderings to get a good starting point
    random.seed(42)
    
    best_set = None
    best_count = 0
    
    # Try natural order first
    selected = set()
    for x in range(1, n + 1):
        if not creates_4ap(selected, x):
            selected.add(x)
    
    current_count = count_3aps_in_set(selected)
    if current_count > best_count:
        best_count = current_count
        best_set = set(selected)
    
    # Try random orderings for better initial solutions
    for trial in range(10):
        if time.time() - start_time > deadline - 5:
            break
        
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        selected = set()
        for x in order:
            if not creates_4ap(selected, x):
                selected.add(x)
        
        current_count = count_3aps_in_set(selected)
        if current_count > best_count:
            best_count = current_count
            best_set = set(selected)
    
    # Simulated annealing
    selected = set(best_set)
    current_count = best_count
    best_set_sa = set(selected)
    best_count_sa = best_count
    
    # Temperature schedule
    T_start = 5.0
    T_end = 0.1
    cooling_rate = 0.999
    
    T = T_start
    no_improve = 0
    
    while time.time() - start_time < deadline:
        # Pick a random move: add, remove, or swap
        move_type = random.random()
        improved = False
        
        if move_type < 0.3:
            # Try adding a random element
            candidates_to_add = [x for x in range(1, n + 1) if x not in selected]
            if not candidates_to_add:
                continue
            x = random.choice(candidates_to_add)
            
            if creates_4ap(selected, x):
                continue
            
            delta = delta_3ap_count_add(selected, x)
            new_count = current_count + delta
            
            if new_count > current_count or random.random() < (1.0 / (1.0 + T)) * max(1, delta):
                selected.add(x)
                current_count = new_count
                improved = True
            else:
                # Accept with probability based on temperature (for non-improving moves)
                if delta < 0 and random.random() < min(1.0, T / max(1, -delta)):
                    selected.add(x)
                    current_count = new_count
                    improved = True
        
        elif move_type < 0.6:
            # Try removing a random element
            candidates_to_remove = list(selected)
            if not candidates_to_remove:
                continue
            x = random.choice(candidates_to_remove)
            
            delta = -delta_3ap_count_remove(selected, x)
            new_count = current_count + delta
            
            if new_count > current_count or random.random() < min(1.0, T / max(1, -delta)):
                selected.remove(x)
                current_count = new_count
                improved = True
        
        else:
            # Try a swap: remove one, add another
            candidates_to_remove = list(selected)
            candidates_to_add = [x for x in range(1, n + 1) if x not in selected]
            
            if not candidates_to_remove or not candidates_to_add:
                continue
            
            x_rm = random.choice(candidates_to_remove)
            x_add = random.choice(candidates_to_add)
            
            # Check if adding x_add creates a 4-AP (considering x_rm is removed)
            trial_set = set(selected)
            trial_set.remove(x_rm)
            
            if creates_4ap(trial_set, x_add):
                continue
            
            # Calculate delta
            delta_remove = -delta_3ap_count_remove(selected, x_rm)
            delta_add = delta_3ap_count_add(trial_set, x_add)
            delta = delta_remove + delta_add
            new_count = current_count + delta
            
            if new_count > current_count or random.random() < min(1.0, T / max(1, -delta)):
                selected = trial_set
                selected.add(x_add)
                current_count = new_count
                improved = True
        
        if current_count > best_count_sa:
            best_count_sa = current_count
            best_set_sa = set(selected)
            no_improve = 0
        else:
            no_improve += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
        
        # Occasionally reset temperature if stuck
        if no_improve > 1000:
            T = max(T, T_start * 0.5)
            no_improve = 0
    
    if best_count_sa > best_count:
        best_set = best_set_sa
        best_count = best_count_sa
    
    return sorted(best_set)
