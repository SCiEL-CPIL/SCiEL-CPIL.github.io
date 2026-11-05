import time
import random
import sys

N = 256
TIME_LIMIT = 85.0

# Precomputation
three_aps = []
three_aps_set = set()
element_to_3aps = {i: [] for i in range(1, N+1)}
for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        ap = (a, b, c)
        three_aps.append(ap)
        three_aps_set.add(ap)
        for elem in ap:
            element_to_3aps[elem].append(ap)

four_aps = []
element_to_4aps = {i: [] for i in range(1, N+1)}
for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        ap = (a, b, c, e)
        four_aps.append(ap)
        for elem in ap:
            element_to_4aps[elem].append(ap)

# Map 4-APs to indices for fast lookup
four_ap_index = {ap: i for i, ap in enumerate(four_aps)}
num_4aps = len(four_aps)

# Precompute for each 3-AP, the 4-APs that contain it (as a subset)
# A 3-AP (a, b, c) is contained in (a, b, c, a+3d) if a+3d <= N
# and in (a-d, a, b, c) if a-d >= 1
three_ap_to_four_aps = {}
for i, (a, b, c) in enumerate(three_aps):
    d = b - a
    contained = []
    # Extend forward
    if a + 3 * d <= N:
        next_ap = (a, b, c, a + 3 * d)
        if next_ap in four_ap_index:
            contained.append(four_ap_index[next_ap])
    # Extend backward
    if a - d >= 1:
        prev_ap = (a - d, a, b, c)
        if prev_ap in four_ap_index:
            contained.append(four_ap_index[prev_ap])
    three_ap_to_four_aps[i] = contained

def count_3aps_from_elements(A):
    A_set = set(A)
    count = 0
    for a, b, c in three_aps:
        if a in A_set and b in A_set and c in A_set:
            count += 1
    return count

def greedy_initial(seed_val=0):
    random.seed(seed_val)
    A = set()
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    for x in candidates:
        valid = True
        for ap in element_to_4aps[x]:
            others = [elem for elem in ap if elem != x]
            if all(o in A for o in others):
                valid = False
                break
        if valid:
            A.add(x)
    return A

def simulated_annealing(A_init, deadline, seed_val=42):
    random.seed(seed_val)
    A = set(A_init)
    
    # State: Set of elements A
    # We track the 3-APs present in A
    # And the 4-APs present in A (violations)
    
    # Initialize counts
    # For efficiency, we maintain:
    # active_3aps: set of indices of 3-APs fully contained in A
    # active_4aps: set of indices of 4-APs fully contained in A
    
    active_3aps = set()
    active_4aps = set()
    
    # Helper to update counts when adding/removing an element
    # This is O(deg(x)) which is small.
    
    # Initial setup
    for x in A:
        for ap_idx in element_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if a in A and b in A and c in A:
                active_3aps.add(ap_idx)
        for ap_idx in element_to_4aps[x]:
            a, b, c, d = four_aps[ap_idx]
            if a in A and b in A and c in A and d in A:
                active_4aps.add(ap_idx)
                
    # Wait, the above initial setup is O(|A| * deg) but checking membership is O(1).
    # Actually, simpler: just iterate all 3-APs and 4-APs once?
    # N=256, 3-APs ~16k, 4-APs ~8k. Iterating all is fast enough for init.
    active_3aps = set()
    for i, (a, b, c) in enumerate(three_aps):
        if a in A and b in A and c in A:
            active_3aps.add(i)
            
    active_4aps = set()
    for i, (a, b, c, d) in enumerate(four_aps):
        if a in A and b in A and c in A and d in A:
            active_4aps.add(i)

    current_3count = len(active_3aps)
    current_4count = len(active_4aps)
    
    best_A = set(A)
    best_score = current_3count - 100 * current_4count # Heuristic score for tracking best valid
    
    T = 50.0
    T_min = 0.01
    cooling = 0.999
    
    start_time = time.time()
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        # Choose a move: Add, Remove, or Swap
        move_type = random.random()
        
        if move_type < 0.3: # Add
            candidates = [x for x in range(1, N + 1) if x not in A]
            if not candidates: continue
            x = random.choice(candidates)
            
            # Calculate Delta
            delta_3 = 0
            for ap_idx in element_to_3aps[x]:
                a, b, c = three_aps[ap_idx]
                others = [e for e in (a, b, c) if e != x]
                if all(o in A for o in others):
                    delta_3 += 1
                    
            delta_4 = 0
            for ap_idx in element_to_4aps[x]:
                a, b, c, d = four_aps[ap_idx]
                others = [e for e in (a, b, c, d) if e != x]
                if all(o in A for o in others):
                    delta_4 += 1
            
            new_3 = current_3count + delta_3
            new_4 = current_4count + delta_4
            
            # Acceptance: We want to maximize 3-APs and minimize 4-APs.
            # Score function: S = 3count - M * 4count
            # Delta S = delta_3 - M * delta_4
            M = 100
            delta_S = delta_3 - M * delta_4
            
            accept = False
            if delta_S >= 0:
                accept = True
            else:
                accept = random.random() < pow(2.718281828, delta_S / T)
                
            if accept:
                A.add(x)
                current_3count = new_3
                current_4count = new_4
                
                # Update active sets (simplified: just re-evaluate affected? No, just add to sets if condition met)
                # Actually, for correctness of active sets, we need to update them.
                # But since we only use them for counting, and we calculate delta, we can just update counts.
                # However, for future moves, we need to know which 3/4-APs are active.
                # So we must update active_3aps and active_4aps.
                
                for ap_idx in element_to_3aps[x]:
                    a, b, c = three_aps[ap_idx]
                    if a in A and b in A and c in A:
                        active_3aps.add(ap_idx)
                for ap_idx in element_to_4aps[x]:
                    a, b, c, d = four_aps[ap_idx]
                    if a in A and b in A and c in A and d in A:
                        active_4aps.add(ap_idx)
                        
                if current_4count == 0 and current_3count > count_3aps_from_elements(best_A):
                    best_A = set(A)
                    best_score = current_3count
                    
        elif move_type < 0.6: # Remove
            if not A: continue
            x = random.choice(list(A))
            
            delta_3 = 0
            for ap_idx in element_to_3aps[x]:
                a, b, c = three_aps[ap_idx]
                others = [e for e in (a, b, c) if e != x]
                if all(o in A for o in others):
                    delta_3 += 1
                    
            delta_4 = 0
            for ap_idx in element_to_4aps[x]:
                a, b, c, d = four_aps[ap_idx]
                others = [e for e in (a, b, c, d) if e != x]
                if all(o in A for o in others):
                    delta_4 += 1
            
            new_3 = current_3count - delta_3
            new_4 = current_4count - delta_4
            
            M = 100
            delta_S = -delta_3 + M * delta_4
            
            accept = False
            if delta_S >= 0:
                accept = True
            else:
                accept = random.random() < pow(2.718281828, delta_S / T)
                
            if accept:
                A.remove(x)
                current_3count = new_3
                current_4count = new_4
                
                for ap_idx in element_to_3aps[x]:
                    a, b, c = three_aps[ap_idx]
                    if a not in A or b not in A or c not in A:
                        active_3aps.discard(ap_idx)
                for ap_idx in element_to_4aps[x]:
                    a, b, c, d = four_aps[ap_idx]
                    if a not in A or b not in A or c not in A or d not in A:
                        active_4aps.discard(ap_idx)
                        
                if current_4count == 0 and current_3count > count_3aps_from_elements(best_A):
                    best_A = set(A)
                    best_score = current_3count
                    
        else: # Swap
            if len(A) < 2: continue
            x = random.choice(list(A))
            candidates = [y for y in range(1, N + 1) if y not in A]
            if not candidates: continue
            y = random.choice(candidates)
            
            # Remove x, Add y
            # Delta 3:
            # Lost 3-APs containing x (but not y)
            # Gained 3-APs containing y (but not x)
            # Note: 3-APs containing both x and y? Impossible since x!=y and 3-AP has distinct elements.
            
            lost_3 = 0
            for ap_idx in element_to_3aps[x]:
                a, b, c = three_aps[ap_idx]
                others = [e for e in (a, b, c) if e != x]
                if all(o in A for o in others):
                    # Check if y is in this AP? No, x and y are distinct.
                    # But if y was already in A, this AP is still active?
                    # Wait, we are removing x. So this AP is broken unless y is in it (impossible).
                    # So it is lost.
                    lost_3 += 1
                    
            gained_3 = 0
            for ap_idx in element_to_3aps[y]:
                a, b, c = three_aps[ap_idx]
                others = [e for e in (a, b, c) if e != y]
                # Check if others are in A (excluding y, and x is still in A for this check? 
                # In a swap, we conceptually remove x then add y.
                # So for y's APs, we check if the other 2 are in A\{x} U {y}?
                # No, standard swap: A' = (A - {x}) U {y}.
                # So for AP containing y, we need other 2 in A' .
                # Since y is not in A, others must be in A - {x}.
                # So check if others are in A and neither is x.
                if all(o in A for o in others) and x not in others:
                    gained_3 += 1
            
            lost_4 = 0
            for ap_idx in element_to_4aps[x]:
                a, b, c, d = four_aps[ap_idx]
                others = [e for e in (a, b, c, d) if e != x]
                if all(o in A for o in others):
                    lost_4 += 1
                    
            gained_4 = 0
            for ap_idx in element_to_4aps[y]:
                a, b, c, d = four_aps[ap_idx]
                others = [e for e in (a, b, c, d) if e != y]
                if all(o in A for o in others) and x not in others:
                    gained_4 += 1
            
            delta_3 = gained_3 - lost_3
            delta_4 = gained_4 - lost_4
            
            M = 100
            delta_S = delta_3 - M * delta_4
            
            accept = False
            if delta_S >= 0:
                accept = True
            else:
                accept = random.random() < pow(2.718281828, delta_S / T)
                
            if accept:
                A.remove(x)
                A.add(y)
                current_3count += delta_3
                current_4count += delta_4
                
                # Update active sets
                for ap_idx in element_to_3aps[x]:
                    a, b, c = three_aps[ap_idx]
                    if a not in A or b not in A or c not in A:
                        active_3aps.discard(ap_idx)
                for ap_idx in element_to_4aps[x]:
                    a, b, c, d = four_aps[ap_idx]
                    if a not in A or b not in A or c not in A or d not in A:
                        active_4aps.discard(ap_idx)
                        
                for ap_idx in element_to_3aps[y]:
                    a, b, c = three_aps[ap_idx]
                    if a in A and b in A and c in A:
                        active_3aps.add(ap_idx)
                for ap_idx in element_to_4aps[y]:
                    a, b, c, d = four_aps[ap_idx]
                    if a in A and b in A and c in A and d in A:
                        active_4aps.add(ap_idx)
                        
                if current_4count == 0 and current_3count > count_3aps_from_elements(best_A):
                    best_A = set(A)
                    best_score = current_3count
                    
        T *= cooling
        
    # Final check
    if current_4count == 0 and current_3count > count_3aps_from_elements(best_A):
        best_A = set(A)
        
    return best_A

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_set = None
    best_score = -1
    
    # Run SA with a few different seeds/initializations
    num_runs = 5
    
    for i in range(num_runs):
        if time.time() > deadline - 15:
            break
            
        init_set = greedy_initial(i)
        
        # Allocate remaining time for SA
        remaining = deadline - time.time()
        if remaining < 2:
            continue
            
        sa_set = simulated_annealing(init_set, time.time() + remaining * 0.9, seed_val=100+i)
        
        # Verify and score
        score = count_3aps_from_elements(sa_set)
        
        # Check validity (should be 4-AP free if SA converged, but ensure)
        # SA might end with 4-APs if T is too high at end?
        # We track best_A which is only updated if current_4count == 0.
        # So best_A returned is 4-AP free.
        
        if score > best_score:
            best_score = score
            best_set = sa_set
            
    if best_set is None:
        best_set = greedy_initial(0)
        
    return sorted(best_set)

if __name__ == "__main__":
    result = construct()
    print(f"Cardinality: {len(result)}")
    print(f"3-AP Count: {count_3aps_from_elements(result)}")
