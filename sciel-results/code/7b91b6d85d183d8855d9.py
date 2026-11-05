<<<<<<< SEARCH
import time
import random
import math

# Precompute 4-APs for fast checking
N = 512
aps_by_element = [[] for _ in range(N + 1)]
for a in range(1, N + 1):
    # d such that a+3d <= N
    max_d = (N - a) // 3
    for d in range(1, max_d + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        # The AP is (a, b, c, e)
        aps_by_element[a].append((b, c, e))
        aps_by_element[b].append((a, c, e))
        aps_by_element[c].append((a, b, e))
        aps_by_element[e].append((a, b, c))

def can_add(x, S):
    """
    Check if x can be added to set S without creating a 4-AP.
    S is a set of integers.
    """
    # Check if any 4-AP containing x has its other 3 elements in S
    for others in aps_by_element[x]:
        if others[0] in S and others[1] in S and others[2] in S:
            return False
    return True

def construct():
    start_time = time.time()
    # Keep a safety margin of 10 seconds
    deadline = start_time + 50.0
    
    # Initial solution: Greedy
    def greedy_init():
        S = set()
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, S):
                S.add(x)
        # Second pass
        for x in range(1, N + 1):
            if x not in S:
                if can_add(x, S):
                    S.add(x)
        return S

    best_set = greedy_init()
    best_size = len(best_set)
    
    # Simulated Annealing
    current_set = set(best_set)
    current_size = best_size
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-6
    
    # Precompute list of elements for faster iteration
    all_elements = list(range(1, N + 1))
    
    while time.time() < deadline and T > min_T:
        # Move: Pick a random element x in current_set and remove it.
        # Then pick a random element y not in current_set and try to add it.
        # If y can be added, accept the move (swap).
        # If y cannot be added, revert (keep x).
        # This keeps size constant.
        
        # To allow size changes, we can also do:
        # 1. Add a random y not in S.
        # 2. Remove a random x in S.
        
        # Let's do a mix:
        # With prob 0.5: Add y
        # With prob 0.5: Remove x
        
        if random.random() < 0.5:
            # Add
            y = random.choice(all_elements)
            if y not in current_set:
                if can_add(y, current_set):
                    current_set.add(y)
                    current_size += 1
                    # Accept if size increased
                    # If size didn't increase (impossible here), use SA
                else:
                    # Cannot add
                    pass
        else:
            # Remove
            if current_size > 0:
                x = random.choice(list(current_set))
                current_set.remove(x)
                current_size -= 1
                # Accept if size decreased? No, SA allows it.
                # We need to track prev_size to decide acceptance.
        
        # This logic is getting messy. Let's use a cleaner SA structure.
        # State: (S, size)
        # Move: generate new state (S', size')
        # Accept if size' > size, or with prob exp((size' - size) / T)
        
        # Let's restart the loop with this structure.
        
        prev_set = set(current_set)
        prev_size = current_size
        
        # Generate move
        if random.random() < 0.5:
            # Add
            y = random.choice(all_elements)
            if y not in current_set:
                if can_add(y, current_set):
                    current_set.add(y)
                    current_size += 1
        
        else:
            # Remove
            if current_size > 0:
                x = random.choice(list(current_set))
                current_set.remove(x)
                current_size -= 1
        
        # Acceptance
        delta = current_size - prev_size
        if delta > 0:
            # Always accept
            pass
        else:
            # Accept with prob exp(delta / T)
            if delta < 0:
                prob = math.exp(delta / T)
                if random.random() > prob:
                    # Reject
                    current_set = prev_set
                    current_size = prev_size
        
        # Update best
        if current_size > best_size:
            best_size = current_size
            best_set = set(current_set)
        
        T *= cooling_rate
    
    return sorted(best_set)
=======
import time
import random
import math

# Precompute 4-APs for fast checking
N = 512
aps_by_element = [[] for _ in range(N + 1)]
for a in range(1, N + 1):
    # d such that a+3d <= N
    max_d = (N - a) // 3
    for d in range(1, max_d + 1):
        b = a + d
        c = a + 2 * d
        e = a + 3 * d
        # The AP is (a, b, c, e)
        aps_by_element[a].append((b, c, e))
        aps_by_element[b].append((a, c, e))
        aps_by_element[c].append((a, b, e))
        aps_by_element[e].append((a, b, c))

def can_add(x, in_S):
    """
    Check if x can be added to set S without creating a 4-AP.
    in_S is a boolean array where in_S[i] is True if i is in S.
    """
    # Check if any 4-AP containing x has its other 3 elements in S
    for others in aps_by_element[x]:
        if in_S[others[0]] and in_S[others[1]] and in_S[others[2]]:
            return False
    return True

def construct():
    start_time = time.time()
    # Keep a safety margin of 10 seconds
    deadline = start_time + 50.0
    
    # Initial solution: Greedy
    def greedy_init():
        in_S = [False] * (N + 1)
        S_list = []
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, in_S):
                in_S[x] = True
                S_list.append(x)
        # Second pass
        for x in range(1, N + 1):
            if not in_S[x]:
                if can_add(x, in_S):
                    in_S[x] = True
                    S_list.append(x)
        return in_S, S_list

    best_in_S, best_S_list = greedy_init()
    best_size = len(best_S_list)
    
    # Simulated Annealing
    in_S = [False] * (N + 1)
    S_list = []
    for x in best_S_list:
        in_S[x] = True
        S_list.append(x)
    current_size = best_size
    
    T = 1.0
    cooling_rate = 0.999
    min_T = 1e-6
    
    # Precompute list of elements for faster iteration
    all_elements = list(range(1, N + 1))
    
    while time.time() < deadline and T > min_T:
        prev_in_S = in_S[:]  # Copy is expensive!
        prev_S_list = S_list[:]
        prev_size = current_size
        
        # Generate move
        if random.random() < 0.5:
            # Add
            y = random.choice(all_elements)
            if not in_S[y]:
                if can_add(y, in_S):
                    in_S[y] = True
                    S_list.append(y)
                    current_size += 1
        
        else:
            # Remove
            if current_size > 0:
                idx = random.randint(0, current_size - 1)
                x = S_list[idx]
                in_S[x] = False
                S_list[idx] = S_list[-1]
                S_list.pop()
                current_size -= 1
        
        # Acceptance
        delta = current_size - prev_size
        if delta > 0:
            # Always accept
            pass
        else:
            # Accept with prob exp(delta / T)
            if delta < 0:
                prob = math.exp(delta / T)
                if random.random() > prob:
                    # Reject
                    in_S = prev_in_S
                    S_list = prev_S_list
                    current_size = prev_size
        
        # Update best
        if current_size > best_size:
            best_size = current_size
            best_in_S = in_S[:]
            best_S_list = S_list[:]
        
        T *= cooling_rate
    
    return sorted(best_S_list)
>>>>>>> REPLACE
