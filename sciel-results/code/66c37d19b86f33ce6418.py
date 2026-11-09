import time

N = 256
TIME_LIMIT = 85.0

# Precompute conflicts
conflicts = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(1, N + 1):
        if i == j:
            continue
        k1 = 2 * j - i
        if 1 <= k1 <= N and k1 != i and k1 != j:
            conflicts[i][j].add(k1)
            conflicts[j][i].add(k1)
        k2 = 2 * i - j
        if 1 <= k2 <= N and k2 != i and k2 != j:
            conflicts[i][j].add(k2)
            conflicts[j][i].add(k2)
        if (i + j) % 2 == 0:
            k3 = (i + j) // 2
            if 1 <= k3 <= N and k3 != i and k3 != j:
                conflicts[i][j].add(k3)
                conflicts[j][i].add(k3)

weights = [0.0] * (N + 1)
for i in range(1, N + 1):
    weights[i] = 1.0 / i

best_weight = 0.0
best_set = []
start_time = time.time()

def construct():
    global best_weight, best_set, start_time
    
    # Greedy Initial Solution
    curr_set = []
    forbidden = set()
    for x in range(1, N + 1):
        if x in forbidden:
            continue
        curr_set.append(x)
        for z in curr_set[:-1]:
            forbidden.update(conflicts[x][z])
            
    best_weight = sum(weights[x] for x in curr_set)
    best_set = list(curr_set)
    
    # B&B with Backtracking
    # State:
    # curr_set: list of chosen
    # forbidden: set of forbidden
    # idx: index in candidates (0 to N)
    # curr_weight: float
    # rem_weight: float (sum of weights of candidates[idx:] not in forbidden)
    
    # We need to track which elements are "decided" (in curr_set or skipped)
    # and which are "forbidden".
    # Candidates are 1..N.
    # We process in order 1..N.
    
    # To calculate rem_weight efficiently, we can maintain it.
    # Initial rem_weight = sum(weights[1:])
    
    # When we skip x: rem_weight -= weights[x]
    # When we take x: rem_weight -= weights[x] - sum(weights[y] for y in newly_forbidden if y > x)
    # Wait, if y < x, it was already processed.
    # So we only care about y > x.
    
    def solve(idx, curr_set, forbidden, curr_weight, rem_weight):
        global best_weight, best_set
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        if idx > N:
            if curr_weight > best_weight:
                best_weight = curr_weight
                best_set = list(curr_set)
            return
            
        x = idx
        
        # Prune
        if curr_weight + rem_weight <= best_weight:
            return
            
        # If x is forbidden, we MUST skip it.
        if x in forbidden:
            # Skip x
            # rem_weight already excludes x? 
            # In my logic, rem_weight is sum of weights of UNPROCESSED candidates not in forbidden.
            # If x is forbidden, it wasn't included in rem_weight?
            # Let's define rem_weight as sum of weights of all y in [idx, N] such that y not in forbidden.
            # If x is in forbidden, then x is not in rem_weight.
            # So we just move to idx+1 with same rem_weight.
            solve(idx + 1, curr_set, forbidden, curr_weight, rem_weight)
            return
            
        # Branch 1: Take x
        # New forbidden = forbidden U (union of conflicts[x][z] for z in curr_set)
        new_forbidden = set(forbidden)
        newly_added = []
        for z in curr_set:
            for y in conflicts[x][z]:
                if y not in new_forbidden:
                    new_forbidden.add(y)
                    if y > x: # Only affects rem_weight if y is in future candidates
                        newly_added.append(y)
        
        # Calculate new rem_weight
        # Old rem_weight included x and all future y not in old forbidden.
        # New rem_weight excludes x.
        # New rem_weight excludes any y > x that is in new_forbidden but was NOT in old_forbidden.
        new_rem_weight = rem_weight - weights[x]
        for y in newly_added:
            new_rem_weight -= weights[y]
            
        curr_set.append(x)
        solve(idx + 1, curr_set, new_forbidden, curr_weight + weights[x], new_rem_weight)
        curr_set.pop()
        
        # Branch 2: Skip x
        # x was in rem_weight (since it was not forbidden).
        # So new rem_weight = rem_weight - weights[x]
        solve(idx + 1, curr_set, forbidden, curr_weight, rem_weight - weights[x])

    # Initialize
    curr_set = []
    forbidden = set()
    curr_weight = 0.0
    rem_weight = sum(weights[1:])
    
    solve(1, curr_set, forbidden, curr_weight, rem_weight)
    
    return best_set
