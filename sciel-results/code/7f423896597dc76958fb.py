"""Optimized construction for max 3-AP, no 4-AP subset of [1,256]."""
import time
import random

N = 256

# Precompute 3-APs and 4-APs
# 3-AP: (a, b, c) such that a+b=2c? No, a, a+d, a+2d.
# 4-AP: a, a+d, a+2d, a+3d.

three_aps = []
four_aps = []

# Precompute membership lists for fast lookup
# three_ap_members[i] = list of indices of 3-APs containing i
# four_ap_members[i] = list of indices of 4-APs containing i

def precompute():
    global three_aps, four_aps
    three_aps = []
    four_aps = []
    
    # Generate all 3-APs
    for d in range(1, (N // 2) + 1):
        for a in range(1, N - 2*d + 1):
            b = a + d
            c = a + 2*d
            three_aps.append((a, b, c))
            
    # Generate all 4-APs
    for d in range(1, (N // 3) + 1):
        for a in range(1, N - 3*d + 1):
            b = a + d
            c = a + 2*d
            e = a + 3*d
            four_aps.append((a, b, c, e))

# Run precomputation once
precompute()

# Build membership indices
three_ap_members = [[] for _ in range(N+1)]
for idx, (a, b, c) in enumerate(three_aps):
    three_ap_members[a].append(idx)
    three_ap_members[b].append(idx)
    three_ap_members[c].append(idx)

four_ap_members = [[] for _ in range(N+1)]
for idx, (a, b, c, e) in enumerate(four_aps):
    four_ap_members[a].append(idx)
    four_ap_members[b].append(idx)
    four_ap_members[c].append(idx)
    four_ap_members[e].append(idx)

def count_3aps_fast(S_set):
    """Count 3-APs in S_set using precomputed list."""
    count = 0
    for a, b, c in three_aps:
        if a in S_set and b in S_set and c in S_set:
            count += 1
    return count

def is_valid_add(S_set, x):
    """Check if adding x to S_set creates a 4-AP."""
    # Check all 4-APs containing x
    for idx in four_ap_members[x]:
        a, b, c, e = four_aps[idx]
        # The other 3 elements must be in S_set
        others = [y for y in (a, b, c, e) if y != x]
        if all(y in S_set for y in others):
            return False
    return True

def get_3ap_gain(S_set, x):
    """Calculate how many new 3-APs are formed by adding x to S_set."""
    gain = 0
    for idx in three_ap_members[x]:
        a, b, c = three_aps[idx]
        others = [y for y in (a, b, c) if y != x]
        if all(y in S_set for y in others):
            gain += 1
    return gain

def get_3ap_loss(S_set, x):
    """Calculate how many 3-APs are lost by removing x from S_set."""
    loss = 0
    for idx in three_ap_members[x]:
        a, b, c = three_aps[idx]
        others = [y for y in (a, b, c) if y != x]
        if all(y in S_set for y in others):
            loss += 1
    return loss

def greedy_construct(seed):
    random.seed(seed)
    S_set = set()
    candidates = list(range(1, N+1))
    random.shuffle(candidates)
    
    for x in candidates:
        if is_valid_add(S_set, x):
            S_set.add(x)
            
    return S_set

def local_search(S_set, time_limit):
    """
    Local search with Add and Swap moves.
    """
    current_score = count_3aps_fast(S_set)
    start_time = time.time()
    deadline = start_time + time_limit
    
    best_S = set(S_set)
    best_score = current_score
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try Add moves
        best_add_gain = 0
        best_add_x = -1
        for x in range(1, N+1):
            if x in S_set:
                continue
            if is_valid_add(S_set, x):
                gain = get_3ap_gain(S_set, x)
                if gain > best_add_gain:
                    best_add_gain = gain
                    best_add_x = x
        
        if best_add_x != -1 and best_add_gain > 0:
            S_set.add(best_add_x)
            current_score += best_add_gain
            improved = True
            if current_score > best_score:
                best_score = current_score
                best_S = set(S_set)
            continue # Restart loop to re-evaluate adds/swaps
        
        # 2. Try Swap moves (x in S, y not in S)
        # To keep it fast, we limit the number of swaps checked or use a heuristic.
        # With N=256, checking all swaps is 256*256 = 65k checks.
        # Each check involves is_valid_add (which checks ~10-20 4-APs) and gain/loss calc.
        # This might be too slow if done every iteration.
        # We will only do swaps if we are stuck on adds.
        
        best_swap_delta = 0
        best_swap_x = -1
        best_swap_y = -1
        
        # Optimization: Only check swaps if we haven't improved in a while or randomly?
        # Let's just check a subset or all if fast enough.
        # is_valid_add for the new set S' = S - {x} + {y}
        # We can't just use is_valid_add(S_set, y) because removing x might break a 4-AP that y would complete?
        # No, removing x can only *remove* 4-APs, not create them.
        # So if S is 4-AP free, S' is 4-AP free IF y doesn't create a new 4-AP in the context of S-{x}.
        # Wait, y might form a 4-AP with elements in S-{x}.
        # The check is: Does S' contain a 4-AP?
        # Since S has no 4-APs, any 4-AP in S' must involve y.
        # So we just need to check if y creates a 4-AP in S-{x}.
        
        # We can speed this up by precomputing for each x, the 4-APs it participates in.
        # But let's just do the check.
        
        # To save time, we'll only check swaps every few iterations or if stuck.
        # Let's try all swaps but break early if time is tight.
        
        for x in S_set:
            # Potential loss from removing x
            loss_x = get_3ap_loss(S_set, x)
            
            for y in range(1, N+1):
                if y in S_set:
                    continue
                
                # Check if S' = S - {x} + {y} is valid
                # We need to check if y creates a 4-AP in S_temp = S - {x}
                # We can simulate this:
                # For each 4-AP containing y, check if the other 3 are in S and not x.
                valid = True
                for idx in four_ap_members[y]:
                    a, b, c, e = four_aps[idx]
                    others = [v for v in (a, b, c, e) if v != y]
                    # All others must be in S, and none can be x (since x is removed)
                    # Actually, if one of the others is x, then it's NOT in S'.
                    # So we need: all(v in S_set for v in others) AND x not in others?
                    # No, if x is in others, then x is not in S', so that 4-AP is not formed.
                    # So the condition for INVALID is: all(v in S_set for v in others) AND x not in others.
                    if all(v in S_set for v in others) and x not in others:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Calculate gain from adding y to S-{x}
                # Gain is number of 3-APs containing y where other 2 are in S-{x}
                gain_y = 0
                for idx in three_ap_members[y]:
                    a, b, c = three_aps[idx]
                    others = [v for v in (a, b, c) if v != y]
                    # Others must be in S and not x
                    if all(v in S_set for v in others) and x not in others:
                        gain_y += 1
                
                delta = gain_y - loss_x
                if delta > best_swap_delta:
                    best_swap_delta = delta
                    best_swap_x = x
                    best_swap_y = y
            
            # Optional: break inner loop if time is very tight
            if time.time() > deadline - 0.1:
                break
        
        if best_swap_x != -1 and best_swap_delta > 0:
            S_set.remove(best_swap_x)
            S_set.add(best_swap_y)
            current_score += best_swap_delta
            improved = True
            if current_score > best_score:
                best_score = current_score
                best_S = set(S_set)
        
        if not improved:
            # Stuck. Try a random perturbation? Or break.
            # Let's try a random swap to escape local optimum, accepting worse score?
            # No, let's just break to allow restarts.
            break
    
    return best_S

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    best_result = set()
    best_score = -1
    
    # Multiple restarts
    for restart in range(50):
        if time.time() > deadline:
            break
        
        seed = restart * 1000 + 42
        S_set = greedy_construct(seed)
        
        # Allocate remaining time to local search
        remaining = deadline - time.time()
        if remaining < 1.0:
            # Just evaluate this one
            score = count_3aps_fast(S_set)
            if score > best_score:
                best_score = score
                best_result = set(S_set)
            continue
            
        # Local search
        S_opt = local_search(S_set, remaining - 1.0)
        score = count_3aps_fast(S_opt)
        
        if score > best_score:
            best_score = score
            best_result = set(S_opt)
            
    return sorted(list(best_result))
