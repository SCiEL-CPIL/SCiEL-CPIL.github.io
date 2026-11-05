import time
import random
import math

N = 256

# Precompute all 3-APs and 4-APs
three_aps = []  # list of (a, b, c) where a < b < c and b-a == c-b
four_aps = []   # list of (a, b, c, d) where a < b < c < d and b-a == c-b == d-c

for d in range(1, N // 2 + 1):
    for start in range(1, N - 2 * d + 1):
        a, b, c = start, start + d, start + 2 * d
        three_aps.append((a, b, c))

for d in range(1, N // 3 + 1):
    for start in range(1, N - 3 * d + 1):
        a, b, c, e = start, start + d, start + 2 * d, start + 3 * d
        four_aps.append((a, b, c, e))

# Build incidence structures
# For each element x, which 3-APs contain it?
three_ap_incidence = [[] for _ in range(N + 1)]
for idx, (a, b, c) in enumerate(three_aps):
    three_ap_incidence[a].append(idx)
    three_ap_incidence[b].append(idx)
    three_ap_incidence[c].append(idx)

# For each element x, which 4-APs contain it?
four_ap_incidence = [[] for _ in range(N + 1)]
for idx, (a, b, c, e) in enumerate(four_aps):
    four_ap_incidence[a].append(idx)
    four_ap_incidence[b].append(idx)
    four_ap_incidence[c].append(idx)
    four_ap_incidence[e].append(idx)

def count_3aps(A):
    """Count distinct 3-APs in set A (represented as a set of ints)."""
    count = 0
    for a, b, c in three_aps:
        if a in A and b in A and c in A:
            count += 1
    return count

def has_4ap(A):
    """Check if set A contains any 4-AP."""
    for a, b, c, e in four_aps:
        if a in A and b in A and c in A and e in A:
            return True
    return False

def greedy_construct(seed_val=0):
    """
    Density-based greedy: at each step, add the element x not in A that
    maximizes (number of new 3-APs formed) - lambda * (number of 4-AP conflicts created).
    We only add if no 4-AP is created.
    """
    rng = random.Random(seed_val)
    A = set()
    
    # Precompute for each x, the list of 3-APs containing x and 4-APs containing x
    # We'll compute gain and risk dynamically.
    
    candidates = list(range(1, N + 1))
    if seed_val % 2 == 0:
        rng.shuffle(candidates)
    else:
        # Sort by some heuristic: prefer elements with high 3-AP degree
        degree = [len(three_ap_incidence[x]) for x in range(N + 1)]
        candidates.sort(key=lambda x: (-degree[x], x))
    
    for x in candidates:
        if x in A:
            continue
        # Check 4-AP freedom
        valid = True
        for ap_idx in four_ap_incidence[x]:
            a, b, c, e = four_aps[ap_idx]
            others = [y for y in (a, b, c, e) if y != x]
            if all(y in A for y in others):
                valid = False
                break
        if not valid:
            continue
        # Count new 3-APs
        new_3aps = 0
        for ap_idx in three_ap_incidence[x]:
            a, b, c = three_aps[ap_idx]
            others = [y for y in (a, b, c) if y != x]
            if all(y in A for y in others):
                new_3aps += 1
        # Only add if it creates at least one new 3-AP, or if A is empty
        if new_3aps > 0 or len(A) == 0:
            A.add(x)
    
    return A

def simulated_annealing(A_init, deadline, seed_val=42):
    """
    Simulated annealing on 4-AP-free sets.
    Moves: add an element (if 4-AP free), remove an element, or swap.
    Objective: maximize number of 3-APs.
    """
    rng = random.Random(seed_val)
    A = set(A_init)
    current_score = count_3aps(A)
    best_A = set(A)
    best_score = current_score
    
    start_temp = 5.0
    end_temp = 0.1
    # Exponential cooling
    total_steps_estimate = 2000
    alpha = (end_temp / start_temp) ** (1.0 / max(total_steps_estimate, 1))
    
    temp = start_temp
    step = 0
    no_improve = 0
    
    while time.time() < deadline:
        if step % 100 == 0 and no_improve > 500:
            break
        step += 1
        
        # Generate a random move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try to add an element
            candidates = [x for x in range(1, N + 1) if x not in A]
            if not candidates:
                continue
            x = rng.choice(candidates)
            # Check 4-AP freedom
            valid = True
            for ap_idx in four_ap_incidence[x]:
                a, b, c, e = four_aps[ap_idx]
                others = [y for y in (a, b, c, e) if y != x]
                if all(y in A for y in others):
                    valid = False
                    break
            if not valid:
                continue
            # Compute delta
            delta = 0
            for ap_idx in three_ap_incidence[x]:
                a, b, c = three_aps[ap_idx]
                others = [y for y in (a, b, c) if y != x]
                if all(y in A for y in others):
                    delta += 1
            if delta > 0 or temp > 1.0:
                A.add(x)
                new_score = current_score + delta
                if new_score > current_score or rng.random() < math.exp(-(-delta) / temp):
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_A = set(A)
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    A.discard(x)
                    no_improve += 1
            else:
                no_improve += 1
                
        elif move_type < 0.7:
            # Try to remove an element
            if not A:
                continue
            x = rng.choice(list(A))
            # Compute delta (loss)
            delta = 0
            for ap_idx in three_ap_incidence[x]:
                a, b, c = three_aps[ap_idx]
                others = [y for y in (a, b, c) if y != x]
                if all(y in A for y in others):
                    delta += 1
            A.discard(x)
            new_score = current_score - delta
            if new_score > current_score or rng.random() < math.exp(-(-delta) / temp):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_A = set(A)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                A.add(x)
                no_improve += 1
                
        else:
            # Try a swap: remove x, add y
            if not A:
                continue
            x = rng.choice(list(A))
            candidates = [y for y in range(1, N + 1) if y not in A]
            if not candidates:
                continue
            y = rng.choice(candidates)
            
            # Check if adding y to A\{x} is 4-AP free
            A_temp = set(A)
            A_temp.discard(x)
            valid = True
            for ap_idx in four_ap_incidence[y]:
                a, b, c, e = four_aps[ap_idx]
                others = [z for z in (a, b, c, e) if z != y]
                if all(z in A_temp for z in others):
                    valid = False
                    break
            if not valid:
                no_improve += 1
                continue
            
            # Compute delta: loss from removing x, gain from adding y to A\{x}
            loss_x = 0
            for ap_idx in three_ap_incidence[x]:
                a, b, c = three_aps[ap_idx]
                others = [z for z in (a, b, c) if z != x]
                if all(z in A for z in others):
                    loss_x += 1
            
            gain_y = 0
            for ap_idx in three_ap_incidence[y]:
                a, b, c = three_aps[ap_idx]
                others = [z for z in (a, b, c) if z != y]
                if all(z in A_temp for z in others):
                    gain_y += 1
            
            delta = gain_y - loss_x
            A.discard(x)
            A.add(y)
            new_score = current_score + delta
            if new_score > current_score or rng.random() < math.exp(-(-delta) / temp):
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_A = set(A)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                A.discard(y)
                A.add(x)
                no_improve += 1
        
        temp *= alpha
        if temp < end_temp:
            temp = end_temp
    
    return best_A

def local_polish(A, deadline):
    """
    Fast local search polish: try all single additions and swaps that improve score.
    """
    A = set(A)
    current_score = count_3aps(A)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try additions
        for x in range(1, N + 1):
            if x in A:
                continue
            valid = True
            for ap_idx in four_ap_incidence[x]:
                a, b, c, e = four_aps[ap_idx]
                others = [y for y in (a, b, c, e) if y != x]
                if all(y in A for y in others):
                    valid = False
                    break
            if not valid:
                continue
            delta = 0
            for ap_idx in three_ap_incidence[x]:
                a, b, c = three_aps[ap_idx]
                others = [y for y in (a, b, c) if y != x]
                if all(y in A for y in others):
                    delta += 1
            if delta > 0:
                A.add(x)
                current_score += delta
                improved = True
                break
        if improved:
            continue
        # Try swaps
        for x in list(A):
            for y in range(1, N + 1):
                if y in A or y == x:
                    continue
                A_temp = set(A)
                A_temp.discard(x)
                valid = True
                for ap_idx in four_ap_incidence[y]:
                    a, b, c, e = four_aps[ap_idx]
                    others = [z for z in (a, b, c, e) if z != y]
                    if all(z in A_temp for z in others):
                        valid = False
                        break
                if not valid:
                    continue
                loss_x = 0
                for ap_idx in three_ap_incidence[x]:
                    a, b, c = three_aps[ap_idx]
                    others = [z for z in (a, b, c) if z != x]
                    if all(z in A for z in others):
                        loss_x += 1
                gain_y = 0
                for ap_idx in three_ap_incidence[y]:
                    a, b, c = three_aps[ap_idx]
                    others = [z for z in (a, b, c) if z != y]
                    if all(z in A_temp for z in others):
                        gain_y += 1
                if gain_y > loss_x:
                    A.discard(x)
                    A.add(y)
                    current_score += (gain_y - loss_x)
                    improved = True
                    break
            if improved:
                break
    return A

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_A = set()
    best_score = -1
    
    # Generate diverse initial candidates
    candidates = []
    
    # Deterministic greedy
    candidates.append(greedy_construct(0))
    
    # Randomized greedies
    for i in range(1, 15):
        if time.time() > deadline - 30:
            break
        candidates.append(greedy_construct(i))
    
    # Evaluate and optimize each candidate
    for idx, init_A in enumerate(candidates):
        if time.time() > deadline - 15:
            break
        
        s = count_3aps(init_A)
        
        # Simulated annealing
        if time.time() < deadline - 10:
            sa_deadline = min(deadline - 5, time.time() + 5.0)
            sa_A = simulated_annealing(init_A, sa_deadline, seed_val=100 + idx)
            sa_score = count_3aps(sa_A)
            if sa_score > best_score:
                best_score = sa_score
                best_A = set(sa_A)
        
        # Local polish on the best so far
        if time.time() < deadline - 2:
            polish_deadline = min(deadline - 1, time.time() + 3.0)
            polished_A = local_polish(best_A, polish_deadline)
            pol_score = count_3aps(polished_A)
            if pol_score > best_score:
                best_score = pol_score
                best_A = polished_A
    
    # Final polish
    if time.time() < deadline - 1 and best_A:
        polished_A = local_polish(best_A, deadline - 0.5)
        pol_score = count_3aps(polished_A)
        if pol_score > best_score:
            best_A = polished_A
    
    if not best_A:
        best_A = candidates[0] if candidates else {1}
    
    return sorted(best_A)

# EVOLVE-BLOCK-END
