import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    time_limit = 85.0  # Safety margin (90s wall-clock limit)
    
    # Precompute 4-APs and their conflict structure
    four_aps = []
    for a in range(1, N - 2):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= N:
                four_aps.append((a, b, c, e))
    
    # For each element, list of 4-APs it participates in
    elem_4aps = {i: [] for i in range(1, N + 1)}
    for idx, ap in enumerate(four_aps):
        for elem in ap:
            elem_4aps[elem].append(idx)
    
    # Precompute 3-AP partner lists for incremental scoring
    # mid_partners[x]: list of (a, c) where a < x < c and a+c=2x (x is middle)
    # left_partners[x]: list of (b, c) where x < b < c and 2b=x+c (x is first)
    # right_partners[x]: list of (a, b) where a < b < x and 2b=a+x (x is last)
    
    mid_partners = [[] for _ in range(N + 1)]
    left_partners = [[] for _ in range(N + 1)]
    right_partners = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        for c in range(a + 2, N + 1, 2):
            x = (a + c) // 2
            if 1 <= x <= N:
                mid_partners[x].append((a, c))
    
    for x in range(1, N + 1):
        for b in range(x + 1, N + 1):
            c = 2 * b - x
            if c > b and c <= N:
                left_partners[x].append((b, c))
    
    for x in range(1, N + 1):
        for b in range(1, x):
            a = 2 * b - x
            if a > 0 and a < b:
                right_partners[x].append((a, b))

    def has_4ap_conflict(elem, A_set):
        """Check if adding elem to A_set creates a 4-AP."""
        for ap_idx in elem_4aps[elem]:
            ap = four_aps[ap_idx]
            others = [v for v in ap if v != elem]
            if all(o in A_set for o in others):
                return True
        return False

    def count_3aps(A_set):
        """Count 3-APs in A_set."""
        count = 0
        for x in A_set:
            for a, c in mid_partners[x]:
                if a in A_set and c in A_set:
                    count += 1
        return count

    def gain_add(x, A_set):
        """Compute gain in 3-APs when adding x to A_set."""
        gain = 0
        for a, c in mid_partners[x]:
            if a in A_set and c in A_set:
                gain += 1
        for b, c in left_partners[x]:
            if b in A_set and c in A_set:
                gain += 1
        for a, b in right_partners[x]:
            if a in A_set and b in A_set:
                gain += 1
        return gain

    def loss_remove(x, A_set):
        """Compute loss in 3-APs when removing x from A_set."""
        loss = 0
        for a, c in mid_partners[x]:
            if a in A_set and c in A_set:
                loss += 1
        for b, c in left_partners[x]:
            if b in A_set and c in A_set:
                loss += 1
        for a, b in right_partners[x]:
            if a in A_set and b in A_set:
                loss += 1
        return loss

    # Build initial solution using greedy with lookahead
    def build_initial():
        A = set()
        # Greedy: add elements in order of maximum 3-AP gain
        remaining = list(range(1, N + 1))
        random.shuffle(remaining)
        
        for x in remaining:
            if not has_4ap_conflict(x, A):
                A.add(x)
        
        # Improve with local search
        improved = True
        while improved:
            improved = False
            # Try adding
            for x in range(1, N + 1):
                if x not in A and not has_4ap_conflict(x, A):
                    g = gain_add(x, A)
                    if g > 0:
                        A.add(x)
                        improved = True
            # Try removing
            for x in list(A):
                l = loss_remove(x, A)
                if l < 0:  # removing increases score (shouldn't happen but check)
                    A.discard(x)
                    improved = True
                elif l == 0:
                    # Check if removing frees up space for better additions
                    A_test = set(A)
                    A_test.discard(x)
                    best_add = -1
                    best_gain = 0
                    for y in range(1, N + 1):
                        if y not in A_test and not has_4ap_conflict(y, A_test):
                            g = gain_add(y, A_test)
                            if g > best_gain:
                                best_gain = g
                                best_add = y
                    if best_add > 0 and best_gain > l:
                        A.discard(x)
                        A.add(best_add)
                        improved = True
        return A

    # Parallel tempering setup
    NUM_REPLICAS = 5
    base_temps = [0.05, 0.2, 0.5, 1.5, 4.0]
    
    # Initialize replicas
    replicas = []
    replica_scores = []
    best_A = set()
    best_score = -1
    
    for i in range(NUM_REPLICAS):
        if i == 0:
            A_rep = build_initial()
        else:
            # Create diverse initial solutions
            A_rep = set()
            # Use different orderings
            order = list(range(1, N + 1))
            random.shuffle(order)
            for x in order:
                if not has_4ap_conflict(x, A_rep):
                    A_rep.add(x)
        
        score = count_3aps(A_rep)
        replicas.append(A_rep)
        replica_scores.append(score)
        
        if score > best_score:
            best_score = score
            best_A = set(A_rep)
    
    # Parallel tempering loop
    T = base_temps[:]
    exchange_interval = 10  # Try exchange every 10 iterations per replica
    
    iteration = 0
    while time.time() - start_time < time_limit:
        for r in range(NUM_REPLICAS):
            if time.time() - start_time > time_limit:
                break
            
            A = replicas[r]
            score = replica_scores[r]
            temp = T[r]
            
            # Choose move type
            move_type = random.random()
            new_A = set(A)
            new_score = score
            accepted = False
            
            if move_type < 0.4:
                # Single add
                candidates = [x for x in range(1, N + 1) if x not in A]
                if candidates:
                    x = random.choice(candidates)
                    if not has_4ap_conflict(x, A):
                        g = gain_add(x, A)
                        new_A.add(x)
                        new_score = score + g
                        if g >= 0 or random.random() < math.exp(g / temp):
                            accepted = True
                            
            elif move_type < 0.7:
                # Single remove
                if A:
                    x = random.choice(list(A))
                    new_A.discard(x)
                    l = loss_remove(x, A)
                    new_score = score - l
                    if l <= 0 or random.random() < math.exp(-l / temp):
                        accepted = True
                        
            else:
                # 2-swap: remove one, add one
                if A:
                    x = random.choice(list(A))
                    candidates = [y for y in range(1, N + 1) if y not in A]
                    if candidates:
                        y = random.choice(candidates)
                        new_A.discard(x)
                        if not has_4ap_conflict(y, new_A):
                            # Loss from removing x (computed on A without x)
                            l = loss_remove(x, A)
                            # Gain from adding y (computed on A without x)
                            g = gain_add(y, new_A)
                            # But we need to account for interactions between x and y
                            # Simplified: just use the net
                            new_score = score - l + g
                            new_A.add(y)
                            net = new_score - score
                            if net >= 0 or random.random() < math.exp(net / temp):
                                accepted = True
            
            if accepted:
                replicas[r] = new_A
                replica_scores[r] = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_A = set(new_A)
            
            iteration += 1
            
            # Parallel tempering: try exchange with neighbor
            if iteration % exchange_interval == 0:
                if r < NUM_REPLICAS - 1:
                    r2 = r + 1
                    # Metropolis criterion for exchange
                    delta = (replica_scores[r] / T[r]) - (replica_scores[r2] / T[r2])
                    # Actually: accept if exp(-delta) > random, where delta = (S2/T2 - S1/T1)
                    # Let's use: accept swap with prob min(1, exp(-(S2/T2 - S1/T1)))
                    arg = -(replica_scores[r2] / T[r2] - replica_scores[r] / T[r])
                    if arg > 0 or random.random() < math.exp(min(arg, 50)):
                        # Swap states
                        replicas[r], replicas[r2] = replicas[r2], replicas[r]
                        replica_scores[r], replica_scores[r2] = replica_scores[r2], replica_scores[r]
    
    return sorted(best_A)

# EVOLVE-BLOCK-END
