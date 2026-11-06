import time
import random
import numpy as np

def construct():
    start_time = time.time()
    n = 256
    # Safety margin: 90s limit, target 85s
    deadline = start_time + 85.0

    # ---------- Precomputation ----------
    # 0-indexed elements 0 to n-1 representing 1 to n
    
    # All 3-APs
    aps3 = []
    for d in range(1, n // 2 + 1):
        for a in range(0, n - 2 * d):
            b = a + d
            c = a + 2 * d
            aps3.append((a, b, c))
            
    # All 4-APs
    aps4 = []
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(0, n - 3 * d):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps4.append((a, b, c, e))

    # partners3[x] = list of (u, v) such that {x, u, v} is a 3-AP
    partners3 = [[] for _ in range(n)]
    for a, b, c in aps3:
        partners3[a].append((b, c))
        partners3[b].append((a, c))
        partners3[c].append((a, b))

    # partners4[x] = list of (u, v, w) such that {x, u, v, w} is a 4-AP
    partners4 = [[] for _ in range(n)]
    for a, b, c, d in aps4:
        partners4[a].append((b, c, d))
        partners4[b].append((a, c, d))
        partners4[c].append((a, b, d))
        partners4[d].append((a, b, c))

    # Degrees for heuristic ordering
    deg3 = [len(p) for p in partners3]
    deg4 = [len(p) for p in partners4]

    # ---------- Core Operations ----------
    
    def count_3aps(S):
        cnt = 0
        for a, b, c in aps3:
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1:
                cnt += 1
        return cnt

    def can_add(S, x):
        """Check if adding x to S creates a 4-AP."""
        for u, v, w in partners4[x]:
            if (S >> u) & 1 and (S >> v) & 1 and (S >> w) & 1:
                return False
        return True

    def gain_add(S, x):
        """Number of new 3-APs formed by adding x to S."""
        g = 0
        for u, v in partners3[x]:
            if (S >> u) & 1 and (S >> v) & 1:
                g += 1
        return g

    def loss_remove(S, x):
        """Number of 3-APs lost by removing x from S."""
        g = 0
        for u, v in partners3[x]:
            if (S >> u) & 1 and (S >> v) & 1:
                g += 1
        return g

    # ---------- Hill Climbing (Steepest Ascent) ----------
    
    def hill_climb(S, current_score, time_limit):
        """
        Iteratively find the best improving move (add, remove, or swap).
        """
        S = int(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            best_delta = 0
            best_move = None
            
            # 1. Try Adds
            for x in range(n):
                if (S >> x) & 1:
                    continue
                if not can_add(S, x):
                    continue
                g = gain_add(S, x)
                if g > best_delta:
                    best_delta = g
                    best_move = ('add', x, -1)
            
            # 2. Try Removes
            for x in range(n):
                if not ((S >> x) & 1):
                    continue
                l = loss_remove(S, x)
                delta = -l
                if delta > best_delta:
                    best_delta = delta
                    best_move = ('remove', x, -1)
            
            # 3. Try Swaps
            # To optimize, we can precompute lists of elements in and out of S
            in_set = [x for x in range(n) if (S >> x) & 1]
            out_set = [x for x in range(n) if not ((S >> x) & 1)]
            
            for u in in_set:
                # Precompute loss for u
                loss_u = loss_remove(S, u)
                S_wo_u = S & ~(1 << u)
                
                for v in out_set:
                    if v == u: # Should not happen
                        continue
                    
                    # Check validity of adding v to S_wo_u
                    valid = True
                    for a, b, c in partners4[v]:
                        # Check if all a,b,c are in S_wo_u
                        if ((S_wo_u >> a) & 1) and ((S_wo_u >> b) & 1) and ((S_wo_u >> c) & 1):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Calculate gain for v in S_wo_u
                    gain_v = 0
                    for a, b in partners3[v]:
                        if ((S_wo_u >> a) & 1) and ((S_wo_u >> b) & 1):
                            gain_v += 1
                    
                    delta = gain_v - loss_u
                    if delta > best_delta:
                        best_delta = delta
                        best_move = ('swap', v, u)
            
            if best_move is not None and best_delta > 0:
                mtype, v, u = best_move
                if mtype == 'add':
                    S |= (1 << v)
                elif mtype == 'remove':
                    S &= ~(1 << v)
                else: # swap
                    S &= ~(1 << u)
                    S |= (1 << v)
                current_score += best_delta
                improved = True

        return S, current_score

    # ---------- Simulated Annealing (Fast Random Moves) ----------
    
    def sa_search(S, current_score, time_limit):
        S = int(S)
        T = 5.0
        T_min = 0.1
        cooling = 0.99
        best_S = S
        best_score = current_score
        
        while time.time() < time_limit:
            if T < T_min:
                break
            
            move_type = random.random()
            in_set = [x for x in range(n) if (S >> x) & 1]
            out_set = [x for x in range(n) if not ((S >> x) & 1)]
            
            if not in_set or not out_set:
                break

            if move_type < 0.3:
                # Add
                x = random.choice(out_set)
                if can_add(S, x):
                    S_new = S | (1 << x)
                    delta = gain_add(S, x)
                else:
                    continue
            elif move_type < 0.6:
                # Remove
                x = random.choice(in_set)
                S_new = S & ~(1 << x)
                delta = -loss_remove(S, x)
            elif move_type < 0.9:
                # Swap
                u = random.choice(in_set)
                v = random.choice(out_set)
                S_wo_u = S & ~(1 << u)
                if can_add(S_wo_u, v):
                    S_new = S_wo_u | (1 << v)
                    # Approximate delta: gain(v in S_wo_u) - loss(u in S)
                    # Note: loss(u in S) is calculated on S, not S_wo_u, but u is removed so it's fine.
                    # gain(v in S_wo_u) needs to be calculated on S_wo_u
                    gain_v = 0
                    for a, b in partners3[v]:
                        if ((S_wo_u >> a) & 1) and ((S_wo_u >> b) & 1):
                            gain_v += 1
                    loss_u = loss_remove(S, u)
                    delta = gain_v - loss_u
                else:
                    continue
            else:
                # 2-Swap (Remove 2, Add 2)
                if len(in_set) < 2 or len(out_set) < 2:
                    continue
                u1, u2 = random.sample(in_set, 2)
                v1, v2 = random.sample(out_set, 2)
                S_wo_2 = S & ~(1 << u1) & ~(1 << u2)
                if can_add(S_wo_2, v1) and can_add(S_wo_2 | (1 << v1), v2):
                    S_new = S_wo_2 | (1 << v1) | (1 << v2)
                    # Calculate delta
                    # Loss of u1, u2
                    l1 = loss_remove(S, u1)
                    l2 = loss_remove(S & ~(1 << u1), u2)
                    # Gain of v1, v2
                    g1 = 0
                    for a, b in partners3[v1]:
                        if ((S_wo_2 >> a) & 1) and ((S_wo_2 >> b) & 1):
                            g1 += 1
                    g2 = 0
                    S_tmp = S_wo_2 | (1 << v1)
                    for a, b in partners3[v2]:
                        if ((S_tmp >> a) & 1) and ((S_tmp >> b) & 1):
                            g2 += 1
                    delta = (g1 + g2) - (l1 + l2)
                else:
                    continue

            if delta > 0 or random.random() < np.exp(delta / T):
                S = S_new
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_S = S
            
            T *= cooling
        
        return best_S, best_score

    # ---------- Initialization Strategies ----------
    
    def greedy_construct(order, use_score=True):
        S = 0
        for x in order:
            if not can_add(S, x):
                continue
            if use_score:
                g = gain_add(S, x)
                if g > 0:
                    S |= (1 << x)
            else:
                S |= (1 << x)
        return S

    def prune_from_full():
        S = (1 << n) - 1
        # We need to break all 4-APs.
        # Iterate through 4-APs, if still present, remove one element with min loss.
        # To be efficient, we can just iterate through aps4 once.
        for ap in aps4:
            a, b, c, d = ap
            if (S >> a) & 1 and (S >> b) & 1 and (S >> c) & 1 and (S >> d) & 1:
                min_loss = float('inf')
                min_elem = a
                for elem in ap:
                    l = loss_remove(S, elem)
                    if l < min_loss:
                        min_loss = l
                        min_elem = elem
                S &= ~(1 << min_elem)
        return S

    # ---------- Main Search Loop ----------
    
    best_S = 0
    best_score = 0
    
    # Generate initial orderings
    orderings = []
    
    # 1. Degree based (high 3-AP degree first)
    order1 = sorted(range(n), key=lambda x: (-deg3[x], x))
    orderings.append(order1)
    
    # 2. Net benefit (3-AP degree - 4-AP degree)
    order2 = sorted(range(n), key=lambda x: (-(deg3[x] - deg4[x]), x))
    orderings.append(order2)
    
    # 3. Natural
    orderings.append(list(range(n)))
    
    # 4. Reverse Natural
    orderings.append(list(range(n-1, -1, -1)))
    
    # 5-12. Random
    for i in range(8):
        rng = random.Random(100 + i)
        ord = list(range(n))
        rng.shuffle(ord)
        orderings.append(ord)

    # Run Hill Climbing on each initialization
    for i, order in enumerate(orderings):
        if time.time() > deadline - 5:
            break
        
        S = greedy_construct(order, use_score=True)
        sc = count_3aps(S)
        
        # Hill Climb
        time_per_hc = max(1.0, (deadline - 5 - time.time()) / (len(orderings) - i))
        S, sc = hill_climb(S, sc, time.time() + time_per_hc)
        
        # SA Polish
        if time.time() < deadline - 2:
            S, sc = sa_search(S, sc, time.time() + min(2.0, deadline - time.time() - 1))
            
        if sc > best_score:
            best_score = sc
            best_S = S

    # Strategy: Prune from Full
    if time.time() < deadline - 15:
        S_p = prune_from_full()
        sc_p = count_3aps(S_p)
        
        # Hill Climb
        time_per_hc = max(1.0, (deadline - 5 - time.time()) / 2)
        S_p, sc_p = hill_climb(S_p, sc_p, time.time() + time_per_hc)
        
        # SA Polish
        if time.time() < deadline - 2:
            S_p, sc_p = sa_search(S_p, sc_p, time.time() + min(2.0, deadline - time.time() - 1))
            
        if sc_p > best_score:
            best_score = sc_p
            best_S = S_p

    # Final Polish on Best
    if time.time() < deadline - 1 and best_S:
        best_S, best_score = hill_climb(best_S, best_score, time.time() + 1.0)
        if time.time() < deadline - 0.1:
            best_S, best_score = sa_search(best_S, best_score, deadline - 0.1)

    # Convert to 1-based list
    result = [x + 1 for x in range(n) if (best_S >> x) & 1]
    if not result:
        result = [1]
        
    return result
