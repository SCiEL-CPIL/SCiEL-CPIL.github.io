"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    n = 512
    deadline = time.time() + 55.0  # Safety margin for 60s limit

    # Precompute all 4-term arithmetic progressions (as tuples of 4 positions 1-indexed)
    # and for each position, the list of AP indices that contain it
    aps = []  # list of tuples (a, b, c, e) where a+d=b+c, i.e., (a, a+d, a+2d, a+3d)
    pos_to_aps = [[] for _ in range(n + 1)]  # 1-indexed: pos -> list of AP indices
    
    ap_id = 0
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                aps.append((a, b, c, e))
                for pos in (a, b, c, e):
                    pos_to_aps[pos].append(ap_id)
                ap_id += 1
    
    num_aps = ap_id
    if num_aps == 0:
        return list(range(1, n + 1))

    # Precompute for each AP, the 4 positions
    # aps is already a list of tuples

    # Greedy initial construction
    def greedy_greedy_order(order):
        """Greedy add elements in given order."""
        selected = set()
        # For fast checking, we'll use the pos_to_aps structure
        # When adding x, check all APs containing x: are the other 3 already selected?
        for x in order:
            valid = True
            for ap_idx in pos_to_aps[x]:
                a, b, c, e = aps[ap_idx]
                # Check if the other 3 are in selected
                if a == x:
                    if b in selected and c in selected and e in selected:
                        valid = False
                        break
                elif b == x:
                    if a in selected and c in selected and e in selected:
                        valid = False
                        break
                elif c == x:
                    if a in selected and b in selected and e in selected:
                        valid = False
                        break
                else:  # e == x
                    if a in selected and b in selected and c in selected:
                        valid = False
                        break
            if valid:
                selected.add(x)
        return selected

    # Deterministic baseline
    best_set = greedy_greedy_order(range(1, n + 1))
    best_size = len(best_set)

    # Randomized greedy with restarts
    candidates = list(range(1, n + 1))
    
    while time.time() < deadline:
        random.shuffle(candidates)
        selected = greedy_greedy_order(candidates)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = set(selected)
        
        # Try some local search improvements (swaps)
        # For each x in selected, try removing it and adding some y not in selected
        # This is expensive, so do it sparingly
        if time.time() > deadline - 5.0:
            break
            
        # Try a few random swaps
        for _ in range(10):
            if time.time() > deadline - 3.0:
                break
            if not selected:
                break
            x = random.choice(list(selected))
            selected.remove(x)
            # Try to add random y's
            added = False
            for y in random.sample(candidates, min(20, len(candidates))):
                if y in selected:
                    continue
                valid = True
                for ap_idx in pos_to_aps[y]:
                    a, b, c, e = aps[ap_idx]
                    if a == y:
                        if b in selected and c in selected and e in selected:
                            valid = False
                            break
                    elif b == y:
                        if a in selected and c in selected and e in selected:
                            valid = False
                            break
                    elif c == y:
                        if a in selected and b in selected and e in selected:
                            valid = False
                            break
                    else:
                        if a in selected and b in selected and c in selected:
                            valid = False
                            break
                if valid:
                    selected.add(y)
                    added = True
                    break
            if not added:
                selected.add(x)  # put x back
            
            if len(selected) > best_size:
                best_size = len(selected)
                best_set = set(selected)

    return sorted(best_set)

# EVOLVE-BLOCK-END
